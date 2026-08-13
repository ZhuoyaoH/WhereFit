#!/usr/bin/env python3
"""Build reusable ERA5 historical-weather caches for every WhereFit city."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_PATH = ROOT / "data" / "city_seed.csv"
DEFAULT_OUTPUT_DIR = ROOT / "release" / "modelscope-studio" / "data" / "cache" / "weather" / "history"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
MODEL = "era5"
CHUNK_YEARS = 5
DAILY_VARIABLES = (
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_mean",
    "apparent_temperature_max",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
)


def parse_date(value: str) -> date:
    """Parse an ISO date for command-line arguments."""

    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def default_end_date() -> date:
    """Return the latest complete historical day allowed by Open-Meteo."""

    return date.today() - timedelta(days=10)


def date_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Split an inclusive date range into deterministic five-year chunks."""

    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = date(min(cursor.year + CHUNK_YEARS - 1, end.year), 12, 31)
        if chunk_end > end:
            chunk_end = end
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def safe_name(value: str) -> str:
    """Convert a city name into the cache-file form used by the application."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def cache_path(output_dir: Path, city_en: str, start: date, end: date, layout: str) -> Path:
    """Return an application-compatible ERA5 cache path for the selected layout."""

    directory = output_dir / "chunks" if layout == "chunks" else output_dir
    return directory / f"{safe_name(city_en)}_{start.isoformat()}_{end.isoformat()}_{MODEL}_daily.csv"


def valid_chunk(path: Path, expected_start: date, expected_end: date) -> bool:
    """Check that an existing CSV contains a complete contiguous daily range."""

    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or set(("time", *DAILY_VARIABLES)) - set(reader.fieldnames):
                return False
            days = [datetime.strptime(str(row["time"]), "%Y-%m-%d").date() for row in reader]
    except (OSError, ValueError, csv.Error):
        return False
    expected_count = (expected_end - expected_start).days + 1
    return len(days) == expected_count and days[0] == expected_start and days[-1] == expected_end


def fetch_daily(
    city: dict[str, str],
    start: date,
    end: date,
    timeout: int,
    retries: int,
    request_delay: float,
) -> dict[str, list[object]]:
    """Fetch and validate one ERA5 daily response with bounded retries."""

    params = {
        "latitude": city["latitude"],
        "longitude": city["longitude"],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "auto",
        "models": MODEL,
        "wind_speed_unit": "ms",
    }
    request = Request(f"{ARCHIVE_URL}?{urlencode(params)}", headers={"User-Agent": "WhereFit-history-builder/1.0"})
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if request_delay > 0:
                time.sleep(request_delay)
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read())
            daily = payload.get("daily") if isinstance(payload, dict) else None
            if not isinstance(daily, dict):
                raise ValueError("response has no daily payload")
            missing = [column for column in ("time", *DAILY_VARIABLES) if column not in daily]
            if missing:
                raise ValueError(f"response is missing fields: {missing}")
            lengths = {len(daily[column]) for column in ("time", *DAILY_VARIABLES)}
            expected_count = (end - start).days + 1
            if lengths != {expected_count}:
                raise ValueError(f"response has unexpected daily lengths: {sorted(lengths)}")
            return {column: list(daily[column]) for column in ("time", *DAILY_VARIABLES)}
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                retry_after = exc.headers.get("Retry-After") if isinstance(exc, HTTPError) else None
                try:
                    pause = float(retry_after) if retry_after else min(60, 2 ** attempt)
                except ValueError:
                    pause = min(60, 2 ** attempt)
                time.sleep(pause)
    raise RuntimeError(f"{city['city']}: request failed after {retries + 1} attempts: {last_error}")


def write_chunk(path: Path, daily: dict[str, list[object]]) -> None:
    """Atomically write one validated cache chunk in the application's CSV schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    try:
        with partial.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("time", *DAILY_VARIABLES))
            writer.writeheader()
            for index in range(len(daily["time"])):
                writer.writerow({column: daily[column][index] for column in ("time", *DAILY_VARIABLES)})
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def build_one(
    city: dict[str, str],
    start: date,
    end: date,
    output_dir: Path,
    layout: str,
    timeout: int,
    retries: int,
    request_delay: float,
) -> tuple[str, int, int]:
    """Build one city cache and return its reuse and download counts."""

    reused = 0
    fetched = 0
    ranges = date_chunks(start, end) if layout == "chunks" else [(start, end)]
    for chunk_start, chunk_end in ranges:
        output = cache_path(output_dir, city["city_en"], chunk_start, chunk_end, layout)
        if valid_chunk(output, chunk_start, chunk_end):
            reused += 1
            continue
        daily = fetch_daily(city, chunk_start, chunk_end, timeout, retries, request_delay)
        write_chunk(output, daily)
        fetched += 1
    return city["city"], reused, fetched


def read_cities(path: Path) -> list[dict[str, str]]:
    """Read the full city seed while enforcing the fields used by the API."""

    with path.open(newline="", encoding="utf-8") as handle:
        cities = list(csv.DictReader(handle))
    required = {"city", "city_en", "latitude", "longitude"}
    if not cities or required - set(cities[0]):
        raise ValueError(f"seed file lacks required columns: {sorted(required)}")
    return cities


def main() -> int:
    """Parse arguments, build all city caches concurrently, and report failures."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", type=parse_date, default=date(2000, 1, 1))
    parser.add_argument("--end-date", type=parse_date, default=default_end_date())
    parser.add_argument(
        "--layout",
        choices=("full", "chunks"),
        default="full",
        help="write one complete cache per city (default) or application chunk files",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument(
        "--city-offset",
        type=int,
        default=0,
        help="skip this many cities from the beginning of the seed list",
    )
    parser.add_argument(
        "--city-limit",
        type=int,
        default=None,
        help="build at most this many cities after applying --city-offset",
    )
    args = parser.parse_args()
    if args.end_date < args.start_date:
        parser.error("--end-date must not be earlier than --start-date")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.request_delay < 0:
        parser.error("--request-delay must not be negative")
    if args.city_offset < 0:
        parser.error("--city-offset must not be negative")
    if args.city_limit is not None and args.city_limit < 1:
        parser.error("--city-limit must be at least 1")

    all_cities = read_cities(args.seed)
    cities = all_cities[args.city_offset :]
    if args.city_limit is not None:
        cities = cities[: args.city_limit]
    if not cities:
        parser.error("selected city range is empty")
    files_per_city = len(date_chunks(args.start_date, args.end_date)) if args.layout == "chunks" else 1
    print(
        f"Building {len(cities)} of {len(all_cities)} cities × {files_per_city} ERA5 {args.layout} cache files into {args.output_dir}",
        flush=True,
    )
    failures: list[str] = []
    reused_total = 0
    fetched_total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                build_one,
                city,
                args.start_date,
                args.end_date,
                args.output_dir,
                args.layout,
                args.timeout,
                args.retries,
                args.request_delay,
            ): city
            for city in cities
        }
        for future in as_completed(futures):
            city = futures[future]
            try:
                name, reused, fetched = future.result()
                reused_total += reused
                fetched_total += fetched
                print(f"{name}: reused {reused}, fetched {fetched}", flush=True)
            except Exception as exc:
                failures.append(f"{city['city']}: {exc}")
                print(f"FAILED {city['city']}: {exc}", file=sys.stderr, flush=True)
    print(f"Completed: reused {reused_total}, fetched {fetched_total}, failed {len(failures)}", flush=True)
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
