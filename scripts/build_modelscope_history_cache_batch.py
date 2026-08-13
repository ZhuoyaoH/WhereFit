#!/usr/bin/env python3
"""Download all city ERA5 daily histories through one multi-coordinate request."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from build_modelscope_history_cache import (
    ARCHIVE_URL,
    DAILY_VARIABLES,
    DEFAULT_OUTPUT_DIR,
    MODEL,
    cache_path,
    default_end_date,
    parse_date,
    read_cities,
    valid_chunk,
    write_chunk,
)

BATCH_SIZE = 5
REQUEST_DELAY_SECONDS = 30.0
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_PAUSE_SECONDS = 60.0


def city_batches(cities: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    """Split pending cities into API-safe, fixed-size coordinate batches."""

    return [cities[index : index + BATCH_SIZE] for index in range(0, len(cities), BATCH_SIZE)]


def fetch_batch(
    cities: list[dict[str, str]], start: date, end: date
) -> list[tuple[dict[str, str], dict[str, list[object]]]]:
    """Request one coordinate batch and return only fully validated daily payloads."""

    params = {
        "latitude": ",".join(city["latitude"] for city in cities),
        "longitude": ",".join(city["longitude"] for city in cities),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "auto",
        "models": MODEL,
        "wind_speed_unit": "ms",
    }
    request = Request(f"{ARCHIVE_URL}?{urlencode(params)}", headers={"User-Agent": "WhereFit-history-builder/1.0"})
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            with urlopen(request, timeout=300) as response:
                payload = json.loads(response.read())
            break
        except HTTPError as exc:
            if exc.code != 429 or attempt >= RATE_LIMIT_RETRIES:
                raise
            retry_after = exc.headers.get("Retry-After")
            try:
                pause = max(RATE_LIMIT_PAUSE_SECONDS, float(retry_after)) if retry_after else RATE_LIMIT_PAUSE_SECONDS
            except ValueError:
                pause = RATE_LIMIT_PAUSE_SECONDS
            print(f"Rate limited; waiting {pause:.0f} seconds before retry {attempt + 1}/{RATE_LIMIT_RETRIES}.", flush=True)
            time.sleep(pause)
    responses = payload if isinstance(payload, list) else [payload]
    if len(responses) != len(cities):
        raise RuntimeError(f"expected {len(cities)} responses, received {len(responses)}")
    expected_count = (end - start).days + 1
    prepared: list[tuple[dict[str, str], dict[str, list[object]]]] = []
    for city, item in zip(cities, responses):
        daily = item.get("daily") if isinstance(item, dict) else None
        if not isinstance(daily, dict):
            raise RuntimeError(f"{city['city']}: response has no daily records")
        columns = ("time", *DAILY_VARIABLES)
        if any(column not in daily or len(daily[column]) != expected_count for column in columns):
            raise RuntimeError(f"{city['city']}: response fields or dates are incomplete")
        prepared.append((city, {column: list(daily[column]) for column in columns}))
    return prepared


def main() -> int:
    """Fetch all pending cities in safe batches and write validated full caches."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=default_end_date(),
        help="inclusive history end date; defaults to the latest complete historical day",
    )
    args = parser.parse_args()
    start = date(2000, 1, 1)
    cities = read_cities(Path(__file__).resolve().parents[1] / "data" / "city_seed.csv")
    end = args.end_date
    if end < start:
        parser.error("--end-date must not be earlier than 2000-01-01")
    pending = [
        city
        for city in cities
        if not valid_chunk(cache_path(DEFAULT_OUTPUT_DIR, city["city_en"], start, end, "full"), start, end)
    ]
    if not pending:
        print("All 77 full history caches are already valid.")
        return 0
    batches = city_batches(pending)
    print(f"Fetching {len(pending)} cities in {len(batches)} batches of up to {BATCH_SIZE}.", flush=True)
    for index, batch in enumerate(batches, start=1):
        if index > 1:
            time.sleep(REQUEST_DELAY_SECONDS)
        print(f"Batch {index}/{len(batches)}: requesting {len(batch)} cities.", flush=True)
        for city, daily in fetch_batch(batch, start, end):
            write_chunk(cache_path(DEFAULT_OUTPUT_DIR, city["city_en"], start, end, "full"), daily)
            print(f"{city['city']}: fetched 1", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
