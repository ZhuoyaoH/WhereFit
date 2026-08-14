#!/usr/bin/env python3
"""Build versioned 77-city climate-normal tables from NASA POWER."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_PATH = ROOT / "data" / "city_seed.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "climate"
DEFAULT_RESUME_DIR = ROOT / "data" / "cache" / "nasa-power-climate-build"
POWER_DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
MODEL = "merra2_power"
MODEL_LABEL = "NASA POWER / MERRA-2"
MODEL_RESOLUTION = "0.5 degree latitude x 0.625 degree longitude (~50 km)"
CACHE_SCHEMA_VERSION = 1
START_DATE = "2000-01-01"
END_DATE = "2025-12-31"
DAILY_VARIABLES = (
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "PRECTOTCORR",
    "WS10M",
    "WS10M_MAX",
)
INTERNAL_VARIABLES = (
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_mean",
    "wind_speed_10m_max",
    "apparent_temperature_mean",
    "snowfall_proxy",
)
POWER_TO_INTERNAL = {
    "T2M": "temperature_2m_mean",
    "T2M_MAX": "temperature_2m_max",
    "T2M_MIN": "temperature_2m_min",
    "RH2M": "relative_humidity_2m_mean",
    "PRECTOTCORR": "precipitation_sum",
    "WS10M": "wind_speed_10m_mean",
    "WS10M_MAX": "wind_speed_10m_max",
}
OUTPUT_COLUMNS = (
    "city",
    "period_month",
    "year_start",
    "year_end",
    "sample_years",
    "temperature_mean_c",
    "daily_max_temperature_mean_c",
    "apparent_temperature_mean_c",
    "relative_humidity_mean_pct",
    "precipitation_days_mean",
    "heavy_rain_days_mean",
    "extreme_rain_days_mean",
    "hot_days_mean",
    "cold_days_mean",
    "windy_days_mean",
    "snow_days_mean",
    "missing_rate",
    "source_product",
    "source_model",
    "spatial_resolution",
    "spatial_method",
)


@dataclass(frozen=True)
class CityBuildResult:
    """Hold one city's aggregates and response provenance."""

    city: str
    monthly_rows: list[dict[str, object]]
    annual_row: dict[str, object]
    provenance: dict[str, object]


def _resume_path(resume_dir: Path, city: str) -> Path:
    """Return a collision-resistant project-local resume path for one city."""

    token = hashlib.sha256(city.encode("utf-8")).hexdigest()[:16]
    return resume_dir / f"{token}.json"


def _write_resume_result(
    path: Path,
    result: CityBuildResult,
    start_date: str,
    end_date: str,
) -> None:
    """Persist a completed aggregate so a rate-limited build can resume safely."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "model": MODEL,
        "start_date": start_date,
        "end_date": end_date,
        "city": result.city,
        "monthly_rows": result.monthly_rows,
        "annual_row": result.annual_row,
        "provenance": result.provenance,
    }
    partial = path.with_suffix(".json.part")
    partial.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    partial.replace(path)


def _read_resume_result(
    path: Path,
    city: str,
    start_date: str,
    end_date: str,
) -> CityBuildResult | None:
    """Read a matching resumable aggregate or ignore incompatible state."""

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("cache_schema_version") != CACHE_SCHEMA_VERSION
        or payload.get("model") != MODEL
        or payload.get("start_date") != start_date
        or payload.get("end_date") != end_date
        or payload.get("city") != city
    ):
        return None
    monthly_rows = payload.get("monthly_rows")
    annual_row = payload.get("annual_row")
    provenance = payload.get("provenance")
    if not isinstance(monthly_rows, list) or len(monthly_rows) != 12:
        return None
    if not isinstance(annual_row, dict) or not isinstance(provenance, dict):
        return None
    return CityBuildResult(city, monthly_rows, annual_row, provenance)


def _sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_json(url: str, timeout: int, retries: int) -> tuple[Any, str]:
    """Fetch JSON with bounded retries and return its payload plus byte hash."""

    request = Request(url, headers={"User-Agent": "WhereFit-climate-builder/1.0"})
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
            payload = json.loads(raw)
            if isinstance(payload, dict) and payload.get("error"):
                raise ValueError(str(payload.get("reason") or "unexpected API response"))
            if not isinstance(payload, (dict, list)):
                raise ValueError("unexpected API response type")
            return payload, hashlib.sha256(raw).hexdigest()
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                retry_after = 0
                if isinstance(exc, HTTPError) and exc.code == 429:
                    try:
                        retry_after = int(exc.headers.get("Retry-After", "0"))
                    except (TypeError, ValueError):
                        retry_after = 0
                time.sleep(min(45, max(retry_after, 2 ** (attempt + 1))))
    raise RuntimeError(f"NASA POWER request failed after {retries + 1} attempts: {last_error}")


def _request_url(row: pd.Series, start_date: str, end_date: str) -> str:
    """Build one documented NASA POWER point request."""

    params = {
        "parameters": ",".join(DAILY_VARIABLES),
        "community": "AG",
        "latitude": str(float(row["latitude"])),
        "longitude": str(float(row["longitude"])),
        "start": start_date.replace("-", ""),
        "end": end_date.replace("-", ""),
        "format": "JSON",
    }
    return f"{POWER_DAILY_URL}?{urlencode(params)}"


def _fetch_city(
    row: pd.Series,
    start_date: str,
    end_date: str,
    timeout: int,
    retries: int,
    request_delay: float,
) -> CityBuildResult:
    """Fetch and validate one city from the NASA POWER point API."""

    url = _request_url(row, start_date, end_date)
    if request_delay > 0:
        time.sleep(request_delay)
    payload, response_hash = _fetch_json(url, timeout=timeout, retries=retries)
    if not isinstance(payload, dict):
        raise ValueError(f"{row['city']}: response is not an object")
    properties = payload.get("properties")
    parameters = properties.get("parameter") if isinstance(properties, dict) else None
    if not isinstance(parameters, dict):
        raise ValueError(f"{row['city']}: response has no properties.parameter object")
    missing = sorted(set(DAILY_VARIABLES) - set(parameters))
    if missing:
        raise ValueError(f"{row['city']}: response is missing parameters {missing}")
    data = pd.DataFrame({name: parameters[name] for name in DAILY_VARIABLES})
    data.index.name = "time"
    data = data.reset_index().rename(columns=POWER_TO_INTERNAL)
    for column in POWER_TO_INTERNAL.values():
        data[column] = pd.to_numeric(data[column], errors="coerce").replace(-999.0, pd.NA)
    data["apparent_temperature_mean"] = _apparent_temperature(
        data["temperature_2m_mean"],
        data["relative_humidity_2m_mean"],
        data["wind_speed_10m_mean"],
    )
    data["snowfall_proxy"] = data["precipitation_sum"].ge(1.0) & data["temperature_2m_mean"].le(0.0)
    _validate_daily(data, str(row["city"]), start_date, end_date)
    monthly_rows, annual_row = _aggregate_city(data, str(row["city"]), start_date, end_date)
    geometry = payload.get("geometry")
    coordinates = geometry.get("coordinates", []) if isinstance(geometry, dict) else []
    header = payload.get("header") if isinstance(payload.get("header"), dict) else {}
    api = header.get("api") if isinstance(header.get("api"), dict) else {}
    return CityBuildResult(
        str(row["city"]),
        monthly_rows,
        annual_row,
        {
            "city": str(row["city"]),
            "requested_latitude": float(row["latitude"]),
            "requested_longitude": float(row["longitude"]),
            "grid_longitude": coordinates[0] if len(coordinates) > 0 else None,
            "grid_latitude": coordinates[1] if len(coordinates) > 1 else None,
            "elevation_m": coordinates[2] if len(coordinates) > 2 else None,
            "time_standard": header.get("time_standard"),
            "api_version": api.get("version"),
            "sources": header.get("sources"),
            "response_sha256": response_hash,
        },
    )


def _apparent_temperature(
    temperature_c: pd.Series,
    relative_humidity_pct: pd.Series,
    wind_speed_ms: pd.Series,
) -> pd.Series:
    """Estimate shade apparent temperature using the BOM/Steadman equation."""

    vapour_pressure_hpa = (
        relative_humidity_pct
        / 100.0
        * 6.105
        * np.exp(17.27 * temperature_c / (237.7 + temperature_c))
    )
    return temperature_c + 0.33 * vapour_pressure_hpa - 0.70 * wind_speed_ms - 4.0


def _validate_daily(data: pd.DataFrame, city: str, start_date: str, end_date: str) -> None:
    """Reject incomplete dates, absent fields, and all-null core variables."""

    required = {"time", *INTERNAL_VARIABLES}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{city}: missing daily fields {missing}")
    dates = pd.to_datetime(data["time"], errors="coerce")
    expected = pd.date_range(start_date, end_date, freq="D")
    if dates.isna().any() or len(dates) != len(expected) or not dates.reset_index(drop=True).equals(pd.Series(expected)):
        raise ValueError(f"{city}: daily date coverage is incomplete or unordered")
    for column in INTERNAL_VARIABLES:
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{city}: {column} contains missing values for {MODEL_LABEL}")


def _aggregate_city(
    data: pd.DataFrame,
    city: str,
    start_date: str,
    end_date: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Create twelve calendar-month normals and one annual normal."""

    work = data.copy()
    work["time"] = pd.to_datetime(work["time"], errors="raise")
    for column in INTERNAL_VARIABLES:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["year"] = work["time"].dt.year
    work["month"] = work["time"].dt.month
    monthly_rows = [
        _aggregate_period(work[work["month"] == month], city, month, start_date, end_date)
        for month in range(1, 13)
    ]
    annual_row = _aggregate_period(work, city, 0, start_date, end_date)
    return monthly_rows, annual_row


def _aggregate_period(
    data: pd.DataFrame,
    city: str,
    period_month: int,
    start_date: str,
    end_date: str,
) -> dict[str, object]:
    """Aggregate temperatures as daily means and thresholds as mean annual counts."""

    sample_years = int(data["year"].nunique())
    if sample_years == 0:
        raise ValueError(f"{city}: no rows for period month {period_month}")
    yearly_counts = pd.DataFrame(
        {
            "precipitation_days": data["precipitation_sum"].ge(1.0).groupby(data["year"]).sum(),
            "heavy_rain_days": data["precipitation_sum"].ge(20.0).groupby(data["year"]).sum(),
            "extreme_rain_days": data["precipitation_sum"].ge(50.0).groupby(data["year"]).sum(),
            "hot_days": data["temperature_2m_max"].ge(35.0).groupby(data["year"]).sum(),
            "cold_days": data["temperature_2m_min"].le(0.0).groupby(data["year"]).sum(),
            "windy_days": data["wind_speed_10m_max"].ge(10.8).groupby(data["year"]).sum(),
            "snow_days": data["snowfall_proxy"].gt(0.0).groupby(data["year"]).sum(),
        }
    ).reindex(range(pd.Timestamp(start_date).year, pd.Timestamp(end_date).year + 1), fill_value=0)
    missing_rate = float(
        data[list(INTERNAL_VARIABLES)].isna().sum().sum() / (len(data) * len(INTERNAL_VARIABLES))
    )
    return {
        "city": city,
        "period_month": period_month,
        "year_start": pd.Timestamp(start_date).year,
        "year_end": pd.Timestamp(end_date).year,
        "sample_years": sample_years,
        "temperature_mean_c": round(float(data["temperature_2m_mean"].mean()), 4),
        "daily_max_temperature_mean_c": round(float(data["temperature_2m_max"].mean()), 4),
        "apparent_temperature_mean_c": round(float(data["apparent_temperature_mean"].mean()), 4),
        "relative_humidity_mean_pct": round(float(data["relative_humidity_2m_mean"].mean()), 4),
        "precipitation_days_mean": round(float(yearly_counts["precipitation_days"].mean()), 4),
        "heavy_rain_days_mean": round(float(yearly_counts["heavy_rain_days"].mean()), 4),
        "extreme_rain_days_mean": round(float(yearly_counts["extreme_rain_days"].mean()), 4),
        "hot_days_mean": round(float(yearly_counts["hot_days"].mean()), 4),
        "cold_days_mean": round(float(yearly_counts["cold_days"].mean()), 4),
        "windy_days_mean": round(float(yearly_counts["windy_days"].mean()), 4),
        "snow_days_mean": round(float(yearly_counts["snow_days"].mean()), 4),
        "missing_rate": round(missing_rate, 6),
        "source_product": "NASA POWER Daily API",
        "source_model": MODEL_LABEL,
        "spatial_resolution": MODEL_RESOLUTION,
        "spatial_method": "source-native model grid cell containing city-centre coordinates",
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a stable-column CSV atomically inside the output directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(partial, index=False)
    partial.replace(path)


def build_tables(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Fetch all seed cities and write monthly, annual, and manifest files."""

    seed_path = Path(args.seed_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    resume_dir = Path(args.resume_dir).resolve()
    seed = pd.read_csv(seed_path)
    required_seed = {"city", "latitude", "longitude"}
    if missing := sorted(required_seed - set(seed.columns)):
        raise ValueError(f"seed file is missing columns: {missing}")
    if seed["city"].duplicated().any():
        raise ValueError("seed file has duplicate city names")

    results: dict[str, CityBuildResult] = {}
    pending_rows: list[pd.Series] = []
    for _, row in seed.iterrows():
        city = str(row["city"])
        cached = _read_resume_result(
            _resume_path(resume_dir, city),
            city,
            args.start_date,
            args.end_date,
        )
        if cached is None:
            pending_rows.append(row)
        else:
            results[city] = cached
            print(f"[resume] {city}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(
                _fetch_city,
                row,
                args.start_date,
                args.end_date,
                int(args.timeout),
                int(args.retries),
                float(args.request_delay),
            ): str(row["city"])
            for row in pending_rows
        }
        for future in as_completed(futures):
            result = future.result()
            results[result.city] = result
            _write_resume_result(
                _resume_path(resume_dir, result.city),
                result,
                args.start_date,
                args.end_date,
            )
            print(f"[{len(results)}/{len(seed)}] {result.city}", flush=True)

    ordered = [results[str(city)] for city in seed["city"]]
    monthly_rows = [row for result in ordered for row in result.monthly_rows]
    annual_rows = [result.annual_row for result in ordered]
    monthly_path = output_dir / f"nasa_power_merra2_city_monthly_{args.start_date[:4]}_{args.end_date[:4]}.csv"
    annual_path = output_dir / f"nasa_power_merra2_city_annual_{args.start_date[:4]}_{args.end_date[:4]}.csv"
    manifest_path = output_dir / "nasa_power_merra2_manifest.json"
    _write_csv(monthly_path, monthly_rows)
    _write_csv(annual_path, annual_rows)
    manifest = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "provider": "NASA POWER",
            "product": "Daily API",
            "endpoint": POWER_DAILY_URL,
            "model": MODEL_LABEL,
            "resolution": MODEL_RESOLUTION,
            "access": "public NASA Earth Science data; attribution requested",
            "documentation_url": "https://power.larc.nasa.gov/docs/services/api/temporal/daily/",
            "citation_url": "https://power.larc.nasa.gov/docs/referencing/",
        },
        "request": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "community": "AG",
            "time_standard": "LST",
            "daily_variables": list(DAILY_VARIABLES),
        },
        "aggregation": {
            "temperature_fields": "mean of daily values across the selected calendar period",
            "day_count_fields": "mean yearly count within the selected calendar period",
            "thresholds": {
                "precipitation_day": "precipitation_sum >= 1 mm/day",
                "heavy_rain_day": "precipitation_sum >= 20 mm/day",
                "extreme_rain_day": "precipitation_sum >= 50 mm/day",
                "hot_day": "temperature_2m_max >= 35 C",
                "cold_day": "temperature_2m_min <= 0 C",
                "windy_day": "WS10M_MAX >= 10.8 m/s (38.88 km/h)",
                "snow_day_proxy": "PRECTOTCORR >= 1 mm/day and T2M <= 0 C",
            },
            "derived_fields": {
                "apparent_temperature_mean_c": "T2M + 0.33*e - 0.70*WS10M - 4.0; e=(RH2M/100)*6.105*exp(17.27*T2M/(237.7+T2M))",
                "snow_days_mean": "temperature-and-precipitation proxy, not observed snowfall",
            },
            "changes_from_source": "daily model-grid values aggregated to city-centre monthly and annual summaries",
        },
        "inputs": {
            "city_seed": str(seed_path.relative_to(ROOT)),
            "city_seed_sha256": _sha256(seed_path),
            "builder": str(Path(__file__).resolve().relative_to(ROOT)),
            "builder_sha256": _sha256(Path(__file__).resolve()),
        },
        "outputs": {
            "monthly": {
                "path": str(monthly_path.relative_to(ROOT)),
                "rows": len(monthly_rows),
                "sha256": _sha256(monthly_path),
            },
            "annual": {
                "path": str(annual_path.relative_to(ROOT)),
                "rows": len(annual_rows),
                "sha256": _sha256(annual_path),
            },
        },
        "city_responses": [result.provenance for result in ordered],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for city in seed["city"].astype(str):
        path = _resume_path(resume_dir, city)
        if path.exists():
            path.unlink()
    if resume_dir.exists() and not any(resume_dir.iterdir()):
        resume_dir.rmdir()
    return monthly_path, annual_path, manifest_path


def parse_args() -> argparse.Namespace:
    """Parse reproducible build settings from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-path", default=DEFAULT_SEED_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume-dir", default=DEFAULT_RESUME_DIR)
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--request-delay", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    """Run the complete climate-normal build."""

    monthly, annual, manifest = build_tables(parse_args())
    print(monthly)
    print(annual)
    print(manifest)


if __name__ == "__main__":
    main()
