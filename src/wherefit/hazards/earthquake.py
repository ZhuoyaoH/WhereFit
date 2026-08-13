"""USGS earthquake history provider."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from math import asin, cos, radians, sin, sqrt

import pandas as pd
import requests

from wherefit.config import HISTORY_START_DATE, USGS_EARTHQUAKE_URL
from wherefit.data_sources.cache import read_csv_cache, safe_cache_name, write_csv_cache
from wherefit.models import EarthquakeSummary, Location


USGS_QUERY_LIMIT = 20_000
USGS_CHUNK_YEARS = 5
USGS_MAX_PAGES_PER_CHUNK = 100


def get_earthquake_summary(
    location: Location,
    cache_dir: Path,
    start_date: str = HISTORY_START_DATE,
    end_date: str | None = None,
    radius_km: int = 500,
    min_magnitude: float = 4.0,
    force_refresh: bool = False,
    timeout: int = 25,
) -> EarthquakeSummary:
    end = end_date or (date.today() - timedelta(days=1)).isoformat()
    cache_path = cache_dir / safe_cache_name(
        location.city_en or location.city,
        start_date,
        end,
        radius_km,
        f"m{min_magnitude:g}",
    )
    cached = None if force_refresh else read_csv_cache(cache_path)
    if cached is not None:
        return summarize_earthquakes(cached, location, "USGS Earthquake Catalog API", "cache")

    rows: list[dict[str, object]] = []
    failures = 0
    successful_chunks = 0
    for chunk_start, chunk_end in _earthquake_date_chunks(start_date, end, USGS_CHUNK_YEARS):
        try:
            features = _fetch_chunk_features(
                location,
                chunk_start,
                chunk_end,
                radius_km,
                min_magnitude,
                timeout,
            )
            rows.extend(_features_to_rows(features))
            successful_chunks += 1
        except Exception:
            failures += 1
    if successful_chunks == 0:
        return EarthquakeSummary(
            event_count_m4=0,
            event_count_m5=0,
            event_count_m6=0,
            max_magnitude=None,
            latest_event_date=None,
            nearest_distance_km=None,
            source="USGS Earthquake Catalog API",
            status="failed",
        )
    data = pd.DataFrame(rows)
    if not data.empty and "event_id" in data.columns:
        data = data.drop_duplicates(subset=["event_id"]).reset_index(drop=True)
    status = "partial" if failures else "live"
    if not failures:
        write_csv_cache(cache_path, data)
    return summarize_earthquakes(data, location, "USGS Earthquake Catalog API", status)


def _fetch_chunk_features(
    location: Location,
    start_date: str,
    end_date: str,
    radius_km: int,
    min_magnitude: float,
    timeout: int,
) -> list[dict[str, object]]:
    """Fetch one time chunk with stable pagination under USGS's result cap."""

    features: list[dict[str, object]] = []
    offset = 1
    for _ in range(USGS_MAX_PAGES_PER_CHUNK):
        params = {
            "format": "geojson",
            "starttime": start_date,
            "endtime": f"{end_date}T23:59:59.999",
            "latitude": location.latitude,
            "longitude": location.longitude,
            "maxradiuskm": radius_km,
            "minmagnitude": min_magnitude,
            "orderby": "time-asc",
            "limit": USGS_QUERY_LIMIT,
            "offset": offset,
        }
        response = requests.get(USGS_EARTHQUAKE_URL, params=params, timeout=timeout)
        response.raise_for_status()
        page = response.json().get("features", [])
        if not isinstance(page, list):
            raise ValueError("USGS response has no feature list")
        features.extend(page)
        if len(page) < USGS_QUERY_LIMIT:
            return features
        offset += USGS_QUERY_LIMIT
    raise RuntimeError("USGS pagination exceeded the defensive page limit")


def _features_to_rows(features: list[dict[str, object]]) -> list[dict[str, object]]:
    """Normalize USGS GeoJSON features for caching and distance summaries."""

    rows: list[dict[str, object]] = []
    for feature in features:
        props = feature.get("properties", {}) or {}
        geometry = feature.get("geometry", {}) or {}
        coords = geometry.get("coordinates", [None, None, None])
        if not isinstance(coords, list) or len(coords) < 2:
            continue
        rows.append(
            {
                "event_id": feature.get("id"),
                "time": pd.to_datetime(props.get("time"), unit="ms", errors="coerce"),
                "magnitude": props.get("mag"),
                "place": props.get("place"),
                "latitude": coords[1],
                "longitude": coords[0],
                "depth_km": coords[2] if len(coords) > 2 else None,
            }
        )
    return rows


def _earthquake_date_chunks(start_date: str, end_date: str, years: int) -> list[tuple[str, str]]:
    """Split a date range into non-overlapping calendar-year chunks."""

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    if end < start:
        raise ValueError("earthquake end date is earlier than start date")
    chunks: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        chunk_end = date(min(cursor.year + years - 1, end.year), 12, 31)
        if chunk_end > end:
            chunk_end = end
        chunks.append((cursor.isoformat(), chunk_end.isoformat()))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def summarize_earthquakes(data: pd.DataFrame, location: Location, source: str, status: str) -> EarthquakeSummary:
    if data.empty:
        return EarthquakeSummary(0, 0, 0, None, None, None, source, status)
    mags = pd.to_numeric(data.get("magnitude"), errors="coerce").fillna(0.0)
    dates = pd.to_datetime(data.get("time"), errors="coerce")
    work = data.copy()
    distances = work.apply(
        lambda row: _distance_km(location.latitude, location.longitude, row.get("latitude"), row.get("longitude")),
        axis=1,
    )
    work["distance_km"] = distances
    events = _event_records(work)
    return EarthquakeSummary(
        event_count_m4=int((mags >= 4.0).sum()),
        event_count_m5=int((mags >= 5.0).sum()),
        event_count_m6=int((mags >= 6.0).sum()),
        max_magnitude=float(mags.max()) if len(mags) else None,
        latest_event_date=dates.max().date().isoformat() if not pd.isna(dates.max()) else None,
        nearest_distance_km=round(float(distances.min()), 1) if len(distances) else None,
        source=source,
        status=status,
        count_100km=int((distances <= 100).sum()),
        count_200km=int((distances <= 200).sum()),
        count_500km=int((distances <= 500).sum()),
        events=events,
    )


def _event_records(data: pd.DataFrame, limit: int = 350) -> list[dict[str, object]]:
    if data.empty:
        return []
    work = data.copy()
    work["time"] = pd.to_datetime(work.get("time"), errors="coerce")
    work["magnitude"] = pd.to_numeric(work.get("magnitude"), errors="coerce")
    work["latitude"] = pd.to_numeric(work.get("latitude"), errors="coerce")
    work["longitude"] = pd.to_numeric(work.get("longitude"), errors="coerce")
    work = work.dropna(subset=["latitude", "longitude", "magnitude"])
    work = work.sort_values(["magnitude", "time"], ascending=[False, False]).head(limit)
    records: list[dict[str, object]] = []
    for _, row in work.iterrows():
        event_time = row.get("time")
        records.append(
            {
                "date": event_time.date().isoformat() if not pd.isna(event_time) else "",
                "magnitude": round(float(row.get("magnitude")), 1),
                "place": str(row.get("place") or ""),
                "latitude": float(row.get("latitude")),
                "longitude": float(row.get("longitude")),
                "distance_km": round(float(row.get("distance_km")), 1),
            }
        )
    return records


def _distance_km(lat1: float, lon1: float, lat2: object, lon2: object) -> float:
    if pd.isna(lat2) or pd.isna(lon2):
        return 99999.0
    radius = 6371.0
    d_lat = radians(float(lat2) - lat1)
    d_lon = radians(float(lon2) - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(float(lat2))) * sin(d_lon / 2) ** 2
    return 2 * radius * asin(sqrt(a))
