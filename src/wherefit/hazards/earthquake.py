"""USGS earthquake history provider."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from math import asin, cos, radians, sin, sqrt

import pandas as pd
import requests

from wherefit.config import HISTORY_START_DATE, USGS_EARTHQUAKE_URL
from wherefit.data_sources.cache import read_csv_cache, safe_cache_name, write_csv_cache
from wherefit.models import EarthquakeSummary, Location


def get_earthquake_summary(
    location: Location,
    cache_dir: Path,
    start_date: str = HISTORY_START_DATE,
    end_date: str | None = None,
    radius_km: int = 300,
    min_magnitude: float = 4.0,
    force_refresh: bool = False,
    timeout: int = 25,
) -> EarthquakeSummary:
    end = end_date or (date.today() - timedelta(days=1)).isoformat()
    cache_path = cache_dir / safe_cache_name(location.city_en or location.city, start_date, end, radius_km, "m4")
    cached = None if force_refresh else read_csv_cache(cache_path)
    if cached is not None:
        return summarize_earthquakes(cached, location, "USGS Earthquake Catalog API", "cache")

    params = {
        "format": "geojson",
        "starttime": start_date,
        "endtime": end,
        "latitude": location.latitude,
        "longitude": location.longitude,
        "maxradiuskm": radius_km,
        "minmagnitude": min_magnitude,
        "orderby": "time",
    }
    try:
        response = requests.get(USGS_EARTHQUAKE_URL, params=params, timeout=timeout)
        response.raise_for_status()
        features = response.json().get("features", [])
        rows = []
        for feature in features:
            props = feature.get("properties", {})
            coords = feature.get("geometry", {}).get("coordinates", [None, None, None])
            lon, lat = coords[0], coords[1]
            rows.append(
                {
                    "time": pd.to_datetime(props.get("time"), unit="ms", errors="coerce"),
                    "magnitude": props.get("mag"),
                    "place": props.get("place"),
                    "latitude": lat,
                    "longitude": lon,
                    "depth_km": coords[2] if len(coords) > 2 else None,
                }
            )
        data = pd.DataFrame(rows)
        write_csv_cache(cache_path, data)
        return summarize_earthquakes(data, location, "USGS Earthquake Catalog API", "live")
    except Exception:
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


def summarize_earthquakes(data: pd.DataFrame, location: Location, source: str, status: str) -> EarthquakeSummary:
    if data.empty:
        return EarthquakeSummary(0, 0, 0, None, None, None, source, status)
    mags = pd.to_numeric(data.get("magnitude"), errors="coerce").fillna(0.0)
    dates = pd.to_datetime(data.get("time"), errors="coerce")
    distances = data.apply(
        lambda row: _distance_km(location.latitude, location.longitude, row.get("latitude"), row.get("longitude")),
        axis=1,
    )
    return EarthquakeSummary(
        event_count_m4=int((mags >= 4.0).sum()),
        event_count_m5=int((mags >= 5.0).sum()),
        event_count_m6=int((mags >= 6.0).sum()),
        max_magnitude=float(mags.max()) if len(mags) else None,
        latest_event_date=dates.max().date().isoformat() if not pd.isna(dates.max()) else None,
        nearest_distance_km=round(float(distances.min()), 1) if len(distances) else None,
        source=source,
        status=status,
    )


def _distance_km(lat1: float, lon1: float, lat2: object, lon2: object) -> float:
    if pd.isna(lat2) or pd.isna(lon2):
        return 99999.0
    radius = 6371.0
    d_lat = radians(float(lat2) - lat1)
    d_lon = radians(float(lon2) - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(float(lat2))) * sin(d_lon / 2) ** 2
    return 2 * radius * asin(sqrt(a))

