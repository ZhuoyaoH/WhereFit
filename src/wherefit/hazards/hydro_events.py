"""Flood and landslide event providers."""

from __future__ import annotations

from datetime import date, timedelta
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from wherefit.config import HISTORY_START_DATE
from wherefit.data_sources.cache import read_csv_cache, safe_cache_name, write_csv_cache
from wherefit.models import Location


EONET_EVENTS_URL = "https://eonet.gsfc.nasa.gov/api/v3/events/geojson"
OPEN_METEO_FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"


def get_eonet_events(
    location: Location,
    category: str,
    cache_dir: Path,
    radius_km: int = 500,
    start_date: str = HISTORY_START_DATE,
    end_date: str | None = None,
    force_refresh: bool = False,
    timeout: int = 25,
) -> tuple[list[dict[str, object]], str]:
    end = end_date or date.today().isoformat()
    cache_path = cache_dir / safe_cache_name(location.city_en or location.city, category, start_date, end, radius_km)
    cached = None if force_refresh else read_csv_cache(cache_path)
    if cached is not None:
        return _event_records(cached, location, category, radius_km), "cache"

    params = {
        "category": category,
        "status": "all",
        "start": start_date,
        "end": end,
        "bbox": _bbox(location.latitude, location.longitude, radius_km),
        "limit": 500,
    }
    try:
        response = requests.get(EONET_EVENTS_URL, params=params, timeout=timeout)
        response.raise_for_status()
        features = response.json().get("features", [])
        rows = [_feature_to_row(feature) for feature in features]
        data = pd.DataFrame([row for row in rows if row is not None])
        write_csv_cache(cache_path, data)
        return _event_records(data, location, category, radius_km), "live"
    except Exception:
        return [], "failed"


def get_flood_discharge_record(
    location: Location,
    cache_dir: Path,
    force_refresh: bool = False,
    timeout: int = 25,
) -> tuple[dict[str, object] | None, str]:
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=365)).isoformat()
    cache_path = cache_dir / safe_cache_name(location.city_en or location.city, start, end, "river_discharge")
    cached = None if force_refresh else read_csv_cache(cache_path)
    if cached is not None:
        return _discharge_record(cached), "cache"

    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "daily": "river_discharge,river_discharge_max",
        "start_date": start,
        "end_date": end,
    }
    try:
        response = requests.get(OPEN_METEO_FLOOD_URL, params=params, timeout=timeout)
        response.raise_for_status()
        daily = response.json().get("daily", {})
        data = pd.DataFrame(daily)
        if data.empty or "river_discharge" not in data.columns:
            raise ValueError("flood response has no river discharge data")
        write_csv_cache(cache_path, data)
        return _discharge_record(data), "live"
    except Exception:
        return None, "failed"


def _feature_to_row(feature: dict[str, Any]) -> dict[str, object] | None:
    """Normalize one EONET GeoJSON feature into the local event schema."""

    props = feature.get("properties", {}) or {}
    coords = _representative_coordinate(feature.get("geometry", {}) or {})
    if coords is None:
        return None
    lon, lat = coords
    event_date = str(props.get("date") or "")
    if not event_date:
        geometry_dates = props.get("geometryDates") or []
        if isinstance(geometry_dates, list) and geometry_dates:
            event_date = str(geometry_dates[-1] or "")
    return {
        "id": props.get("id") or feature.get("id") or "",
        "title": props.get("title") or "",
        "date": event_date,
        "closed": props.get("closed") or "",
        "source": ",".join(str(item.get("id") or item.get("title") or "") for item in props.get("sources", []) if isinstance(item, dict)),
        "latitude": lat,
        "longitude": lon,
    }


def _representative_coordinate(geometry: dict[str, Any]) -> tuple[float, float] | None:
    coords = geometry.get("coordinates")
    if coords is None:
        return None
    flat = list(_flatten_coordinates(coords))
    if not flat:
        return None
    lon = sum(item[0] for item in flat) / len(flat)
    lat = sum(item[1] for item in flat) / len(flat)
    return float(lon), float(lat)


def _flatten_coordinates(coords: Any):
    if isinstance(coords, list) and len(coords) >= 2 and all(isinstance(value, (int, float)) for value in coords[:2]):
        yield float(coords[0]), float(coords[1])
        return
    if isinstance(coords, list):
        for item in coords:
            yield from _flatten_coordinates(item)


def _event_records(data: pd.DataFrame, location: Location, category: str, radius_km: int) -> list[dict[str, object]]:
    if data.empty:
        return []
    work = data.copy()
    work["latitude"] = pd.to_numeric(work.get("latitude"), errors="coerce")
    work["longitude"] = pd.to_numeric(work.get("longitude"), errors="coerce")
    work = work.dropna(subset=["latitude", "longitude"])
    if work.empty:
        return []
    work["distance_km"] = work.apply(
        lambda row: _distance_km(location.latitude, location.longitude, row.get("latitude"), row.get("longitude")),
        axis=1,
    )
    work = work[work["distance_km"] <= radius_km].copy()
    if work.empty:
        return []
    work["date_sort"] = pd.to_datetime(work.get("date"), errors="coerce")
    work = work.sort_values(["date_sort", "distance_km"], ascending=[False, True]).head(100)
    category_label = "洪涝事件" if category == "floods" else "滑坡事件"
    records: list[dict[str, object]] = []
    for _, row in work.iterrows():
        records.append(
            {
                "类型": category_label,
                "记录": str(row.get("title") or category_label),
                "日期": str(row.get("date") or ""),
                "距离": round(float(row.get("distance_km")), 1),
                "数据口径": "美国航天局地球观测自然事件接口",
                "证据类型": "公开目录记录",
            }
        )
    return records


def _discharge_record(data: pd.DataFrame) -> dict[str, object] | None:
    discharge = pd.to_numeric(data.get("river_discharge"), errors="coerce").dropna()
    discharge_max = pd.to_numeric(data.get("river_discharge_max"), errors="coerce").dropna()
    if discharge.empty:
        return None
    peak = float(discharge_max.max()) if not discharge_max.empty else float(discharge.max())
    latest = float(discharge.iloc[-1])
    return {
        "类型": "河流流量数据",
        "记录": f"城市附近区域近一年平均河流流量约 {discharge.mean():.1f} 立方米/秒，峰值约 {peak:.1f} 立方米/秒，最近值约 {latest:.1f} 立方米/秒；不能据此判断城市洪水概率",
        "日期": "",
        "距离": "",
        "数据口径": "开放气象洪水资料提供的城市附近区域河流流量估算",
        "证据类型": "区域估算数据",
    }


def _bbox(latitude: float, longitude: float, radius_km: int) -> str:
    """Return EONET's west,north,east,south bounding-box order."""

    lat_delta = radius_km / 111.0
    lon_delta = radius_km / max(20.0, 111.0 * cos(radians(latitude)))
    west = longitude - lon_delta
    east = longitude + lon_delta
    south = latitude - lat_delta
    north = latitude + lat_delta
    return f"{west:.4f},{north:.4f},{east:.4f},{south:.4f}"


def _distance_km(lat1: float, lon1: float, lat2: object, lon2: object) -> float:
    if pd.isna(lat2) or pd.isna(lon2):
        return 99999.0
    radius = 6371.0
    d_lat = radians(float(lat2) - lat1)
    d_lon = radians(float(lon2) - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(float(lat2))) * sin(d_lon / 2) ** 2
    return 2 * radius * asin(sqrt(a))
