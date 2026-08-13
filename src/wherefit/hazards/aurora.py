"""Aurora opportunity nowcast for candidate cities."""

from __future__ import annotations

from datetime import datetime, timezone
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd
import requests

from wherefit.config import AURORA_CACHE_TTL_MINUTES, SWPC_AURORA_OVATION_URL
from wherefit.data_sources.cache import read_csv_cache, write_csv_cache
from wherefit.models import AuroraSummary, Location


def build_aurora_summary(
    location: Location,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
    include_live: bool = False,
    timeout: int = 20,
) -> AuroraSummary:
    if include_live and cache_dir is not None:
        live = _get_live_ovation_summary(location, cache_dir, force_refresh, timeout)
        if live is not None:
            return live
    return _heuristic_summary(location)


def _heuristic_summary(location: Location) -> AuroraSummary:
    lat = location.latitude
    if lat >= 52:
        label = "高纬相对较高"
        score = 62.0
        explanation = "该城市处于较高纬度区域，强地磁活动期间存在低概率极光机会，仍取决于云量、光污染和夜间条件。"
    elif lat >= 48:
        label = "低到中等"
        score = 42.0
        explanation = "该城市纬度较高，极强地磁活动期间可能有异常低纬极光机会，但常规可见性仍然较低。"
    elif lat >= 42:
        label = "较低"
        score = 18.0
        explanation = "该城市纬度不足以稳定观测极光，只有极强地磁暴期间才可能出现非常低概率机会。"
    else:
        label = "极低"
        score = 5.0
        explanation = "该城市纬度较低，常规情况下极光可见性极低；当前仅作旅行兴趣提示。"
    return AuroraSummary(
        city=location.city,
        opportunity_label=label,
        opportunity_score=score,
        explanation=explanation,
        source="按城市纬度粗略估算；可查询官方未来 30 分钟提示",
        status="heuristic",
    )


def _get_live_ovation_summary(
    location: Location,
    cache_dir: Path,
    force_refresh: bool,
    timeout: int,
) -> AuroraSummary | None:
    cache_path = cache_dir / "ovation_aurora_latest.csv"
    cached = None if force_refresh else _read_fresh_cache(cache_path)
    data = cached[0] if cached is not None else None
    forecast_time = cached[1] if cached is not None else None
    status = "cache" if cached is not None else "live"
    if data is None:
        try:
            response = requests.get(SWPC_AURORA_OVATION_URL, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            forecast_time = payload.get("Forecast Time") or payload.get("forecast_time")
            coordinates = payload.get("coordinates") or payload.get("Coordinates") or payload.get("data")
            if not coordinates:
                return None
            data = pd.DataFrame(coordinates, columns=["longitude", "latitude", "probability"])
            data["forecast_time"] = forecast_time or ""
            data["cached_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_csv_cache(cache_path, data)
        except Exception:
            return None
    if data.empty:
        return None
    try:
        nearest = _nearest_aurora_row(data, location)
    except (KeyError, IndexError, ValueError):
        return None
    probability = float(nearest["probability"])
    distance = float(nearest["distance_km"])
    label = _live_label(probability, location.latitude)
    explanation = f"官方未来 30 分钟提示显示，附近区域机会约为 {probability:.1f}。实际能否看到仍取决于是否天黑、云量、光污染和地磁活动。"
    return AuroraSummary(
        city=location.city,
        opportunity_label=label,
        opportunity_score=min(100.0, max(0.0, probability)),
        explanation=explanation,
        source="NOAA SWPC OVATION Aurora 30 Minute Forecast",
        status=status,
        forecast_time=forecast_time,
        nearest_probability=probability,
        nearest_distance_km=round(distance, 1),
    )


def _read_fresh_cache(
    cache_path: Path,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, str | None] | None:
    """Return OVATION cache data only while its recorded fetch time is fresh."""

    data = read_csv_cache(cache_path)
    if data is None or data.empty or "cached_at_utc" not in data.columns:
        return None
    cached_values = data["cached_at_utc"].dropna().astype(str)
    if cached_values.empty:
        return None
    cached_at = pd.to_datetime(cached_values.iloc[0], utc=True, errors="coerce")
    if pd.isna(cached_at):
        return None
    current = now or datetime.now(timezone.utc)
    age_minutes = (pd.Timestamp(current) - cached_at).total_seconds() / 60.0
    if age_minutes < 0 or age_minutes > AURORA_CACHE_TTL_MINUTES:
        return None
    forecast_time = None
    if "forecast_time" in data.columns:
        values = data["forecast_time"].dropna().astype(str)
        if not values.empty and values.iloc[0].strip():
            forecast_time = values.iloc[0]
    return data, forecast_time


def _nearest_aurora_row(data: pd.DataFrame, location: Location) -> pd.Series:
    work = data.copy()
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    work["probability"] = pd.to_numeric(work["probability"], errors="coerce").fillna(0.0)
    work = work.dropna(subset=["latitude", "longitude"])
    work["distance_km"] = work.apply(
        lambda row: _distance_km(location.latitude, location.longitude, row["latitude"], row["longitude"]),
        axis=1,
    )
    return work.sort_values("distance_km").iloc[0]


def _live_label(probability: float, latitude: float) -> str:
    if probability >= 30:
        return "未来 30 分钟机会较高"
    if latitude >= 50 and probability >= 10:
        return "低到中等"
    if probability >= 5:
        return "较低"
    return "极低"


def _distance_km(lat1: float, lon1: float, lat2: object, lon2: object) -> float:
    radius = 6371.0
    d_lat = radians(float(lat2) - lat1)
    d_lon = radians(float(lon2) - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(float(lat2))) * sin(d_lon / 2) ** 2
    return 2 * radius * asin(sqrt(a))
