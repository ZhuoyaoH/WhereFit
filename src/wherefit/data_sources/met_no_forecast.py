"""MET Norway Locationforecast provider for short-term travel fit."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from wherefit.config import MET_NO_LOCATIONFORECAST_URL
from wherefit.data_sources.cache import read_csv_cache, safe_cache_name, write_csv_cache
from wherefit.data_sources.open_meteo_forecast import (
    UNIT_LABELS,
    _forecast_confidence,
    _forecast_horizon_weight,
    _is_supported_forecast_window,
    _normalize_unit_system,
    _thresholds,
)
from wherefit.models import ForecastSummary, Location


PROVIDER_NAME = "MET Norway Locationforecast API"
USER_AGENT = "WhereFit/1.0 (public ModelScope Studio)"


def get_met_no_forecast_summary(
    location: Location,
    cache_dir: Path,
    start_date: str,
    end_date: str,
    unit_system: str = "metric",
    force_refresh: bool = False,
    timeout: int = 20,
) -> ForecastSummary:
    unit_system = _normalize_unit_system(unit_system)
    if not _is_supported_forecast_window(start_date, end_date, max_days_ahead=9):
        return _failed_summary(location, start_date, end_date, unit_system, "MET Norway 日期必须从今天起且不超过未来 9 天。")
    cache_path = cache_dir / safe_cache_name(location.city_en or location.city, start_date, end_date, "met_no_forecast", unit_system)
    cached = None if force_refresh else read_csv_cache(cache_path)
    if cached is not None:
        summary = _summarize_met_no(cached, location, start_date, end_date, "cache", "读取 MET Norway 预报缓存", unit_system)
        return replace(summary, confidence=_forecast_horizon_weight(start_date, end_date, "cache") * 0.9)

    params = {"lat": round(location.latitude, 4), "lon": round(location.longitude, 4)}
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    try:
        response = requests.get(MET_NO_LOCATIONFORECAST_URL, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        raw = _payload_to_hourly_frame(response.json(), start_date, end_date)
        if raw.empty:
            raise ValueError("MET Norway returned no timeseries data in the selected date range")
        write_csv_cache(cache_path, raw)
        summary = _summarize_met_no(raw, location, start_date, end_date, "live", "已获取并缓存 MET Norway 未来天气预报", unit_system)
        return replace(summary, confidence=_forecast_horizon_weight(start_date, end_date, "live") * 0.9)
    except Exception as exc:
        return _failed_summary(location, start_date, end_date, unit_system, f"MET Norway 预报请求失败：{exc}")


def _payload_to_hourly_frame(payload: dict, start_date: str, end_date: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    for item in payload.get("properties", {}).get("timeseries", []):
        timestamp = item.get("time")
        if not timestamp:
            continue
        day = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).date()
        if day < start or day > end:
            continue
        data = item.get("data", {})
        instant = data.get("instant", {}).get("details", {})
        next_1h = data.get("next_1_hours", {}).get("details", {})
        next_6h = data.get("next_6_hours", {}).get("details", {})
        rows.append(
            {
                "time": timestamp,
                "date": day.isoformat(),
                "temperature_2m": instant.get("air_temperature"),
                "wind_speed_10m": instant.get("wind_speed"),
                "precipitation_1h": next_1h.get("precipitation_amount"),
                "precipitation_6h": next_6h.get("precipitation_amount"),
            }
        )
    return pd.DataFrame(rows)


def _summarize_met_no(
    raw: pd.DataFrame,
    location: Location,
    start_date: str,
    end_date: str,
    status: str,
    message: str,
    unit_system: str,
) -> ForecastSummary:
    data = raw.copy()
    data["date"] = data.get("date", pd.Series(dtype=str)).astype(str)
    data["temperature_2m"] = pd.to_numeric(data.get("temperature_2m"), errors="coerce")
    data["wind_speed_10m"] = pd.to_numeric(data.get("wind_speed_10m"), errors="coerce").fillna(0.0)
    data["precipitation_1h"] = pd.to_numeric(data.get("precipitation_1h"), errors="coerce")
    data["precipitation_6h"] = pd.to_numeric(data.get("precipitation_6h"), errors="coerce")

    daily = (
        data.groupby("date", as_index=False)
        .agg(
            temperature_2m_max=("temperature_2m", "max"),
            temperature_2m_min=("temperature_2m", "min"),
            wind_speed_10m_max=("wind_speed_10m", "max"),
            precipitation_1h=("precipitation_1h", "sum"),
            precipitation_6h=("precipitation_6h", "max"),
        )
        .sort_values("date")
    )
    precipitation_mm = daily["precipitation_1h"].where(daily["precipitation_1h"].fillna(0.0) > 0, daily["precipitation_6h"].fillna(0.0))
    temp_max_c = daily["temperature_2m_max"].fillna(0.0)
    temp_min_c = daily["temperature_2m_min"].fillna(0.0)
    wind_ms = daily["wind_speed_10m_max"].fillna(0.0)
    thresholds = _thresholds("metric")
    days = max(1, len(daily))
    labels = UNIT_LABELS[unit_system]

    temp_output = temp_max_c
    temp_min_output = temp_min_c
    precip_output = precipitation_mm
    if unit_system == "imperial":
        temp_output = temp_max_c * 9 / 5 + 32
        temp_min_output = temp_min_c * 9 / 5 + 32
        precip_output = precipitation_mm / 25.4

    return ForecastSummary(
        city=location.city,
        start_date=start_date,
        end_date=end_date,
        days=days,
        temp_max_mean=float(temp_output.mean()) if not temp_output.empty else 0.0,
        temp_min_mean=float(temp_min_output.mean()) if not temp_min_output.empty else 0.0,
        apparent_temp_max_mean=float(temp_output.mean()) if not temp_output.empty else 0.0,
        precipitation_days=int((precipitation_mm >= thresholds["rain_day"]).sum()),
        precipitation_probability_max=0.0,
        heavy_rain_days=int((precipitation_mm >= thresholds["heavy_rain"]).sum()),
        windy_days=int((wind_ms >= thresholds["windy"]).sum()),
        confidence=_forecast_confidence(days, status) * 0.9,
        source=PROVIDER_NAME,
        status=status,
        message=message + "；该源不提供体感温度和降水概率，体感列暂用最高温近似。",
        provider=PROVIDER_NAME,
        temperature_unit=labels["temperature"],
        precipitation_unit=labels["precipitation"],
        wind_speed_unit=labels["wind"],
    )


def _failed_summary(location: Location, start_date: str, end_date: str, unit_system: str, message: str) -> ForecastSummary:
    labels = UNIT_LABELS[_normalize_unit_system(unit_system)]
    return ForecastSummary(
        city=location.city,
        start_date=start_date,
        end_date=end_date,
        days=0,
        temp_max_mean=0.0,
        temp_min_mean=0.0,
        apparent_temp_max_mean=0.0,
        precipitation_days=0,
        precipitation_probability_max=0.0,
        heavy_rain_days=0,
        windy_days=0,
        confidence=0.0,
        source=PROVIDER_NAME,
        status="failed",
        message=message,
        provider=PROVIDER_NAME,
        temperature_unit=labels["temperature"],
        precipitation_unit=labels["precipitation"],
        wind_speed_unit=labels["wind"],
    )
