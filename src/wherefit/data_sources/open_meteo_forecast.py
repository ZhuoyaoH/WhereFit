"""Open-Meteo forecast provider for short-term travel fit."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from wherefit.config import OPEN_METEO_FORECAST_URL
from wherefit.data_sources.cache import read_csv_cache, safe_cache_name, write_csv_cache
from wherefit.models import ForecastSummary, Location


DAILY_FORECAST_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
    "precipitation_sum",
    "precipitation_probability_max",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "weather_code",
]

PROVIDER_NAME = "Open-Meteo Forecast API"
UNIT_LABELS = {
    "metric": {"temperature": "°C", "precipitation": "mm", "wind": "m/s"},
    "imperial": {"temperature": "°F", "precipitation": "inch", "wind": "mph"},
}


def get_forecast_summary(
    location: Location,
    cache_dir: Path,
    start_date: str,
    end_date: str,
    unit_system: str = "metric",
    force_refresh: bool = False,
    timeout: int = 20,
) -> ForecastSummary:
    days = _date_span_days(start_date, end_date)
    if days < 1 or days > 16:
        return _failed_summary(location, start_date, end_date, unit_system, f"Forecast 只支持 1-16 天范围，当前为 {days} 天。")
    if not _is_supported_forecast_window(start_date, end_date, max_days_ahead=15):
        return _failed_summary(location, start_date, end_date, unit_system, "Forecast 日期必须从今天起且不超过未来 15 天。")

    unit_system = _normalize_unit_system(unit_system)
    cache_path = cache_dir / safe_cache_name(location.city_en or location.city, start_date, end_date, "open_meteo_forecast", unit_system)
    cached = None if force_refresh else read_csv_cache(cache_path)
    if cached is not None:
        summary = summarize_forecast(cached, location, start_date, end_date, "cache", "读取未来预报缓存", unit_system=unit_system)
        return replace(summary, confidence=_forecast_horizon_weight(start_date, end_date, "cache"))

    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "daily": ",".join(DAILY_FORECAST_VARIABLES),
        "timezone": "auto",
        "start_date": start_date,
        "end_date": end_date,
        **_unit_params(unit_system),
    }
    try:
        response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=timeout)
        response.raise_for_status()
        raw = pd.DataFrame(response.json().get("daily", {}))
        if raw.empty or "time" not in raw.columns:
            raise ValueError("Open-Meteo returned no forecast daily data")
        write_csv_cache(cache_path, raw)
        summary = summarize_forecast(raw, location, start_date, end_date, "live", "已获取并缓存未来天气预报", unit_system=unit_system)
        return replace(summary, confidence=_forecast_horizon_weight(start_date, end_date, "live"))
    except Exception as exc:
        return _failed_summary(location, start_date, end_date, unit_system, f"未来预报请求失败：{exc}")


def summarize_forecast(
    raw: pd.DataFrame,
    location: Location,
    start_date: str,
    end_date: str,
    status: str,
    message: str,
    unit_system: str = "metric",
    provider: str = PROVIDER_NAME,
    source: str = PROVIDER_NAME,
) -> ForecastSummary:
    unit_system = _normalize_unit_system(unit_system)
    thresholds = _thresholds(unit_system)
    labels = UNIT_LABELS[unit_system]
    data = raw.copy()
    days = max(1, len(data))
    precip = pd.to_numeric(data.get("precipitation_sum"), errors="coerce").fillna(0.0)
    temp_max = pd.to_numeric(data.get("temperature_2m_max"), errors="coerce")
    temp_min = pd.to_numeric(data.get("temperature_2m_min"), errors="coerce")
    apparent = pd.to_numeric(data.get("apparent_temperature_max"), errors="coerce")
    precip_prob = pd.to_numeric(data.get("precipitation_probability_max"), errors="coerce").fillna(0.0)
    wind = pd.to_numeric(data.get("wind_speed_10m_max"), errors="coerce").fillna(0.0)
    confidence = _forecast_confidence(days, status)
    return ForecastSummary(
        city=location.city,
        start_date=start_date,
        end_date=end_date,
        days=days,
        temp_max_mean=float(temp_max.mean()) if not temp_max.empty else 0.0,
        temp_min_mean=float(temp_min.mean()) if not temp_min.empty else 0.0,
        apparent_temp_max_mean=float(apparent.mean()) if not apparent.empty else 0.0,
        precipitation_days=int((precip >= thresholds["rain_day"]).sum()),
        precipitation_probability_max=float(precip_prob.max()) if not precip_prob.empty else 0.0,
        heavy_rain_days=int((precip >= thresholds["heavy_rain"]).sum()),
        windy_days=int((wind >= thresholds["windy"]).sum()),
        confidence=confidence,
        source=source,
        status=status,
        message=message,
        provider=provider,
        temperature_unit=labels["temperature"],
        precipitation_unit=labels["precipitation"],
        wind_speed_unit=labels["wind"],
    )


def _failed_summary(location: Location, start_date: str, end_date: str, unit_system: str, message: str) -> ForecastSummary:
    unit_system = _normalize_unit_system(unit_system)
    labels = UNIT_LABELS[unit_system]
    return ForecastSummary(
        city=location.city,
        start_date=start_date,
        end_date=end_date,
        days=max(0, _date_span_days(start_date, end_date)),
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


def _date_span_days(start_date: str, end_date: str) -> int:
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return 0
    return (end - start).days + 1


def default_forecast_dates(today: date | None = None) -> tuple[str, str]:
    current = today or date.today()
    return current.isoformat(), (current + timedelta(days=6)).isoformat()


def _forecast_confidence(days: int, status: str) -> float:
    if status == "failed":
        return 0.0
    if days <= 3:
        return 0.85
    if days <= 7:
        return 0.72
    return 0.58


def _forecast_horizon_weight(
    start_date: str,
    end_date: str,
    status: str,
    today: date | None = None,
) -> float:
    """Return a recency weight based on the farthest forecast lead time."""

    if status == "failed":
        return 0.0
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return 0.0
    current = today or date.today()
    if start < current or end < start:
        return 0.0
    lead_days = (end - current).days
    if lead_days <= 3:
        return 0.85
    if lead_days <= 7:
        return 0.72
    return 0.58


def _is_supported_forecast_window(
    start_date: str,
    end_date: str,
    max_days_ahead: int,
    today: date | None = None,
) -> bool:
    """Validate that a date range is a present or future provider window."""

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return False
    current = today or date.today()
    return current <= start <= end <= current + timedelta(days=max_days_ahead)


def _normalize_unit_system(unit_system: str) -> str:
    return "imperial" if unit_system == "imperial" else "metric"


def _unit_params(unit_system: str) -> dict[str, str]:
    if _normalize_unit_system(unit_system) == "imperial":
        return {
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
        }
    return {
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }


def _thresholds(unit_system: str) -> dict[str, float]:
    if _normalize_unit_system(unit_system) == "imperial":
        return {
            "rain_day": 0.0393701,
            "heavy_rain": 0.787402,
            "windy": 24.1594,
        }
    return {
        "rain_day": 1.0,
        "heavy_rain": 20.0,
        "windy": 10.8,
    }
