"""Open-Meteo forecast provider for short-term travel fit."""

from __future__ import annotations

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


def get_forecast_summary(
    location: Location,
    cache_dir: Path,
    start_date: str,
    end_date: str,
    force_refresh: bool = False,
    timeout: int = 20,
) -> ForecastSummary:
    days = _date_span_days(start_date, end_date)
    if days < 1 or days > 16:
        return _failed_summary(location, start_date, end_date, f"Forecast 只支持 1-16 天范围，当前为 {days} 天。")

    cache_path = cache_dir / safe_cache_name(location.city_en or location.city, start_date, end_date, "forecast")
    cached = None if force_refresh else read_csv_cache(cache_path)
    if cached is not None:
        return summarize_forecast(cached, location, start_date, end_date, "cache", "读取未来预报缓存")

    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "daily": ",".join(DAILY_FORECAST_VARIABLES),
        "timezone": "auto",
        "start_date": start_date,
        "end_date": end_date,
    }
    try:
        response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=timeout)
        response.raise_for_status()
        raw = pd.DataFrame(response.json().get("daily", {}))
        if raw.empty or "time" not in raw.columns:
            raise ValueError("Open-Meteo returned no forecast daily data")
        write_csv_cache(cache_path, raw)
        return summarize_forecast(raw, location, start_date, end_date, "live", "已获取并缓存未来天气预报")
    except Exception as exc:
        return _failed_summary(location, start_date, end_date, f"未来预报请求失败：{exc}")


def summarize_forecast(
    raw: pd.DataFrame,
    location: Location,
    start_date: str,
    end_date: str,
    status: str,
    message: str,
) -> ForecastSummary:
    data = raw.copy()
    days = max(1, len(data))
    precip = pd.to_numeric(data.get("precipitation_sum"), errors="coerce").fillna(0.0)
    temp_max = pd.to_numeric(data.get("temperature_2m_max"), errors="coerce")
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
        apparent_temp_max_mean=float(apparent.mean()) if not apparent.empty else 0.0,
        precipitation_days=int((precip >= 1.0).sum()),
        precipitation_probability_max=float(precip_prob.max()) if not precip_prob.empty else 0.0,
        heavy_rain_days=int((precip >= 20.0).sum()),
        windy_days=int((wind >= 10.8).sum()),
        confidence=confidence,
        source="Open-Meteo Forecast API",
        status=status,
        message=message,
    )


def _failed_summary(location: Location, start_date: str, end_date: str, message: str) -> ForecastSummary:
    return ForecastSummary(
        city=location.city,
        start_date=start_date,
        end_date=end_date,
        days=max(0, _date_span_days(start_date, end_date)),
        temp_max_mean=0.0,
        apparent_temp_max_mean=0.0,
        precipitation_days=0,
        precipitation_probability_max=0.0,
        heavy_rain_days=0,
        windy_days=0,
        confidence=0.0,
        source="Open-Meteo Forecast API",
        status="failed",
        message=message,
    )


def _date_span_days(start_date: str, end_date: str) -> int:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
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
