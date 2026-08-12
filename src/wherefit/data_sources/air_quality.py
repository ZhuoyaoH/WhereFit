"""Open-Meteo air-quality provider."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from wherefit.data_sources.cache import read_csv_cache, safe_cache_name, write_csv_cache
from wherefit.models import AirQualitySummary, Location


PROVIDER_NAME = "Open-Meteo Air Quality API"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def get_air_quality_summary(
    location: Location,
    cache_dir: Path,
    force_refresh: bool = False,
    timeout: int = 20,
) -> AirQualitySummary:
    period_end = date.today()
    period_start = period_end - timedelta(days=7)
    cache_path = cache_dir / safe_cache_name(location.city_en or location.city, date.today().isoformat(), "air_quality")
    cached = None if force_refresh else read_csv_cache(cache_path)
    if cached is not None:
        return summarize_air_quality(
            cached,
            location,
            "cache",
            "读取近期空气质量缓存",
            period_start=period_start.isoformat(),
            period_end=period_end.isoformat(),
        )

    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "hourly": "pm2_5,us_aqi",
        "past_days": 7,
        "forecast_days": 1,
        "timezone": "auto",
    }
    try:
        response = requests.get(AIR_QUALITY_URL, params=params, timeout=timeout)
        response.raise_for_status()
        hourly = response.json().get("hourly", {})
        data = pd.DataFrame(hourly)
        if data.empty or "pm2_5" not in data.columns:
            raise ValueError("air-quality response has no PM2.5 data")
        write_csv_cache(cache_path, data)
        return summarize_air_quality(
            data,
            location,
            "live",
            "已获取并缓存近期空气质量数据",
            period_start=period_start.isoformat(),
            period_end=period_end.isoformat(),
        )
    except Exception as exc:
        return AirQualitySummary(
            city=location.city,
            pm25_mean=None,
            us_aqi_mean=None,
            source=PROVIDER_NAME,
            status="failed",
            message=f"空气质量数据请求失败：{exc}",
            sample_hours=0,
            period_start=period_start.isoformat(),
            period_end=period_end.isoformat(),
            time_basis="recent_8_day_window",
        )


def summarize_air_quality(
    data: pd.DataFrame,
    location: Location,
    status: str,
    message: str,
    period_start: str | None = None,
    period_end: str | None = None,
) -> AirQualitySummary:
    """Aggregate recent hourly PM2.5 and AQI without implying a long-term normal."""

    pm25 = pd.to_numeric(data.get("pm2_5"), errors="coerce").dropna()
    aqi = pd.to_numeric(data.get("us_aqi"), errors="coerce").dropna()
    if pm25.empty:
        return AirQualitySummary(
            location.city,
            None,
            None,
            PROVIDER_NAME,
            "failed",
            "空气质量缓存缺少 PM2.5 数据",
            0,
            period_start,
            period_end,
            "recent_8_day_window",
        )
    return AirQualitySummary(
        city=location.city,
        pm25_mean=round(float(pm25.mean()), 1),
        us_aqi_mean=round(float(aqi.mean()), 1) if not aqi.empty else None,
        source=PROVIDER_NAME,
        status=status,
        message=message,
        sample_hours=int(pm25.count()),
        period_start=period_start,
        period_end=period_end,
        time_basis="recent_8_day_window",
    )
