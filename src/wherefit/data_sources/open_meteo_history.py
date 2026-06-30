"""Open-Meteo historical weather provider."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from wherefit.config import CONFIG, HISTORY_START_DATE, OPEN_METEO_ARCHIVE_URL
from wherefit.data_sources.cache import read_csv_cache, safe_cache_name, write_csv_cache
from wherefit.models import ClimateMetrics, Location


DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_mean",
    "apparent_temperature_max",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
]

CHUNK_YEARS = 5


@dataclass(frozen=True)
class HistoricalFetchResult:
    metrics: ClimateMetrics | None
    raw: pd.DataFrame | None
    source: str
    status: str
    message: str
    cache_path: Path


def default_history_end_date(today: date | None = None) -> str:
    current = today or date.today()
    return (current - timedelta(days=10)).isoformat()


def get_history_metrics(
    location: Location,
    month: int,
    cache_dir: Path,
    start_date: str = HISTORY_START_DATE,
    end_date: str | None = None,
    force_refresh: bool = False,
    timeout: int = 30,
    fallback_pm25: float = 25.0,
) -> HistoricalFetchResult:
    end = end_date or default_history_end_date()
    cache_path = cache_dir / safe_cache_name(location.city_en or location.city, start_date, end, "daily")
    cached = None if force_refresh else read_csv_cache(cache_path)
    if cached is not None:
        return HistoricalFetchResult(
            metrics=metrics_from_history(cached, location, month, fallback_pm25=fallback_pm25),
            raw=cached,
            source="Open-Meteo Historical Weather API",
            status="cache",
            message=f"读取历史天气缓存：{cache_path.name}",
            cache_path=cache_path,
        )

    return _fetch_chunked_history(
        location=location,
        month=month,
        cache_dir=cache_dir,
        full_cache_path=cache_path,
        start_date=start_date,
        end_date=end,
        force_refresh=force_refresh,
        timeout=timeout,
        fallback_pm25=fallback_pm25,
    )


def _fetch_chunked_history(
    location: Location,
    month: int,
    cache_dir: Path,
    full_cache_path: Path,
    start_date: str,
    end_date: str,
    force_refresh: bool,
    timeout: int,
    fallback_pm25: float,
) -> HistoricalFetchResult:
    chunks: list[pd.DataFrame] = []
    failures: list[str] = []
    for chunk_start, chunk_end in _date_chunks(start_date, end_date, years=CHUNK_YEARS):
        chunk_path = cache_dir / "chunks" / safe_cache_name(
            location.city_en or location.city,
            chunk_start,
            chunk_end,
            "daily",
        )
        chunk = None if force_refresh else read_csv_cache(chunk_path)
        if chunk is None:
            try:
                chunk = _fetch_history_chunk(location, chunk_start, chunk_end, timeout)
                write_csv_cache(chunk_path, chunk)
            except Exception as exc:
                failures.append(f"{chunk_start}~{chunk_end}: {exc}")
                continue
        chunks.append(chunk)

    if not chunks:
        return HistoricalFetchResult(
            metrics=None,
            raw=None,
            source="Open-Meteo Historical Weather API",
            status="failed",
            message="历史天气请求失败，且没有可用分块缓存，已回退静态数据：" + "; ".join(failures[:2]),
            cache_path=full_cache_path,
        )

    raw = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")
    write_csv_cache(full_cache_path, raw)
    status = "partial" if failures else "live"
    message = f"已聚合 Open-Meteo 历史天气：{len(chunks)} 个分块，缓存 {full_cache_path.name}"
    if failures:
        message += f"；{len(failures)} 个分块失败，已用可用年份计算"
    return HistoricalFetchResult(
        metrics=metrics_from_history(raw, location, month, fallback_pm25=fallback_pm25),
        raw=raw,
        source="Open-Meteo Historical Weather API",
        status=status,
        message=message,
        cache_path=full_cache_path,
    )


def _fetch_history_chunk(location: Location, start_date: str, end_date: str, timeout: int) -> pd.DataFrame:
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "auto",
    }
    try:
        response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        daily = payload.get("daily", {})
        raw = pd.DataFrame(daily)
        if raw.empty or "time" not in raw.columns:
            raise ValueError("Open-Meteo returned no daily data")
    except Exception as exc:
        raise RuntimeError(exc) from exc
    return raw


def _date_chunks(start_date: str, end_date: str, years: int) -> list[tuple[str, str]]:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    if end < start:
        raise ValueError("history end date is earlier than start date")
    chunks: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        chunk_end = date(min(cursor.year + years - 1, end.year), 12, 31)
        if chunk_end > end:
            chunk_end = end
        chunks.append((cursor.isoformat(), chunk_end.isoformat()))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def metrics_from_history(
    raw: pd.DataFrame,
    location: Location,
    month: int,
    fallback_pm25: float = 25.0,
) -> ClimateMetrics:
    data = raw.copy()
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    data = data[data["time"].dt.month == int(month)]
    if data.empty:
        raise ValueError(f"no historical weather rows for month {month}")

    temp_mean = _mean(data, "temperature_2m_mean", default=20.0)
    temp_max = _mean(data, "temperature_2m_max", default=temp_mean + 5.0)
    app_mean = _mean(data, "apparent_temperature_mean", default=temp_mean)
    app_max = _quantile(data, "apparent_temperature_max", 0.75, default=app_mean)
    precipitation = _series(data, "precipitation_sum")
    rain_days = float((precipitation >= 1.0).sum() / _sample_years(data))
    heavy_rain_days = float((precipitation >= 20.0).sum() / _sample_years(data))
    extreme_rain_days = float((precipitation >= 50.0).sum() / _sample_years(data))
    hot_days = float((_series(data, "temperature_2m_max") >= 35.0).sum() / _sample_years(data))
    cold_days = float((_series(data, "temperature_2m_min") <= 0.0).sum() / _sample_years(data))
    windy_days = float((_series(data, "wind_speed_10m_max") >= 10.8).sum() / _sample_years(data))
    snow_days = float((_series(data, "snowfall_sum") > 0.0).sum() / _sample_years(data))
    missing_rate = float(data.isna().sum().sum() / max(1, data.shape[0] * data.shape[1]))

    # Daily humidity is not always available in Open-Meteo archive responses; estimate conservatively.
    humidity = _estimate_humidity(location, month, rain_days)
    confidence = max(0.55, CONFIG.historical_data_confidence - min(0.25, missing_rate))
    return ClimateMetrics(
        temperature_mean=temp_mean,
        temperature_max=max(temp_max, app_max),
        apparent_temperature=max(app_mean, app_max),
        relative_humidity_mean=humidity,
        precipitation_days=rain_days,
        heavy_rain_days=heavy_rain_days,
        pm25=fallback_pm25,
        hot_days=hot_days,
        winter_cold_level=_winter_cold_proxy(cold_days),
        coastal=location.coastal,
        typhoon_region=location.typhoon_region,
        data_source="Open-Meteo Historical Weather API",
        data_status="cache/live",
        sample_years=_sample_years(data),
        missing_rate=round(missing_rate, 3),
        precipitation_extreme_days=extreme_rain_days,
        cold_days=cold_days,
        windy_days=windy_days,
        snow_days=snow_days,
    )


def _series(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        return pd.Series([0.0] * len(data), index=data.index)
    return pd.to_numeric(data[column], errors="coerce").fillna(0.0)


def _mean(data: pd.DataFrame, column: str, default: float) -> float:
    if column not in data.columns:
        return default
    value = pd.to_numeric(data[column], errors="coerce").mean()
    if pd.isna(value):
        return default
    return float(value)


def _quantile(data: pd.DataFrame, column: str, q: float, default: float) -> float:
    if column not in data.columns:
        return default
    value = pd.to_numeric(data[column], errors="coerce").quantile(q)
    if pd.isna(value):
        return default
    return float(value)


def _sample_years(data: pd.DataFrame) -> int:
    years = data["time"].dt.year.dropna().nunique()
    return max(1, int(years))


def _estimate_humidity(location: Location, month: int, rain_days: float) -> float:
    base = 52.0 + min(24.0, rain_days * 1.2)
    if location.coastal:
        base += 8.0
    if location.region_type in {"south_coast", "east_coast", "southwest_basin"}:
        base += 5.0
    if month in {6, 7, 8, 9}:
        base += 4.0
    if location.region_type in {"northwest", "plateau", "north_plateau"}:
        base -= 12.0
    return float(max(30.0, min(92.0, base)))


def _winter_cold_proxy(cold_days: float) -> float:
    if cold_days >= 18:
        return 5.0
    if cold_days >= 10:
        return 4.0
    if cold_days >= 4:
        return 3.0
    if cold_days >= 1:
        return 2.0
    return 1.0
