"""Open-Meteo historical weather provider."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from wherefit.config import HISTORY_START_DATE, OPEN_METEO_ARCHIVE_URL, OPEN_METEO_HISTORY_MODEL
from wherefit.data_sources.cache import read_csv_cache, safe_cache_name, write_csv_cache
from wherefit.models import ClimateMetrics, Location


DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_mean",
    "apparent_temperature_max",
    "relative_humidity_2m_mean",
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


@dataclass(frozen=True)
class _ReusableChunk:
    data: pd.DataFrame
    path: Path
    end_date: date


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
    cache_path = cache_dir / safe_cache_name(
        location.city_en or location.city,
        start_date,
        end,
        OPEN_METEO_HISTORY_MODEL,
        "daily",
    )
    try:
        start_value = _parse_date(start_date)
        end_value = _parse_date(end)
        if end_value < start_value:
            raise ValueError("history end date is earlier than start date")
    except ValueError as exc:
        return HistoricalFetchResult(
            metrics=None,
            raw=None,
            source="Open-Meteo Historical Weather API",
            status="failed",
            message=f"历史日期范围无效，未发起请求：{exc}",
            cache_path=cache_path,
        )
    cached = None if force_refresh else read_csv_cache(cache_path)
    if cached is not None and _cache_covers_range(cached, start_date, end):
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
    cache_hits = 0
    downloads = 0
    location_key = location.city_en or location.city
    for chunk_start, chunk_end in _date_chunks(start_date, end_date, years=CHUNK_YEARS):
        chunk_path = _chunk_cache_path(cache_dir, location_key, chunk_start, chunk_end)
        reusable = None if force_refresh else _read_reusable_chunk(cache_dir, location_key, chunk_start, chunk_end)
        chunk = None
        if reusable is not None:
            cache_hits += 1
            chunk = reusable.data
            if reusable.end_date < _parse_date(chunk_end):
                missing_start = (reusable.end_date + timedelta(days=1)).isoformat()
                try:
                    tail = _fetch_history_chunk(location, missing_start, chunk_end, timeout)
                    chunk = _merge_daily_frames([chunk, tail], chunk_start, chunk_end)
                    downloads += 1
                    write_csv_cache(chunk_path, chunk)
                except Exception as exc:
                    failures.append(f"{missing_start}~{chunk_end}: {exc}")
            elif reusable.path != chunk_path:
                write_csv_cache(chunk_path, chunk)
        if chunk is None:
            try:
                chunk = _fetch_history_chunk(location, chunk_start, chunk_end, timeout)
                downloads += 1
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
            message="历史天气暂时无法获取，也没有已保存的数据，当前改用多年平均数据：" + "; ".join(failures[:2]),
            cache_path=full_cache_path,
        )

    raw = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")
    has_full_coverage = _cache_covers_range(raw, start_date, end_date)
    if has_full_coverage:
        write_csv_cache(full_cache_path, raw)
    status = "cache" if downloads == 0 and not failures else "partial" if failures else "live"
    message = f"已聚合 Open-Meteo 历史天气：{len(chunks)} 个分块（缓存 {cache_hits}，下载 {downloads}）"
    if has_full_coverage:
        message += f"，完整缓存 {full_cache_path.name}"
    else:
        message += "，当前可用分块未覆盖完整日期范围"
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
        "models": OPEN_METEO_HISTORY_MODEL,
        "wind_speed_unit": "ms",
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


def _chunk_cache_path(cache_dir: Path, location_key: str, start_date: str, end_date: str) -> Path:
    return cache_dir / "chunks" / safe_cache_name(
        location_key,
        start_date,
        end_date,
        OPEN_METEO_HISTORY_MODEL,
        "daily",
    )


def _read_reusable_chunk(cache_dir: Path, location_key: str, chunk_start: str, chunk_end: str) -> _ReusableChunk | None:
    exact_path = _chunk_cache_path(cache_dir, location_key, chunk_start, chunk_end)
    exact = read_csv_cache(exact_path)
    if exact is not None:
        trimmed = _trim_dates(exact, chunk_start, chunk_end)
        actual_end = _contiguous_coverage_end(trimmed, chunk_start)
        if actual_end is not None:
            return _ReusableChunk(data=trimmed, path=exact_path, end_date=actual_end)

    safe_location = safe_cache_name(location_key, suffix="")
    candidates: list[tuple[date, Path]] = []
    pattern = f"{safe_location}_{chunk_start}_*_{OPEN_METEO_HISTORY_MODEL}_daily.csv"
    for path in (cache_dir / "chunks").glob(pattern):
        parsed_end = _parse_chunk_end(path, safe_location, chunk_start)
        if parsed_end is not None and parsed_end >= _parse_date(chunk_start):
            candidates.append((parsed_end, path))
    if not candidates:
        return None

    target_end = _parse_date(chunk_end)
    usable = [(end, path) for end, path in candidates if end <= target_end]
    if usable:
        reusable_end, reusable_path = max(usable, key=lambda item: item[0])
    else:
        reusable_end, reusable_path = min(candidates, key=lambda item: item[0])
    data = read_csv_cache(reusable_path)
    if data is None:
        return None
    trimmed = _trim_dates(data, chunk_start, chunk_end)
    actual_end = _contiguous_coverage_end(trimmed, chunk_start)
    if actual_end is None:
        return None
    return _ReusableChunk(data=trimmed, path=reusable_path, end_date=min(reusable_end, actual_end))


def _parse_chunk_end(path: Path, safe_location: str, chunk_start: str) -> date | None:
    prefix = f"{safe_location}_{chunk_start}_"
    suffix = f"_{OPEN_METEO_HISTORY_MODEL}_daily.csv"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    raw_end = name[len(prefix) : -len(suffix)]
    try:
        return _parse_date(raw_end)
    except ValueError:
        return None


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _trim_dates(data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    work = data.copy()
    if "time" not in work.columns:
        return work
    time = pd.to_datetime(work["time"], errors="coerce")
    mask = (time >= pd.Timestamp(start_date)) & (time <= pd.Timestamp(end_date))
    return work.loc[mask].copy()


def _merge_daily_frames(frames: list[pd.DataFrame], start_date: str, end_date: str) -> pd.DataFrame:
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")
    return _trim_dates(merged, start_date, end_date)


def _cache_covers_range(data: pd.DataFrame, start_date: str, end_date: str) -> bool:
    if data.empty or "time" not in data.columns:
        return False
    times = pd.to_datetime(data["time"], errors="coerce").dropna().dt.normalize()
    if times.empty:
        return False
    expected = pd.date_range(start_date, end_date, freq="D")
    actual = pd.DatetimeIndex(times.unique())
    return bool(expected.isin(actual).all())


def _contiguous_coverage_end(data: pd.DataFrame, start_date: str) -> date | None:
    """Return the actual complete end date instead of trusting a cache filename."""

    if data.empty or "time" not in data.columns:
        return None
    times = pd.to_datetime(data["time"], errors="coerce").dropna().dt.normalize()
    if times.empty:
        return None
    actual = pd.DatetimeIndex(sorted(times.unique()))
    start = pd.Timestamp(start_date)
    if actual[0] != start:
        return None
    expected = pd.date_range(start, actual[-1], freq="D")
    if len(actual) != len(expected) or not expected.isin(actual).all():
        return None
    return actual[-1].date()


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
    required = {
        "time",
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
    }
    missing_required = sorted(required - set(raw.columns))
    if missing_required:
        raise ValueError(f"historical weather is missing required fields: {missing_required}")
    data = raw.copy()
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    if int(month) != 0:
        data = data[data["time"].dt.month == int(month)]
    if data.empty:
        raise ValueError(f"no historical weather rows for month {month}")

    temp_mean = _mean(data, "temperature_2m_mean", default=20.0)
    temp_max = _mean(data, "temperature_2m_max", default=temp_mean + 5.0)
    app_mean = _mean(data, "apparent_temperature_mean", default=temp_mean)
    precipitation = _series(data, "precipitation_sum")
    rain_days = float((precipitation >= 1.0).sum() / _sample_years(data))
    heavy_rain_days = float((precipitation >= 20.0).sum() / _sample_years(data))
    extreme_rain_days = float((precipitation >= 50.0).sum() / _sample_years(data))
    hot_days = float((_series(data, "temperature_2m_max") >= 35.0).sum() / _sample_years(data))
    cold_days = float((_series(data, "temperature_2m_min") <= 0.0).sum() / _sample_years(data))
    windy_days = float((_series(data, "wind_speed_10m_max") >= 10.8).sum() / _sample_years(data))
    snow_days = float((_series(data, "snowfall_sum") > 0.0).sum() / _sample_years(data))
    missing_rate = float(data.isna().sum().sum() / max(1, data.shape[0] * data.shape[1]))

    humidity_values = (
        pd.to_numeric(data["relative_humidity_2m_mean"], errors="coerce").dropna()
        if "relative_humidity_2m_mean" in data.columns
        else pd.Series(dtype=float)
    )
    estimated_fields: list[str] = []
    if humidity_values.empty:
        humidity_month = 0 if int(month) == 0 else int(month)
        humidity_rain_days = rain_days / 12.0 if int(month) == 0 else rain_days
        humidity = _estimate_humidity(location, humidity_month, humidity_rain_days)
        estimated_fields.append("relative_humidity_mean")
    else:
        humidity = float(humidity_values.mean())
    for field, source_column in {
        "apparent_temperature": "apparent_temperature_mean",
        "windy_days": "wind_speed_10m_max",
        "snow_days": "snowfall_sum",
    }.items():
        if source_column not in data.columns:
            estimated_fields.append(field)
    return ClimateMetrics(
        temperature_mean=temp_mean,
        temperature_max=temp_max,
        apparent_temperature=app_mean,
        relative_humidity_mean=humidity,
        precipitation_days=rain_days,
        heavy_rain_days=heavy_rain_days,
        pm25=fallback_pm25,
        hot_days=hot_days,
        winter_cold_level=_winter_cold_proxy(cold_days),
        coastal=location.coastal,
        typhoon_region=location.typhoon_region,
        data_source="Open-Meteo Historical Weather API (ERA5)",
        data_status="cache/live",
        sample_years=_sample_years(data),
        missing_rate=round(missing_rate, 3),
        precipitation_extreme_days=extreme_rain_days,
        cold_days=cold_days,
        windy_days=windy_days,
        snow_days=snow_days,
        period_months=12 if int(month) == 0 else 1,
        estimated_fields=tuple(estimated_fields),
        fallback_fields=("pm25",),
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


def _sample_years(data: pd.DataFrame) -> int:
    years = data["time"].dt.year.dropna().nunique()
    return max(1, int(years))


def _estimate_humidity(location: Location, month: int, rain_days: float) -> float:
    base = 52.0 + min(24.0, rain_days * 1.2)
    if location.coastal:
        base += 8.0
    if location.region_type in {"south_coast", "east_coast", "southwest_basin"}:
        base += 5.0
    warm_months = {12, 1, 2, 3} if location.latitude < 0 else {6, 7, 8, 9}
    if month in warm_months:
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
