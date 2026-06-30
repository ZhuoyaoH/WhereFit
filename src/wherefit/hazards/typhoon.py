"""IBTrACS Western Pacific typhoon track summaries."""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd
import requests

from wherefit.config import HISTORY_START_DATE, IBTRACS_WP_CSV_URL
from wherefit.models import Location, TyphoonSummary


IBTRACS_COLUMNS = ["SID", "SEASON", "NAME", "ISO_TIME", "LAT", "LON", "WMO_WIND", "USA_WIND", "BASIN"]


def get_typhoon_summary(
    location: Location,
    cache_dir: Path,
    start_year: int = 2000,
    force_refresh: bool = False,
    timeout: int = 90,
) -> TyphoonSummary:
    cache_path = cache_dir / "ibtracs.WP.list.v04r01.csv"
    try:
        if force_refresh or not cache_path.exists():
            _download_ibtracs(cache_path, timeout=timeout)
        data = _load_ibtracs(cache_path, start_year)
        return summarize_typhoon_tracks(data, location)
    except Exception as exc:
        return TyphoonSummary(
            city=location.city,
            count_100km=0,
            count_200km=0,
            count_500km=0,
            nearest_distance_km=None,
            strongest_name=None,
            strongest_year=None,
            strongest_wind=None,
            latest_nearby_name=None,
            latest_nearby_year=None,
            source="NOAA IBTrACS Western Pacific",
            status="failed",
            message=f"IBTrACS 台风数据不可用：{exc}",
        )


def summarize_typhoon_tracks(data: pd.DataFrame, location: Location) -> TyphoonSummary:
    if data.empty:
        return _empty_summary(location, "cache", "IBTrACS 数据为空或无 2000 年后记录")
    work = data.copy()
    work["distance_km"] = work.apply(
        lambda row: _distance_km(location.latitude, location.longitude, row["LAT"], row["LON"]),
        axis=1,
    )
    work["wind"] = pd.to_numeric(work["WMO_WIND"], errors="coerce").fillna(
        pd.to_numeric(work["USA_WIND"], errors="coerce")
    )
    nearby_500 = work[work["distance_km"] <= 500].copy()
    if nearby_500.empty:
        nearest = float(work["distance_km"].min()) if not work.empty else None
        return TyphoonSummary(
            city=location.city,
            count_100km=0,
            count_200km=0,
            count_500km=0,
            nearest_distance_km=round(nearest, 1) if nearest is not None else None,
            strongest_name=None,
            strongest_year=None,
            strongest_wind=None,
            latest_nearby_name=None,
            latest_nearby_year=None,
            source="NOAA IBTrACS Western Pacific",
            status="cache/live",
            message="2000 年以来 500km 内未匹配到台风路径点。",
        )
    storm_min = nearby_500.groupby("SID")["distance_km"].min()
    count_100 = int((storm_min <= 100).sum())
    count_200 = int((storm_min <= 200).sum())
    count_500 = int((storm_min <= 500).sum())
    strongest = nearby_500.sort_values("wind", ascending=False, na_position="last").iloc[0]
    latest = nearby_500.sort_values("SEASON", ascending=False).iloc[0]
    return TyphoonSummary(
        city=location.city,
        count_100km=count_100,
        count_200km=count_200,
        count_500km=count_500,
        nearest_distance_km=round(float(storm_min.min()), 1),
        strongest_name=_clean_name(strongest.get("NAME")),
        strongest_year=int(strongest.get("SEASON")) if pd.notna(strongest.get("SEASON")) else None,
        strongest_wind=float(strongest.get("wind")) if pd.notna(strongest.get("wind")) else None,
        latest_nearby_name=_clean_name(latest.get("NAME")),
        latest_nearby_year=int(latest.get("SEASON")) if pd.notna(latest.get("SEASON")) else None,
        source="NOAA IBTrACS Western Pacific",
        status="cache/live",
        message="已基于 IBTrACS 西北太平洋路径点统计 100/200/500km 接近次数。",
    )


def _download_ibtracs(path: Path, timeout: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(IBTRACS_WP_CSV_URL, timeout=timeout)
    response.raise_for_status()
    path.write_bytes(response.content)


def _load_ibtracs(path: Path, start_year: int) -> pd.DataFrame:
    data = pd.read_csv(path, skiprows=[1], usecols=lambda col: col in IBTRACS_COLUMNS, low_memory=False)
    data["SEASON"] = pd.to_numeric(data["SEASON"], errors="coerce")
    data["LAT"] = pd.to_numeric(data["LAT"], errors="coerce")
    data["LON"] = pd.to_numeric(data["LON"], errors="coerce")
    data = data[(data["SEASON"] >= start_year) & data["LAT"].notna() & data["LON"].notna()]
    return data


def _empty_summary(location: Location, status: str, message: str) -> TyphoonSummary:
    return TyphoonSummary(
        city=location.city,
        count_100km=0,
        count_200km=0,
        count_500km=0,
        nearest_distance_km=None,
        strongest_name=None,
        strongest_year=None,
        strongest_wind=None,
        latest_nearby_name=None,
        latest_nearby_year=None,
        source="NOAA IBTrACS Western Pacific",
        status=status,
        message=message,
    )


def _clean_name(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    name = str(value).strip()
    if not name or name.lower() == "nan":
        return None
    return name


def _distance_km(lat1: float, lon1: float, lat2: object, lon2: object) -> float:
    if pd.isna(lat2) or pd.isna(lon2):
        return 99999.0
    radius = 6371.0
    d_lat = radians(float(lat2) - lat1)
    d_lon = radians(float(lon2) - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(float(lat2))) * sin(d_lon / 2) ** 2
    return 2 * radius * asin(sqrt(a))
