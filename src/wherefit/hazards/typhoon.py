"""Location-aware IBTrACS tropical-cyclone track summaries."""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd
import requests

from wherefit.config import IBTRACS_CSV_URL_TEMPLATE
from wherefit.models import Location, TyphoonSummary


IBTRACS_COLUMNS = ["SID", "SEASON", "NAME", "ISO_TIME", "LAT", "LON", "WMO_WIND", "USA_WIND", "BASIN"]
IBTRACS_BASIN_NAMES = {
    "NA": "North Atlantic",
    "EP": "Eastern Pacific",
    "WP": "Western Pacific",
    "NI": "North Indian",
    "SI": "South Indian",
    "SP": "South Pacific",
}


def get_typhoon_summary(
    location: Location,
    cache_dir: Path,
    start_year: int = 2000,
    force_refresh: bool = False,
    timeout: int = 90,
) -> TyphoonSummary:
    """Load the relevant official basin subset(s) and summarize nearby tracks."""

    basins = basins_for_location(location)
    source = _source_label(basins)
    try:
        frames: list[pd.DataFrame] = []
        downloaded = False
        for basin, cache_path in zip(basins, typhoon_cache_paths(location, cache_dir)):
            if force_refresh or not cache_path.exists():
                _download_ibtracs(cache_path, basin=basin, timeout=timeout)
                downloaded = True
            frames.append(_load_ibtracs(cache_path, start_year))
        data = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["SID", "ISO_TIME"])
        status = "live" if downloaded else "cache"
        return summarize_typhoon_tracks(data, location, source=source, status=status)
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
            source=source,
            status="failed",
            message=f"IBTrACS 台风数据不可用：{exc}",
        )


def basins_for_location(location: Location) -> tuple[str, ...]:
    """Select the IBTrACS basin subset nearest to a city coordinate."""

    longitude = ((float(location.longitude) + 180.0) % 360.0) - 180.0
    latitude = float(location.latitude)
    if latitude < 0:
        if longitude >= 135.0 or longitude < -120.0:
            return ("SP",)
        if longitude >= 10.0:
            return ("SI",)
        return ("SP", "SI")
    if longitude >= 100.0 or longitude < -160.0:
        return ("WP",)
    if latitude >= 35.0 and -100.0 <= longitude < 45.0:
        return ("NA",)
    if 80.0 <= longitude < 100.0:
        return ("NI", "WP")
    if 20.0 <= longitude < 100.0:
        return ("NI",)
    if -100.0 <= longitude < 20.0:
        return ("NA",)
    return ("EP",)


def typhoon_cache_paths(location: Location, cache_dir: Path) -> list[Path]:
    """Return deterministic cache files for the city's selected basin subsets."""

    return [cache_dir / f"ibtracs.{basin}.list.v04r01.csv" for basin in basins_for_location(location)]


def summarize_typhoon_tracks(
    data: pd.DataFrame,
    location: Location,
    source: str = "NOAA IBTrACS v04r01",
    status: str = "cache/live",
) -> TyphoonSummary:
    """Count unique storms by closest approach and retain map-ready track samples."""

    if data.empty:
        return _empty_summary(location, source, status, "IBTrACS 数据为空或无 2000 年后记录")
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
            source=source,
            status=status,
            message="2000 年以来所选海盆数据在 500km 内未匹配到热带气旋路径点。",
            track_points=_track_records(work.sort_values("distance_km").head(80)),
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
        source=source,
        status=status,
        message="已基于 IBTrACS 对应海盆路径点统计 100/200/500km 接近次数。",
        track_points=_track_records(nearby_500),
    )


def _download_ibtracs(path: Path, basin: str, timeout: int) -> None:
    """Download one official IBTrACS basin CSV to the runtime cache."""

    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(IBTRACS_CSV_URL_TEMPLATE.format(basin=basin), timeout=timeout)
    response.raise_for_status()
    path.write_bytes(response.content)


def _load_ibtracs(path: Path, start_year: int) -> pd.DataFrame:
    data = pd.read_csv(path, skiprows=[1], usecols=lambda col: col in IBTRACS_COLUMNS, low_memory=False)
    data["SEASON"] = pd.to_numeric(data["SEASON"], errors="coerce")
    data["LAT"] = pd.to_numeric(data["LAT"], errors="coerce")
    data["LON"] = pd.to_numeric(data["LON"], errors="coerce")
    data = data[(data["SEASON"] >= start_year) & data["LAT"].notna() & data["LON"].notna()]
    return data


def _empty_summary(location: Location, source: str, status: str, message: str) -> TyphoonSummary:
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
        source=source,
        status=status,
        message=message,
        track_points=[],
    )


def _source_label(basins: tuple[str, ...]) -> str:
    """Build a human-readable provenance label for selected basin files."""

    names = ", ".join(IBTRACS_BASIN_NAMES[basin] for basin in basins)
    return f"NOAA IBTrACS v04r01 ({names})"


def _clean_name(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    name = str(value).strip()
    if not name or name.lower() == "nan":
        return None
    return name


def _track_records(data: pd.DataFrame, limit: int = 900) -> list[dict[str, object]]:
    if data.empty:
        return []
    work = data.copy()
    work["ISO_TIME"] = pd.to_datetime(work.get("ISO_TIME"), errors="coerce")
    work["LAT"] = pd.to_numeric(work.get("LAT"), errors="coerce")
    work["LON"] = pd.to_numeric(work.get("LON"), errors="coerce")
    work["distance_km"] = pd.to_numeric(work.get("distance_km"), errors="coerce")
    work["wind"] = pd.to_numeric(work.get("wind"), errors="coerce")
    work = work.dropna(subset=["LAT", "LON"])
    if "SID" in work.columns:
        work = work.sort_values(["SID", "ISO_TIME"], ascending=[True, True])
    records: list[dict[str, object]] = []
    for _, row in work.head(limit).iterrows():
        event_time = row.get("ISO_TIME")
        season = row.get("SEASON")
        records.append(
            {
                "sid": str(row.get("SID") or ""),
                "name": _clean_name(row.get("NAME")) or "未命名",
                "season": int(season) if pd.notna(season) else None,
                "time": event_time.date().isoformat() if not pd.isna(event_time) else "",
                "latitude": float(row.get("LAT")),
                "longitude": float(row.get("LON")),
                "distance_km": round(float(row.get("distance_km")), 1) if pd.notna(row.get("distance_km")) else None,
                "wind": float(row.get("wind")) if pd.notna(row.get("wind")) else None,
            }
        )
    return records


def _distance_km(lat1: float, lon1: float, lat2: object, lon2: object) -> float:
    if pd.isna(lat2) or pd.isna(lon2):
        return 99999.0
    radius = 6371.0
    d_lat = radians(float(lat2) - lat1)
    d_lon = radians(float(lon2) - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(float(lat2))) * sin(d_lon / 2) ** 2
    return 2 * radius * asin(sqrt(a))
