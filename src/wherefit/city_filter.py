"""Candidate city filtering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


REPRESENTATIVE_ADMIN_LEVELS = {"provincial_capital", "municipality", "capital"}


@dataclass(frozen=True)
class CityFilterOptions:
    representative_only: bool = True
    countries: tuple[str, ...] = ()
    provinces: tuple[str, ...] = ()
    cities: tuple[str, ...] = ()
    regions: tuple[str, ...] = ()
    max_heat_level: int = 5
    max_humidity_level: int = 5
    max_precipitation_level: int = 5
    max_air_quality_level: int = 5
    max_winter_cold_level: int = 5
    exclude_typhoon_region: bool = False
    exclude_coastal: bool = False


def filter_seed_cities(data: pd.DataFrame, options: CityFilterOptions) -> pd.DataFrame:
    filtered = data.copy()
    if options.representative_only:
        filtered = filtered[filtered.get("admin_level", "").isin(REPRESENTATIVE_ADMIN_LEVELS)]
    filtered = _filter_in(filtered, "country", options.countries)
    filtered = _filter_in(filtered, "province", options.provinces)
    filtered = _filter_in(filtered, "city", options.cities)
    filtered = _filter_in(filtered, "region_type", options.regions)
    filtered = _filter_numeric_max(filtered, "summer_heat_level", options.max_heat_level)
    filtered = _filter_numeric_max(filtered, "humidity_level", options.max_humidity_level)
    filtered = _filter_numeric_max(filtered, "precipitation_level", options.max_precipitation_level)
    filtered = _filter_numeric_max(filtered, "air_quality_level", options.max_air_quality_level)
    filtered = _filter_numeric_max(filtered, "winter_cold_level", options.max_winter_cold_level)
    if options.exclude_typhoon_region and "typhoon_region" in filtered.columns:
        filtered = filtered[~filtered["typhoon_region"].map(_to_bool)]
    if options.exclude_coastal and "coastal" in filtered.columns:
        filtered = filtered[~filtered["coastal"].map(_to_bool)]
    return _sort_candidates(filtered).reset_index(drop=True)


def candidate_city_input(data: pd.DataFrame, limit: int | None = None, lang: str = "zh") -> str:
    rows = data if limit is None else data.head(limit)
    if lang == "en" and "city_en" in rows.columns:
        name_column = "city_en"
    else:
        name_column = "city_zh" if "city_zh" in rows.columns else "city"
    return ", ".join(str(city) for city in rows[name_column].tolist())


def _filter_in(data: pd.DataFrame, column: str, values: Iterable[str]) -> pd.DataFrame:
    chosen = [value for value in values if value]
    if not chosen or column not in data.columns:
        return data
    return data[data[column].isin(chosen)]


def _filter_numeric_max(data: pd.DataFrame, column: str, max_value: int) -> pd.DataFrame:
    if column not in data.columns:
        return data
    values = pd.to_numeric(data[column], errors="coerce")
    return data[values <= int(max_value)]


def _sort_candidates(data: pd.DataFrame) -> pd.DataFrame:
    priority = {
        "municipality": 0,
        "provincial_capital": 1,
        "capital": 2,
        "special_admin": 3,
        "city": 4,
    }
    work = data.copy()
    work["_admin_priority"] = work.get("admin_level", "").map(priority).fillna(9)
    sort_columns = ["_admin_priority"]
    if "province" in work.columns:
        sort_columns.append("province")
    if "city_zh" in work.columns:
        sort_columns.append("city_zh")
    elif "city" in work.columns:
        sort_columns.append("city")
    return work.sort_values(sort_columns).drop(columns=["_admin_priority"])


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}
