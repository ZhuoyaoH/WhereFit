"""Load static seed data and convert it into scoring inputs."""

from __future__ import annotations

from dataclasses import replace
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from wherefit.models import ClimateMetrics, Location


REQUIRED_COLUMNS = {
    "city",
    "country",
    "latitude",
    "longitude",
    "timezone",
    "coastal",
    "typhoon_region",
    "summer_heat_level",
    "humidity_level",
    "air_quality_level",
    "precipitation_level",
    "winter_cold_level",
}

LONG_TERM_AIR_QUALITY_COLUMNS = {
    "city",
    "year",
    "pm25_mean_ug_m3",
    "valid_grid_cells",
    "grid_cell_count",
    "source_product",
    "source_version",
    "spatial_method",
}

CLIMATE_BASELINE_COLUMNS = {
    "city",
    "period_month",
    "year_start",
    "year_end",
    "sample_years",
    "temperature_mean_c",
    "daily_max_temperature_mean_c",
    "apparent_temperature_mean_c",
    "relative_humidity_mean_pct",
    "precipitation_days_mean",
    "heavy_rain_days_mean",
    "extreme_rain_days_mean",
    "hot_days_mean",
    "cold_days_mean",
    "windy_days_mean",
    "snow_days_mean",
    "missing_rate",
    "source_product",
    "source_model",
    "spatial_resolution",
    "spatial_method",
}

CITY_ALIASES = {
    "beijing": "Beijing",
    "北京": "Beijing",
    "shanghai": "Shanghai",
    "上海": "Shanghai",
    "guangzhou": "Guangzhou",
    "广州": "Guangzhou",
    "canton": "Guangzhou",
    "shenzhen": "Shenzhen",
    "深圳": "Shenzhen",
    "hangzhou": "Hangzhou",
    "杭州": "Hangzhou",
    "nanjing": "Nanjing",
    "南京": "Nanjing",
    "suzhou": "Suzhou",
    "苏州": "Suzhou",
    "qingdao": "Qingdao",
    "青岛": "Qingdao",
    "dalian": "Dalian",
    "大连": "Dalian",
    "tianjin": "Tianjin",
    "天津": "Tianjin",
    "chengdu": "Chengdu",
    "成都": "Chengdu",
    "chongqing": "Chongqing",
    "重庆": "Chongqing",
    "wuhan": "Wuhan",
    "武汉": "Wuhan",
    "changsha": "Changsha",
    "长沙": "Changsha",
    "xian": "Xian",
    "xi'an": "Xian",
    "西安": "Xian",
    "zhengzhou": "Zhengzhou",
    "郑州": "Zhengzhou",
    "jinan": "Jinan",
    "济南": "Jinan",
    "hefei": "Hefei",
    "合肥": "Hefei",
    "fuzhou": "Fuzhou",
    "福州": "Fuzhou",
    "xiamen": "Xiamen",
    "厦门": "Xiamen",
    "kunming": "Kunming",
    "昆明": "Kunming",
    "guiyang": "Guiyang",
    "贵阳": "Guiyang",
    "nanning": "Nanning",
    "南宁": "Nanning",
    "haikou": "Haikou",
    "海口": "Haikou",
    "sanya": "Sanya",
    "三亚": "Sanya",
    "harbin": "Harbin",
    "哈尔滨": "Harbin",
    "changchun": "Changchun",
    "长春": "Changchun",
    "shenyang": "Shenyang",
    "沈阳": "Shenyang",
    "hohhot": "Hohhot",
    "huhehaote": "Hohhot",
    "呼和浩特": "Hohhot",
    "urumqi": "Urumqi",
    "wulumuqi": "Urumqi",
    "乌鲁木齐": "Urumqi",
    "lanzhou": "Lanzhou",
    "兰州": "Lanzhou",
    "xining": "Xining",
    "西宁": "Xining",
    "yinchuan": "Yinchuan",
    "银川": "Yinchuan",
    "lhasa": "Lhasa",
    "拉萨": "Lhasa",
    "taiyuan": "Taiyuan",
    "太原": "Taiyuan",
    "shijiazhuang": "Shijiazhuang",
    "石家庄": "Shijiazhuang",
    "ningbo": "Ningbo",
    "宁波": "Ningbo",
    "wenzhou": "Wenzhou",
    "温州": "Wenzhou",
    "zhuhai": "Zhuhai",
    "珠海": "Zhuhai",
    "hong kong": "Hong Kong",
    "hongkong": "Hong Kong",
    "香港": "Hong Kong",
    "香港特别行政区": "Hong Kong",
    "macau": "Macau",
    "macao": "Macau",
    "澳门": "Macau",
    "澳门特别行政区": "Macau",
    "taipei": "Taipei",
    "台北": "Taipei",
    "mohe": "Mohe",
    "漠河": "Mohe",
    "heihe": "Heihe",
    "黑河": "Heihe",
    "hulunbuir": "Hulunbuir",
    "呼伦贝尔": "Hulunbuir",
    "genhe": "Genhe",
    "根河": "Genhe",
    "altay": "Altay",
    "aletai": "Altay",
    "阿勒泰": "Altay",
    "tokyo": "Tokyo",
    "东京": "Tokyo",
    "osaka": "Osaka",
    "大阪": "Osaka",
    "kyoto": "Kyoto",
    "京都": "Kyoto",
    "sapporo": "Sapporo",
    "札幌": "Sapporo",
    "fukuoka": "Fukuoka",
    "福冈": "Fukuoka",
    "naha": "Naha",
    "那霸": "Naha",
    "seoul": "Seoul",
    "首尔": "Seoul",
    "singapore": "Singapore",
    "新加坡": "Singapore",
    "bangkok": "Bangkok",
    "曼谷": "Bangkok",
}

ZH_CITY_DISPLAY_NAMES = {
    "Hong Kong": "香港特别行政区",
    "Macau": "澳门特别行政区",
}


def display_city_name(city: object, city_en: object = "", lang: str = "zh") -> str:
    """Return the localized city label without changing the stable city identifier."""

    chinese_name = str(city)
    english_name = str(city_en) if str(city_en).strip() else chinese_name
    if lang == "en":
        return english_name
    return ZH_CITY_DISPLAY_NAMES.get(english_name, chinese_name)


def load_seed_cities(path: str | Path) -> pd.DataFrame:
    """Load and validate the curated city baseline before any scoring."""

    data = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"city seed data missing columns: {sorted(missing)}")
    _validate_seed_data(data)
    return data


def attach_long_term_air_quality(seed_data: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Attach a validated city-level multi-year PM2.5 summary to seed rows."""

    source_path = Path(path)
    if not source_path.exists():
        return seed_data.copy()
    data = pd.read_csv(source_path)
    missing = LONG_TERM_AIR_QUALITY_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"long-term air-quality data missing columns: {sorted(missing)}")
    _validate_long_term_air_quality(data)
    summaries: list[dict[str, object]] = []
    for city, group in data.groupby("city", sort=True):
        ordered = group.sort_values("year")
        years = pd.to_numeric(ordered["year"], errors="raise").astype(int)
        values = pd.to_numeric(ordered["pm25_mean_ug_m3"], errors="raise").astype(float)
        summaries.append(
            {
                "city": str(city),
                "long_term_pm25": float(values.mean()),
                "long_term_pm25_std": float(values.std(ddof=0)),
                "long_term_pm25_trend": _linear_slope(years.tolist(), values.tolist()),
                "pm25_year_start": int(years.min()),
                "pm25_year_end": int(years.max()),
                "pm25_sample_years": int(years.nunique()),
                "pm25_source": f"{ordered['source_product'].iloc[0]} {ordered['source_version'].iloc[0]}",
                "pm25_status": "dataset",
                "pm25_spatial_method": str(ordered["spatial_method"].iloc[0]),
            }
        )
    summary = pd.DataFrame(summaries)
    return seed_data.merge(summary, on="city", how="left", validate="one_to_one")


def load_climate_baseline(monthly_path: str | Path, annual_path: str | Path) -> pd.DataFrame:
    """Load validated precomputed monthly and annual climate-normal rows."""

    monthly_source = Path(monthly_path)
    annual_source = Path(annual_path)
    if not monthly_source.exists() or not annual_source.exists():
        return pd.DataFrame(columns=sorted(CLIMATE_BASELINE_COLUMNS))
    data = pd.concat(
        [pd.read_csv(monthly_source), pd.read_csv(annual_source)],
        ignore_index=True,
    )
    missing = CLIMATE_BASELINE_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"climate baseline data missing columns: {sorted(missing)}")
    _validate_climate_baseline(data)
    return data


def parse_city_input(raw_text: str) -> list[str]:
    pieces = re.split(r"[,，\n]+", raw_text)
    cities: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        name = piece.strip()
        if not name:
            continue
        canonical = canonical_city_name(name)
        key = canonical.lower()
        if key not in seen:
            cities.append(canonical)
            seen.add(key)
    return cities


def canonical_city_name(name: str) -> str:
    key = name.strip().lower()
    return CITY_ALIASES.get(key, name.strip())


def match_cities(data: pd.DataFrame, requested: Iterable[str]) -> tuple[pd.DataFrame, list[str]]:
    canonical_to_requested = {canonical_city_name(city).lower(): city for city in requested}
    available = data.copy()
    available["_match_keys"] = available.apply(_row_match_keys, axis=1)
    matched = available[available["_match_keys"].apply(lambda keys: bool(keys & set(canonical_to_requested.keys())))]
    found: set[str] = set()
    for keys in matched["_match_keys"]:
        found.update(keys)
    matched = matched.drop(columns=["_match_keys"])
    missing = [city for key, city in canonical_to_requested.items() if key not in found]
    return matched.reset_index(drop=True), missing


def row_to_location(row: pd.Series) -> Location:
    city_en = str(row.get("city_en", row["city"]))
    display_city = str(row.get("city_zh", row["city"]))
    return Location(
        city=display_city,
        country=str(row["country"]),
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"]),
        timezone=str(row["timezone"]),
        coastal=_to_bool(row["coastal"]),
        typhoon_region=_to_bool(row["typhoon_region"]),
        province=str(row.get("province", "")),
        city_en=city_en,
        region_type=str(row.get("region_type", "")),
        admin_level=str(row.get("admin_level", "")),
    )


def row_to_metrics(
    row: pd.Series,
    month: int,
    climate_baseline: pd.DataFrame | None = None,
) -> ClimateMetrics:
    """Convert one city row to measured baseline metrics or an explicit fallback."""

    if climate_baseline is not None and not climate_baseline.empty:
        matched = climate_baseline[
            (climate_baseline["city"].astype(str) == str(row["city"]))
            & (pd.to_numeric(climate_baseline["period_month"], errors="coerce") == int(month))
        ]
        if len(matched) == 1:
            return _metrics_from_climate_baseline(row, matched.iloc[0], month)

    if int(month) == 0:
        return _annual_metrics_from_seed_row(row)
    heat_level = float(row["summer_heat_level"])
    humidity_level = float(row["humidity_level"])
    precipitation_level = float(row["precipitation_level"])
    air_quality_level = float(row["air_quality_level"])
    winter_cold_level = float(row["winter_cold_level"])

    season_factor = _season_factor(month, float(row["latitude"]))
    summer_factor = max(0.25, season_factor)
    winter_factor = max(0.0, -season_factor)

    temperature_max = 18 + heat_level * 3.0 + summer_factor * 7.0 - winter_factor * winter_cold_level * 2.5
    temperature_mean = temperature_max - 5.0 - winter_factor * 2.0
    humidity = 42 + humidity_level * 8.0 + summer_factor * 4.0
    precipitation_days = precipitation_level * (3.2 + summer_factor * 1.1)
    pm25, pm25_metadata = _pm25_from_row(row, air_quality_level)
    hot_days = max(0.0, (heat_level - 2.0) * 5.5 * summer_factor)
    heavy_rain_days = max(0.0, (precipitation_level - 2.0) * (0.9 + summer_factor * 0.6))
    apparent_temperature = temperature_mean + max(0.0, humidity - 60.0) * 0.08 + hot_days * 0.08

    return ClimateMetrics(
        temperature_mean=temperature_mean,
        temperature_max=temperature_max,
        apparent_temperature=apparent_temperature,
        relative_humidity_mean=humidity,
        precipitation_days=precipitation_days,
        heavy_rain_days=heavy_rain_days,
        pm25=pm25,
        hot_days=hot_days,
        winter_cold_level=winter_cold_level,
        coastal=_to_bool(row["coastal"]),
        typhoon_region=_to_bool(row["typhoon_region"]),
        data_source="基础城市数据",
        data_status="fallback",
        sample_years=0,
        missing_rate=0.0,
        precipitation_extreme_days=max(0.0, (precipitation_level - 3.0) * 0.5),
        cold_days=max(0.0, (winter_cold_level - 2.0) * 4.0 * winter_factor),
        windy_days=2.0 if _to_bool(row["coastal"]) else 1.0,
        snow_days=max(0.0, (winter_cold_level - 3.0) * 3.0 * winter_factor),
        period_months=1,
        fallback_fields=(
            "temperature_mean",
            "temperature_max",
            "apparent_temperature",
            "relative_humidity_mean",
            "precipitation_days",
            "heavy_rain_days",
            *(("pm25",) if pm25_metadata["pm25_status"] == "fallback" else ()),
            "hot_days",
            "cold_days",
            "windy_days",
            "snow_days",
        ),
        **pm25_metadata,
    )


def _metrics_from_climate_baseline(row: pd.Series, baseline: pd.Series, month: int) -> ClimateMetrics:
    """Map one versioned NASA POWER aggregate row into the scoring model."""

    pm25, pm25_metadata = _pm25_from_row(row, float(row["air_quality_level"]))
    fallback_fields: tuple[str, ...] = () if pm25_metadata["pm25_status"] != "fallback" else ("pm25",)
    cold_days = float(baseline["cold_days_mean"])
    return ClimateMetrics(
        temperature_mean=float(baseline["temperature_mean_c"]),
        temperature_max=float(baseline["daily_max_temperature_mean_c"]),
        apparent_temperature=float(baseline["apparent_temperature_mean_c"]),
        relative_humidity_mean=float(baseline["relative_humidity_mean_pct"]),
        precipitation_days=float(baseline["precipitation_days_mean"]),
        heavy_rain_days=float(baseline["heavy_rain_days_mean"]),
        pm25=pm25,
        hot_days=float(baseline["hot_days_mean"]),
        winter_cold_level=_winter_cold_level(cold_days),
        coastal=_to_bool(row["coastal"]),
        typhoon_region=_to_bool(row["typhoon_region"]),
        data_source=(
            f"{baseline['source_product']} "
            f"({baseline['source_model']}, {int(baseline['year_start'])}-{int(baseline['year_end'])} precomputed)"
        ),
        data_status="dataset",
        sample_years=int(baseline["sample_years"]),
        missing_rate=float(baseline["missing_rate"]),
        precipitation_extreme_days=float(baseline["extreme_rain_days_mean"]),
        cold_days=cold_days,
        windy_days=float(baseline["windy_days_mean"]),
        snow_days=float(baseline["snow_days_mean"]),
        period_months=12 if int(month) == 0 else 1,
        estimated_fields=("apparent_temperature", "snow_days"),
        fallback_fields=fallback_fields,
        **pm25_metadata,
    )


def _annual_metrics_from_seed_row(row: pd.Series) -> ClimateMetrics:
    """Aggregate all twelve monthly seed estimates into a full-year baseline."""

    monthly = [row_to_metrics(row, month) for month in range(1, 13)]
    first = monthly[0]
    return ClimateMetrics(
        temperature_mean=sum(item.temperature_mean for item in monthly) / 12.0,
        temperature_max=sum(item.temperature_max for item in monthly) / 12.0,
        apparent_temperature=sum(item.apparent_temperature for item in monthly) / 12.0,
        relative_humidity_mean=sum(item.relative_humidity_mean for item in monthly) / 12.0,
        precipitation_days=sum(item.precipitation_days for item in monthly),
        heavy_rain_days=sum(item.heavy_rain_days for item in monthly),
        pm25=sum(item.pm25 for item in monthly) / 12.0,
        hot_days=sum(item.hot_days for item in monthly),
        winter_cold_level=sum(item.winter_cold_level for item in monthly) / 12.0,
        coastal=first.coastal,
        typhoon_region=first.typhoon_region,
        data_source="基础城市数据（全年聚合）",
        data_status="fallback",
        sample_years=0,
        missing_rate=0.0,
        precipitation_extreme_days=sum(item.precipitation_extreme_days for item in monthly),
        cold_days=sum(item.cold_days for item in monthly),
        windy_days=sum(item.windy_days for item in monthly),
        snow_days=sum(item.snow_days for item in monthly),
        period_months=12,
        fallback_fields=first.fallback_fields,
        pm25_source=first.pm25_source,
        pm25_status=first.pm25_status,
        pm25_year_start=first.pm25_year_start,
        pm25_year_end=first.pm25_year_end,
        pm25_sample_years=first.pm25_sample_years,
        pm25_trend_per_year=first.pm25_trend_per_year,
        pm25_spatial_method=first.pm25_spatial_method,
    )


def with_long_term_air_quality(metrics: ClimateMetrics, source: ClimateMetrics) -> ClimateMetrics:
    """Copy long-term PM2.5 values and provenance into another climate record."""

    fallback_fields = set(metrics.fallback_fields)
    if source.pm25_status == "fallback":
        fallback_fields.add("pm25")
    else:
        fallback_fields.discard("pm25")
    return replace(
        metrics,
        pm25=source.pm25,
        fallback_fields=tuple(sorted(fallback_fields)),
        pm25_source=source.pm25_source,
        pm25_status=source.pm25_status,
        pm25_year_start=source.pm25_year_start,
        pm25_year_end=source.pm25_year_end,
        pm25_sample_years=source.pm25_sample_years,
        pm25_trend_per_year=source.pm25_trend_per_year,
        pm25_spatial_method=source.pm25_spatial_method,
    )


def _pm25_from_row(row: pd.Series, air_quality_level: float) -> tuple[float, dict[str, object]]:
    """Return continuous long-term PM2.5 data or an explicit seed fallback."""

    value = row.get("long_term_pm25")
    if value is not None and pd.notna(value):
        return float(value), {
            "pm25_source": str(row.get("pm25_source", "ACAG SatPM2.5")),
            "pm25_status": str(row.get("pm25_status", "dataset")),
            "pm25_year_start": int(row["pm25_year_start"]),
            "pm25_year_end": int(row["pm25_year_end"]),
            "pm25_sample_years": int(row["pm25_sample_years"]),
            "pm25_trend_per_year": float(row["long_term_pm25_trend"]),
            "pm25_spatial_method": str(row.get("pm25_spatial_method", "city-centre grid mean")),
        }
    return 5 + air_quality_level * 8.0, {
        "pm25_source": "curated seed baseline",
        "pm25_status": "fallback",
        "pm25_year_start": None,
        "pm25_year_end": None,
        "pm25_sample_years": 0,
        "pm25_trend_per_year": None,
        "pm25_spatial_method": "curated level",
    }


def _linear_slope(years: list[int], values: list[float]) -> float:
    """Return the ordinary least-squares slope in concentration units per year."""

    if len(years) < 2 or len(years) != len(values):
        return 0.0
    year_mean = sum(years) / len(years)
    value_mean = sum(values) / len(values)
    denominator = sum((year - year_mean) ** 2 for year in years)
    if denominator == 0:
        return 0.0
    numerator = sum((year - year_mean) * (value - value_mean) for year, value in zip(years, values))
    return float(numerator / denominator)


def _row_match_keys(row: pd.Series) -> set[str]:
    keys = {str(row["city"]).lower()}
    if "city_en" in row and pd.notna(row["city_en"]):
        keys.add(str(row["city_en"]).lower())
    if "city_zh" in row and pd.notna(row["city_zh"]):
        keys.add(str(row["city_zh"]).lower())
    if "aliases" in row and pd.notna(row["aliases"]):
        for alias in str(row["aliases"]).split("|"):
            alias = alias.strip()
            if alias:
                keys.add(alias.lower())
                keys.add(canonical_city_name(alias).lower())
    keys.add(canonical_city_name(str(row["city"])).lower())
    return keys


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _season_factor(month: int, latitude: float = 0.0) -> float:
    """Return a coarse seasonal phase with southern-hemisphere inversion."""

    monthly = {
        1: -0.90,
        2: -0.75,
        3: -0.35,
        4: 0.15,
        5: 0.45,
        6: 0.75,
        7: 1.00,
        8: 0.95,
        9: 0.65,
        10: 0.25,
        11: -0.25,
        12: -0.70,
    }
    value = monthly.get(int(month), 0.6)
    return -value if float(latitude) < 0 else value


def _winter_cold_level(cold_days: float) -> float:
    """Map mean freezing-day counts to the existing ordinal scoring input."""

    if cold_days >= 18:
        return 5.0
    if cold_days >= 10:
        return 4.0
    if cold_days >= 4:
        return 3.0
    if cold_days >= 1:
        return 2.0
    return 1.0


def _validate_seed_data(data: pd.DataFrame) -> None:
    """Reject duplicate cities, invalid coordinates, and out-of-range levels."""

    for column in ("city", "city_zh", "city_en"):
        if column in data.columns and data[column].duplicated().any():
            duplicates = sorted(data.loc[data[column].duplicated(keep=False), column].astype(str).unique())
            raise ValueError(f"city seed data has duplicate {column} values: {duplicates}")
    if not pd.to_numeric(data["latitude"], errors="coerce").between(-90, 90).all():
        raise ValueError("city seed data has invalid latitude values")
    if not pd.to_numeric(data["longitude"], errors="coerce").between(-180, 180).all():
        raise ValueError("city seed data has invalid longitude values")
    for column in (
        "summer_heat_level",
        "humidity_level",
        "air_quality_level",
        "precipitation_level",
        "winter_cold_level",
    ):
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any() or not values.between(1, 5).all():
            raise ValueError(f"city seed data has invalid 1-5 levels in {column}")


def _validate_long_term_air_quality(data: pd.DataFrame) -> None:
    """Reject duplicate, incomplete, or physically implausible PM2.5 records."""

    if data.duplicated(subset=["city", "year"]).any():
        raise ValueError("long-term air-quality data has duplicate city-year rows")
    years = pd.to_numeric(data["year"], errors="coerce")
    values = pd.to_numeric(data["pm25_mean_ug_m3"], errors="coerce")
    valid_cells = pd.to_numeric(data["valid_grid_cells"], errors="coerce")
    total_cells = pd.to_numeric(data["grid_cell_count"], errors="coerce")
    if years.isna().any() or not years.between(1980, 2100).all():
        raise ValueError("long-term air-quality data has invalid years")
    if values.isna().any() or not values.between(0, 500).all():
        raise ValueError("long-term air-quality data has invalid PM2.5 values")
    if valid_cells.isna().any() or total_cells.isna().any() or not ((valid_cells > 0) & (valid_cells <= total_cells)).all():
        raise ValueError("long-term air-quality data has invalid grid-cell counts")
    for column in ("source_product", "source_version", "spatial_method"):
        if data[column].isna().any() or data[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"long-term air-quality data has empty {column}")
        if data[column].nunique() != 1:
            raise ValueError(f"long-term air-quality data mixes multiple {column} values")


def _validate_climate_baseline(data: pd.DataFrame) -> None:
    """Reject incomplete coverage, duplicate periods, and implausible aggregates."""

    if data.duplicated(subset=["city", "period_month"]).any():
        raise ValueError("climate baseline data has duplicate city-period rows")
    months = pd.to_numeric(data["period_month"], errors="coerce")
    if months.isna().any() or not months.between(0, 12).all():
        raise ValueError("climate baseline data has invalid period months")
    expected_periods = set(range(13))
    for city, group in data.groupby("city", sort=False):
        actual_periods = set(pd.to_numeric(group["period_month"], errors="raise").astype(int))
        if actual_periods != expected_periods:
            raise ValueError(f"climate baseline data has incomplete periods for {city}")
    years_start = pd.to_numeric(data["year_start"], errors="coerce")
    years_end = pd.to_numeric(data["year_end"], errors="coerce")
    sample_years = pd.to_numeric(data["sample_years"], errors="coerce")
    if years_start.isna().any() or years_end.isna().any() or (years_end < years_start).any():
        raise ValueError("climate baseline data has invalid year ranges")
    if sample_years.isna().any() or not sample_years.between(1, 200).all():
        raise ValueError("climate baseline data has invalid sample-year counts")
    ranges = {
        "temperature_mean_c": (-90.0, 60.0),
        "daily_max_temperature_mean_c": (-90.0, 70.0),
        "apparent_temperature_mean_c": (-100.0, 75.0),
        "relative_humidity_mean_pct": (0.0, 100.0),
        "precipitation_days_mean": (0.0, 366.0),
        "heavy_rain_days_mean": (0.0, 366.0),
        "extreme_rain_days_mean": (0.0, 366.0),
        "hot_days_mean": (0.0, 366.0),
        "cold_days_mean": (0.0, 366.0),
        "windy_days_mean": (0.0, 366.0),
        "snow_days_mean": (0.0, 366.0),
        "missing_rate": (0.0, 1.0),
    }
    for column, (lower, upper) in ranges.items():
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any() or not values.between(lower, upper).all():
            raise ValueError(f"climate baseline data has invalid values in {column}")
