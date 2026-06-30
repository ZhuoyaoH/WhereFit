"""Load static seed data and convert it into scoring inputs."""

from __future__ import annotations

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
    "macau": "Macau",
    "macao": "Macau",
    "澳门": "Macau",
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


def load_seed_cities(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"city seed data missing columns: {sorted(missing)}")
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


def row_to_metrics(row: pd.Series, month: int) -> ClimateMetrics:
    heat_level = float(row["summer_heat_level"])
    humidity_level = float(row["humidity_level"])
    precipitation_level = float(row["precipitation_level"])
    air_quality_level = float(row["air_quality_level"])
    winter_cold_level = float(row["winter_cold_level"])

    season_factor = _season_factor(month)
    summer_factor = max(0.25, season_factor)
    winter_factor = max(0.0, -season_factor)

    temperature_max = 18 + heat_level * 3.0 + summer_factor * 7.0 - winter_factor * winter_cold_level * 2.5
    temperature_mean = temperature_max - 5.0 - winter_factor * 2.0
    humidity = 42 + humidity_level * 8.0 + summer_factor * 4.0
    precipitation_days = precipitation_level * (3.2 + summer_factor * 1.1)
    pm25 = 5 + air_quality_level * 8.0
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
        data_source="静态种子数据",
        data_status="fallback",
        sample_years=0,
        missing_rate=0.0,
        precipitation_extreme_days=max(0.0, (precipitation_level - 3.0) * 0.5),
        cold_days=max(0.0, (winter_cold_level - 2.0) * 4.0 * winter_factor),
        windy_days=2.0 if _to_bool(row["coastal"]) else 1.0,
        snow_days=max(0.0, (winter_cold_level - 3.0) * 3.0 * winter_factor),
    )


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


def _season_factor(month: int) -> float:
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
    return monthly.get(int(month), 0.6)
