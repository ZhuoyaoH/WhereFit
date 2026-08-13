from __future__ import annotations

import pandas as pd

from wherefit.city_filter import CityFilterOptions, candidate_city_input, filter_seed_cities
from wherefit.data_loader import load_seed_cities, parse_city_input, row_to_location, row_to_metrics
from wherefit.models import ClimateMetrics, Location, UserPreference
from wherefit.scoring.comfort import score_humidity, score_temperature
from wherefit.scoring.overall import compute_personal_fit, evaluate_city, rank_cities
from wherefit.scoring.preference import normalize_score
from wherefit.visualization.maps import AMAP_ATTRIBUTION, _display_position
from wherefit.visualization.maps import make_map


def pref(mode: str = "Travel", heat: int = 3, humidity: int = 3) -> UserPreference:
    return UserPreference(
        mode=mode,
        month=7,
        heat_sensitivity=heat,
        cold_sensitivity=2,
        humidity_sensitivity=humidity,
        rain_sensitivity=3,
        air_quality_sensitivity=3,
        extreme_weather_sensitivity=3,
    )


def metrics(heat: float = 4, humidity: float = 4, pm25: float = 25) -> ClimateMetrics:
    return ClimateMetrics(
        temperature_mean=20 + heat * 2.5,
        temperature_max=24 + heat * 3,
        apparent_temperature=22 + heat * 3,
        relative_humidity_mean=45 + humidity * 8,
        precipitation_days=humidity * 4,
        heavy_rain_days=max(0, humidity - 2),
        pm25=pm25,
        hot_days=max(0, (heat - 2) * 6),
        winter_cold_level=2,
        coastal=False,
        typhoon_region=False,
    )


def location(city: str) -> Location:
    return Location(city=city, country="Test", latitude=0, longitude=0, timezone="UTC", coastal=False, typhoon_region=False)


def test_normalize_score_clips_to_0_100() -> None:
    assert normalize_score(-10) == 0
    assert normalize_score(130) == 100
    assert normalize_score(None) == 50


def test_heat_sensitive_user_penalizes_hot_city_more() -> None:
    hot_metrics = metrics(heat=5, humidity=3)
    tolerant = pref(heat=0)
    sensitive = pref(heat=5)
    assert score_temperature(hot_metrics, sensitive) < score_temperature(hot_metrics, tolerant)


def test_humidity_sensitive_user_penalizes_humid_city_more() -> None:
    humid_metrics = metrics(heat=3, humidity=5)
    tolerant = pref(humidity=0)
    sensitive = pref(humidity=5)
    assert score_humidity(humid_metrics, sensitive) < score_humidity(humid_metrics, tolerant)


def test_living_mode_weights_risk_more_than_travel() -> None:
    travel_score = 80
    high_risk = 70
    travel_fit = compute_personal_fit(travel_score, high_risk, pref(mode="Travel"))
    living_fit = compute_personal_fit(travel_score, high_risk, pref(mode="Living"))
    assert living_fit < travel_fit


def test_missing_like_values_do_not_crash() -> None:
    result = evaluate_city(location("Test City"), metrics(heat=3, humidity=3, pm25=30), pref())
    assert 0 <= result.score.personal_fit_score <= 100
    assert result.score.warnings


def test_ranking_descends_by_personal_fit() -> None:
    results = [
        evaluate_city(location("Hot"), metrics(heat=5, humidity=5, pm25=45), pref(heat=5, humidity=5)),
        evaluate_city(location("Mild"), metrics(heat=2, humidity=2, pm25=15), pref(heat=5, humidity=5)),
    ]
    ranked = rank_cities(results)
    assert ranked[0].score.personal_fit_score >= ranked[1].score.personal_fit_score
    assert ranked[0].location.city == "Mild"


def test_city_input_accepts_chinese_and_english_commas() -> None:
    assert parse_city_input("Tokyo，青岛, Chengdu") == ["Tokyo", "Qingdao", "Chengdu"]


def test_seed_row_conversion_smoke() -> None:
    row = pd.Series(
        {
            "city": "Qingdao",
            "country": "China",
            "latitude": 36.0671,
            "longitude": 120.3826,
            "timezone": "Asia/Shanghai",
            "coastal": True,
            "typhoon_region": False,
            "summer_heat_level": 3,
            "humidity_level": 3,
            "air_quality_level": 2,
            "precipitation_level": 3,
            "winter_cold_level": 3,
        }
    )
    assert row_to_location(row).city == "Qingdao"
    assert row_to_metrics(row, 7).temperature_max > 0


def test_default_city_filter_uses_representative_cities() -> None:
    data = pd.DataFrame(
        {
            "city": ["Beijing", "Shenzhen", "Guangzhou"],
            "city_zh": ["北京", "深圳", "广州"],
            "province": ["北京", "广东", "广东"],
            "admin_level": ["municipality", "city", "provincial_capital"],
            "region_type": ["north", "south_coast", "south_coast"],
            "summer_heat_level": [4, 5, 5],
            "humidity_level": [2, 5, 5],
            "precipitation_level": [2, 5, 5],
            "air_quality_level": [4, 3, 3],
            "winter_cold_level": [4, 1, 1],
            "coastal": [False, True, True],
            "typhoon_region": [False, True, True],
        }
    )
    filtered = filter_seed_cities(data, CityFilterOptions())
    assert set(filtered["city"]) == {"Beijing", "Guangzhou"}
    assert "深圳" not in candidate_city_input(filtered)


def test_city_filter_can_exclude_typhoon_and_humid_cities() -> None:
    data = pd.DataFrame(
        {
            "city": ["Beijing", "Guangzhou"],
            "city_zh": ["北京", "广州"],
            "province": ["北京", "广东"],
            "admin_level": ["municipality", "provincial_capital"],
            "region_type": ["north", "south_coast"],
            "summer_heat_level": [4, 5],
            "humidity_level": [2, 5],
            "precipitation_level": [2, 5],
            "air_quality_level": [4, 3],
            "winter_cold_level": [4, 1],
            "coastal": [False, True],
            "typhoon_region": [False, True],
        }
    )
    filtered = filter_seed_cities(
        data,
        CityFilterOptions(max_humidity_level=3, exclude_typhoon_region=True),
    )
    assert filtered["city"].tolist() == ["Beijing"]


def test_city_filter_all_cities_keeps_non_capital_entries() -> None:
    data = pd.DataFrame(
        {
            "city": ["Beijing", "Shenzhen"],
            "city_zh": ["北京", "深圳"],
            "province": ["北京", "广东"],
            "admin_level": ["municipality", "city"],
            "region_type": ["north", "south_coast"],
            "summer_heat_level": [4, 5],
            "humidity_level": [2, 5],
            "precipitation_level": [2, 5],
            "air_quality_level": [4, 3],
            "winter_cold_level": [4, 1],
            "coastal": [False, True],
            "typhoon_region": [False, True],
        }
    )
    filtered = filter_seed_cities(data, CityFilterOptions(representative_only=False))
    assert set(filtered["city"]) == {"Beijing", "Shenzhen"}


def test_city_filter_can_limit_candidates_to_selected_countries() -> None:
    data = pd.DataFrame(
        {
            "city": ["Beijing", "Tokyo", "Singapore"],
            "country": ["China", "Japan", "Singapore"],
            "admin_level": ["municipality", "global_city", "global_city"],
        }
    )
    filtered = filter_seed_cities(
        data,
        CityFilterOptions(representative_only=False, countries=("Japan", "Singapore")),
    )
    assert filtered["city"].tolist() == ["Singapore", "Tokyo"]


def test_city_filter_combines_country_and_city_conditions() -> None:
    data = pd.DataFrame(
        {
            "city": ["Toronto", "Vancouver", "Tokyo"],
            "country": ["Canada", "Canada", "Japan"],
            "province": ["加拿大", "加拿大", "日本"],
            "admin_level": ["global_city", "global_city", "global_city"],
        }
    )
    filtered = filter_seed_cities(
        data,
        CityFilterOptions(representative_only=False, countries=("Canada",), cities=("Toronto",)),
    )
    assert filtered["city"].tolist() == ["Toronto"]


def test_international_city_seed_options_are_available_only_when_all_cities_enabled() -> None:
    data = load_seed_cities("data/city_seed.csv")
    international = data[data["admin_level"] == "global_city"]
    assert len(international) == 30
    assert {"纽约", "伦敦", "东京", "新加坡"}.issubset(set(international["city_zh"]))

    default_filtered = filter_seed_cities(data, CityFilterOptions(representative_only=True))
    assert "纽约" not in set(default_filtered["city_zh"])

    all_filtered = filter_seed_cities(data, CityFilterOptions(representative_only=False))
    candidates = candidate_city_input(all_filtered, limit=len(all_filtered), lang="zh")
    assert "纽约" in candidates
    assert "New York" not in candidates


def test_china_map_points_are_converted_for_amap_tiles() -> None:
    lat, lon = _display_position(39.9042, 116.4074, "China")
    assert abs(lat - 39.9042) > 0.001
    assert abs(lon - 116.4074) > 0.001
    foreign_lat, foreign_lon = _display_position(35.6762, 139.6503, "Japan")
    assert (foreign_lat, foreign_lon) == (35.6762, 139.6503)


def test_map_uses_amap_tile_layer_without_carto_provider() -> None:
    result = evaluate_city(location("北京"), metrics(heat=3, humidity=2), pref())
    deck_json = make_map([result]).to_json()
    assert "appmaptile" in deck_json
    assert "amap-raster" in deck_json
    assert AMAP_ATTRIBUTION == "@高德"
    assert "carto" not in deck_json.lower()
    assert "openstreetmap" not in deck_json.lower()
