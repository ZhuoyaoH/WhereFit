from __future__ import annotations

import pandas as pd

from wherefit.data_loader import load_seed_cities, match_cities, parse_city_input, row_to_location
from wherefit.data_sources.open_meteo_forecast import summarize_forecast
from wherefit.data_sources.open_meteo_history import _date_chunks, metrics_from_history
from wherefit.hazards.aurora import _nearest_aurora_row, build_aurora_summary
from wherefit.hazards.earthquake import summarize_earthquakes
from wherefit.hazards.typhoon import summarize_typhoon_tracks


def test_domestic_city_seed_supports_chinese_aliases() -> None:
    data = load_seed_cities("data/city_seed.csv")
    requested = parse_city_input("北京，广州, mohe")
    matched, missing = match_cities(data, requested)
    assert missing == []
    assert set(matched["city"]) == {"Beijing", "Guangzhou", "Mohe"}
    assert row_to_location(matched.iloc[0]).city in {"北京", "广州", "漠河"}


def test_historical_weather_metrics_are_aggregated_by_month() -> None:
    location = row_to_location(load_seed_cities("data/city_seed.csv").query("city == 'Beijing'").iloc[0])
    raw = pd.DataFrame(
        {
            "time": ["2000-07-01", "2000-07-02", "2001-07-01", "2001-08-01"],
            "temperature_2m_mean": [25.0, 28.0, 27.0, 30.0],
            "temperature_2m_max": [32.0, 36.0, 35.0, 38.0],
            "temperature_2m_min": [20.0, 22.0, 21.0, 24.0],
            "apparent_temperature_mean": [27.0, 31.0, 30.0, 33.0],
            "apparent_temperature_max": [34.0, 39.0, 38.0, 41.0],
            "precipitation_sum": [0.0, 22.0, 55.0, 3.0],
            "snowfall_sum": [0.0, 0.0, 0.0, 0.0],
            "wind_speed_10m_max": [5.0, 12.0, 8.0, 3.0],
        }
    )
    metrics = metrics_from_history(raw, location, month=7, fallback_pm25=30)
    assert metrics.sample_years == 2
    assert metrics.heavy_rain_days == 1.0
    assert metrics.precipitation_extreme_days == 0.5
    assert metrics.hot_days == 1.0
    assert metrics.pm25 == 30


def test_history_date_chunks_use_five_year_windows() -> None:
    chunks = _date_chunks("2000-01-01", "2011-06-30", years=5)
    assert chunks == [
        ("2000-01-01", "2004-12-31"),
        ("2005-01-01", "2009-12-31"),
        ("2010-01-01", "2011-06-30"),
    ]


def test_earthquake_summary_counts_magnitudes_and_distance() -> None:
    location = row_to_location(load_seed_cities("data/city_seed.csv").query("city == 'Chengdu'").iloc[0])
    data = pd.DataFrame(
        {
            "time": ["2020-01-01", "2021-01-01", "2022-01-01"],
            "magnitude": [4.2, 5.4, 6.1],
            "latitude": [30.6, 31.0, 32.0],
            "longitude": [104.1, 104.5, 105.0],
        }
    )
    summary = summarize_earthquakes(data, location, "test", "cache")
    assert summary.event_count_m4 == 3
    assert summary.event_count_m5 == 2
    assert summary.event_count_m6 == 1
    assert summary.max_magnitude == 6.1
    assert summary.nearest_distance_km is not None


def test_aurora_summary_prioritizes_high_latitude_domestic_cities() -> None:
    data = load_seed_cities("data/city_seed.csv")
    mohe = row_to_location(data.query("city == 'Mohe'").iloc[0])
    guangzhou = row_to_location(data.query("city == 'Guangzhou'").iloc[0])
    assert build_aurora_summary(mohe).opportunity_score > build_aurora_summary(guangzhou).opportunity_score


def test_forecast_summary_counts_short_term_weather_flags() -> None:
    location = row_to_location(load_seed_cities("data/city_seed.csv").query("city == 'Shanghai'").iloc[0])
    raw = pd.DataFrame(
        {
            "time": ["2026-07-01", "2026-07-02", "2026-07-03"],
            "temperature_2m_max": [32.0, 36.0, 34.0],
            "apparent_temperature_max": [35.0, 40.0, 38.0],
            "precipitation_sum": [0.0, 24.0, 3.0],
            "precipitation_probability_max": [10.0, 80.0, 35.0],
            "wind_speed_10m_max": [5.0, 12.0, 8.0],
        }
    )
    summary = summarize_forecast(raw, location, "2026-07-01", "2026-07-03", "test", "ok")
    assert summary.days == 3
    assert summary.precipitation_days == 2
    assert summary.heavy_rain_days == 1
    assert summary.windy_days == 1
    assert summary.confidence == 0.85


def test_typhoon_tracks_count_unique_storms_by_distance_band() -> None:
    location = row_to_location(load_seed_cities("data/city_seed.csv").query("city == 'Shanghai'").iloc[0])
    data = pd.DataFrame(
        {
            "SID": ["A", "A", "B", "C"],
            "SEASON": [2005, 2005, 2010, 2020],
            "NAME": ["ALPHA", "ALPHA", "BETA", "GAMMA"],
            "ISO_TIME": ["2005-07-01", "2005-07-02", "2010-08-01", "2020-09-01"],
            "LAT": [31.25, 31.0, 30.0, 26.0],
            "LON": [121.5, 123.0, 124.0, 130.0],
            "WMO_WIND": [50, 60, 80, 90],
            "USA_WIND": [55, 65, 85, 95],
            "BASIN": ["WP", "WP", "WP", "WP"],
        }
    )
    summary = summarize_typhoon_tracks(data, location)
    assert summary.count_100km == 1
    assert summary.count_200km == 1
    assert summary.count_500km == 2
    assert summary.strongest_name == "BETA"
    assert summary.latest_nearby_name == "BETA"


def test_aurora_nearest_grid_prefers_closest_ovation_point() -> None:
    location = row_to_location(load_seed_cities("data/city_seed.csv").query("city == 'Mohe'").iloc[0])
    data = pd.DataFrame(
        {
            "longitude": [80.0, 122.5],
            "latitude": [20.0, 53.5],
            "probability": [99.0, 12.0],
        }
    )
    nearest = _nearest_aurora_row(data, location)
    assert nearest["probability"] == 12.0
