from __future__ import annotations

import pandas as pd

from wherefit.data_loader import load_seed_cities, match_cities, parse_city_input, row_to_location
from wherefit.data_sources.met_no_forecast import _payload_to_hourly_frame, _summarize_met_no
from wherefit.data_sources.open_meteo_forecast import summarize_forecast
from wherefit.data_sources.cache import write_csv_cache
from wherefit.data_sources.open_meteo_history import (
    _date_chunks,
    _fetch_history_chunk,
    get_history_metrics,
    metrics_from_history,
)
from wherefit.hazards.aurora import _nearest_aurora_row, build_aurora_summary
from wherefit.hazards.earthquake import summarize_earthquakes
from wherefit.hazards.typhoon import summarize_typhoon_tracks
from wherefit.models import ForecastSummary, UserPreference
from wherefit.scoring.forecast import compute_forecast_trip_fit
from wherefit.visualization.maps import make_earthquake_map, make_typhoon_track_map


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
            "relative_humidity_2m_mean": [50.0, 60.0, 70.0, 80.0],
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
    assert round(metrics.temperature_max, 1) == 34.3
    assert round(metrics.apparent_temperature, 1) == 29.3
    assert metrics.relative_humidity_mean == 60.0
    assert "relative_humidity_mean" not in metrics.estimated_fields
    assert metrics.pm25 == 30


def test_history_date_chunks_use_five_year_windows() -> None:
    chunks = _date_chunks("2000-01-01", "2011-06-30", years=5)
    assert chunks == [
        ("2000-01-01", "2004-12-31"),
        ("2005-01-01", "2009-12-31"),
        ("2010-01-01", "2011-06-30"),
    ]


def test_history_uses_chunk_cache_without_network(tmp_path, monkeypatch) -> None:
    location = row_to_location(load_seed_cities("data/city_seed.csv").query("city == 'Beijing'").iloc[0])
    cached = pd.DataFrame(
        {
            "time": ["2000-07-01", "2000-07-02"],
            "temperature_2m_mean": [25.0, 26.0],
            "temperature_2m_max": [31.0, 32.0],
            "temperature_2m_min": [20.0, 21.0],
            "apparent_temperature_mean": [27.0, 28.0],
            "apparent_temperature_max": [33.0, 34.0],
            "precipitation_sum": [0.0, 2.0],
            "snowfall_sum": [0.0, 0.0],
            "wind_speed_10m_max": [4.0, 5.0],
        }
    )
    write_csv_cache(tmp_path / "chunks" / "Beijing_2000-07-01_2000-07-02_era5_daily.csv", cached)

    def fail_fetch(*args, **kwargs):
        raise AssertionError("network should not be called when exact chunk cache exists")

    monkeypatch.setattr("wherefit.data_sources.open_meteo_history._fetch_history_chunk", fail_fetch)
    result = get_history_metrics(location, 7, tmp_path, start_date="2000-07-01", end_date="2000-07-02")

    assert result.status == "cache"
    assert result.metrics is not None
    assert result.metrics.sample_years == 1


def test_history_ignores_incomplete_full_cache(tmp_path, monkeypatch) -> None:
    location = row_to_location(load_seed_cities("data/city_seed.csv").query("city == 'Beijing'").iloc[0])
    incomplete_full_cache = pd.DataFrame(
        {
            "time": ["2000-07-01", "2000-07-02"],
            "temperature_2m_mean": [25.0, 26.0],
            "temperature_2m_max": [31.0, 32.0],
            "temperature_2m_min": [20.0, 21.0],
            "apparent_temperature_mean": [27.0, 28.0],
            "apparent_temperature_max": [33.0, 34.0],
            "precipitation_sum": [0.0, 2.0],
            "snowfall_sum": [0.0, 0.0],
            "wind_speed_10m_max": [4.0, 5.0],
        }
    )
    complete_chunk_cache = pd.DataFrame(
        {
            "time": ["2000-07-01", "2000-07-02", "2000-07-03"],
            "temperature_2m_mean": [25.0, 26.0, 27.0],
            "temperature_2m_max": [31.0, 32.0, 33.0],
            "temperature_2m_min": [20.0, 21.0, 22.0],
            "apparent_temperature_mean": [27.0, 28.0, 29.0],
            "apparent_temperature_max": [33.0, 34.0, 35.0],
            "precipitation_sum": [0.0, 2.0, 0.0],
            "snowfall_sum": [0.0, 0.0, 0.0],
            "wind_speed_10m_max": [4.0, 5.0, 6.0],
        }
    )
    write_csv_cache(tmp_path / "Beijing_2000-07-01_2000-07-03_era5_daily.csv", incomplete_full_cache)
    write_csv_cache(tmp_path / "chunks" / "Beijing_2000-07-01_2000-07-03_era5_daily.csv", complete_chunk_cache)

    def fail_fetch(*args, **kwargs):
        raise AssertionError("network should not be called when complete chunk cache exists")

    monkeypatch.setattr("wherefit.data_sources.open_meteo_history._fetch_history_chunk", fail_fetch)
    result = get_history_metrics(location, 7, tmp_path, start_date="2000-07-01", end_date="2000-07-03")

    assert result.status == "cache"
    assert result.raw is not None
    assert list(result.raw["time"]) == ["2000-07-01", "2000-07-02", "2000-07-03"]


def test_history_reuses_partial_chunk_and_fetches_only_missing_tail(tmp_path, monkeypatch) -> None:
    location = row_to_location(load_seed_cities("data/city_seed.csv").query("city == 'Beijing'").iloc[0])
    cached = pd.DataFrame(
        {
            "time": ["2000-07-01", "2000-07-02"],
            "temperature_2m_mean": [25.0, 26.0],
            "temperature_2m_max": [31.0, 32.0],
            "temperature_2m_min": [20.0, 21.0],
            "apparent_temperature_mean": [27.0, 28.0],
            "apparent_temperature_max": [33.0, 34.0],
            "precipitation_sum": [0.0, 2.0],
            "snowfall_sum": [0.0, 0.0],
            "wind_speed_10m_max": [4.0, 5.0],
        }
    )
    write_csv_cache(tmp_path / "chunks" / "Beijing_2000-07-01_2000-07-02_era5_daily.csv", cached)
    calls: list[tuple[str, str]] = []

    def fetch_tail(location, start_date, end_date, timeout):
        """Return the one missing day and record the requested range."""

        calls.append((start_date, end_date))
        return pd.DataFrame(
            {
                "time": ["2000-07-03"],
                "temperature_2m_mean": [27.0],
                "temperature_2m_max": [33.0],
                "temperature_2m_min": [22.0],
                "apparent_temperature_mean": [29.0],
                "apparent_temperature_max": [35.0],
                "precipitation_sum": [0.0],
                "snowfall_sum": [0.0],
                "wind_speed_10m_max": [6.0],
            }
        )

    monkeypatch.setattr("wherefit.data_sources.open_meteo_history._fetch_history_chunk", fetch_tail)
    result = get_history_metrics(location, 7, tmp_path, start_date="2000-07-01", end_date="2000-07-03")

    assert calls == [("2000-07-03", "2000-07-03")]
    assert result.status == "live"
    assert result.raw is not None
    assert list(result.raw["time"]) == ["2000-07-01", "2000-07-02", "2000-07-03"]


def test_history_exact_cache_filename_does_not_hide_missing_tail(tmp_path, monkeypatch) -> None:
    """An exact-looking filename must be checked against actual cached dates."""

    location = row_to_location(load_seed_cities("data/city_seed.csv").query("city == 'Beijing'").iloc[0])
    cached = pd.DataFrame(
        {
            "time": ["2000-07-01", "2000-07-02"],
            "temperature_2m_mean": [25.0, 26.0],
            "temperature_2m_max": [31.0, 32.0],
            "temperature_2m_min": [20.0, 21.0],
            "apparent_temperature_mean": [27.0, 28.0],
            "precipitation_sum": [0.0, 2.0],
        }
    )
    write_csv_cache(tmp_path / "chunks" / "Beijing_2000-07-01_2000-07-03_era5_daily.csv", cached)
    calls: list[tuple[str, str]] = []

    def fetch_tail(location, start_date, end_date, timeout):
        """Return the tail hidden by an inaccurate exact-cache filename."""

        calls.append((start_date, end_date))
        return pd.DataFrame(
            {
                "time": ["2000-07-03"],
                "temperature_2m_mean": [27.0],
                "temperature_2m_max": [33.0],
                "temperature_2m_min": [22.0],
                "apparent_temperature_mean": [29.0],
                "precipitation_sum": [0.0],
            }
        )

    monkeypatch.setattr("wherefit.data_sources.open_meteo_history._fetch_history_chunk", fetch_tail)
    result = get_history_metrics(location, 7, tmp_path, start_date="2000-07-01", end_date="2000-07-03")

    assert calls == [("2000-07-03", "2000-07-03")]
    assert result.raw is not None
    assert list(result.raw["time"]) == ["2000-07-01", "2000-07-02", "2000-07-03"]


def test_history_request_fixes_model_and_wind_units(monkeypatch) -> None:
    """Long historical requests must not mix Best Match models or wind units."""

    location = row_to_location(load_seed_cities("data/city_seed.csv").query("city == 'Beijing'").iloc[0])
    captured: dict[str, object] = {}

    def fake_get(url, params, timeout):
        """Capture the provider parameters and return a minimal daily payload."""

        captured.update(params)

        class Response:
            """Minimal successful response for request-contract inspection."""

            def raise_for_status(self) -> None:
                """Represent a successful HTTP status."""

                return None

            def json(self):
                """Return one daily row accepted by the fetch helper."""

                return {"daily": {"time": ["2000-07-01"], "temperature_2m_mean": [25.0]}}

        return Response()

    monkeypatch.setattr("wherefit.data_sources.open_meteo_history.requests.get", fake_get)
    _fetch_history_chunk(location, "2000-07-01", "2000-07-01", timeout=5)

    assert captured["models"] == "era5"
    assert captured["wind_speed_unit"] == "ms"
    assert "relative_humidity_2m_mean" in str(captured["daily"])


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
    assert summary.count_100km >= 1
    assert summary.count_500km == 3
    assert summary.events


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
            "temperature_2m_min": [25.0, 28.0, 27.0],
            "apparent_temperature_max": [35.0, 40.0, 38.0],
            "precipitation_sum": [0.0, 24.0, 3.0],
            "precipitation_probability_max": [10.0, 80.0, 35.0],
            "wind_speed_10m_max": [5.0, 12.0, 8.0],
        }
    )
    summary = summarize_forecast(raw, location, "2026-07-01", "2026-07-03", "test", "ok")
    assert summary.days == 3
    assert summary.precipitation_days == 2
    assert summary.temp_min_mean == 26.666666666666668
    assert summary.heavy_rain_days == 1
    assert summary.windy_days == 1
    assert summary.confidence == 0.85


def test_forecast_trip_fit_penalizes_hot_rainy_forecast() -> None:
    pref = UserPreference(
        mode="Travel",
        month=7,
        heat_sensitivity=5,
        cold_sensitivity=1,
        humidity_sensitivity=3,
        rain_sensitivity=5,
        air_quality_sensitivity=3,
        extreme_weather_sensitivity=4,
    )
    good = ForecastSummary(
        city="Good",
        start_date="2026-07-01",
        end_date="2026-07-03",
        days=3,
        temp_max_mean=25,
        temp_min_mean=18,
        apparent_temp_max_mean=26,
        precipitation_days=0,
        precipitation_probability_max=10,
        heavy_rain_days=0,
        windy_days=0,
        confidence=0.85,
        source="test",
        status="live",
        message="ok",
    )
    bad = ForecastSummary(
        city="Bad",
        start_date="2026-07-01",
        end_date="2026-07-03",
        days=3,
        temp_max_mean=36,
        temp_min_mean=29,
        apparent_temp_max_mean=41,
        precipitation_days=3,
        precipitation_probability_max=90,
        heavy_rain_days=1,
        windy_days=1,
        confidence=0.85,
        source="test",
        status="live",
        message="ok",
    )
    assert compute_forecast_trip_fit(good, pref)[0] > compute_forecast_trip_fit(bad, pref)[0]


def test_forecast_summary_supports_imperial_thresholds() -> None:
    location = row_to_location(load_seed_cities("data/city_seed.csv").query("city == 'Shanghai'").iloc[0])
    raw = pd.DataFrame(
        {
            "time": ["2026-07-01", "2026-07-02", "2026-07-03"],
            "temperature_2m_max": [90.0, 95.0, 93.0],
            "temperature_2m_min": [78.0, 80.0, 79.0],
            "apparent_temperature_max": [95.0, 100.0, 99.0],
            "precipitation_sum": [0.0, 0.9, 0.05],
            "precipitation_probability_max": [10.0, 80.0, 35.0],
            "wind_speed_10m_max": [12.0, 25.0, 18.0],
        }
    )
    summary = summarize_forecast(raw, location, "2026-07-01", "2026-07-03", "test", "ok", unit_system="imperial")
    assert summary.temperature_unit == "°F"
    assert summary.precipitation_unit == "inch"
    assert summary.wind_speed_unit == "mph"
    assert summary.temp_min_mean == 79.0
    assert summary.precipitation_days == 2
    assert summary.heavy_rain_days == 1
    assert summary.windy_days == 1


def test_met_no_payload_is_aggregated_to_daily_summary() -> None:
    location = row_to_location(load_seed_cities("data/city_seed.csv").query("city == 'Beijing'").iloc[0])
    payload = {
        "properties": {
            "timeseries": [
                {
                    "time": "2026-07-01T00:00:00Z",
                    "data": {
                        "instant": {"details": {"air_temperature": 25.0, "wind_speed": 4.0}},
                        "next_1_hours": {"details": {"precipitation_amount": 0.0}},
                    },
                },
                {
                    "time": "2026-07-01T12:00:00Z",
                    "data": {
                        "instant": {"details": {"air_temperature": 31.0, "wind_speed": 12.0}},
                        "next_1_hours": {"details": {"precipitation_amount": 22.0}},
                    },
                },
            ]
        }
    }
    raw = _payload_to_hourly_frame(payload, "2026-07-01", "2026-07-01")
    summary = _summarize_met_no(raw, location, "2026-07-01", "2026-07-01", "test", "ok", "metric")
    assert summary.source == "MET Norway Locationforecast API"
    assert summary.temp_max_mean == 31.0
    assert summary.temp_min_mean == 25.0
    assert summary.precipitation_days == 1
    assert summary.heavy_rain_days == 1
    assert summary.windy_days == 1


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
    assert summary.track_points


def test_hazard_maps_use_amap_tiles() -> None:
    location = row_to_location(load_seed_cities("data/city_seed.csv").query("city == 'Shanghai'").iloc[0])
    earthquake_deck = make_earthquake_map(
        location,
        [
            {
                "date": "2020-01-01",
                "magnitude": 5.2,
                "place": "test",
                "latitude": 31.0,
                "longitude": 122.0,
                "distance_km": 80,
            }
        ],
    ).to_json()
    typhoon_deck = make_typhoon_track_map(
        location,
        [
            {
                "sid": "A",
                "name": "ALPHA",
                "time": "2020-08-01",
                "latitude": 31.0,
                "longitude": 122.0,
                "distance_km": 80,
                "wind": 60,
            },
            {
                "sid": "A",
                "name": "ALPHA",
                "time": "2020-08-02",
                "latitude": 31.5,
                "longitude": 123.0,
                "distance_km": 120,
                "wind": 65,
            },
        ],
    ).to_json()
    assert "appmaptile" in earthquake_deck
    assert "appmaptile" in typhoon_deck
    assert "carto" not in earthquake_deck.lower()
    assert "openstreetmap" not in earthquake_deck.lower()
    assert "carto" not in typhoon_deck.lower()
    assert "openstreetmap" not in typhoon_deck.lower()


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
