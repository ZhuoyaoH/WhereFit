from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from threading import Barrier

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import app
from wherefit.data_loader import display_city_name, load_seed_cities, row_to_location, row_to_metrics
from wherefit.data_quality import build_data_quality_records
from wherefit.data_sources.air_quality import summarize_air_quality
from wherefit.data_sources.open_meteo_forecast import _failed_summary, _forecast_horizon_weight, _is_supported_forecast_window
from wherefit.data_sources.open_meteo_history import metrics_from_history
from wherefit.hazards.aurora import build_aurora_summary
from wherefit.hazards.hydro_events import _bbox, _feature_to_row
from wherefit.hazards.typhoon import basins_for_location
from wherefit.models import ForecastSummary, Location, UserPreference
from wherefit.report.template_report import _period_label
from wherefit.scoring.forecast import apply_forecast_score, compute_forecast_trip_fit
from wherefit.scoring.overall import evaluate_city


def _preference(mode: str = "Compare", month: int = 7) -> UserPreference:
    """Return a stable preference fixture for product-logic tests."""

    return UserPreference(mode, month, 3, 2, 3, 3, 3, 3)


def _beijing_row() -> pd.Series:
    """Return the Beijing seed row."""

    return load_seed_cities("data/city_seed.csv").query("city == 'Beijing'").iloc[0]


def _valid_forecast(location) -> ForecastSummary:
    """Return a small valid forecast summary."""

    return ForecastSummary(
        city=location.city,
        start_date="2026-07-31",
        end_date="2026-08-02",
        days=3,
        temp_max_mean=30.0,
        temp_min_mean=21.0,
        apparent_temp_max_mean=31.0,
        precipitation_days=1,
        precipitation_probability_max=40.0,
        heavy_rain_days=0,
        windy_days=0,
        confidence=0.85,
        source="test",
        status="live",
        message="ok",
    )


def _location(city: str, latitude: float, longitude: float) -> Location:
    """Return a minimal location for provider-selection tests."""

    return Location(city, "test", latitude, longitude, "UTC", False, False)


def test_static_evaluation_does_not_request_recent_air_quality(monkeypatch) -> None:
    """Static and long-term paths must not mix in a recent PM2.5 window."""

    monkeypatch.setattr(app, "get_air_quality_summary", lambda *args, **kwargs: pytest.fail("unexpected recent AQ request"))
    results, missing, messages = app._evaluate("北京", _preference(), data_mode=app.DATA_MODE_STATIC)

    assert missing == []
    assert messages == []
    assert results[0].air_quality is None
    assert results[0].metrics.pm25_status == "dataset"
    assert results[0].metrics.pm25 == pytest.approx(52.58641)


def test_forecast_failure_preserves_climate_fallback_score() -> None:
    """A provider outage must not turn an otherwise valid city into a zero."""

    row = _beijing_row()
    location = row_to_location(row)
    result = evaluate_city(location, row_to_metrics(row, 7), _preference("Travel"))
    failure = _failed_summary(location, "2026-07-31", "2026-08-02", "metric", "offline")

    updated = apply_forecast_score(result, failure, _preference("Travel"))

    assert updated.score.personal_fit_score == result.score.personal_fit_score
    assert updated.score.forecast_trip_fit_score is None
    assert "回退到气候适配分" in updated.score.warnings[-1]


def test_recent_air_quality_affects_only_short_term_forecast_score() -> None:
    """Recent PM2.5 should affect Travel forecast fit without replacing baseline metrics."""

    location = row_to_location(_beijing_row())
    forecast = _valid_forecast(location)
    clean = summarize_air_quality(pd.DataFrame({"pm2_5": [5.0], "us_aqi": [20.0]}), location, "live", "ok")
    polluted = summarize_air_quality(pd.DataFrame({"pm2_5": [90.0], "us_aqi": [180.0]}), location, "live", "ok")

    clean_score = compute_forecast_trip_fit(forecast, _preference("Travel"), clean)[0]
    polluted_score = compute_forecast_trip_fit(forecast, _preference("Travel"), polluted)[0]

    assert clean_score > polluted_score


def test_data_quality_separates_scored_inputs_from_hazard_records() -> None:
    """Seed fallback provenance must state that hazard records do not alter ranking."""

    row = _beijing_row()
    result = evaluate_city(row_to_location(row), row_to_metrics(row, 0), _preference(month=0))
    records = build_data_quality_records(result)
    by_category = {record.category: record for record in records}

    assert by_category["long_term_air_quality"].fallback_used is True
    assert by_category["long_term_air_quality"].time_scope.startswith("long-term")
    assert by_category["hazard_records"].affects_score is False


def test_full_year_seed_metrics_are_twelve_month_aggregate() -> None:
    """Month zero must not silently reuse a summer season factor."""

    row = _beijing_row()
    annual = row_to_metrics(row, 0)
    july = row_to_metrics(row, 7)

    assert annual.data_source.endswith("全年聚合）")
    assert annual.precipitation_days > july.precipitation_days
    assert annual.hot_days >= july.hot_days
    annual_result = evaluate_city(row_to_location(row), annual, _preference(month=0))
    assert annual_result.score.personal_fit_score > 60


def test_history_metrics_reject_missing_core_fields() -> None:
    """Incomplete provider payloads must fail visibly instead of becoming zero-valued climate data."""

    raw = pd.DataFrame({"time": ["2020-01-01"], "temperature_2m_mean": [20.0]})
    with pytest.raises(ValueError, match="missing required fields"):
        metrics_from_history(raw, row_to_location(_beijing_row()), 1)


def test_live_aurora_indicator_does_not_inherit_latitude_heuristic(tmp_path) -> None:
    """A live nowcast value must not be inflated by the fallback latitude score."""

    mohe = row_to_location(load_seed_cities("data/city_seed.csv").query("city == 'Mohe'").iloc[0])
    cache = pd.DataFrame(
        {
            "longitude": [mohe.longitude],
            "latitude": [mohe.latitude],
            "probability": [2.0],
            "forecast_time": [datetime.now(timezone.utc).isoformat()],
            "cached_at_utc": [datetime.now(timezone.utc).isoformat()],
        }
    )
    cache_dir = tmp_path / "aurora"
    cache_dir.mkdir()
    cache.to_csv(cache_dir / "ovation_aurora_latest.csv", index=False)

    summary = build_aurora_summary(mohe, cache_dir=cache_dir, include_live=True)

    assert summary.nearest_probability == 2.0
    assert summary.opportunity_score == 2.0
    assert summary.status == "cache"
    assert summary.forecast_time is not None


def test_stale_aurora_cache_is_not_presented_as_current(tmp_path, monkeypatch) -> None:
    """An expired nowcast must refresh or fall back instead of looking live."""

    mohe = row_to_location(load_seed_cities("data/city_seed.csv").query("city == 'Mohe'").iloc[0])
    stale = pd.DataFrame(
        {
            "longitude": [mohe.longitude],
            "latitude": [mohe.latitude],
            "probability": [80.0],
            "forecast_time": ["stale"],
            "cached_at_utc": [(datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()],
        }
    )
    cache_dir = tmp_path / "aurora"
    cache_dir.mkdir()
    stale.to_csv(cache_dir / "ovation_aurora_latest.csv", index=False)

    def fail_get(*args, **kwargs):
        """Simulate an unavailable NOAA refresh."""

        raise RuntimeError("offline")

    monkeypatch.setattr("wherefit.hazards.aurora.requests.get", fail_get)
    summary = build_aurora_summary(mohe, cache_dir=cache_dir, include_live=True)

    assert summary.status == "heuristic"
    assert summary.nearest_probability is None
    assert summary.opportunity_score != 80.0


def test_manual_forecast_queries_run_concurrently_and_keep_order() -> None:
    """The manual table should not wait for cities sequentially or reorder them."""

    results, _, _ = app._evaluate("北京, 上海, 广州, 成都", _preference(), data_mode=app.DATA_MODE_STATIC)
    barrier = Barrier(4)

    def provider(location, **kwargs):
        """Block each worker until all four city requests have started."""

        barrier.wait(timeout=3)
        return _valid_forecast(location)

    summaries = app._query_forecast_summaries(
        results,
        provider,
        "2026-08-01",
        "2026-08-03",
        "metric",
        False,
    )

    assert [item.city for item in summaries] == [item.location.city for item in results]


def test_default_page_and_static_compare_flow_are_clear() -> None:
    """The default page should present a demo set and the completed flow should expose quality boundaries."""

    page = AppTest.from_file("app.py", default_timeout=30).run()
    assert not page.exception
    assert next(item for item in page.radio if item.label == "选择方式").value == "demo"
    next(item for item in page.button if item.label == "开始比较").click()
    page.run()
    assert not page.exception
    assert any("主排名已经使用随项目发布的 NASA POWER" in item.value for item in page.info)

    next(item for item in page.radio if item.label == "使用场景").set_value("Compare")
    page.run()
    next(item for item in page.button if item.label == "开始比较").click()
    page.run()

    assert not page.exception
    assert "数据来源" in [item.label for item in page.tabs]
    assert "数据来源和使用说明" in [item.value for item in page.subheader]
    assert any("3/3 城市使用历史气候" in item.value for item in page.markdown)

    next(item for item in page.radio if item.label == "语言").set_value("英文")
    page.run()
    assert not page.exception
    assert "Data Sources" in [item.label for item in page.tabs]

    next(item for item in page.radio if item.label == "Theme").set_value("night")
    page.run()
    assert not page.exception
    assert "Data Sources" in [item.label for item in page.tabs]


def test_result_signature_is_language_independent() -> None:
    """Changing display language must not invalidate an otherwise identical result."""

    pref = _preference()
    zh = app._evaluation_signature("北京, 上海", pref, app.DATA_MODE_STATIC, "2000-01-01", "2026-07-20", False)
    en = app._evaluation_signature("Beijing, Shanghai", pref, app.DATA_MODE_STATIC, "2000-01-01", "2026-07-20", False)

    assert zh == en


def test_forecast_weight_uses_lead_time_and_rejects_unsupported_dates() -> None:
    """Forecast reliability should reflect horizon rather than only selected duration."""

    today = date(2026, 7, 31)
    assert _forecast_horizon_weight("2026-07-31", "2026-08-02", "live", today=today) == 0.85
    assert _forecast_horizon_weight("2026-08-08", "2026-08-10", "live", today=today) == 0.58
    assert _is_supported_forecast_window("2026-07-31", "2026-08-10", 15, today=today)
    assert not _is_supported_forecast_window("2026-07-30", "2026-08-01", 15, today=today)
    assert not _is_supported_forecast_window("2026-07-31", "2026-08-20", 15, today=today)


def test_hazard_table_calls_provenance_evidence_type_not_confidence() -> None:
    """Hazard provenance labels must not imply statistical confidence."""

    events = [{"类型": "极端降水提示", "记录": "test", "数据口径": "test", "证据类型": "参考提示"}]
    zh_table = app._event_table(events, app.LANG_ZH)
    en_table = app._event_table(events, app.LANG_EN)

    assert "证据类型" in zh_table.columns
    assert "置信度" not in zh_table.columns
    assert en_table.loc[0, "Evidence Type"] == "Reference reminder"


def test_cross_month_forecast_uses_only_dates_in_selected_month() -> None:
    """A forecast crossing month-end must not reuse the whole window twice."""

    assert app._forecast_window_for_month("2026-07-30", "2026-08-05", 7) == ("2026-07-30", "2026-07-31")
    assert app._forecast_window_for_month("2026-07-30", "2026-08-05", 8) == ("2026-08-01", "2026-08-05")
    assert app._forecast_window_for_month("2026-07-30", "2026-08-05", 6) is None


def test_travel_month_outside_forecast_window_uses_bundled_baseline() -> None:
    """A month without a short-term forecast must still produce an immediate local comparison."""

    data_mode, forecast_window = app._travel_data_mode_for_month("2026-07-30", "2026-08-05", 10)

    assert data_mode == app.DATA_MODE_STATIC
    assert forecast_window is None


def test_chinese_city_labels_keep_special_administrative_regions_clear() -> None:
    """Chinese labels must not leak English names or abbreviate Hong Kong and Macau."""

    assert display_city_name("拉萨", "Lhasa") == "拉萨"
    assert display_city_name("香港", "Hong Kong") == "香港特别行政区"
    assert display_city_name("澳门", "Macau") == "澳门特别行政区"
    assert app._city_display_from_summary("香港", app.LANG_ZH) == "香港特别行政区"
    hong_kong = row_to_location(load_seed_cities("data/city_seed.csv").query("city == 'Hong Kong'").iloc[0])
    result = evaluate_city(hong_kong, row_to_metrics(_beijing_row(), 7), _preference())
    assert app._city_place_label(result, app.LANG_ZH) == "香港特别行政区"


def test_english_report_uses_calendar_month_name() -> None:
    """English prose should say August rather than the misleading '8-month'."""

    assert _period_label(8, "en") == "August"


def test_eonet_bbox_and_event_date_follow_v3_geojson_contract() -> None:
    """EONET v3 uses west,north,east,south and stores date in properties."""

    west, north, east, south = (float(value) for value in _bbox(39.9, 116.4, 500).split(","))
    assert west < east
    assert north > south
    row = _feature_to_row(
        {
            "id": "event-1",
            "properties": {"title": "Flood", "date": "2026-07-20T00:00:00Z", "sources": []},
            "geometry": {"type": "Point", "coordinates": [116.4, 39.9]},
        }
    )
    assert row is not None
    assert row["date"] == "2026-07-20T00:00:00Z"


def test_ibtracs_basin_selection_covers_global_project_cities() -> None:
    """Cyclone queries must select a relevant official basin outside East Asia too."""

    assert basins_for_location(_location("Beijing", 39.9, 116.4)) == ("WP",)
    assert basins_for_location(_location("New York", 40.7, -74.0)) == ("NA",)
    assert basins_for_location(_location("Los Angeles", 34.1, -118.2)) == ("EP",)
    assert basins_for_location(_location("Sydney", -33.9, 151.2)) == ("SP",)
    assert basins_for_location(_location("Dubai", 25.2, 55.3)) == ("NI",)
    assert basins_for_location(_location("Helsinki", 60.2, 24.9)) == ("NA",)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("空气质量指标偏弱", "air-quality indicator may be weaker"),
        ("降水或强降水天数偏多", "rain or heavy-rain days may be frequent"),
        ("长期空气污染指标偏高", "long-term air-pollution indicator is higher"),
    ],
)
def test_english_city_card_translates_watch_out_chips(phrase: str, expected: str) -> None:
    """English city cards must not leak Chinese scoring phrases."""

    assert app._display_phrase(phrase, app.LANG_EN) == expected
