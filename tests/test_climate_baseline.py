from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from wherefit.data_loader import (
    attach_long_term_air_quality,
    load_climate_baseline,
    load_seed_cities,
    row_to_location,
    row_to_metrics,
)
from wherefit.data_quality import build_data_quality_records
from wherefit.models import UserPreference
from wherefit.scoring.overall import evaluate_city
from wherefit.visualization.charts import make_radar_chart, make_ranking_bar_chart
from scripts.build_nasa_power_climate import DAILY_VARIABLES, _fetch_city


MONTHLY_PATH = Path("data/climate/nasa_power_merra2_city_monthly_2000_2025.csv")
ANNUAL_PATH = Path("data/climate/nasa_power_merra2_city_annual_2000_2025.csv")
MANIFEST_PATH = Path("data/climate/nasa_power_merra2_manifest.json")
BUILDER_PATH = Path("scripts/build_nasa_power_climate.py")
PM25_PATH = Path("data/air_quality/acag_v6gl03_city_annual_2015_2024.csv")


def _sha256(path: Path) -> str:
    """Return a stable digest for manifest verification."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preference(month: int) -> UserPreference:
    """Return a neutral comparison preference for chart tests."""

    return UserPreference("Compare", month, 3, 2, 3, 3, 3, 3)


def _baseline() -> pd.DataFrame:
    """Load the bundled monthly and annual baseline."""

    return load_climate_baseline(MONTHLY_PATH, ANNUAL_PATH)


def test_bundled_power_baseline_has_complete_77_city_coverage() -> None:
    """Every seed city must have twelve months and one annual row."""

    baseline = _baseline()
    seed = load_seed_cities("data/city_seed.csv")

    assert baseline.shape[0] == 77 * 13
    assert baseline["city"].nunique() == 77
    assert set(baseline["city"]) == set(seed["city"])
    assert set(baseline["period_month"]) == set(range(13))
    assert baseline.groupby("city")["period_month"].nunique().eq(13).all()
    assert baseline["year_start"].eq(2000).all()
    assert baseline["year_end"].eq(2025).all()
    assert baseline["sample_years"].eq(26).all()
    assert baseline["missing_rate"].eq(0).all()
    assert baseline["source_model"].eq("NASA POWER / MERRA-2").all()


def test_builder_maps_power_payload_to_city_and_derives_fields(monkeypatch) -> None:
    """A POWER response must retain provenance and produce finite derived fields."""

    row = pd.Series({"city": "Alpha", "latitude": 10.0, "longitude": 20.0})
    dates = pd.date_range("2025-01-01", "2025-12-31", freq="D").strftime("%Y%m%d")
    values = {
        "T2M": 15.0,
        "T2M_MAX": 20.0,
        "T2M_MIN": 10.0,
        "RH2M": 60.0,
        "PRECTOTCORR": 0.0,
        "WS10M": 3.0,
        "WS10M_MAX": 5.0,
    }
    parameters = {name: dict.fromkeys(dates, values[name]) for name in DAILY_VARIABLES}
    payload = {
        "geometry": {"coordinates": [20.0, 10.0, 100.0]},
        "properties": {"parameter": parameters},
        "header": {"api": {"version": "test"}, "sources": ["MERRA2"], "time_standard": "LST"},
    }

    def fake_fetch(url: str, timeout: int, retries: int) -> tuple[dict[str, object], str]:
        """Return one complete NASA POWER response and its byte-hash stand-in."""

        assert "parameters=" in url
        assert "latitude=10.0" in url
        return payload, "response-hash"

    monkeypatch.setattr("scripts.build_nasa_power_climate._fetch_json", fake_fetch)
    result = _fetch_city(row, "2025-01-01", "2025-12-31", 5, 0, 0)

    assert result.city == "Alpha"
    assert result.annual_row["temperature_mean_c"] == 15.0
    assert result.annual_row["apparent_temperature_mean_c"] != 15.0
    assert result.provenance["grid_latitude"] == 10.0
    assert result.provenance["response_sha256"] == "response-hash"


def test_emergency_seed_fallback_inverts_southern_seasons() -> None:
    """Missing versioned files must not make Sydney hotter in July than January."""

    row = load_seed_cities("data/city_seed.csv").query("city == 'Sydney'").iloc[0]
    january = row_to_metrics(row, 1)
    july = row_to_metrics(row, 7)

    assert january.temperature_mean > july.temperature_mean
    assert january.temperature_max > july.temperature_max


def test_baseline_seasons_are_geographically_plausible() -> None:
    """Southern seasons and Beijing winter must not follow a northern seed formula."""

    baseline = _baseline().set_index(["city", "period_month"])

    assert baseline.loc[("Sydney", 1), "daily_max_temperature_mean_c"] > baseline.loc[("Sydney", 7), "daily_max_temperature_mean_c"]
    assert baseline.loc[("Melbourne", 1), "temperature_mean_c"] > baseline.loc[("Melbourne", 7), "temperature_mean_c"]
    assert baseline.loc[("Beijing", 1), "temperature_mean_c"] < 5.0
    assert baseline.loc[("Beijing", 7), "temperature_mean_c"] > 20.0


def test_annual_threshold_counts_match_sum_of_month_normals() -> None:
    """Annual mean event counts should equal the twelve calendar-month means."""

    baseline = _baseline()
    for city in ("Beijing", "Sydney", "Singapore"):
        rows = baseline[baseline["city"] == city]
        annual = rows[rows["period_month"] == 0].iloc[0]
        monthly = rows[rows["period_month"] != 0]
        for column in (
            "precipitation_days_mean",
            "heavy_rain_days_mean",
            "extreme_rain_days_mean",
            "hot_days_mean",
            "cold_days_mean",
            "windy_days_mean",
            "snow_days_mean",
        ):
            assert float(annual[column]) == pytest.approx(float(monthly[column].sum()), abs=0.01)


def test_static_metrics_use_power_and_acag_sources() -> None:
    """The default offline path must no longer present artificial climate values."""

    seed = attach_long_term_air_quality(load_seed_cities("data/city_seed.csv"), PM25_PATH)
    row = seed.query("city == 'Beijing'").iloc[0]
    metrics = row_to_metrics(row, 1, _baseline())
    result = evaluate_city(row_to_location(row), metrics, _preference(1))
    quality = {record.category: record for record in build_data_quality_records(result)}

    assert metrics.data_status == "dataset"
    assert metrics.sample_years == 26
    assert metrics.pm25_status == "dataset"
    assert metrics.estimated_fields == ("apparent_temperature", "snow_days")
    assert metrics.fallback_fields == ()
    assert 0 < metrics.relative_humidity_mean <= 100
    assert quality["climate"].fallback_used is False
    assert quality["humidity"].fallback_used is False
    assert quality["long_term_air_quality"].fallback_used is False
    assert not any("湿度仍为估算" in warning for warning in result.score.warnings)
    assert any("可能下雪日" in warning for warning in result.score.warnings)


def test_charts_use_one_higher_is_better_direction() -> None:
    """Charts must not mix positive fit bars with a higher-is-worse exposure bar."""

    seed = attach_long_term_air_quality(load_seed_cities("data/city_seed.csv"), PM25_PATH)
    baseline = _baseline()
    results = []
    for city in ("Beijing", "Sydney"):
        row = seed.query("city == @city").iloc[0]
        metrics = row_to_metrics(row, 7, baseline)
        results.append(evaluate_city(row_to_location(row), metrics, _preference(7)))

    bars = make_ranking_bar_chart(results, "zh")
    radar = make_radar_chart(results[0], "zh")

    assert [trace.name for trace in bars.data] == ["你的偏好匹配分", "气候舒适", "糟糕天气更少"]
    assert bars.layout.yaxis.title.text == "分数（越高越匹配）"
    assert "糟糕天气指数" not in [trace.name for trace in bars.data]
    assert all("安全" not in str(label) for label in radar.data[0].theta)


def test_climate_manifest_hashes_and_provenance_match_outputs() -> None:
    """The manifest must describe the exact files bundled with the app."""

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["source"]["model"] == "NASA POWER / MERRA-2"
    assert manifest["source"]["access"].startswith("public NASA Earth Science data")
    assert manifest["request"]["start_date"] == "2000-01-01"
    assert manifest["request"]["end_date"] == "2025-12-31"
    assert manifest["request"]["community"] == "AG"
    assert manifest["request"]["time_standard"] == "LST"
    assert set(manifest["request"]["daily_variables"]) == set(DAILY_VARIABLES)
    assert "snow_day_proxy" in manifest["aggregation"]["thresholds"]
    apparent_formula = manifest["aggregation"]["derived_fields"]["apparent_temperature_mean_c"]
    assert "WS10M - 4.0" in apparent_formula
    assert "WS10M_MAX" not in apparent_formula
    assert manifest["inputs"]["builder_sha256"] == _sha256(BUILDER_PATH)
    assert manifest["outputs"]["monthly"]["rows"] == 924
    assert manifest["outputs"]["annual"]["rows"] == 77
    assert manifest["outputs"]["monthly"]["sha256"] == _sha256(MONTHLY_PATH)
    assert manifest["outputs"]["annual"]["sha256"] == _sha256(ANNUAL_PATH)
    assert len(manifest["city_responses"]) == 77
    assert all(item["response_sha256"] for item in manifest["city_responses"])
