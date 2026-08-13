from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_acag_pm25 import _parse_h5dump_values, grid_window, parse_years
from wherefit.data_loader import attach_long_term_air_quality, load_seed_cities, row_to_location, row_to_metrics, with_long_term_air_quality
from wherefit.data_quality import build_data_quality_records
from wherefit.data_sources.open_meteo_history import metrics_from_history
from wherefit.models import CityResult, ScoreResult


ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / "data" / "city_seed.csv"
PM25_PATH = ROOT / "data" / "air_quality" / "acag_v6gl03_city_annual_2015_2024.csv"
MANIFEST_PATH = ROOT / "data" / "air_quality" / "acag_v6gl03_manifest.json"


def _score() -> ScoreResult:
    """Return a minimal score record for provenance tests."""

    return ScoreResult(50.0, 50.0, 50.0, {})


def test_bundled_acag_table_has_complete_ten_year_city_coverage() -> None:
    """The committed table must contain one finite annual record per city-year."""

    data = pd.read_csv(PM25_PATH)

    assert data.shape[0] == 770
    assert data["city"].nunique() == 77
    assert set(data["year"].unique()) == set(range(2015, 2025))
    assert not data["pm25_mean_ug_m3"].isna().any()
    assert data["pm25_mean_ug_m3"].between(0, 500).all()
    assert (data["valid_grid_cells"] == 9).all()


def test_manifest_pins_version_sources_and_output_hash() -> None:
    """The provenance manifest must cryptographically identify the bundled table."""

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    output_hash = hashlib.sha256(PM25_PATH.read_bytes()).hexdigest()

    assert manifest["source_version"] == "V6.GL.03"
    assert manifest["license"] == "CC BY 4.0"
    assert manifest["year_start"] == 2015
    assert manifest["year_end"] == 2024
    assert manifest["output_sha256"] == output_hash
    assert len(manifest["sources"]) == 10


def test_acag_summary_replaces_seed_level_and_exposes_provenance() -> None:
    """A covered city should use continuous ACAG PM2.5 without a fallback flag."""

    seed = load_seed_cities(SEED_PATH)
    enriched = attach_long_term_air_quality(seed, PM25_PATH)
    row = enriched.query("city == 'Beijing'").iloc[0]
    metrics = row_to_metrics(row, 7)
    result = CityResult(row_to_location(row), metrics, _score())
    quality = {record.category: record for record in build_data_quality_records(result)}

    assert metrics.pm25 == pytest.approx(52.58641)
    assert metrics.pm25_status == "dataset"
    assert metrics.pm25_sample_years == 10
    assert metrics.pm25_year_start == 2015
    assert metrics.pm25_year_end == 2024
    assert metrics.pm25_trend_per_year == pytest.approx(-3.2921618)
    assert "pm25" not in metrics.fallback_fields
    assert quality["long_term_air_quality"].fallback_used is False
    assert quality["long_term_air_quality"].source == "ACAG SatPM2.5 V6.GL.03"


def test_historical_weather_keeps_independent_acag_pm25_metadata() -> None:
    """Replacing seed climate with weather history must not discard long-term PM2.5."""

    enriched = attach_long_term_air_quality(load_seed_cities(SEED_PATH), PM25_PATH)
    row = enriched.query("city == 'Shanghai'").iloc[0]
    source_metrics = row_to_metrics(row, 0)
    raw = pd.DataFrame(
        {
            "time": ["2020-01-01", "2021-01-01"],
            "temperature_2m_mean": [8.0, 9.0],
            "temperature_2m_max": [11.0, 12.0],
            "temperature_2m_min": [5.0, 6.0],
            "precipitation_sum": [1.0, 0.0],
        }
    )
    history = metrics_from_history(raw, row_to_location(row), 0)
    merged = with_long_term_air_quality(history, source_metrics)

    assert merged.pm25 == source_metrics.pm25
    assert merged.pm25_status == "dataset"
    assert merged.pm25_year_start == 2015
    assert "pm25" not in merged.fallback_fields


def test_missing_acag_file_preserves_explicit_seed_fallback(tmp_path: Path) -> None:
    """A missing optional table should keep the app usable with visible fallback metadata."""

    seed = load_seed_cities(SEED_PATH)
    unchanged = attach_long_term_air_quality(seed, tmp_path / "missing.csv")
    metrics = row_to_metrics(unchanged.query("city == 'Beijing'").iloc[0], 7)

    assert metrics.pm25 == 37.0
    assert metrics.pm25_status == "fallback"
    assert "pm25" in metrics.fallback_fields


def test_acag_grid_and_parser_helpers_are_deterministic() -> None:
    """Coordinate mapping and HDF5 value parsing must remain stable."""

    assert parse_years("2015:2017") == (2015, 2016, 2017)
    assert grid_window(39.9042, 116.4074) == (998, 2963, 3, 3)
    output = "header DATA { 1.0, nan, 3.5 } ATTRIBUTE DATA { 999 }"
    values = _parse_h5dump_values(output)
    assert values[0] == 1.0
    assert pd.isna(values[1])
    assert values[2] == 3.5
