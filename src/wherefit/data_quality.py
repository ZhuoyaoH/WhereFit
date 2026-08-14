"""Build transparent provenance records for each evaluated city."""

from __future__ import annotations

from wherefit.models import AirQualitySummary, CityResult, DataQualityRecord


def build_data_quality_records(result: CityResult) -> tuple[DataQualityRecord, ...]:
    """Return data-source, time-scope, fallback, and scoring-role records."""

    metrics = result.metrics
    humidity_estimated = "relative_humidity_mean" in metrics.estimated_fields
    humidity_fallback = humidity_estimated or metrics.data_status == "fallback"
    pm25_fallback = metrics.pm25_status == "fallback"
    pm25_scope = "long-term baseline; not a recent observation"
    if metrics.pm25_year_start is not None and metrics.pm25_year_end is not None:
        pm25_scope = f"annual estimates, {metrics.pm25_year_start}-{metrics.pm25_year_end}"
    records = [
        DataQualityRecord(
            category="climate",
            source=metrics.data_source,
            status=metrics.data_status,
            time_scope=_climate_scope(metrics.sample_years),
            sample_size=f"{metrics.sample_years} years" if metrics.sample_years else "curated levels",
            affects_score=True,
            fallback_used=metrics.data_status == "fallback",
            note=(
                "Temperature, humidity, precipitation, and wind come from area-wide weather estimates; "
                "possible snow days are estimated from temperature and precipitation."
                if metrics.data_status == "dataset" and "apparent_temperature" in metrics.estimated_fields
                else "Temperature, precipitation, and related information used to calculate the preference match."
            ),
        ),
        DataQualityRecord(
            category="humidity",
            source=metrics.data_source if not humidity_estimated else "estimated from region and rainfall",
            status="estimated" if humidity_estimated else metrics.data_status,
            time_scope=_climate_scope(metrics.sample_years),
            sample_size="derived field" if humidity_estimated else _sample_label(metrics.sample_years),
            affects_score=True,
            fallback_used=humidity_fallback,
            note=(
                "Humidity is estimated from rainfall and regional context because direct humidity data is unavailable for this result."
                if humidity_estimated
                else (
                    "Humidity is provided directly by the fixed-model historical dataset."
                    if metrics.sample_years
                    else "Humidity comes from the basic city reference data."
                )
            ),
        ),
        DataQualityRecord(
            category="long_term_air_quality",
            source=metrics.pm25_source,
            status=metrics.pm25_status,
            time_scope=pm25_scope,
            sample_size=f"{metrics.pm25_sample_years} annual values" if metrics.pm25_sample_years else "curated level",
            affects_score=True,
            fallback_used=pm25_fallback,
            note=(
                "A city-area estimate combining satellite, simulation, and monitoring data; not a single station observation. "
                "Recent 8-day PM2.5 is not substituted for this long-term baseline."
                if not pm25_fallback
                else "Basic city reference data is used because long-term PM2.5 data is unavailable for this city."
            ),
        ),
        DataQualityRecord(
            category="hazard_records",
            source="USGS / IBTrACS / NASA EONET / Open-Meteo Flood when queried",
            status="not_requested",
            time_scope="provider-specific historical or recent window",
            sample_size="on demand",
            affects_score=False,
            fallback_used=False,
            note="Historical records are displayed separately and do not alter the main ranking.",
        ),
    ]
    if result.forecast is not None:
        records.append(_forecast_record(result))
    if result.air_quality is not None:
        records.append(_recent_air_quality_record(result.air_quality))
    return tuple(records)


def _forecast_record(result: CityResult) -> DataQualityRecord:
    """Build the short-term forecast provenance row."""

    forecast = result.forecast
    assert forecast is not None
    return DataQualityRecord(
        category="forecast",
        source=forecast.provider or forecast.source,
        status=forecast.status,
        time_scope=f"{forecast.start_date} to {forecast.end_date}",
        sample_size=f"{forecast.days} days",
        affects_score=forecast.status != "failed",
        fallback_used=False,
        note="Used for the Travel ranking only when the selected month is in the valid forecast window.",
    )


def _recent_air_quality_record(summary: AirQualitySummary) -> DataQualityRecord:
    """Build the recent-air-quality provenance row."""

    scope = "recent 8-day window"
    if summary.period_start and summary.period_end:
        scope = f"{summary.period_start} to {summary.period_end}"
    return DataQualityRecord(
        category="recent_air_quality",
        source=summary.source,
        status=summary.status,
        time_scope=scope,
        sample_size=f"{summary.sample_hours} hours",
        affects_score=summary.status != "failed" and summary.pm25_mean is not None,
        fallback_used=False,
        note="Used only in short-term Travel scoring; never treated as a long-term air-quality normal.",
    )


def _climate_scope(sample_years: int) -> str:
    """Return a compact climate time-scope label."""

    return "multi-year historical window" if sample_years else "curated baseline without observation years"


def _sample_label(sample_years: int) -> str:
    """Return a compact sample-size label."""

    return f"{sample_years} years" if sample_years else "curated level"
