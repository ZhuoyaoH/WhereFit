"""Long-term risk proxy scores."""

from __future__ import annotations

from wherefit.models import ClimateMetrics, UserPreference
from wherefit.scoring.preference import normalize_score


def score_heat_risk(metrics: ClimateMetrics, pref: UserPreference) -> float:
    risk = metrics.hot_days * (4.8 + pref.extreme_weather_sensitivity * 0.4)
    risk += max(0.0, metrics.temperature_max - 34.0) * 3.5
    return normalize_score(risk)


def score_heavy_rain_risk(metrics: ClimateMetrics, pref: UserPreference) -> float:
    risk = metrics.heavy_rain_days * (15.0 + pref.extreme_weather_sensitivity * 1.2)
    risk += max(0.0, metrics.precipitation_days - 12.0) * 1.8
    return normalize_score(risk)


def score_air_pollution_risk(metrics: ClimateMetrics, pref: UserPreference) -> float:
    risk = max(0.0, metrics.pm25 - 10.0) * (1.2 + pref.air_quality_sensitivity * 0.18)
    return normalize_score(risk)


def score_typhoon_coastal_risk(metrics: ClimateMetrics, pref: UserPreference) -> float:
    base = 0.0
    if metrics.coastal:
        base += 12.0
    if metrics.typhoon_region:
        base += 28.0
    return normalize_score(base * (0.75 + pref.extreme_weather_sensitivity * 0.12))


def compute_long_term_risk(metrics: ClimateMetrics, pref: UserPreference) -> tuple[float, dict[str, float]]:
    components = {
        "高温风险": score_heat_risk(metrics, pref),
        "强降水风险": score_heavy_rain_risk(metrics, pref),
        "空气污染风险": score_air_pollution_risk(metrics, pref),
        "沿海台风风险": score_typhoon_coastal_risk(metrics, pref),
    }
    risk = (
        components["高温风险"] * 0.34
        + components["强降水风险"] * 0.25
        + components["空气污染风险"] * 0.23
        + components["沿海台风风险"] * 0.18
    )
    return normalize_score(risk), components
