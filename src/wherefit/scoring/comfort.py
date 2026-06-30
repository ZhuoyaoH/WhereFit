"""Travel comfort component scores."""

from __future__ import annotations

from wherefit.config import CONFIG
from wherefit.models import ClimateMetrics, UserPreference
from wherefit.scoring.preference import normalize_score


def score_temperature(metrics: ClimateMetrics, pref: UserPreference) -> float:
    ideal_lower = CONFIG.ideal_temp_lower + 0.6 * pref.cold_sensitivity
    ideal_upper = CONFIG.ideal_temp_upper - 0.8 * pref.heat_sensitivity

    cold_penalty = max(0.0, ideal_lower - metrics.temperature_mean) * (2.0 + pref.cold_sensitivity)
    heat_penalty = max(0.0, metrics.apparent_temperature - ideal_upper) * (2.0 + pref.heat_sensitivity)
    hot_day_penalty = metrics.hot_days * (0.6 + pref.heat_sensitivity * 0.25)
    return normalize_score(100.0 - cold_penalty - heat_penalty - hot_day_penalty)


def score_humidity(metrics: ClimateMetrics, pref: UserPreference) -> float:
    humidity = metrics.relative_humidity_mean
    if humidity <= CONFIG.humidity_comfort_ceiling:
        return 100.0
    penalty = (humidity - CONFIG.humidity_comfort_ceiling) * (0.8 + 0.3 * pref.humidity_sensitivity)
    return normalize_score(100.0 - penalty)


def score_precipitation(metrics: ClimateMetrics, pref: UserPreference) -> float:
    rain_penalty = metrics.precipitation_days * (1.5 + 0.4 * pref.rain_sensitivity)
    heavy_penalty = metrics.heavy_rain_days * (4.0 + 0.8 * pref.rain_sensitivity)
    return normalize_score(100.0 - rain_penalty - heavy_penalty)


def score_air_quality(metrics: ClimateMetrics, pref: UserPreference) -> float:
    pm25 = metrics.pm25
    if pm25 <= CONFIG.clean_pm25_threshold:
        score = 100.0
    elif pm25 <= CONFIG.moderate_pm25_threshold:
        score = 100.0 - (pm25 - CONFIG.clean_pm25_threshold) * 1.2
    else:
        score = 70.0 - (pm25 - CONFIG.moderate_pm25_threshold) * (
            1.0 + 0.3 * pref.air_quality_sensitivity
        )
    return normalize_score(score)


def compute_travel_comfort(metrics: ClimateMetrics, pref: UserPreference) -> tuple[float, dict[str, float]]:
    components = {
        "温度舒适": score_temperature(metrics, pref),
        "湿度舒适": score_humidity(metrics, pref),
        "降水友好": score_precipitation(metrics, pref),
        "空气质量": score_air_quality(metrics, pref),
    }
    score = (
        components["温度舒适"] * 0.38
        + components["湿度舒适"] * 0.22
        + components["降水友好"] * 0.22
        + components["空气质量"] * 0.18
    )
    return normalize_score(score), components
