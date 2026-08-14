"""Short-term forecast scoring for travel mode."""

from __future__ import annotations

from dataclasses import replace

from wherefit.models import AirQualitySummary, CityResult, ForecastSummary, ScoreResult, UserPreference
from wherefit.scoring.preference import normalize_score


def compute_forecast_trip_fit(
    summary: ForecastSummary,
    pref: UserPreference,
    air_quality: AirQualitySummary | None = None,
) -> tuple[float, dict[str, float]]:
    """Score short-term travel weather and optional recent air quality."""

    if summary.status == "failed" or summary.days <= 0:
        return 0.0, {
            "预报温度舒适": 0.0,
            "预报降水友好": 0.0,
            "预报时效": 0.0,
        }

    temp_max_c = _to_celsius(summary.temp_max_mean, summary.temperature_unit)
    temp_min_c = _to_celsius(summary.temp_min_mean, summary.temperature_unit)
    apparent_c = _to_celsius(summary.apparent_temp_max_mean, summary.temperature_unit)

    ideal_lower = 16.0 + pref.cold_sensitivity * 0.6
    ideal_upper = 27.0 - pref.heat_sensitivity * 0.7
    heat_penalty = max(0.0, apparent_c - ideal_upper) * (2.6 + pref.heat_sensitivity * 0.8)
    cold_penalty = max(0.0, ideal_lower - temp_min_c) * (2.2 + pref.cold_sensitivity * 0.7)
    temp_spread_penalty = max(0.0, temp_max_c - temp_min_c - 13.0) * 0.9
    temperature_score = normalize_score(100.0 - heat_penalty - cold_penalty - temp_spread_penalty)

    rain_penalty = summary.precipitation_days * (4.5 + pref.rain_sensitivity * 1.1)
    probability_penalty = summary.precipitation_probability_max * (0.08 + pref.rain_sensitivity * 0.025)
    heavy_rain_penalty = summary.heavy_rain_days * (9.0 + pref.extreme_weather_sensitivity * 1.2)
    precipitation_score = normalize_score(100.0 - rain_penalty - probability_penalty - heavy_rain_penalty)

    wind_score = normalize_score(100.0 - summary.windy_days * (7.0 + pref.extreme_weather_sensitivity * 1.4))
    horizon_score = normalize_score(summary.confidence * 100.0)

    components = {
        "预报温度舒适": temperature_score,
        "预报降水友好": precipitation_score,
        "预报大风影响": wind_score,
        "预报时效": horizon_score,
    }
    if air_quality is not None and air_quality.status != "failed" and air_quality.pm25_mean is not None:
        air_score = _score_recent_air_quality(air_quality.pm25_mean, pref)
        components["近期空气质量"] = air_score
        score = (
            temperature_score * 0.36
            + precipitation_score * 0.27
            + wind_score * 0.12
            + air_score * 0.15
            + horizon_score * 0.10
        )
    else:
        score = (
            temperature_score * 0.42
            + precipitation_score * 0.32
            + wind_score * 0.14
            + horizon_score * 0.12
        )
    return normalize_score(score), components


def apply_forecast_score(
    result: CityResult,
    summary: ForecastSummary,
    pref: UserPreference,
    air_quality: AirQualitySummary | None = None,
) -> CityResult:
    """Apply valid forecast scoring or preserve the climate fallback when it fails."""

    forecast_score, forecast_components = compute_forecast_trip_fit(summary, pref, air_quality)
    score = result.score
    warnings = list(score.warnings)
    if summary.status == "failed":
        warnings.append("未来天气暂时无法获取，当前排序改用多年气候匹配分")
        return replace(
            result,
            forecast=summary,
            air_quality=air_quality,
            score=replace(
                score,
                forecast_trip_fit_score=None,
                warnings=warnings,
            ),
        )
    elif summary.days > 7:
        warnings.append("预报范围超过 7 天，远期不确定性较高")
    source = summary.provider or summary.source
    if air_quality is not None and air_quality.status != "failed" and air_quality.pm25_mean is not None:
        source = f"{source} + {air_quality.source} (recent)"
    updated_score = ScoreResult(
        travel_comfort_score=score.travel_comfort_score,
        long_term_risk_score=score.long_term_risk_score,
        personal_fit_score=forecast_score,
        component_scores={**score.component_scores, **forecast_components},
        strengths=_forecast_strengths(forecast_components),
        weaknesses=_forecast_weaknesses(forecast_components),
        warnings=warnings,
        confidence=min(score.confidence, summary.confidence) if summary.confidence else score.confidence,
        data_source=source,
        data_status=summary.status,
        climate_normal_fit_score=score.climate_normal_fit_score,
        forecast_trip_fit_score=forecast_score,
        historical_hazard_exposure_score=score.historical_hazard_exposure_score,
        earthquake_history_score=score.earthquake_history_score,
        typhoon_history_score=score.typhoon_history_score,
        rainfall_extreme_score=score.rainfall_extreme_score,
        aurora_opportunity_score=score.aurora_opportunity_score,
    )
    return replace(result, score=updated_score, forecast=summary, air_quality=air_quality)


def _forecast_strengths(components: dict[str, float]) -> list[str]:
    strengths: list[str] = []
    if components["预报温度舒适"] >= 78:
        strengths.append("未来几天气温较舒适")
    if components["预报降水友好"] >= 78:
        strengths.append("预报降水干扰较少")
    return strengths[:2] or ["未来天气出行条件相对均衡"]


def _forecast_weaknesses(components: dict[str, float]) -> list[str]:
    labels = {
        "预报温度舒适": "未来几天气温不理想",
        "预报降水友好": "未来几天可能有降水干扰",
        "预报时效": "预报时间窗较远，时效权重较低",
        "近期空气质量": "近期空气质量可能影响出行体验",
    }
    return [labels[key] for key, value in components.items() if key in labels and value < 62][:4]


def _to_celsius(value: float, unit: str) -> float:
    if unit == "°F":
        return (value - 32.0) * 5.0 / 9.0
    return value


def _score_recent_air_quality(pm25: float, pref: UserPreference) -> float:
    """Convert recent PM2.5 into a short-term travel comfort component."""

    if pm25 <= 10.0:
        return 100.0
    if pm25 <= 35.0:
        return normalize_score(100.0 - (pm25 - 10.0) * 1.2)
    penalty = (pm25 - 35.0) * (1.0 + 0.3 * pref.air_quality_sensitivity)
    return normalize_score(70.0 - penalty)
