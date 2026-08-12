"""Combine component scores into city-level results."""

from __future__ import annotations

from wherefit.config import CONFIG
from wherefit.models import CityResult, ClimateMetrics, Location, ScoreResult, UserPreference
from wherefit.scoring.comfort import compute_travel_comfort
from wherefit.scoring.preference import mode_weights, normalize_score
from wherefit.scoring.risk import compute_long_term_risk


def compute_personal_fit(travel_score: float, risk_score: float, pref: UserPreference) -> float:
    weights = mode_weights(pref.mode)
    return normalize_score(travel_score * weights["comfort"] + (100.0 - risk_score) * weights["risk"])


def evaluate_city(location: Location, metrics: ClimateMetrics, pref: UserPreference) -> CityResult:
    comfort_score, comfort_components = compute_travel_comfort(metrics, pref)
    risk_score, risk_components = compute_long_term_risk(metrics, pref)
    personal_score = compute_personal_fit(comfort_score, risk_score, pref)
    components = {**comfort_components, **risk_components}
    score = ScoreResult(
        travel_comfort_score=comfort_score,
        long_term_risk_score=risk_score,
        personal_fit_score=personal_score,
        component_scores=components,
        strengths=_strengths(components, metrics),
        weaknesses=_weaknesses(components),
        warnings=_warnings(metrics, risk_score),
        confidence=_evidence_quality(metrics),
        data_source=metrics.data_source,
        data_status=metrics.data_status,
        climate_normal_fit_score=personal_score,
        historical_hazard_exposure_score=None,
        typhoon_history_score=None,
        rainfall_extreme_score=components.get("强降水风险"),
    )
    return CityResult(location=location, metrics=metrics, score=score)


def rank_cities(results: list[CityResult]) -> list[CityResult]:
    return sorted(results, key=lambda item: item.score.personal_fit_score, reverse=True)


def _strengths(components: dict[str, float], metrics: ClimateMetrics) -> list[str]:
    strengths: list[str] = []
    if components["温度舒适"] >= 78:
        strengths.append("温度体感较友好")
    if components["湿度舒适"] >= 78:
        strengths.append("湿度压力较低")
    if components["降水友好"] >= 78:
        strengths.append("降水干扰较少")
    if components["空气质量"] >= 78:
        strengths.append("空气质量指标较好")
    if metrics.coastal and components["温度舒适"] >= 70:
        strengths.append("沿海调节有助于缓和高温")
    return strengths[:4] or ["综合表现相对均衡"]


def _weaknesses(components: dict[str, float]) -> list[str]:
    labels = {
        "温度舒适": "温度或体感温度可能不理想",
        "湿度舒适": "湿度偏高，闷热感可能明显",
        "降水友好": "降水或强降水天数偏多",
        "空气质量": "空气质量指标偏弱",
        "高温风险": "长期高温天数较多",
        "强降水风险": "强降水风险指标偏高",
        "空气污染风险": "长期空气污染指标偏高",
        "沿海台风风险": "沿海或台风影响需要留意",
    }
    weak: list[str] = []
    for key, value in components.items():
        if key.endswith("风险") and value >= 55:
            weak.append(labels[key])
        elif not key.endswith("风险") and value < 62:
            weak.append(labels[key])
    return weak[:4]


def _warnings(metrics: ClimateMetrics, risk_score: float) -> list[str]:
    warnings: list[str] = []
    if metrics.typhoon_region:
        warnings.append("该城市位于台风影响区域，当前台风项仍是简化提示")
    if risk_score >= 65:
        warnings.append("糟糕天气指数偏高，建议进一步了解高温、强降雨和空气质量情况")
    if metrics.data_status == "fallback":
        if metrics.pm25_status == "dataset":
            warnings.append("温度、湿度和降水等气候项使用城市参考数据；长期 PM2.5 已使用多年融合数据")
        else:
            warnings.append("当前城市使用人工基础等级，不代表观测常态或权威评级")
    elif metrics.estimated_fields or metrics.fallback_fields:
        if metrics.data_status == "dataset" and {
            "apparent_temperature",
            "snow_days",
        }.issubset(metrics.estimated_fields):
            warnings.append("多年气候数据中的体感温度由温度、湿度和风速计算；可能下雪日由温度和降水推算")
        elif metrics.pm25_status == "dataset" and "relative_humidity_mean" in metrics.estimated_fields:
            warnings.append("历史天气来自公开资料汇总，湿度为估算；长期 PM2.5 使用多年融合数据")
        else:
            warnings.append("历史天气来自公开资料汇总，部分项目为估算或备用参考值")
    return warnings


def _evidence_quality(metrics: ClimateMetrics) -> float:
    """Estimate source completeness; this is not a statistical confidence interval."""

    if metrics.data_status in {"live", "cache", "cache/live", "partial", "dataset"}:
        base = CONFIG.historical_data_confidence - (0.08 if metrics.data_status == "partial" else 0.0)
        base -= min(0.16, len(metrics.estimated_fields) * 0.04)
        base -= min(0.16, len(metrics.fallback_fields) * 0.08)
        return max(0.40, min(CONFIG.historical_data_confidence, base - metrics.missing_rate))
    return 0.45
