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
        confidence=_confidence(metrics),
        data_source=metrics.data_source,
        data_status=metrics.data_status,
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
        strengths.append("空气质量 proxy 较好")
    if metrics.coastal and components["温度舒适"] >= 70:
        strengths.append("沿海调节有助于缓和高温")
    return strengths[:4] or ["综合表现相对均衡"]


def _weaknesses(components: dict[str, float]) -> list[str]:
    labels = {
        "温度舒适": "温度或体感温度可能不理想",
        "湿度舒适": "湿度偏高，闷热感可能明显",
        "降水友好": "降水或强降水天数偏多",
        "空气质量": "空气质量 proxy 偏弱",
        "高温风险": "长期高温暴露 proxy 偏高",
        "强降水风险": "强降水风险 proxy 偏高",
        "空气污染风险": "长期空气污染 proxy 偏高",
        "沿海台风风险": "沿海或台风暴露 proxy 偏高",
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
        warnings.append("长期风险 proxy 偏高，建议结合真实气候数据复核")
    if metrics.data_status == "fallback":
        warnings.append("当前城市使用静态等级 fallback，不代表权威气候风险评级")
    else:
        warnings.append("历史天气来自再分析/公开 API 聚合，仍不代表权威气候风险评级")
    return warnings


def _confidence(metrics: ClimateMetrics) -> float:
    if metrics.data_status in {"live", "cache", "cache/live", "partial"}:
        base = CONFIG.historical_data_confidence - (0.08 if metrics.data_status == "partial" else 0.0)
        return max(0.55, min(CONFIG.historical_data_confidence, base - metrics.missing_rate))
    return CONFIG.static_data_confidence
