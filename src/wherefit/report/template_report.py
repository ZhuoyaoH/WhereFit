"""Generate concise Chinese explanations for the app."""

from __future__ import annotations

from wherefit.config import MODE_LABELS
from wherefit.models import CityResult, UserPreference


def generate_city_report(city_result: CityResult, pref: UserPreference) -> str:
    city = city_result.location.city
    score = city_result.score
    mode_label = MODE_LABELS.get(pref.mode, "城市对比")
    strengths = "、".join(score.strengths[:2])
    weaknesses = "、".join(score.weaknesses[:2]) if score.weaknesses else "暂无特别突出的短板"
    return (
        f"{city} 在{pref.month}月的{mode_label}匹配度为 {score.personal_fit_score:.0f}/100。"
        f"主要优势是{strengths}；需要注意的是{weaknesses}。"
        f"其中旅行舒适度为 {score.travel_comfort_score:.0f}/100，"
        f"长期风险 proxy 为 {score.long_term_risk_score:.0f}/100。结果仅供参考。"
    )


def generate_comparison_report(results: list[CityResult], pref: UserPreference) -> str:
    if not results:
        return "请输入至少一个可识别城市。"
    best = results[0]
    mode_label = MODE_LABELS.get(pref.mode, "城市对比")
    return (
        f"根据当前偏好，{best.location.city} 是这组城市中最适合{pref.month}月{mode_label}的选择，"
        f"匹配度为 {best.score.personal_fit_score:.0f}/100。"
        f"它的主要优势是{'、'.join(best.score.strengths[:2])}。"
        "如果调高怕热、讨厌潮湿或关注极端天气，排序会随对应分项重新计算。"
    )
