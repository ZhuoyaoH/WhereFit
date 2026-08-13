"""Generate concise Chinese explanations for the app."""

from __future__ import annotations

from calendar import month_name

from wherefit.data_loader import display_city_name
from wherefit.models import CityResult, UserPreference


MODE_LABELS_ZH = {
    "Travel": "旅行",
    "Living": "长期居住",
    "Compare": "城市对比",
}
MODE_LABELS_EN = {
    "Travel": "travel",
    "Living": "living",
    "Compare": "comparison",
}

PHRASE_EN = {
    "温度体感较友好": "comfortable perceived temperature",
    "湿度压力较低": "lower humidity pressure",
    "降水干扰较少": "less rain disruption",
    "空气质量指标较好": "better air-quality indicator",
    "沿海调节有助于缓和高温": "coastal moderation helps reduce heat",
    "综合表现相对均衡": "balanced overall profile",
    "未来几天气温体感较友好": "comfortable short-term temperature",
    "预报降水干扰较少": "less forecast rain disruption",
    "未来天气出行条件相对均衡": "balanced short-term travel weather",
    "温度或体感温度可能不理想": "temperature or perceived temperature may be uncomfortable",
    "湿度偏高，闷热感可能明显": "humidity may feel high",
    "降水或强降水天数偏多": "rain or heavy-rain days may be frequent",
    "空气质量指标偏弱": "air-quality indicator may be weaker",
    "长期高温天数较多": "long-term heat may be a concern",
    "强降水风险指标偏高": "heavy rain may be a concern",
    "长期空气污染指标偏高": "long-term air quality may be a concern",
    "沿海或台风影响需要留意": "coastal or typhoon conditions may need attention",
    "未来几天气温或体感温度不理想": "short-term temperature may be uncomfortable",
    "未来几天可能有降水干扰": "short-term rain may disrupt travel",
    "预报时间窗较远，时效权重较低": "forecast horizon weight is lower",
    "近期空气质量可能影响出行体验": "recent air quality may affect travel comfort",
}


def generate_city_report(city_result: CityResult, pref: UserPreference, lang: str = "zh") -> str:
    city = _city_name(city_result, lang)
    score = city_result.score
    mode_label = _mode_label(pref.mode, lang)
    period = _period_label(pref.month, lang)
    strengths = _join(score.strengths[:2], lang)
    weaknesses = _join(score.weaknesses[:2], lang) if score.weaknesses else ("no major weakness" if lang == "en" else "暂无特别突出的短板")
    pm25_context = _pm25_context(city_result, lang)
    if lang == "en":
        return (
            f"{city}'s {period} {mode_label} preference match is {score.personal_fit_score:.0f}/100. "
            f"Main strengths: {strengths}. Watch-outs: {weaknesses}. "
            f"Climate comfort is {score.travel_comfort_score:.0f}/100, and the Bad Weather Index is "
            f"{score.long_term_risk_score:.0f}/100. {pm25_context}"
        )
    return (
        f"{city} 在{period}的{mode_label}偏好匹配分为 {score.personal_fit_score:.0f}/100。"
        f"主要优势是{strengths}；需要注意的是{weaknesses}。"
        f"其中气候舒适指标为 {score.travel_comfort_score:.0f}/100，"
        f"糟糕天气指数为 {score.long_term_risk_score:.0f}/100。{pm25_context}"
    )


def generate_comparison_report(results: list[CityResult], pref: UserPreference, lang: str = "zh") -> str:
    if not results:
        return "Enter at least one recognized city." if lang == "en" else "请输入至少一个可识别城市。"
    best = results[0]
    mode_label = _mode_label(pref.mode, lang)
    best_city = _city_name(best, lang)
    strengths = _join(best.score.strengths[:2], lang)
    period = _period_label(pref.month, lang)
    if lang == "en":
        return (
            f"Given the current preferences, {best_city} is the best {period} {mode_label} option "
            f"among the cities selected for this comparison, with a preference match score of {best.score.personal_fit_score:.0f}/100. "
            f"Its main strengths are {strengths}. Raising heat, humidity, rain, air-quality, or extreme-weather "
            f"sensitivity will recalculate the ranking."
        )
    return (
        f"根据当前偏好，{best_city} 在本次待比较城市中的{period}{mode_label}偏好匹配分最高，"
        f"分数为 {best.score.personal_fit_score:.0f}/100。"
        f"它的主要优势是{strengths}。"
        "如果调高怕热、讨厌潮湿或关注极端天气，排序会随对应分项重新计算。"
    )


def _city_name(city_result: CityResult, lang: str) -> str:
    return display_city_name(city_result.location.city, city_result.location.city_en, lang)


def _pm25_context(city_result: CityResult, lang: str) -> str:
    """Describe the long-term PM2.5 input without implying a station observation."""

    metrics = city_result.metrics
    if metrics.pm25_year_start is not None and metrics.pm25_year_end is not None:
        if lang == "en":
            return (
                f"The long-term PM2.5 input is {metrics.pm25:.1f} ug/m3 "
                f"({metrics.pm25_year_start}-{metrics.pm25_year_end} fused grid estimate)."
            )
        return (
            f"长期 PM2.5 输入为 {metrics.pm25:.1f} 微克/立方米"
            f"（{metrics.pm25_year_start}–{metrics.pm25_year_end} 年网格融合估计）。"
        )
    if lang == "en":
        return f"The long-term PM2.5 input is a {metrics.pm25:.1f} ug/m3 seed fallback."
    return f"长期 PM2.5 输入为 {metrics.pm25:.1f} 微克/立方米的备用参考值。"


def _mode_label(mode: str, lang: str) -> str:
    return (MODE_LABELS_EN if lang == "en" else MODE_LABELS_ZH).get(mode, "comparison" if lang == "en" else "城市对比")


def _period_label(month: int, lang: str) -> str:
    """Return a natural-language full-year or calendar-month label."""

    if int(month) == 0:
        return "full-year" if lang == "en" else "全年"
    return month_name[int(month)] if lang == "en" else f"{int(month)}月"


def _join(items: list[str], lang: str) -> str:
    if lang == "en":
        translated = [PHRASE_EN.get(item, item) for item in items]
        return ", ".join(translated) if translated else "balanced overall profile"
    return "、".join(items) if items else "综合表现相对均衡"
