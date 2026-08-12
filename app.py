from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date, timedelta
from functools import partial
from html import escape
from pathlib import Path
import sys
from typing import Callable

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wherefit.config import (
    CLIMATE_BASELINE_END_YEAR,
    CLIMATE_BASELINE_START_YEAR,
    DISCLAIMER,
    HAZARD_DISCLAIMER,
    HISTORY_START_DATE,
)
from wherefit.city_filter import CityFilterOptions, candidate_city_input, filter_seed_cities
from wherefit.data_loader import (
    attach_long_term_air_quality,
    load_climate_baseline,
    load_seed_cities,
    match_cities,
    parse_city_input,
    row_to_location,
    row_to_metrics,
    with_long_term_air_quality,
)
from wherefit.data_quality import build_data_quality_records
from wherefit.data_sources.air_quality import get_air_quality_summary
from wherefit.data_sources.met_no_forecast import get_met_no_forecast_summary
from wherefit.data_sources.open_meteo_forecast import default_forecast_dates, get_forecast_summary
from wherefit.data_sources.open_meteo_history import default_history_end_date, get_history_metrics
from wherefit.hazards.aurora import build_aurora_summary
from wherefit.hazards.summary import build_hazard_summary
from wherefit.hazards.typhoon import typhoon_cache_paths
from wherefit.models import AirQualitySummary, CityResult, ForecastSummary, HazardSummary, TyphoonSummary, UserPreference
from wherefit.scoring.forecast import apply_forecast_score
from wherefit.report.template_report import generate_city_report, generate_comparison_report
from wherefit.scoring.overall import evaluate_city, rank_cities
from wherefit.visualization.cards import risk_label, score_label
from wherefit.visualization.charts import make_radar_chart, make_ranking_bar_chart
from wherefit.visualization.maps import make_earthquake_map, make_map, make_typhoon_track_map


DATA_PATH = ROOT / "data" / "city_seed.csv"
LONG_TERM_PM25_PATH = ROOT / "data" / "air_quality" / "acag_v6gl03_city_annual_2015_2024.csv"
CLIMATE_MONTHLY_PATH = ROOT / "data" / "climate" / "nasa_power_merra2_city_monthly_2000_2025.csv"
CLIMATE_ANNUAL_PATH = ROOT / "data" / "climate" / "nasa_power_merra2_city_annual_2000_2025.csv"
HISTORY_CACHE_DIR = ROOT / "data" / "cache" / "weather" / "history"
FORECAST_CACHE_DIR = ROOT / "data" / "cache" / "weather" / "forecast"
HAZARD_CACHE_DIR = ROOT / "data" / "cache" / "hazards"
AURORA_CACHE_DIR = ROOT / "data" / "cache" / "aurora"
AIR_QUALITY_CACHE_DIR = ROOT / "data" / "cache" / "air_quality"
MAX_PARALLEL_CITY_REQUESTS = 4
DEFAULT_CITIES = "北京, 上海, 广州, 成都, 昆明, 哈尔滨, 青岛"
DEMO_CITY_SETS = {
    "east_china": {
        "zh": "东部城市：北京、上海、广州",
        "en": "Eastern China: Beijing, Shanghai, Guangzhou",
        "cities_zh": "北京, 上海, 广州",
        "cities_en": "Beijing, Shanghai, Guangzhou",
    },
    "southwest": {
        "zh": "西南与高原：成都、昆明、拉萨",
        "en": "Southwest and Plateau: Chengdu, Kunming, Lhasa",
        "cities_zh": "成都, 昆明, 拉萨",
        "cities_en": "Chengdu, Kunming, Lhasa",
    },
    "north": {
        "zh": "高纬与沿海：漠河、哈尔滨、青岛",
        "en": "High Latitude and Coast: Mohe, Harbin, Qingdao",
        "cities_zh": "漠河, 哈尔滨, 青岛",
        "cities_en": "Mohe, Harbin, Qingdao",
    },
}
DATA_MODE_STATIC = "演示静态数据"
DATA_MODE_HISTORY = "历史气候常态（2000 至今）"
DATA_MODE_FORECAST = "未来天气预报"
DATA_MODE_HAZARD = "历史灾害档案"
DATA_MODE_AURORA = "极光机会"
MONTH_ALL = 0
REGION_TYPE_LABELS = {
    "arid": "干旱/半干旱",
    "central": "中部内陆",
    "east": "华东内陆/平原",
    "east_coast": "东部沿海",
    "high_latitude": "高纬寒冷",
    "mediterranean": "地中海气候",
    "north": "北方内陆",
    "north_coast": "北方沿海",
    "north_plateau": "北方高原",
    "northeast": "东北内陆",
    "northwest": "西北内陆",
    "plateau": "高原",
    "south": "南方内陆",
    "south_coast": "南方沿海",
    "southwest_basin": "西南盆地",
    "southwest_mountain": "西南山地",
    "southwest_plateau": "西南高原",
    "temperate_oceanic": "温带海洋性",
    "tropical": "热带",
}
ADMIN_LEVEL_LABELS = {
    "municipality": "直辖市",
    "provincial_capital": "省会城市",
    "autonomous_region_capital": "自治区首府",
    "special_administrative_region": "特别行政区",
    "global_city": "国际城市",
    "city": "普通城市",
}
COUNTRY_LABELS = {
    "China": "中国",
    "Australia": "澳大利亚",
    "Canada": "加拿大",
    "Denmark": "丹麦",
    "Finland": "芬兰",
    "France": "法国",
    "Germany": "德国",
    "Ireland": "爱尔兰",
    "Italy": "意大利",
    "Japan": "日本",
    "Kenya": "肯尼亚",
    "Netherlands": "荷兰",
    "New Zealand": "新西兰",
    "Singapore": "新加坡",
    "South Africa": "南非",
    "South Korea": "韩国",
    "Spain": "西班牙",
    "Sweden": "瑞典",
    "Switzerland": "瑞士",
    "Thailand": "泰国",
    "Turkey": "土耳其",
    "United Arab Emirates": "阿联酋",
    "United Kingdom": "英国",
    "United States": "美国",
}
DISPLAY_MODE_OPTIONS = {
    "system": {"zh": "跟随系统", "en": "System"},
    "day": {"zh": "白天", "en": "Day"},
    "night": {"zh": "夜间", "en": "Night"},
}
FORECAST_UNITS = {
    "公制（摄氏度、毫米、米/秒）": "metric",
    "英制（华氏度、英寸、英里/小时）": "imperial",
}
MAP_ATTRIBUTION = "@高德"
MAP_ATTRIBUTION_EN = "Map: Amap"
LANG_ZH = "zh"
LANG_EN = "en"
LANG_OPTIONS = {"中文": LANG_ZH, "英文": LANG_EN}

MODE_LABELS_LOCAL = {
    "Travel": {"zh": "旅行", "en": "Travel"},
    "Living": {"zh": "长期居住", "en": "Living"},
    "Compare": {"zh": "城市对比", "en": "Compare"},
}

STATUS_LABELS = {
    "fallback": {"zh": "备用参考值", "en": "backup reference"},
    "live": {"zh": "刚刚获取", "en": "live"},
    "cache": {"zh": "已保存数据", "en": "cache"},
    "cache/live": {"zh": "已保存/刚刚获取", "en": "cache/live"},
    "partial": {"zh": "信息不完整", "en": "partial"},
    "failed": {"zh": "失败", "en": "failed"},
    "not_requested": {"zh": "未查询", "en": "not queried"},
    "heuristic": {"zh": "简化估算", "en": "simplified estimate"},
    "estimated": {"zh": "估算结果", "en": "estimated"},
    "dataset": {"zh": "多年参考数据", "en": "fixed dataset"},
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
    "预报时间窗较远，时效权重较低": "the forecast dates are farther away",
    "近期空气质量可能影响出行体验": "recent air quality may affect travel comfort",
}

RISK_LABELS_EN = {
    "糟糕天气很多": "many weather and environmental concerns",
    "糟糕天气较多": "several weather and environmental concerns",
    "糟糕天气中等": "some weather and environmental concerns",
    "糟糕天气较少": "few weather and environmental concerns",
    "糟糕天气很少": "very few weather and environmental concerns",
}

SCORE_LABELS_EN = {
    "高度匹配": "high fit",
    "较高匹配": "higher fit",
    "中等匹配": "moderate fit",
    "较低匹配": "lower fit",
    "低匹配": "low fit",
}


def _forecast_providers(lang: str = LANG_ZH) -> dict[str, object]:
    if lang == LANG_EN:
        return {
            "Open-Meteo": get_forecast_summary,
            "MET Norway": get_met_no_forecast_summary,
        }
    return {
        "开放气象": get_forecast_summary,
        "挪威气象局": get_met_no_forecast_summary,
    }


def _forecast_units(lang: str = LANG_ZH) -> dict[str, str]:
    if lang == LANG_EN:
        return {
            "Metric (C, mm, m/s)": "metric",
            "Imperial (F, inch, mph)": "imperial",
        }
    return FORECAST_UNITS


def _top_bar() -> str:
    """Render the title with compact language and theme popover controls."""

    current_lang = st.session_state.get("language_choice", "中文")
    lang_for_labels = LANG_OPTIONS.get(current_lang, LANG_ZH)
    title_col, language_col, theme_col = st.columns([8.4, 1, 1], gap="small", vertical_alignment="center")
    with title_col:
        st.title(_t(lang_for_labels, "WhereFit：找一座更适合你的城市", "WhereFit: Find a City That Fits You Better"))
    with language_col:
        with st.popover(_t(lang_for_labels, "语言", "Language"), use_container_width=True):
            lang_label = st.radio(
                _t(lang_for_labels, "语言", "Language"),
                list(LANG_OPTIONS.keys()),
                key="language_choice",
                label_visibility="collapsed",
            )
    with theme_col:
        with st.popover(_t(lang_for_labels, "主题", "Theme"), use_container_width=True):
            st.radio(
                _t(lang_for_labels, "主题", "Theme"),
                list(DISPLAY_MODE_OPTIONS.keys()),
                index=0,
                format_func=lambda value: DISPLAY_MODE_OPTIONS[value][lang_for_labels],
                key="display_mode",
                label_visibility="collapsed",
            )
    return str(lang_label)


def main() -> None:
    st.set_page_config(page_title="WhereFit AI", page_icon="🌏", layout="wide")

    lang_label = _top_bar()
    lang = LANG_OPTIONS[lang_label]
    _inject_css()

    st.caption(_t(lang, "按你的气候偏好比较不同城市。", "Compare cities around your climate preferences."))

    _inline_preference_inputs(lang)
    pref, data_mode, history_start, history_end, force_refresh, forecast_settings = _sidebar_inputs(lang)
    seed_data = _load_data()
    city_input = _city_selection_inputs(seed_data, lang)
    run_clicked = st.button(_t(lang, "开始比较", "Compare"), type="primary", key="run_comparison")

    if not city_input.strip():
        st.session_state.pop("city_evaluation", None)
        st.info(_t(lang, "请选择待比较城市，或切换到手动输入城市，然后点击“开始比较”。", "Select cities to compare, or switch to manual entry, then click Compare."))
        _page_notes(lang)
        return

    signature = _evaluation_signature(city_input, pref, data_mode, history_start, history_end, force_refresh, forecast_settings)
    cached_evaluation = st.session_state.get("city_evaluation")
    if run_clicked:
        with st.spinner(_t(lang, "正在读取城市数据并计算排名...", "Loading city data and calculating the ranking...")):
            results, missing, messages = _evaluate(
                city_input,
                pref,
                data_mode,
                history_start,
                history_end,
                force_refresh,
                forecast_settings,
            )
        cached_evaluation = {
            "signature": signature,
            "results": results,
            "missing": missing,
            "messages": messages,
        }
        st.session_state["city_evaluation"] = cached_evaluation
    elif cached_evaluation is None or cached_evaluation.get("signature") != signature:
        _page_notes(lang)
        return
    else:
        results = cached_evaluation["results"]
        missing = cached_evaluation["missing"]
        messages = cached_evaluation["messages"]

    if cached_evaluation is not None:
        if missing:
            st.warning(_t(lang, f"这些城市暂未收录：{', '.join(missing)}。请检查名称，或从给定城市中选择已收录城市。", f"These cities are not in the seed list yet: {', '.join(missing)}. Check the names, or select included cities from the provided list."))
        if messages:
            with st.expander(_t(lang, "数据读取记录", "Data Loading Log"), expanded=False):
                for message in messages:
                    st.caption(_message_for_display(message, lang))
        if not results:
            st.error(_t(lang, "没有匹配到可评分城市。请尝试输入北京、上海、广州、成都、昆明、青岛等。", "No scorable city matched. Try Beijing, Shanghai, Guangzhou, Chengdu, Kunming, or Qingdao."))
            return
        _render_results(results, pref, data_mode, lang)

    _page_notes(lang)


def _sync_inline_preference(inline_key: str, sidebar_key: str) -> None:
    """Copy one inline preference value into the matching sidebar state."""

    value = st.session_state[inline_key]
    st.session_state[sidebar_key] = value
    st.session_state[f"{inline_key}_synced"] = value


def _prepare_inline_preference(inline_key: str, sidebar_key: str, default: object) -> None:
    """Keep an inline preference control aligned with its sidebar counterpart."""

    current = st.session_state.get(sidebar_key, default)
    if st.session_state.get(f"{inline_key}_synced") != current:
        st.session_state[inline_key] = current
        st.session_state[f"{inline_key}_synced"] = current


def _inline_preference_inputs(lang: str) -> None:
    """Render a cross-platform preference panel that mirrors core sidebar inputs."""

    with st.expander(_t(lang, "调整偏好", "Adjust Preferences"), expanded=False):
        st.markdown("<div class='wf-inline-preference-marker'></div>", unsafe_allow_html=True)
        st.caption(_t(lang, "设置后，重新点击“开始比较”查看结果。", "Update settings, then select Compare to view results."))
        _prepare_inline_preference("inline_scenario", "scenario", "Compare")
        st.radio(
            _t(lang, "使用场景", "Scenario"),
            ["Travel", "Living", "Compare"],
            horizontal=True,
            format_func=lambda value: _mode_label(value, lang),
            key="inline_scenario",
            on_change=_sync_inline_preference,
            args=("inline_scenario", "scenario"),
        )
        preference_specs = [
            ("heat_sensitivity", "怕热程度", "Heat Sensitivity", 3),
            ("cold_sensitivity", "怕冷程度", "Cold Sensitivity", 2),
            ("humidity_sensitivity", "讨厌潮湿程度", "Humidity Sensitivity", 3),
            ("rain_sensitivity", "讨厌下雨程度", "Rain Sensitivity", 3),
            ("air_quality_sensitivity", "关注空气质量程度", "Air Quality Sensitivity", 3),
            ("extreme_weather_sensitivity", "关注极端天气程度", "Extreme Weather Sensitivity", 3),
        ]
        for sidebar_key, zh_label, en_label, default in preference_specs:
            inline_key = f"inline_{sidebar_key}"
            _prepare_inline_preference(inline_key, sidebar_key, default)
            st.slider(
                _t(lang, zh_label, en_label),
                0,
                5,
                key=inline_key,
                on_change=_sync_inline_preference,
                args=(inline_key, sidebar_key),
            )
        st.caption(_t(lang, "0 = 较少关注 · 5 = 高度关注", "0 = lower concern · 5 = high concern"))


def _sidebar_inputs(lang: str) -> tuple[UserPreference, str, str, str, bool, dict[str, object]]:
    with st.sidebar:
        st.markdown("<div class='wf-sidebar-top-spacer'></div>", unsafe_allow_html=True)
        st.header(_t(lang, "你的气候偏好", "Climate Preferences"))
        st.markdown(f"<div class='wf-sidebar-section'>{escape(_t(lang, '使用场景', 'Scenario'))}</div>", unsafe_allow_html=True)
        mode = st.radio(
            _t(lang, "使用场景", "Scenario"),
            ["Travel", "Living", "Compare"],
            horizontal=False,
            format_func=lambda value: _mode_label(value, lang),
            label_visibility="collapsed",
            key="scenario",
        )

        history_start = HISTORY_START_DATE
        history_end = default_history_end_date()
        force_refresh = False
        forecast_settings: dict[str, object] = {}
        data_mode = DATA_MODE_STATIC
        start_default, end_default = default_forecast_dates()
        if mode == "Travel":
            default_month = date.fromisoformat(start_default).month
            month = st.selectbox(
                _t(lang, "目标月份", "Target Month"),
                list(range(1, 13)),
                index=default_month - 1,
                format_func=lambda m: _month_label(m, lang),
                key="travel_month",
            )
            forecast_window = _forecast_window_for_month(start_default, end_default, month)
            if forecast_window is not None:
                forecast_start, forecast_end = forecast_window
                data_mode = DATA_MODE_FORECAST
                forecast_settings = {
                    "start_date": forecast_start,
                    "end_date": forecast_end,
                    "provider_label": next(iter(_forecast_providers(lang).keys())),
                    "unit_system": "metric",
                    "force_refresh": False,
                }
            else:
                data_mode = DATA_MODE_HISTORY
                st.info(
                    _t(
                        lang,
                        "所选月份不在当前 1-16 天有效天气预报范围内，只显示该月份的历史气候数据。",
                        "The selected month is outside the current 1-16 day forecast window, so only historical climate data is shown for that month.",
                    )
                )
                history_start, history_end, force_refresh = _history_settings(lang, expanded=False)
        elif mode == "Living":
            month = MONTH_ALL
            data_mode = DATA_MODE_HISTORY
            history_start, history_end, force_refresh = _history_settings(lang, expanded=False)
        else:
            month_options = [MONTH_ALL] + list(range(1, 13))
            month = st.selectbox(
                _t(lang, "目标月份（可选）", "Target Month (optional)"),
                month_options,
                index=0,
                format_func=lambda m: _month_label(m, lang),
                key="compare_month",
            )
            basis = st.radio(
                _t(lang, "比较数据基础", "Comparison Data Basis"),
                ["static", "history"],
                horizontal=True,
                key="comparison_basis",
                format_func=lambda value: {
                    "static": _t(lang, "2000—2025 年多年平均", "Bundled Climate Baseline"),
                    "history": _t(lang, "多年历史天气", "Multi-year History"),
                }[value],
            )
            data_mode = DATA_MODE_HISTORY if basis == "history" else DATA_MODE_STATIC
            if data_mode == DATA_MODE_HISTORY:
                history_start, history_end, force_refresh = _history_settings(lang, expanded=False)

        st.markdown(f"<div class='wf-sidebar-section'>{escape(_t(lang, '偏好权重', 'Preference Weights'))}</div>", unsafe_allow_html=True)
        heat = st.slider(_t(lang, "怕热程度", "Heat Sensitivity"), 0, 5, 3, key="heat_sensitivity")
        cold = st.slider(_t(lang, "怕冷程度", "Cold Sensitivity"), 0, 5, 2, key="cold_sensitivity")
        humidity = st.slider(_t(lang, "讨厌潮湿程度", "Humidity Sensitivity"), 0, 5, 3, key="humidity_sensitivity")
        rain = st.slider(_t(lang, "讨厌下雨程度", "Rain Sensitivity"), 0, 5, 3, key="rain_sensitivity")
        air = st.slider(_t(lang, "关注空气质量程度", "Air Quality Sensitivity"), 0, 5, 3, key="air_quality_sensitivity")
        extreme = st.slider(_t(lang, "关注极端天气程度", "Extreme Weather Sensitivity"), 0, 5, 3, key="extreme_weather_sensitivity")
        st.caption(_t(lang, "0 = 较少关注 · 5 = 高度关注", "0 = lower concern · 5 = high concern"))
    pref = UserPreference(
        mode=mode,
        month=month,
        heat_sensitivity=heat,
        cold_sensitivity=cold,
        humidity_sensitivity=humidity,
        rain_sensitivity=rain,
        air_quality_sensitivity=air,
        extreme_weather_sensitivity=extreme,
    )
    return pref, data_mode, history_start, history_end, force_refresh, forecast_settings


def _date_years_before(end_date: str, years: int) -> str:
    end = date.fromisoformat(end_date)
    try:
        return end.replace(year=end.year - years).isoformat()
    except ValueError:
        return (end - timedelta(days=365 * years)).isoformat()


def _forecast_window_for_month(start_date: str, end_date: str, month: int) -> tuple[str, str] | None:
    """Return the part of a forecast window that belongs to one month."""

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    matching_dates: list[date] = []
    cursor = start
    while cursor <= end:
        if cursor.month == int(month):
            matching_dates.append(cursor)
        cursor += timedelta(days=1)
    if not matching_dates:
        return None
    return matching_dates[0].isoformat(), matching_dates[-1].isoformat()


def _localized_date_input(lang: str, zh_label: str, en_label: str, value: date, key: str) -> date:
    if lang == LANG_EN:
        selected = st.date_input(en_label, value=value, key=key)
        return selected if isinstance(selected, date) else value
    raw = st.text_input(zh_label, value=value.isoformat(), key=key, help="格式：年-月-日")
    try:
        return date.fromisoformat(raw.strip())
    except ValueError:
        st.warning(f"{zh_label}格式无效，请使用类似 {value.isoformat()} 的日期。")
        return value


def _history_settings(lang: str, expanded: bool) -> tuple[str, str, bool]:
    history_start = HISTORY_START_DATE
    history_end = default_history_end_date()
    force_refresh = False
    with st.expander(_t(lang, "历史数据设置", "Historical Data Settings"), expanded=expanded):
        history_windows = ["all", "10y", "5y", "custom"]
        history_window = st.selectbox(
            _t(lang, "历史窗口", "History Window"),
            history_windows,
            index=0,
            format_func=lambda value: {
                "all": _t(lang, "2000 至今", "2000 to now"),
                "10y": _t(lang, "近 10 年", "Last 10 years"),
                "5y": _t(lang, "近 5 年", "Last 5 years"),
                "custom": _t(lang, "自定义", "Custom"),
            }[value],
            key="history_window",
        )
        if history_window == "10y":
            history_start = _date_years_before(history_end, 10)
        elif history_window == "5y":
            history_start = _date_years_before(history_end, 5)
        elif history_window == "custom":
            history_start = st.text_input(_t(lang, "历史开始日期", "Start Date"), HISTORY_START_DATE, key="history_start")
            history_end = st.text_input(_t(lang, "历史结束日期", "End Date"), default_history_end_date(), key="history_end")
        history_start, history_end = _validated_history_range(history_start, history_end, lang)
        force_refresh = st.checkbox(_t(lang, "强制重新下载历史天气（忽略缓存）", "Force refresh historical weather"), value=False, key="history_force_refresh")
    return history_start, history_end, force_refresh


def _validated_history_range(start_value: str, end_value: str, lang: str) -> tuple[str, str]:
    """Validate custom history dates and fall back to a safe default range."""

    try:
        start = date.fromisoformat(str(start_value).strip())
        end = date.fromisoformat(str(end_value).strip())
        if end < start:
            raise ValueError("end before start")
        if end >= date.today():
            end = date.fromisoformat(default_history_end_date())
            st.warning(
                _t(
                    lang,
                    f"历史结束日期需早于今天，已调整为 {end.isoformat()}。",
                    f"Historical end date must be before today; it was adjusted to {end.isoformat()}.",
                )
            )
        return start.isoformat(), end.isoformat()
    except ValueError:
        safe_end = default_history_end_date()
        st.warning(
            _t(
                lang,
                f"历史日期无效，已使用默认窗口 {HISTORY_START_DATE} 至 {safe_end}。",
                f"Invalid historical date range; using {HISTORY_START_DATE} to {safe_end}.",
            )
        )
        return HISTORY_START_DATE, safe_end


def _city_selection_inputs(data: pd.DataFrame, lang: str) -> str:
    """Render one unambiguous candidate-selection path and return seed-city names."""

    st.subheader(_t(lang, "待比较城市", "Cities to Compare"))
    selection_mode = st.radio(
        _t(lang, "选择方式", "Selection Method"),
        ["demo", "browse", "manual"],
        horizontal=True,
        format_func=lambda value: {
            "demo": _t(lang, "示例组合", "Demo Set"),
            "browse": _t(lang, "筛选已收录城市", "Browse Included Cities"),
            "manual": _t(lang, "手动输入", "Manual Entry"),
        }[value],
        key="city_selection_mode",
    )
    if selection_mode == "manual":
        default_value = DEFAULT_CITIES if lang == LANG_ZH else "Beijing, Shanghai, Guangzhou, Chengdu, Kunming, Harbin, Qingdao"
        return st.text_input(
            _t(lang, "待比较城市", "Cities to Compare"),
            value=default_value,
            key="manual_city_input",
            help=_t(
                lang,
                "仅匹配当前收录的 77 个城市；支持中文、英文、英文逗号、中文逗号和换行。",
                "Matches only the 77 currently included cities. Chinese/English names, commas, and line breaks are supported.",
            ),
        )
    if selection_mode == "demo":
        shared_demo_key = st.session_state.get("demo_city_set_choice", "east_china")
        widget_key = f"demo_city_set_{lang}"
        if st.session_state.get("demo_city_set_render_lang") != lang:
            st.session_state[widget_key] = shared_demo_key
        st.session_state["demo_city_set_render_lang"] = lang
        demo_key = st.selectbox(
            _t(lang, "示例组合", "Demo Set"),
            list(DEMO_CITY_SETS),
            format_func=lambda value: DEMO_CITY_SETS[value][lang],
            key=widget_key,
        )
        st.session_state["demo_city_set_choice"] = demo_key
        demo = DEMO_CITY_SETS[demo_key]
        return str(demo["cities_en" if lang == LANG_EN else "cities_zh"])

    representative_only = st.checkbox(
        _t(lang, "仅显示代表城市", "Representative Cities Only"),
        value=True,
        key="representative_cities_only",
        help=_t(lang, "包括省会、自治区首府和直辖市。关闭后可浏览全部 77 个城市。", "Includes provincial capitals, autonomous-region capitals, and municipalities. Disable to browse all 77 cities."),
    )
    filter_options = CityFilterOptions(representative_only=representative_only)
    candidate_limit = min(12, max(3, len(data)))
    with st.expander(
        _t(lang, "城市筛选", "City Filter"),
        expanded=True,
    ):
        filter_options, candidate_limit = _city_filter_controls(data, representative_only=representative_only, lang=lang)

    filtered = filter_seed_cities(data, filter_options)
    if filtered.empty:
        st.warning(_t(lang, "当前筛选没有匹配城市。请调整筛选条件，或切换到手动输入。", "No city matches the current filters. Adjust filters or switch to manual input."))
        return ""

    preview = filtered.head(candidate_limit)
    st.markdown(_candidate_selection_html(filtered, preview, lang), unsafe_allow_html=True)
    return _candidate_city_input_for_lang(filtered, candidate_limit, lang)


def _candidate_city_input_for_lang(data: pd.DataFrame, limit: int | None, lang: str) -> str:
    rows = data if limit is None else data.head(limit)
    if lang == LANG_EN and "city_en" in rows.columns:
        return ", ".join(str(city) for city in rows["city_en"].tolist())
    if "city_zh" in rows.columns:
        return ", ".join(str(city) for city in rows["city_zh"].tolist())
    return candidate_city_input(rows, limit=None)


def _filter_category_values(data: pd.DataFrame, column: str, values: list[str]) -> pd.DataFrame:
    """Return rows matching selected category values, or all rows when unselected."""

    chosen = [value for value in values if value]
    if not chosen or column not in data.columns:
        return data
    return data[data[column].isin(chosen)]


def _location_filter_labels(data: pd.DataFrame, lang: str) -> dict[str, str]:
    """Build second-level province or international-city choices for the current countries."""

    labels: dict[str, str] = {}
    for _, row in data.iterrows():
        if str(row.get("country", "")) == "China":
            province = str(row.get("province", ""))
            if province:
                labels[f"province:{province}"] = province
        else:
            city = str(row.get("city", ""))
            if city:
                display = str(row.get("city_en" if lang == LANG_EN else "city_zh", city))
                labels[f"city:{city}"] = display
    return dict(sorted(labels.items(), key=lambda item: item[1]))


def _location_values(values: list[str], kind: str) -> list[str]:
    """Extract stable province or city values from encoded second-level choices."""

    prefix = f"{kind}:"
    return [value.removeprefix(prefix) for value in values if value.startswith(prefix)]


def _filter_location_values(data: pd.DataFrame, values: list[str]) -> pd.DataFrame:
    """Return rows matching selected province and international-city choices."""

    provinces = _location_values(values, "province")
    cities = _location_values(values, "city")
    if not provinces and not cities:
        return data
    matches = pd.Series(False, index=data.index)
    if provinces and "province" in data.columns:
        matches |= data["province"].isin(provinces)
    if cities and "city" in data.columns:
        matches |= data["city"].isin(cities)
    return data[matches]


def _drop_invalid_multiselect_values(key: str, valid_values: object) -> None:
    """Clear dependent Streamlit selections that no longer exist in valid choices."""

    valid = set(valid_values)
    current = list(st.session_state.get(key, []))
    cleaned = [value for value in current if value in valid]
    if cleaned != current:
        st.session_state[key] = cleaned


def _city_filter_controls(data: pd.DataFrame, representative_only: bool, lang: str) -> tuple[CityFilterOptions, int]:
    col_a, col_b, col_c = st.columns(3)
    country_options = sorted(str(value) for value in data.get("country", pd.Series(dtype=str)).dropna().unique())
    selected_countries = col_a.multiselect(
        _t(lang, "国家/地区", "Country or Region"),
        country_options,
        default=[],
        key="filter_countries",
        format_func=lambda value: _country_label(value, lang),
    )
    country_filtered = _filter_category_values(data, "country", selected_countries)
    location_labels = _location_filter_labels(country_filtered, lang)
    _drop_invalid_multiselect_values("filter_locations", location_labels)
    selected_locations = col_b.multiselect(
        _t(lang, "省份/地区 / 城市", "Province/Region or City"),
        list(location_labels),
        default=[],
        key="filter_locations",
        format_func=lambda value: location_labels[value],
    )
    location_filtered = _filter_location_values(country_filtered, selected_locations)
    region_options = sorted(str(value) for value in location_filtered.get("region_type", pd.Series(dtype=str)).dropna().unique())
    _drop_invalid_multiselect_values("filter_regions", set(region_options))
    selected_regions = col_c.multiselect(
        _t(lang, "地理气候分区（可选）", "Geo-climate Zone (optional)"),
        region_options,
        default=[],
        key="filter_regions",
        format_func=lambda value: _region_type_label(value, lang),
        help=_t(lang, "这是基础城市表里的粗粒度地理/气候标签，不是行政区，适合和省份/地区搭配缩小候选范围。", "This is a coarse geo-climate label from the seed table, such as south_coast, northwest, or plateau. It is not an administrative region."),
    )
    has_international_city = bool(_location_values(selected_locations, "city"))
    effective_representative_only = representative_only and not (selected_countries or has_international_city)

    hazard_cols = st.columns([1, 1, 1])
    exclude_typhoon = hazard_cols[0].checkbox(
        _t(lang, "排除台风区域", "Exclude typhoon zone"),
        value=False,
        key="filter_exclude_typhoon",
    )
    exclude_coastal = hazard_cols[1].checkbox(
        _t(lang, "排除沿海城市", "Exclude coastal cities"),
        value=False,
        key="filter_exclude_coastal",
    )
    default_limit = min(12, max(3, len(data))) if representative_only else max(3, len(data))
    candidate_limit = hazard_cols[2].slider(
        _t(lang, "最多候选数", "Max Candidates"),
        3,
        max(3, len(data)),
        default_limit,
        key="filter_candidate_limit",
    )
    return (
        CityFilterOptions(
            representative_only=effective_representative_only,
            countries=tuple(selected_countries),
            provinces=tuple(_location_values(selected_locations, "province")),
            cities=tuple(_location_values(selected_locations, "city")),
            regions=tuple(selected_regions),
            exclude_typhoon_region=exclude_typhoon,
            exclude_coastal=exclude_coastal,
        ),
        candidate_limit,
    )


def _candidate_preview_table(data: pd.DataFrame, lang: str) -> pd.DataFrame:
    columns = {
        ("city_en" if lang == LANG_EN else "city_zh"): _t(lang, "城市", "City"),
        "country": _t(lang, "国家", "Country"),
        "admin_level": _t(lang, "城市层级", "Admin Level"),
        "region_type": _t(lang, "地理气候分区", "Geo-climate Zone"),
        "coastal": _t(lang, "沿海", "Coastal"),
        "typhoon_region": _t(lang, "台风区域", "Typhoon Zone"),
    }
    if lang == LANG_ZH:
        columns["province"] = "省份/地区"
    available = [column for column in columns if column in data.columns]
    table = data[available].rename(columns=columns)
    bool_columns = [column for column in [_t(lang, "沿海", "Coastal"), _t(lang, "台风区域", "Typhoon Zone")] if column in table.columns]
    for column in bool_columns:
        table[column] = table[column].map(lambda value: _yes_no(value, lang))
    country_col = _t(lang, "国家", "Country")
    if country_col in table.columns:
        table[country_col] = table[country_col].map(lambda value: _country_label(value, lang))
    admin_col = _t(lang, "城市层级", "Admin Level")
    if admin_col in table.columns:
        table[admin_col] = table[admin_col].map(lambda value: _admin_level_label(value, lang))
    zone_col = _t(lang, "地理气候分区", "Geo-climate Zone")
    if zone_col in table.columns:
        table[zone_col] = table[zone_col].map(lambda value: _region_type_label(value, lang))
    return table


def _candidate_selection_html(filtered: pd.DataFrame, preview: pd.DataFrame, lang: str) -> str:
    table = _candidate_preview_table(preview, lang)
    rows = []
    for _, row in table.iterrows():
        cells = "".join(f"<td>{escape(str(value))}</td>" for value in row.tolist())
        rows.append(f"<tr>{cells}</tr>")
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in table.columns)
    overflow = len(filtered) - len(preview)
    overflow_note = ""
    if overflow > 0:
        overflow_text = _t(lang, f"另有 {overflow} 个待比较城市未在预览中显示。", f"{overflow} additional cities are not shown in the preview.")
        overflow_note = f'<div class="wf-candidate-note">{escape(overflow_text)}</div>'
    return f"""
    <div class="wf-candidate-panel">
      <div class="wf-candidate-head">
        <div>
          <div class="wf-eyebrow">{escape(_t(lang, "比较范围", "Comparison Set"))}</div>
          <div class="wf-candidate-title">{escape(_t(lang, f"已选择 {len(filtered)} 个待比较城市", f"{len(filtered)} cities selected"))}</div>
        </div>
        <div class="wf-candidate-count">{escape(_t(lang, f"本次比较 {len(preview)} 个", f"Comparing {len(preview)}"))}</div>
      </div>
      {overflow_note}
      <div class="wf-table-wrap">
        <table class="wf-table">
          <thead><tr>{headers}</tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </div>
    """


def _region_type_label(value: object, lang: str = LANG_ZH) -> str:
    raw = str(value)
    if lang == LANG_EN:
        return raw.replace("_", " ").title()
    return REGION_TYPE_LABELS.get(raw, raw)


def _country_label(value: object, lang: str = LANG_ZH) -> str:
    raw = str(value)
    if lang == LANG_EN:
        return raw
    return COUNTRY_LABELS.get(raw, raw)


def _admin_level_label(value: object, lang: str = LANG_ZH) -> str:
    raw = str(value)
    if lang == LANG_EN:
        return raw.replace("_", " ").title()
    return ADMIN_LEVEL_LABELS.get(raw, raw)


def _evaluation_signature(
    city_input: str,
    pref: UserPreference,
    data_mode: str,
    history_start: str,
    history_end: str | None,
    force_refresh: bool,
    forecast_settings: dict[str, object] | None = None,
) -> tuple[object, ...]:
    """Build a language-independent signature for result-cache invalidation."""

    forecast_settings = forecast_settings or {}
    normalized_cities = tuple(city.lower() for city in parse_city_input(city_input))
    normalized_forecast_settings = dict(forecast_settings)
    provider_label = str(normalized_forecast_settings.get("provider_label") or "")
    if provider_label:
        normalized_forecast_settings["provider_label"] = (
            "met_no" if provider_label in {"挪威气象局", "MET Norway"} else "open_meteo"
        )
    return (
        normalized_cities,
        data_mode,
        history_start,
        history_end,
        force_refresh,
        tuple(sorted(normalized_forecast_settings.items())),
        pref.mode,
        pref.month,
        pref.heat_sensitivity,
        pref.cold_sensitivity,
        pref.humidity_sensitivity,
        pref.rain_sensitivity,
        pref.air_quality_sensitivity,
        pref.extreme_weather_sensitivity,
    )


@st.cache_data
def _load_data() -> pd.DataFrame:
    return attach_long_term_air_quality(load_seed_cities(DATA_PATH), LONG_TERM_PM25_PATH)


@st.cache_data
def _load_climate_data() -> pd.DataFrame:
    """Load the bundled NASA POWER monthly and annual aggregates once per process."""

    return load_climate_baseline(CLIMATE_MONTHLY_PATH, CLIMATE_ANNUAL_PATH)


def _evaluate(
    city_input: str,
    pref: UserPreference,
    data_mode: str = DATA_MODE_STATIC,
    history_start: str = HISTORY_START_DATE,
    history_end: str | None = None,
    force_refresh: bool = False,
    forecast_settings: dict[str, object] | None = None,
) -> tuple[list[CityResult], list[str], list[str]]:
    """Evaluate matched cities, parallelizing independent network-backed work."""

    data = _load_data()
    climate_baseline = _load_climate_data()
    requested = parse_city_input(city_input)
    matched, missing = match_cities(data, requested)
    rows = [row for _, row in matched.iterrows()]
    worker = partial(
        _evaluate_city_row,
        pref=pref,
        data_mode=data_mode,
        history_start=history_start,
        history_end=history_end,
        force_refresh=force_refresh,
        forecast_settings=forecast_settings or {},
        climate_baseline=climate_baseline,
    )
    if data_mode in {DATA_MODE_HISTORY, DATA_MODE_FORECAST} and len(rows) > 1:
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_CITY_REQUESTS, len(rows))) as executor:
            evaluated = list(executor.map(worker, rows))
    else:
        evaluated = [worker(row) for row in rows]
    results = [result for result, _ in evaluated]
    messages = [message for _, city_messages in evaluated for message in city_messages]
    return rank_cities(results), missing, messages


def _evaluate_city_row(
    row: pd.Series,
    pref: UserPreference,
    data_mode: str,
    history_start: str,
    history_end: str | None,
    force_refresh: bool,
    forecast_settings: dict[str, object],
    climate_baseline: pd.DataFrame,
) -> tuple[CityResult, list[str]]:
    """Evaluate one city without making Streamlit calls from worker threads."""

    location = row_to_location(row)
    fallback_metrics = row_to_metrics(row, pref.month, climate_baseline)
    metrics = fallback_metrics
    messages: list[str] = []
    if data_mode == DATA_MODE_HISTORY:
        history = get_history_metrics(
            location=location,
            month=pref.month,
            cache_dir=HISTORY_CACHE_DIR,
            start_date=history_start,
            end_date=history_end,
            force_refresh=force_refresh,
            fallback_pm25=fallback_metrics.pm25,
        )
        messages.append(f"{location.city}: {history.message}")
        if history.metrics is not None:
            metrics = with_long_term_air_quality(
                replace(history.metrics, data_status=history.status),
                fallback_metrics,
            )
    result = evaluate_city(location, metrics, pref)
    if data_mode == DATA_MODE_FORECAST:
        air_quality = get_air_quality_summary(
            location,
            cache_dir=AIR_QUALITY_CACHE_DIR,
            force_refresh=False,
        )
        messages.append(f"{location.city}: {air_quality.message}")
        result = _apply_forecast_mode(result, pref, forecast_settings, air_quality)
        if result.forecast is not None:
            messages.append(f"{location.city}: {result.forecast.message}")
    return replace(result, data_quality=build_data_quality_records(result)), messages


def _apply_forecast_mode(
    result: CityResult,
    pref: UserPreference,
    settings: dict[str, object],
    air_quality: AirQualitySummary,
) -> CityResult:
    """Fetch the selected forecast and apply short-term travel scoring."""

    start_default, end_default = default_forecast_dates()
    provider_label = str(settings.get("provider_label") or "开放气象")
    provider_map = {**_forecast_providers(LANG_ZH), **_forecast_providers(LANG_EN)}
    provider = provider_map.get(provider_label, get_forecast_summary)
    summary = provider(
        result.location,
        cache_dir=FORECAST_CACHE_DIR,
        start_date=str(settings.get("start_date") or start_default),
        end_date=str(settings.get("end_date") or end_default),
        unit_system=str(settings.get("unit_system") or "metric"),
        force_refresh=bool(settings.get("force_refresh", False)),
    )
    return apply_forecast_score(result, summary, pref, air_quality)


def _render_results(results: list[CityResult], pref: UserPreference, data_mode: str, lang: str) -> None:
    best = results[0]
    _render_decision_panel(results, best, pref, data_mode, lang)
    tab_specs = _tab_specs(pref, data_mode, lang)
    tabs = st.tabs([label for label, _ in tab_specs])
    for tab, (_, kind) in zip(tabs, tab_specs):
        with tab:
            if kind == "overview":
                _render_overview(results, pref, data_mode, lang)
            elif kind == "history":
                _render_history_tab(results, pref, data_mode, lang)
            elif kind == "forecast":
                _render_forecast_tab(results, lang)
            elif kind == "hazard":
                _render_hazard_tab(results, lang)
            elif kind == "aurora":
                _render_aurora_tab(results, lang)
            elif kind == "quality":
                _render_data_quality_tab(results, lang)
            elif kind == "report":
                _render_report_tab(results, pref, best, lang)


def _tab_specs(pref: UserPreference, data_mode: str, lang: str) -> list[tuple[str, str]]:
    specs = [
        (_t(lang, "城市适配", "City Fit"), "overview"),
        (_t(lang, "历史天气", "Historical Weather"), "history"),
        (_t(lang, "未来预报", "Forecast"), "forecast"),
        (_t(lang, "历史灾害", "Historical Hazards"), "hazard"),
        (_t(lang, "极光机会", "Aurora"), "aurora"),
        (_t(lang, "数据来源", "Data Sources"), "quality"),
        (_t(lang, "解释报告", "Report"), "report"),
    ]
    if pref.mode == "Travel" and data_mode == DATA_MODE_FORECAST:
        order = ["forecast", "overview", "quality", "history", "hazard", "aurora", "report"]
    elif pref.mode in {"Travel", "Living"} and data_mode == DATA_MODE_HISTORY:
        order = ["history", "overview", "quality", "forecast", "hazard", "aurora", "report"]
    else:
        order = ["overview", "quality", "history", "forecast", "hazard", "aurora", "report"]
    by_kind = {kind: (label, kind) for label, kind in specs}
    return [by_kind[kind] for kind in order]


def _render_decision_panel(results: list[CityResult], best: CityResult, pref: UserPreference, data_mode: str, lang: str) -> None:
    forecast_count = sum(1 for item in results if item.forecast is not None and item.forecast.status != "failed")
    status_text = _status_label(best.score.data_status, lang)
    top_reasons = best.score.strengths[:3] or [_t(lang, "综合表现相对均衡", "Balanced overall profile")]
    warning_text = _join_display(best.score.warnings[:2], lang) if best.score.warnings else _t(lang, "暂无突出提示", "No major watch-out")
    chips = "".join(_chip_html(_display_phrase(reason, lang), "good") for reason in top_reasons)
    html = f"""
    <div class="wf-decision">
      <div class="wf-decision-main">
        <div class="wf-eyebrow">{escape(_t(lang, "本次最符合你偏好的城市", "Best Match for Your Preferences"))}</div>
        <div class="wf-city-title">{escape(_city_name(best, lang))}</div>
        <div class="wf-city-subtitle">{escape(generate_comparison_report(results, pref, lang))}</div>
        <div class="wf-chip-row">{chips}</div>
      </div>
      <div class="wf-decision-grid">
        {_stat_tile_html(_t(lang, "你的偏好匹配分", "Your Preference Match"), f"{best.score.personal_fit_score:.0f}/100", _score_label(score_label(best.score.personal_fit_score)[0], lang), "teal")}
        {_stat_tile_html(_t(lang, "场景", "Scenario"), _mode_label(pref.mode, lang), _time_scope_label(pref, data_mode, lang), "blue")}
        {_stat_tile_html(_t(lang, "主要数据", "Main Data"), status_text, _t(lang, "本次排序所用资料", "data used for this ranking"), "slate")}
        {_stat_tile_html(_t(lang, "未来天气", "Forecast"), f"{forecast_count}/{len(results)}", _t(lang, "城市有可用预报", "cities have a forecast"), "amber")}
      </div>
      <div class="wf-watchout"><span>{escape(_t(lang, "注意", "Watch-out"))}</span>{escape(warning_text)}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def _stat_tile_html(label: str, value: str, detail: str, tone: str) -> str:
    return f"""
    <div class="wf-stat wf-stat-{tone}">
      <div class="wf-stat-label">{escape(str(label))}</div>
      <div class="wf-stat-value">{escape(str(value))}</div>
      <div class="wf-stat-detail">{escape(str(detail))}</div>
    </div>
    """


def _summary_callout_html(text: str, lang: str) -> str:
    return f"""
    <div class="wf-callout">
      <span>{escape(_t(lang, "结论", "Summary"))}</span>
      <p>{escape(text)}</p>
    </div>
    """


def _render_data_quality_tab(results: list[CityResult], lang: str) -> None:
    """Render provenance, time scale, fallback use, and scoring impact."""

    st.subheader(_t(lang, "数据来源和使用说明", "Data Sources and Usage Notes"))
    st.info(
        _t(
            lang,
            "这里说明每项数据从哪里来、对应哪个时间范围，以及是否参与评分；这不是统计准确率。历史灾害记录在独立页面按需查询，不会改变主排名。",
            "This page shows each source, its time range, and whether it affects the score. It is not a measure of statistical accuracy. On-demand hazard records never alter the main ranking.",
        )
    )
    rows: list[dict[str, object]] = []
    for item in results:
        for record in item.data_quality:
            rows.append(
                {
                    _t(lang, "城市", "City"): _city_name(item, lang),
                    _t(lang, "数据项目", "Data Item"): _quality_category_label(record.category, lang),
                    _t(lang, "数据源", "Source"): _source_label(record.source, lang),
                    _t(lang, "状态", "Status"): _status_label(record.status, lang),
                    _t(lang, "时间范围", "Time Scope"): _quality_text(record.time_scope, lang),
                    _t(lang, "数据覆盖时间", "Time Covered"): _quality_text(record.sample_size, lang),
                    _t(lang, "用于本次排序", "Used in This Ranking"): _yes_no(record.affects_score, lang),
                    _t(lang, "是否为估算/备用值", "Estimate or Backup"): _yes_no(record.fallback_used, lang),
                    _t(lang, "说明", "Note"): _quality_text(record.note, lang),
                }
            )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _render_report_tab(results: list[CityResult], pref: UserPreference, best: CityResult, lang: str) -> None:
    st.subheader(_t(lang, "解释报告", "Explanation Report"))
    comparison_report = generate_comparison_report(results, pref, lang)
    report_text = "\n\n".join(
        [comparison_report]
        + [
            f"{_city_name(item, lang)} {_t(lang, '解释', 'explanation')}\n{generate_city_report(item, pref, lang)}\n{_t(lang, '提示', 'Notes')}: {_join_display(item.score.warnings, lang)}"
            for item in results
        ]
    )
    st.markdown(_summary_callout_html(comparison_report, lang), unsafe_allow_html=True)
    report_cols = st.columns([1, 1, 1])
    report_cols[0].metric(_t(lang, "首选城市", "Top City"), _city_name(best, lang))
    report_cols[1].metric(_t(lang, "你的偏好匹配分", "Your Preference Match"), f"{best.score.personal_fit_score:.0f}/100")
    report_cols[2].metric(_t(lang, "场景", "Scenario"), _mode_label(pref.mode, lang))
    st.download_button(_t(lang, "下载文本报告", "Download TXT Report"), report_text, file_name="wherefit_report.txt", mime="text/plain")
    for item in results:
        with st.expander(f"{_city_name(item, lang)} {_t(lang, '解释', 'explanation')}", expanded=item == best):
            st.write(generate_city_report(item, pref, lang))
    st.text_area(_t(lang, "可复制报告文本", "Copyable Report Text"), report_text, height=260)


def _render_overview(results: list[CityResult], pref: UserPreference, data_mode: str, lang: str) -> None:
    best = results[0]
    st.subheader(_t(lang, "比较总结", "Comparison Summary"))
    st.markdown(_summary_callout_html(generate_comparison_report(results, pref, lang), lang), unsafe_allow_html=True)

    st.subheader(_t(lang, "城市排名", "City Ranking"))
    ranking = _ranking_table(results, lang)
    st.dataframe(
        ranking,
        hide_index=True,
        width="stretch",
        column_config=_ranking_column_config(lang),
    )
    st.caption(
        _t(
            lang,
            "糟糕天气指数综合高温、强降雨、长期空气质量，以及沿海/台风等提示；数值越低，和气候舒适分一起越有利于提高你的偏好匹配分。",
            "The Bad Weather Index combines heat, heavy rain, long-term air quality, and broad coastal/typhoon cues. A lower value, together with higher climate comfort, supports a higher preference match.",
        )
    )

    chart_col, radar_col = st.columns([1.25, 1])
    with chart_col:
        st.plotly_chart(make_ranking_bar_chart(results, lang), width="stretch")
    with radar_col:
        selected_key = _localized_city_selectbox(
            _t(lang, "查看城市画像", "City Profile"),
            results,
            lang,
            "profile_city",
        )
        selected = next(item for item in results if (item.location.city_en or item.location.city) == selected_key)
        st.plotly_chart(make_radar_chart(selected, lang), width="stretch")

    st.subheader(_t(lang, "地图", "Map"))
    st.pydeck_chart(make_map(results), width="stretch")
    st.caption(_map_attribution(lang))

    st.subheader(_t(lang, "城市卡片", "City Cards"))
    for index, item in enumerate(results, start=1):
        _city_card(index, item, pref, lang)


def _render_history_tab(results: list[CityResult], pref: UserPreference, data_mode: str, lang: str) -> None:
    st.subheader(_t(lang, "历史天气数据概览", "Historical Weather Overview"))
    display_results = results
    if data_mode == DATA_MODE_FORECAST:
        cache_key = _history_tab_cache_key(results, pref)
        loaded = st.session_state.get(cache_key)
        st.caption(_t(lang, "此处用于查看历史明细，不会改写上方排名。", "This view shows historical detail and does not change the ranking above."))
        if st.button(_t(lang, "加载历史天气数据", "Load Historical Weather"), key=f"{cache_key}_button"):
            loaded = _load_history_results_for_tab(results, pref, lang)
            st.session_state[cache_key] = loaded
        if not loaded:
            return
        display_results = loaded
    if pref.month == MONTH_ALL:
        st.caption(_t(lang, "全年多年平均", "Full-year multi-year average"))
    else:
        st.caption(_t(lang, "所选月份的多年平均", "Multi-year average for the selected month"))
    _render_history_table(display_results, lang)
    if data_mode == DATA_MODE_STATIC:
        st.markdown(
            _t(
                lang,
                "数据来源：[NASA POWER](https://power.larc.nasa.gov/)（项目将 MERRA-2/POWER 的资料整理为月度和年度平均；体感温度为计算值，可能下雪日为推算值）。",
                "Source: [NASA POWER](https://power.larc.nasa.gov/) (MERRA-2/POWER data summarized into monthly and annual averages; apparent temperature is calculated and snow days are estimated).",
            )
        )
        return
    if lang == LANG_EN:
        st.markdown("Sources: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), [Open-Meteo Weather Variables](https://open-meteo.com/en/docs), [Open-Meteo Licence](https://open-meteo.com/en/licence).")
    else:
        st.markdown(
            "数据来源：[开放气象历史天气文档](https://open-meteo.com/en/docs/historical-weather-api)；"
            "变量参考：[开放气象变量说明](https://open-meteo.com/en/docs)；"
            "许可说明：[开放气象许可说明](https://open-meteo.com/en/licence)。"
        )


def _history_tab_cache_key(results: list[CityResult], pref: UserPreference) -> str:
    cities = ",".join(item.location.city for item in results)
    return f"history_tab_results:{cities}:{pref.month}"


def _load_history_results_for_tab(results: list[CityResult], pref: UserPreference, lang: str) -> list[CityResult]:
    if not results:
        st.warning(_t(lang, "没有可读取历史天气的城市。", "No cities are available for historical weather loading."))
        return []
    loaded: list[CityResult] = []
    with st.spinner(_t(lang, "正在读取或下载历史天气数据...", "Reading or downloading historical weather data...")):
        worker = partial(_load_history_result_for_city, pref=pref)
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_CITY_REQUESTS, len(results))) as executor:
            history_results = list(executor.map(worker, results))
        for item, evaluated, message in history_results:
            st.caption(f"{_city_name(item, lang)}: {_message_for_display(message, lang)}")
            if evaluated is not None:
                loaded.append(evaluated)
    if not loaded:
        st.warning(_t(lang, "历史天气数据暂时不可用，请稍后重试。", "Historical weather data is temporarily unavailable. Try again later."))
    return rank_cities(loaded)


def _load_history_result_for_city(
    item: CityResult,
    pref: UserPreference,
) -> tuple[CityResult, CityResult | None, str]:
    """Load and score one city's history for the history tab."""

    history = get_history_metrics(
        location=item.location,
        month=pref.month,
        cache_dir=HISTORY_CACHE_DIR,
        start_date=HISTORY_START_DATE,
        end_date=default_history_end_date(),
        force_refresh=False,
        fallback_pm25=item.metrics.pm25,
    )
    if history.metrics is None:
        return item, None, history.message
    metrics = with_long_term_air_quality(
        replace(history.metrics, data_status=history.status),
        item.metrics,
    )
    evaluated = evaluate_city(item.location, metrics, pref)
    return item, replace(evaluated, data_quality=build_data_quality_records(evaluated)), history.message


def _render_history_table(results: list[CityResult], lang: str) -> None:
    rows = []
    for item in results:
        metrics = item.metrics
        rows.append(
            {
                _t(lang, "城市", "City"): _city_name(item, lang),
                _t(lang, "数据源", "Source"): _source_label(metrics.data_source, lang),
                _t(lang, "状态", "Status"): _status_label(metrics.data_status, lang),
                _t(lang, "样本年数", "Sample Years"): metrics.sample_years,
                _t(lang, "平均气温（摄氏度）", "Mean Temp (C)"): round(metrics.temperature_mean, 1),
                _t(lang, "体感温度（摄氏度）", "Apparent Temp (C)"): round(metrics.apparent_temperature, 1),
                _t(lang, "降水日", "Rain Days"): round(metrics.precipitation_days, 1),
                _t(lang, "强降水日", "Heavy-rain Days"): round(metrics.heavy_rain_days, 1),
                _t(lang, "极端降水日", "Extreme-rain Days"): round(metrics.precipitation_extreme_days, 1),
                _t(lang, "高温日", "Hot Days"): round(metrics.hot_days, 1),
                _t(lang, "长期 PM2.5（微克/立方米）", "Long-term PM2.5 (ug/m3)"): round(metrics.pm25, 1),
                _t(lang, "PM2.5 年趋势（微克/立方米/年）", "PM2.5 Trend (ug/m3/year)"): (
                    round(metrics.pm25_trend_per_year, 2) if metrics.pm25_trend_per_year is not None else None
                ),
                _t(lang, "缺失率", "Missing Rate"): round(metrics.missing_rate, 3),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _render_forecast_tab(results: list[CityResult], lang: str) -> None:
    st.subheader(_t(lang, "未来天气预报", "Weather Forecast"))
    summaries = [item.forecast for item in results if item.forecast is not None]
    city_keys = tuple(item.location.city_en or item.location.city for item in results)
    manual_state = st.session_state.get("manual_forecast_result")
    if isinstance(manual_state, dict) and manual_state.get("cities") == city_keys:
        summaries = list(manual_state.get("summaries") or summaries)
    start_default, end_default = default_forecast_dates()
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        start = _localized_date_input(lang, "出行开始日期", "Trip Start Date", date.fromisoformat(start_default), "forecast_start")
    with col_b:
        end = _localized_date_input(lang, "出行结束日期", "Trip End Date", date.fromisoformat(end_default), "forecast_end")
    force_refresh = col_c.checkbox(_t(lang, "刷新预报缓存", "Refresh forecast cache"), value=False, key="forecast_force_refresh")
    col_d, col_e = st.columns([1.4, 1])
    providers = _forecast_providers(lang)
    unit_options = _forecast_units(lang)
    provider_label = col_d.selectbox(
        _t(lang, "预报数据源", "Forecast Provider"),
        list(providers.keys()),
        key=f"forecast_provider_{lang}",
    )
    unit_label = col_e.selectbox(
        _t(lang, "预报单位", "Units"),
        list(unit_options.keys()),
        key=f"forecast_units_{lang}",
    )
    st.caption(_t(lang, "可查看未来短期天气趋势。日期越远，不确定性越高。", "Use this for near-term weather trends. Uncertainty increases for later dates."))
    if st.button(_t(lang, "查询未来天气预报", "Query Forecast"), key="query_forecast"):
        provider = providers[provider_label]
        unit_system = unit_options[unit_label]
        summaries = _query_forecast_summaries(
            results,
            provider,
            start.isoformat(),
            end.isoformat(),
            unit_system,
            force_refresh,
        )
        manual_state = {
            "cities": city_keys,
            "summaries": summaries,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "provider": (summaries[0].provider or summaries[0].source) if summaries else provider_label,
        }
        st.session_state["manual_forecast_result"] = manual_state
    if summaries:
        if isinstance(manual_state, dict) and manual_state.get("cities") == city_keys:
            st.caption(
                _t(
                    lang,
                    f"当前下表保留最近一次手动查询：{manual_state.get('start_date')} 至 {manual_state.get('end_date')}，数据源 {_source_label(str(manual_state.get('provider') or ''), lang)}。",
                    f"The table keeps the latest manual query: {manual_state.get('start_date')} to {manual_state.get('end_date')}, provider {_source_label(str(manual_state.get('provider') or ''), lang)}.",
                )
            )
        st.caption(
            _t(
                lang,
                "页面首次比较时自动加载的有效预报会参与旅行主排名；在本页手动更改日期或数据源仅更新下表，不会静默改写上方主排名，若要按新窗口排名请重新开始比较。",
                "The valid forecast loaded during the initial comparison affects the Travel ranking. Manual date/provider queries update only this table and never silently rewrite the main ranking; rerun Compare to rank a new primary window.",
            )
        )
        _render_forecast_summaries(summaries, lang)
        _render_recent_air_quality(results, lang)
    else:
        st.info(_t(lang, "点击“查询未来天气预报”后，会为当前待比较城市请求或读取未来天气缓存。", "Click Query Forecast to request or read forecast cache for the selected cities."))


def _query_forecast_summaries(
    results: list[CityResult],
    provider: Callable[..., ForecastSummary],
    start_date: str,
    end_date: str,
    unit_system: str,
    force_refresh: bool,
) -> list[ForecastSummary]:
    """Query independent city forecasts concurrently without Streamlit calls."""

    def query(item: CityResult) -> ForecastSummary:
        """Query one city with the shared manual settings."""

        return provider(
            item.location,
            cache_dir=FORECAST_CACHE_DIR,
            start_date=start_date,
            end_date=end_date,
            unit_system=unit_system,
            force_refresh=force_refresh,
        )

    if len(results) <= 1:
        return [query(item) for item in results]
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_CITY_REQUESTS, len(results))) as executor:
        return list(executor.map(query, results))


def _render_forecast_summaries(summaries: list[ForecastSummary], lang: str) -> None:
    table = pd.DataFrame(
        {
            _t(lang, "城市", "City"): [item.city if lang == LANG_ZH else _city_en_from_summary(item.city) for item in summaries],
            _t(lang, "数据源", "Source"): [_source_label(item.provider or item.source, lang) for item in summaries],
            _t(lang, "状态", "Status"): [_status_label(item.status, lang) for item in summaries],
            _t(lang, "天数", "Days"): [item.days for item in summaries],
            _t(lang, "最高温均值", "Mean High Temp"): [f"{round(item.temp_max_mean, 1)} {item.temperature_unit}" for item in summaries],
            _t(lang, "最低温均值", "Mean Low Temp"): [f"{round(item.temp_min_mean, 1)} {item.temperature_unit}" for item in summaries],
            _t(lang, "最高体感均值", "Mean Apparent High"): [f"{round(item.apparent_temp_max_mean, 1)} {item.temperature_unit}" for item in summaries],
            _t(lang, "降水天数", "Rain Days"): [_days_label(item.precipitation_days, lang) for item in summaries],
            _t(lang, "最大降水概率", "Max Rain Probability"): [round(item.precipitation_probability_max, 1) for item in summaries],
            _t(lang, "强降水天数", "Heavy-rain Days"): [_days_label(item.heavy_rain_days, lang) for item in summaries],
            _t(lang, "大风天数", "Windy Days"): [_days_label(item.windy_days, lang) for item in summaries],
            _t(lang, "预报时效权重", "Forecast Horizon Weight"): [round(item.confidence, 2) for item in summaries],
            _t(lang, "说明", "Notes"): [_message_for_display(item.message, lang) for item in summaries],
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")
    st.markdown(_t(lang, "预报来源：[开放气象预报文档](https://open-meteo.com/en/docs)、[挪威气象局预报文档](https://api.met.no/weatherapi/locationforecast/2.0/documentation)。", "Forecast sources: [Open-Meteo Forecast API](https://open-meteo.com/en/docs), [MET Norway Locationforecast](https://api.met.no/weatherapi/locationforecast/2.0/documentation)."))


def _render_recent_air_quality(results: list[CityResult], lang: str) -> None:
    """Show recent air quality separately from long-term climate baselines."""

    summaries = [item.air_quality for item in results if item.air_quality is not None]
    if not summaries:
        return
    st.markdown(f"**{_t(lang, '近期空气质量（仅旅行评分）', 'Recent Air Quality (Travel scoring only)')}**")
    table = pd.DataFrame(
        {
            _t(lang, "城市", "City"): [item.city if lang == LANG_ZH else _city_en_from_summary(item.city) for item in summaries],
            "PM2.5": [item.pm25_mean for item in summaries],
            "US AQI": [item.us_aqi_mean for item in summaries],
            _t(lang, "时间窗口", "Time Window"): [
                f"{item.period_start or '--'} – {item.period_end or '--'}" for item in summaries
            ],
            _t(lang, "样本小时", "Sample Hours"): [item.sample_hours for item in summaries],
            _t(lang, "状态", "Status"): [_status_label(item.status, lang) for item in summaries],
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        _t(
            lang,
            "该近期窗口只用于短期旅行体验，不作为长期居住空气质量常态。",
            "This recent window is used only for short-term travel comfort and is not treated as a living-climate normal.",
        )
    )


def _render_hazard_tab(results: list[CityResult], lang: str) -> None:
    st.subheader(_t(lang, "历史灾害记录", "Historical Hazard Records"))
    st.warning(_hazard_disclaimer(lang))
    selected_key = _localized_city_selectbox(
        _t(lang, "选择城市", "Select City"),
        results,
        lang,
        "hazard_city",
    )
    selected = next(item for item in results if (item.location.city_en or item.location.city) == selected_key)
    col_a, col_b, col_c = st.columns(3)
    query_earthquake = col_a.checkbox(_t(lang, "查询地震历史", "Query USGS Earthquakes"), value=False, key="hazard_query_earthquake", help=_t(lang, "默认优先使用已保存的数据；没有时查询 2000 年至今城市周边 500km 内 4 级及以上地震，并分别统计 100/200/500km 内的次数。", "Reads saved data first; otherwise queries M4+ earthquakes since 2000 within 500 km, with 100/200/500 km counts."))
    query_typhoon = col_b.checkbox(_t(lang, "查询热带气旋路径", "Query IBTrACS Cyclone Tracks"), value=False, key="hazard_query_typhoon", help=_t(lang, "默认优先读取缓存；无缓存时按城市坐标下载对应海盆的官方最佳路径子集，统计城市周边 100/200/500km 接近次数。", "Reads cache first; without cache, downloads the official basin subset selected for the city and counts 100/200/500 km approaches."))
    query_hydro = col_c.checkbox(_t(lang, "查询洪涝/滑坡事件", "Query Flood/Landslide Events"), value=False, key="hazard_query_hydro", help=_t(lang, "默认优先读取缓存；无缓存时查询公开事件接口和洪水流量接口。", "Reads cache first; without cache, queries public event and flood-discharge APIs."))
    force_refresh_hazard = st.checkbox(
        _t(lang, "忽略缓存重新请求灾害数据", "Ignore cache and refresh hazard data"),
        value=False,
        key="hazard_force_refresh",
        help=_t(lang, "通常不需要勾选。热带气旋路径文件较大，强制刷新可能耗时较久。", "Usually leave this off. Tropical-cyclone track data is large, so forced refresh can take a while."),
    )
    run_hazard_query = st.button(
        _t(lang, "查询所选灾害记录", "Query Selected Hazard Records"),
        type="secondary",
        key="run_hazard_query",
    )
    selected_query = query_earthquake or query_typhoon or query_hydro
    typhoon_caches = typhoon_cache_paths(selected.location, HAZARD_CACHE_DIR / "typhoon")
    if query_typhoon and force_refresh_hazard:
        st.info(_t(lang, "正在强制重新下载热带气旋路径数据，文件较大，可能需要等待。", "Forcing a fresh tropical-cyclone track download. The file is large and may take a while."))
    elif query_typhoon and not all(path.exists() for path in typhoon_caches):
        st.info(_t(lang, "首次查询热带气旋路径需要下载较大的历史路径文件，完成后会写入本地缓存。", "The first tropical-cyclone query downloads a large historical track file and then caches it locally."))
    summary_key = f"hazard_summary:{selected.location.city}"
    summary = st.session_state.get(summary_key)
    if run_hazard_query and not selected_query:
        st.warning(_t(lang, "请先选择至少一种要查询的灾害记录。", "Select at least one hazard record type first."))
    elif run_hazard_query:
        with st.spinner(_t(lang, "正在查询历史灾害数据...", "Querying historical hazard records...")):
            summary = build_hazard_summary(
                selected.location,
                selected.metrics,
                cache_dir=HAZARD_CACHE_DIR,
                force_refresh=force_refresh_hazard,
                include_earthquake=query_earthquake,
                include_typhoon=query_typhoon,
                include_hydro_events=query_hydro,
                refresh_hydro_events=force_refresh_hazard,
            )
            st.session_state[summary_key] = summary
    if summary is None:
        summary = build_hazard_summary(
            selected.location,
            selected.metrics,
            cache_dir=HAZARD_CACHE_DIR,
            force_refresh=False,
            include_earthquake=False,
            include_typhoon=False,
            include_hydro_events=False,
            refresh_hydro_events=False,
        )
    elif selected_query and not run_hazard_query:
        st.caption(
            _t(
                lang,
                "查询选项已准备好；点击“查询所选灾害记录”后才会联网或读取缓存。",
                "Query options are ready. No network/cache request occurs until Query Selected Hazard Records is clicked.",
            )
        )
    _render_hazard_summary(summary, selected.location, lang)


def _render_hazard_summary(summary: HazardSummary, location, lang: str) -> None:
    eq = summary.earthquake
    cols = st.columns(5)
    not_queried = _t(lang, "未查询", "Not queried")
    cols[0].metric(_t(lang, "100km 地震", "100 km Quakes"), eq.count_100km if eq.status not in {"not_requested", "failed"} else not_queried)
    cols[1].metric(_t(lang, "200km 地震", "200 km Quakes"), eq.count_200km if eq.status not in {"not_requested", "failed"} else not_queried)
    cols[2].metric(_t(lang, "500km 地震", "500 km Quakes"), eq.count_500km if eq.status not in {"not_requested", "failed"} else not_queried)
    cols[3].metric(_t(lang, "6级及以上地震", "M6+ Quakes"), eq.event_count_m6 if eq.status not in {"not_requested", "failed"} else not_queried)
    cols[4].metric(_t(lang, "最大震级", "Max Magnitude"), f"{eq.max_magnitude:.1f}" if eq.max_magnitude else _t(lang, "无数据", "No data"))
    st.caption(_hazard_scope_note("earthquake", lang))
    if eq.status == "not_requested":
        st.info(_t(lang, "勾选“查询地震历史”后，会请求该城市周边历史地震记录并写入本地缓存。", "Enable Query USGS Earthquakes to request nearby historical events and cache them locally."))
    elif eq.status == "failed":
        st.warning(_t(lang, "地震数据暂时无法获取；当前只展示简单的灾害参考提示。", "USGS earthquake request failed; only simple hazard reminders are shown."))
    else:
        st.caption(_t(lang, f"地震数据源：{_source_label(eq.source, lang)}；状态：{_status_label(eq.status, lang)}；最近事件：{eq.latest_event_date or '无'}；最近距离：{eq.nearest_distance_km or '无'} km", f"Earthquake source: {eq.source}; status: {_status_label(eq.status, lang)}; latest event: {eq.latest_event_date or 'none'}; nearest distance: {eq.nearest_distance_km or 'none'} km"))
        if eq.events:
            st.pydeck_chart(make_earthquake_map(location, eq.events), width="stretch")
            st.caption(_map_attribution(lang))
            with st.expander(_t(lang, "部分地震记录", "Earthquake Event Sample"), expanded=False):
                st.dataframe(pd.DataFrame(eq.events), hide_index=True, width="stretch")
    if summary.typhoon is None:
        st.info(_t(lang, "勾选“查询热带气旋路径”后，会按城市周边 100/200/500km 统计 2000 年以来热带气旋路径接近次数。", "Enable Query IBTrACS Cyclone Tracks to count 100/200/500 km approaches since 2000."))
    else:
        _render_typhoon_summary(summary.typhoon, location, lang)
    st.markdown(f"**{_t(lang, '热带气旋摘要', 'Tropical Cyclone Summary')}:** {_hazard_note(summary.typhoon_note, lang, 'typhoon')}")
    st.markdown(f"**{_t(lang, '洪涝/强降水摘要', 'Flood/Heavy-rain Summary')}:** {_hazard_note(summary.rainfall_extreme_note, lang, 'rain')}")
    st.markdown(f"**{_t(lang, '滑坡摘要', 'Landslide Summary')}:** {_hazard_note(summary.landslide_note, lang, 'landslide')}")
    if summary.hazard_exposure_score is not None:
        st.metric(
            _t(lang, "灾害与环境参考", "Hazard and Environmental Reference"),
            f"{summary.hazard_exposure_score:.0f}/100",
            help=_t(
                lang,
                "这里汇总公开灾害记录和环境信息，便于进一步了解；不参与上方城市主排名，也不是风险概率。",
                "This supplementary indicator is used only inside the hazard view. It does not affect the main city ranking and is not a risk probability.",
            ),
        )
    event_cols = st.columns(2)
    with event_cols[0]:
        st.markdown(f"**{_t(lang, '洪涝事件与河流流量', 'Flood Events and River Discharge')}**")
        st.caption(_hazard_scope_note("flood", lang))
        st.dataframe(_event_table(summary.flood_events, lang), hide_index=True, width="stretch")
    with event_cols[1]:
        st.markdown(f"**{_t(lang, '滑坡事件', 'Landslide Events')}**")
        st.caption(_hazard_scope_note("landslide", lang))
        st.dataframe(_event_table(summary.landslide_events, lang), hide_index=True, width="stretch")
    with st.expander(_t(lang, "数据源说明", "Source Notes")):
        for note in _source_notes(lang):
            st.write("- " + note)


def _render_typhoon_summary(summary: TyphoonSummary, location, lang: str) -> None:
    cols = st.columns(4)
    cols[0].metric(_t(lang, "100km 内热带气旋", "Cyclones within 100 km"), summary.count_100km if summary.status != "failed" else _t(lang, "失败", "Failed"))
    cols[1].metric(_t(lang, "200km 内热带气旋", "Cyclones within 200 km"), summary.count_200km if summary.status != "failed" else _t(lang, "失败", "Failed"))
    cols[2].metric(_t(lang, "500km 内热带气旋", "Cyclones within 500 km"), summary.count_500km if summary.status != "failed" else _t(lang, "失败", "Failed"))
    cols[3].metric(_t(lang, "最近路径点", "Nearest Track Point"), f"{summary.nearest_distance_km:.0f} km" if summary.nearest_distance_km is not None else _t(lang, "无数据", "No data"))
    st.caption(_hazard_scope_note("typhoon", lang))
    if summary.status == "failed":
        st.warning(_message_for_display(summary.message, lang))
        return
    st.caption(
        ("; " if lang == LANG_EN else "；").join(
            [
                _t(lang, f"热带气旋数据源：{_source_label(summary.source, lang)}", f"Tropical cyclone source: {summary.source}"),
                _t(lang, f"状态：{_status_label(summary.status, lang)}", f"Status: {_status_label(summary.status, lang)}"),
                _t(lang, f"最强接近：{summary.strongest_name or '无'} {summary.strongest_year or ''}".strip(), f"Strongest nearby: {summary.strongest_name or 'none'} {summary.strongest_year or ''}".strip()),
                _t(lang, f"最近年份：{summary.latest_nearby_name or '无'} {summary.latest_nearby_year or ''}".strip(), f"Latest nearby year: {summary.latest_nearby_name or 'none'} {summary.latest_nearby_year or ''}".strip()),
            ]
        )
    )
    st.caption(_message_for_display(summary.message, lang))
    if summary.track_points:
        st.pydeck_chart(make_typhoon_track_map(location, summary.track_points), width="stretch")
        st.caption(_map_attribution(lang))
        with st.expander(_t(lang, "热带气旋路径点样本", "Tropical Cyclone Track Point Sample"), expanded=False):
            st.dataframe(pd.DataFrame(summary.track_points).head(200), hide_index=True, width="stretch")


def _render_aurora_tab(results: list[CityResult], lang: str) -> None:
    st.subheader(_t(lang, "极光机会", "Aurora Opportunity"))
    col_a, col_b = st.columns(2)
    include_live = col_a.checkbox(_t(lang, "查询未来 30 分钟极光提示", "Query NOAA SWPC OVATION Nowcast"), value=False, key="aurora_include_live", help=_t(lang, "读取官方未来 30 分钟极光提示，估算当前看到极光的机会。", "Read the official 30-minute aurora probability grid and estimate opportunity from the nearest grid point."))
    force_refresh = col_b.checkbox(_t(lang, "刷新极光缓存", "Refresh aurora cache"), value=False, key="aurora_force_refresh")
    summaries = [
        build_aurora_summary(
            item.location,
            cache_dir=AURORA_CACHE_DIR,
            force_refresh=force_refresh,
            include_live=include_live,
        )
        for item in results
    ]
    table = pd.DataFrame(
        {
            _t(lang, "城市", "City"): [item.city if lang == LANG_ZH else _city_en_from_summary(item.city) for item in summaries],
            _t(lang, "机会等级", "Opportunity Level"): [_aurora_label(item.opportunity_label, lang) for item in summaries],
            _t(lang, "极光机会分", "Aurora Opportunity"): [item.opportunity_score for item in summaries],
            _t(lang, "附近区域机会", "Nearby Area Chance"): [item.nearest_probability for item in summaries],
            _t(lang, "附近区域距离 km", "Nearby Area Distance km"): [item.nearest_distance_km for item in summaries],
            _t(lang, "预报时间", "Forecast Time"): [item.forecast_time for item in summaries],
            _t(lang, "说明", "Notes"): [_aurora_explanation(item, lang) for item in summaries],
            _t(lang, "数据状态", "Status"): [_status_label(item.status, lang) for item in summaries],
        }
    ).sort_values(_t(lang, "极光机会分", "Aurora Opportunity"), ascending=False)
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(_t(lang, "不查询未来 30 分钟提示时，结果只按城市纬度粗略估算；实际能否看到极光还受云量、夜晚时段和光污染影响。", "Without nowcast, the app uses a latitude-based estimate. Actual visibility also depends on clouds, darkness, and light pollution."))


def _ranking_table(results: list[CityResult], lang: str) -> pd.DataFrame:
    rows = []
    for index, item in enumerate(results, start=1):
        rows.append(
            {
                _t(lang, "排名", "Rank"): index,
                _t(lang, "城市", "City"): _city_name(item, lang),
                _t(lang, "省份/地区", "Country"): item.location.province if lang == LANG_ZH else item.location.country,
                _t(lang, "你的偏好匹配分", "Your Preference Match"): round(item.score.personal_fit_score, 1),
                _t(lang, "多年气候匹配分", "Climate Match"): round(item.score.climate_normal_fit_score, 1),
                _t(lang, "短期预报", "Short-term Forecast"): None if item.score.forecast_trip_fit_score is None else round(item.score.forecast_trip_fit_score, 1),
                _t(lang, "气候舒适", "Climate Comfort"): round(item.score.travel_comfort_score, 1),
                _t(lang, "糟糕天气指数", "Bad Weather Index"): round(item.score.long_term_risk_score, 1),
                _t(lang, "状态", "Status"): _status_label(item.score.data_status, lang),
                _t(lang, "主要理由", "Main Reasons"): _join_display(item.score.strengths[:2], lang),
            }
        )
    return pd.DataFrame(rows)


def _ranking_column_config(lang: str) -> dict[str, object]:
    progress_color = "#a78bfa"
    return {
        _t(lang, "你的偏好匹配分", "Your Preference Match"): st.column_config.ProgressColumn(
            _t(lang, "你的偏好匹配分", "Your Preference Match"),
            min_value=0,
            max_value=100,
            format="%.0f",
            color=progress_color,
        ),
        _t(lang, "多年气候匹配分", "Climate Match"): st.column_config.ProgressColumn(
            _t(lang, "多年气候匹配分", "Climate Match"),
            min_value=0,
            max_value=100,
            format="%.0f",
            color=progress_color,
        ),
        _t(lang, "短期预报", "Short-term Forecast"): st.column_config.ProgressColumn(
            _t(lang, "短期预报", "Short-term Forecast"),
            min_value=0,
            max_value=100,
            format="%.0f",
            color=progress_color,
        ),
        _t(lang, "气候舒适", "Climate Comfort"): st.column_config.ProgressColumn(
            _t(lang, "气候舒适", "Climate Comfort"),
            min_value=0,
            max_value=100,
            format="%.0f",
            color=progress_color,
        ),
        _t(lang, "糟糕天气指数", "Bad Weather Index"): st.column_config.ProgressColumn(
            _t(lang, "糟糕天气指数", "Bad Weather Index"),
            min_value=0,
            max_value=100,
            format="%.0f",
            color=progress_color,
        ),
    }


def _data_mode_short_label(data_mode: str, lang: str) -> str:
    labels = {
        DATA_MODE_STATIC: {"zh": "多年平均", "en": "Bundled Baseline"},
        DATA_MODE_HISTORY: {"zh": "历史", "en": "History"},
        DATA_MODE_FORECAST: {"zh": "预报", "en": "Forecast"},
        DATA_MODE_HAZARD: {"zh": "灾害", "en": "Hazard"},
        DATA_MODE_AURORA: {"zh": "极光", "en": "Aurora"},
    }
    return labels.get(data_mode, {"zh": data_mode, "en": data_mode})[lang]


def _time_scope_label(pref: UserPreference, data_mode: str, lang: str) -> str:
    if pref.month == MONTH_ALL:
        return _t(lang, "全年", "Full Year")
    if pref.mode == "Travel" and data_mode == DATA_MODE_FORECAST:
        return _t(lang, "有效预报窗口", "Forecast Window")
    return _month_label(pref.month, lang)


def _city_card(index: int, item: CityResult, pref: UserPreference, lang: str) -> None:
    label, color = score_label(item.score.personal_fit_score)
    display_label = _score_label(label, lang)
    place = item.location.province if lang == LANG_ZH else item.location.country
    status = _status_badge_html(item.score.data_status, lang)
    strengths = "".join(_chip_html(_display_phrase(text, lang), "good") for text in item.score.strengths[:3])
    weaknesses = "".join(_chip_html(_display_phrase(text, lang), "warn") for text in item.score.weaknesses[:2])
    forecast_score = item.score.forecast_trip_fit_score
    forecast_text = "--" if forecast_score is None else f"{forecast_score:.0f}"
    html = f"""
    <div class="wf-city-card">
      <div class="wf-city-card-head">
        <div>
          <div class="wf-card-rank">#{index}</div>
          <h3>{escape(_city_name(item, lang))}</h3>
          <p>{escape(place)} · {status}</p>
        </div>
        <div class="wf-score-pill" style="border-color:{escape(color)}; color:{escape(color)}">{escape(display_label)} · {item.score.personal_fit_score:.0f}</div>
      </div>
      <div class="wf-score-grid">
        {_score_bar_html(_t(lang, "你的偏好匹配分", "Your Preference Match"), item.score.personal_fit_score, "teal")}
        {_score_bar_html(_t(lang, "气候舒适", "Climate Comfort"), item.score.travel_comfort_score, "blue")}
        {_score_bar_html(_t(lang, "糟糕天气指数", "Bad Weather Index"), item.score.long_term_risk_score, "violet")}
        {_score_bar_html(_t(lang, "预报", "Forecast"), forecast_score, "amber", fallback=forecast_text)}
      </div>
      <div class="wf-card-body">{escape(generate_city_report(item, pref, lang))}</div>
      <div class="wf-chip-row">{strengths}{weaknesses}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def _method_section(lang: str) -> None:
    with st.expander(_t(lang, "方法说明", "Method Notes")):
        if lang == LANG_EN:
            st.markdown(
                """
                **What the score means**

                - Your Preference Match compares only the cities currently selected and follows the preferences you set. It is not a probability, universal city ranking, or safety model.
                - The Bad Weather Index combines heat, heavy rain, long-term PM2.5, and broad coastal/typhoon labels. A higher value means more weather or environmental factors may need attention; it is not a safety rating or a disaster probability. Public hazard records do not change the main ranking.

                **How the score is calculated**

                - Each item is first converted to a 0–100 score and limited to that range. Higher Climate Comfort is better; a higher Bad Weather Index means more factors to watch out for.
                - `Climate Comfort = Temperature × 38% + Humidity × 22% + Rain × 22% + Air Quality × 18%`
                - `Your Preference Match = Climate Comfort × comfort weight + (100 − Bad Weather Index) × weather weight`

                | Scenario | Comfort weight | Weather weight |
                | --- | ---: | ---: |
                | Travel | 75% | 25% |
                | Long-term living | 45% | 55% |
                | Compare | 60% | 40% |

                - In Travel mode, when a valid short-term forecast is available, the final match uses the forecast score instead: temperature 36–42%, precipitation 27–32%, wind 12–14%, forecast timeliness 10–12%, and recent PM2.5 15% when available. The exact weights depend on whether recent PM2.5 was returned.

                **Time-scale rules**

                - The offline comparison uses bundled 2000-2025 NASA POWER / MERRA-2 area estimates. Historical mode separately requests ERA5 for the dates you choose.
                - Full-year rain, heat, cold, and similar day counts are displayed as annual totals, but converted to monthly-equivalent rates before applying thresholds calibrated for a month.
                - POWER humidity is provided directly; apparent temperature is calculated and snow days are estimated from temperature and precipitation. Long-term air quality uses ACAG SatPM2.5 annual data for 2015-2024.
                - Recent Open-Meteo PM2.5 is used only for Travel scoring in a valid forecast window and is never substituted for a long-term normal.
                - If forecast data is unavailable, the comparison continues with the multi-year climate match rather than showing a misleading zero.

                **Separate evidence views**

                - USGS earthquakes, IBTrACS tracks, NASA EONET records, Open-Meteo river-flow estimates, and NOAA OVATION are shown only when you choose to check them.
                - Track proximity, missing records, river-flow estimates, and aurora chances are not damage, safety, or event probabilities.

                Sources: [NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/),
                [Open-Meteo Historical Weather](https://open-meteo.com/en/docs/historical-weather-api),
                [Open-Meteo Forecast](https://open-meteo.com/en/docs),
                [Open-Meteo Licence](https://open-meteo.com/en/licence),
                [MET Norway Locationforecast](https://api.met.no/weatherapi/locationforecast/2.0/documentation),
                [ACAG SatPM2.5 on AWS](https://registry.opendata.aws/surface-pm2-5-v6gl/).
                """
            )
            return
        st.markdown(
            """
            **分数的含义**

            - “你的偏好匹配分”只比较本次选中的城市，并按你设置的偏好计算；它不是概率、全球城市排名或安全模型。
            - “糟糕天气指数”综合高温、强降雨、长期 PM2.5，以及较粗的沿海/台风标签。数值越高，表示可能需要进一步了解的天气或环境因素越多；它不是安全评级或灾害概率。公开灾害记录不会改变主排名。

            **评分怎么算**

            - 每个项目都会先换算为 0—100 分，并限制在这个范围内。气候舒适分越高越好；糟糕天气指数越高，表示需要留意的因素越多。
            - `气候舒适分 = 温度舒适 × 38% + 湿度舒适 × 22% + 降水友好 × 22% + 空气质量 × 18%`
            - `你的偏好匹配分 = 气候舒适分 × 舒适权重 +（100 − 糟糕天气指数）× 天气权重`

            | 使用场景 | 舒适权重 | 天气权重 |
            | --- | ---: | ---: |
            | 旅行 | 75% | 25% |
            | 长期居住 | 45% | 55% |
            | 城市对比 | 60% | 40% |

            - 旅行模式中，如果有可用的近期预报，最终匹配分会改用“近期预报分”：温度占 36—42%，降水占 27—32%，大风占 12—14%，预报时效占 10—12%；如果有近期 PM2.5，则再占 15%。具体比例会随近期 PM2.5 是否返回而调整。

            **时间尺度规则**

            - 多年气候数据覆盖 2000—2025 年；如果你选择查看指定日期范围，应用会另外读取那段时期的历史天气。
            - 全年模式会把全年下雨、高温和寒冷天数按月平均后再比较，避免因为时间更长而显得“更糟”。
            - 体感温度由温度、湿度和风速算出；“可能下雪日”由温度和降水推算，不是实测雪天。多年空气质量使用 2015—2024 年的数据。
            - 近期 PM2.5 只用于近期旅行比较，不会替代多年的空气质量情况。
            - 如果未来天气暂时取不到，应用会使用多年气候数据继续比较，不会把城市直接算成 0 分。

            **独立证据页面**

            - 地震、台风路径、洪涝/滑坡记录、河流水量和极光情况都在你需要时才查询。
            - 路径靠近、暂未查到记录、河流水量和极光机会都不等于损失、安全程度或事件发生概率。

            数据来源参考：
            [NASA POWER 逐日 API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)、
            [开放气象历史天气文档](https://open-meteo.com/en/docs/historical-weather-api)、
            [开放气象预报文档](https://open-meteo.com/en/docs)、
            [开放气象许可说明](https://open-meteo.com/en/licence)、
            [挪威气象局预报文档](https://api.met.no/weatherapi/locationforecast/2.0/documentation)、
            [ACAG SatPM2.5 AWS 开放数据](https://registry.opendata.aws/surface-pm2-5-v6gl/)。
            """
        )


def _page_notes(lang: str) -> None:
    """Keep method detail and the safety boundary available without interrupting the workflow."""

    with st.expander(_t(lang, "数据与使用说明", "Data and Usage Notes"), expanded=False):
        st.caption(_disclaimer(lang))
        _method_section(lang)


def _t(lang: str, zh: str, en: str) -> str:
    return en if lang == LANG_EN else zh


def _city_name(item: CityResult, lang: str) -> str:
    return item.location.city_en or item.location.city if lang == LANG_EN else item.location.city


def _localized_city_selectbox(label: str, results: list[CityResult], lang: str, state_key: str) -> str:
    """Render a language-sensitive city selector while preserving its canonical choice."""

    options = [item.location.city_en or item.location.city for item in results]
    shared_key = f"{state_key}_choice"
    widget_key = f"{state_key}_{lang}"
    render_lang_key = f"{state_key}_render_lang"
    shared_value = str(st.session_state.get(shared_key, options[0]))
    if shared_value not in options:
        shared_value = options[0]
    if st.session_state.get(render_lang_key) != lang:
        st.session_state[widget_key] = shared_value
    st.session_state[render_lang_key] = lang
    selected = st.selectbox(
        label,
        options,
        format_func=lambda value: next(
            _city_name(item, lang) for item in results if (item.location.city_en or item.location.city) == value
        ),
        key=widget_key,
    )
    st.session_state[shared_key] = selected
    return str(selected)


def _display_phrase(text: str, lang: str) -> str:
    """Translate a score explanation phrase for chips and compact labels."""

    return PHRASE_EN.get(text, _warning_en(text)) if lang == LANG_EN else text


def _city_en_from_summary(city: str) -> str:
    data = _load_data()
    matched = data[data["city_zh"] == city]
    if not matched.empty:
        return str(matched.iloc[0].get("city_en") or matched.iloc[0].get("city") or city)
    return city


def _mode_label(value: str, lang: str) -> str:
    return MODE_LABELS_LOCAL.get(value, {"zh": value, "en": value})[lang]


def _month_label(month: int, lang: str) -> str:
    if int(month) == MONTH_ALL:
        return _t(lang, "全年", "Full Year")
    if lang == LANG_EN:
        return date(2026, int(month), 1).strftime("%B")
    return f"{month} 月"


def _yes_no(value: object, lang: str) -> str:
    truthy = str(value).lower() in {"true", "1", "yes"}
    if lang == LANG_EN:
        return "Yes" if truthy else "No"
    return "是" if truthy else "否"


def _status_label(status: str, lang: str) -> str:
    return STATUS_LABELS.get(status, {"zh": status, "en": status})[lang]


def _status_badge_html(status: str, lang: str) -> str:
    tone = {
        "live": "good",
        "cache": "good",
        "cache/live": "good",
        "live/cache": "good",
        "dataset": "good",
        "partial": "warn",
        "fallback": "muted",
        "heuristic": "muted",
        "not_requested": "muted",
        "failed": "bad",
    }.get(status, "muted")
    return f"<span class='wf-status wf-status-{tone}'>{escape(_status_label(status, lang))}</span>"


def _quality_category_label(category: str, lang: str) -> str:
    """Localize data-quality categories."""

    labels = {
        "climate": {"zh": "气候指标", "en": "Climate metrics"},
        "humidity": {"zh": "湿度", "en": "Humidity"},
        "long_term_air_quality": {"zh": "长期空气质量", "en": "Long-term air quality"},
        "recent_air_quality": {"zh": "近期空气质量", "en": "Recent air quality"},
        "forecast": {"zh": "短期天气预报", "en": "Short-term forecast"},
        "hazard_records": {"zh": "历史灾害记录", "en": "Historical hazard records"},
    }
    return labels.get(category, {"zh": category, "en": category})[lang]


def _quality_text(value: str, lang: str) -> str:
    """Translate compact data-quality notes used by the provenance table."""

    if lang == LANG_EN:
        return value
    translations = {
        "multi-year historical window": "多年平均数据",
        "curated baseline without observation years": "整理好的城市参考数据；没有逐年记录",
        "curated levels": "整理好的参考等级",
        "curated level": "整理好的参考等级",
        "derived field": "根据其他数据算出",
        "estimated from region and rainfall": "根据地区和降水情况估算",
        "curated seed baseline": "基础城市参考数据",
        "long-term baseline; not a recent observation": "多年平均数据；不是近期实时情况",
        "provider-specific historical or recent window": "不同来源对应的时间范围",
        "on demand": "需要时查询",
        "recent 8-day window": "最近 8 天",
        "Temperature, precipitation, and related information used to calculate the preference match.": "温度、降水等信息会用于计算你的偏好匹配分。",
        "Humidity is estimated from rainfall and regional context because direct humidity data is unavailable for this result.": "这次没有直接湿度数据，所以根据降水和地区情况估算。",
        "Humidity is provided directly by the fixed-model historical dataset.": "湿度来自多年天气数据。",
        "Humidity comes from the basic city reference data.": "湿度来自基础城市参考数据。",
        "The long-term exposure score does not substitute a recent 8-day PM2.5 window for a climate baseline.": "糟糕天气指数不会拿最近 8 天的 PM2.5 代替多年的空气质量情况。",
        "A city-area estimate combining satellite, simulation, and monitoring data; not a single station observation. Recent 8-day PM2.5 is not substituted for this long-term baseline.": "结合卫星、模拟和地面监测得出的城市周边估算值，不是单个监测站读数；最近 8 天的 PM2.5 不会替代多年的空气质量情况。",
        "Basic city reference data is used because long-term PM2.5 data is unavailable for this city.": "该城市暂缺多年 PM2.5 数据，因此使用整理好的参考等级。",
        "Historical records are displayed separately and do not alter the main ranking.": "历史记录单独展示，不改变主排名。",
        "Used for the Travel ranking only when the selected month is in the valid forecast window.": "只在所选日期有可用预报时用于旅行比较。",
        "Used only in short-term Travel scoring; never treated as a long-term air-quality normal.": "只用于近期旅行比较，不当作多年的空气质量水平。",
    }
    if value.endswith(" years") and value.split()[0].isdigit():
        return value.replace(" years", " 年")
    if value.endswith(" annual values") and value.split()[0].isdigit():
        return value.replace(" annual values", " 个年度值")
    if value.startswith("annual estimates, "):
        return value.replace("annual estimates, ", "年度融合估计：")
    if value.endswith(" days") and value.split()[0].isdigit():
        return value.replace(" days", " 天")
    if value.endswith(" hours") and value.split()[0].isdigit():
        return value.replace(" hours", " 小时")
    return translations.get(value, value)


def _chip_html(text: str, tone: str = "good") -> str:
    return f"<span class='wf-chip wf-chip-{tone}'>{escape(str(text))}</span>"


def _score_bar_html(label: str, value: float | None, tone: str, fallback: str | None = None) -> str:
    if value is None:
        display = fallback or "--"
        width = 0
    else:
        display = f"{value:.0f}"
        width = max(0, min(100, int(round(value))))
    return f"""
    <div class="wf-score-row">
      <div class="wf-score-row-top"><span>{escape(label)}</span><b>{escape(display)}</b></div>
      <div class="wf-score-track"><div class="wf-score-fill wf-score-{tone}" style="width:{width}%"></div></div>
    </div>
    """


def _source_label(source: str, lang: str) -> str:
    if lang == LANG_EN:
        replacements = {
            "基础城市数据（全年聚合）": "Seed city data (full-year aggregation)",
            "基础城市数据": "Seed city data",
            "静态种子数据": "Static seed data",
            "开放气象": "Open-Meteo",
            "挪威气象局": "MET Norway",
            "美国地质调查局": "USGS",
            "美国海洋和大气管理局": "NOAA",
            "美国空间天气中心": "NOAA Space Weather Prediction Center",
            "国际热带气旋最佳路径档案": "IBTrACS tropical-cyclone best-track archive",
            "接口": "API",
        }
        text = source
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    if source.startswith("NOAA IBTrACS"):
        return "美国海洋和大气管理局国际热带气旋最佳路径档案（按城市选择海盆）"
    replacements = {
        "静态种子数据": "基础城市数据",
        "Open-Meteo Historical Weather API": "开放气象历史天气数据",
        "Open-Meteo Forecast API": "开放气象未来天气预报",
        "Open-Meteo Air Quality API": "开放气象空气质量数据",
        "ACAG SatPM2.5": "华盛顿大学 ACAG 卫星融合 PM2.5",
        "MET Norway Locationforecast API": "挪威气象局未来天气预报",
        "USGS Earthquake Catalog API": "美国地质调查局地震目录",
        "NOAA SWPC OVATION Aurora 30 Minute Forecast": "美国空间天气中心未来 30 分钟极光提示",
        "NOAA SWPC OVATION 30-minute forecast": "美国空间天气中心未来 30 分钟极光提示",
        "Open-Meteo": "开放气象",
        "MET Norway": "挪威气象局",
        "IBTrACS": "国际热带气旋最佳路径档案",
        "USGS": "美国地质调查局",
        "NOAA": "美国海洋和大气管理局",
        "API": "接口",
        "proxy": "参考信息",
        "fallback": "备用值",
    }
    text = source
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _risk_label(value: float, lang: str) -> str:
    label = risk_label(value)
    return RISK_LABELS_EN.get(label, label) if lang == LANG_EN else label


def _score_label(label: str, lang: str) -> str:
    return SCORE_LABELS_EN.get(label, label) if lang == LANG_EN else label


def _join_display(items: list[str], lang: str) -> str:
    if lang == LANG_EN:
        translated = [PHRASE_EN.get(item, _warning_en(item)) for item in items]
        return ", ".join(translated)
    return "；".join(items)


def _warning_en(text: str) -> str:
    replacements = {
        "该城市位于台风影响区域，当前台风项仍是简化提示": "This city is in a typhoon-affected area; the current result is only a simple reminder.",
        "糟糕天气指数偏高，建议进一步了解高温、强降雨和空气质量情况": "The Bad Weather Index is higher; review heat, heavy rain, and air quality in more detail.",
        "当前城市使用人工基础等级，不代表观测常态或权威评级": "This city uses curated baseline levels, not observed normals or an authoritative rating.",
        "温度、湿度和降水等气候项使用城市参考数据；长期 PM2.5 已使用多年融合数据": "Temperature, humidity, and precipitation use city reference data; long-term PM2.5 uses multi-year fused data.",
        "多年气候数据中的体感温度由温度、湿度和风速计算；可能下雪日由温度和降水推算": "Apparent temperature in the multi-year climate data is calculated from temperature, humidity, and wind speed; possible snow days are estimated from temperature and precipitation.",
        "历史天气来自公开资料汇总，部分项目为估算或备用参考值": "Historical weather comes from public sources; some values are estimated or use backup references.",
        "历史天气来自公开资料汇总，湿度为估算；长期 PM2.5 使用多年融合数据": "Historical weather comes from public sources and humidity is estimated; long-term PM2.5 uses a multi-year fused dataset.",
        "未来天气暂时无法获取，当前排序改用多年气候匹配分": "Forecast data is unavailable, so the ranking uses the multi-year climate match instead.",
        "预报范围超过 7 天，远期不确定性较高": "Forecast range exceeds 7 days, so uncertainty is higher.",
    }
    return replacements.get(text, text)


def _map_attribution(lang: str) -> str:
    """Return the map-provider caption in the active interface language."""

    return MAP_ATTRIBUTION_EN if lang == LANG_EN else MAP_ATTRIBUTION


def _message_for_display(message: str, lang: str) -> str:
    if lang == LANG_ZH:
        replacements = {
            "Open-Meteo": "开放气象",
            "Air Quality": "空气质量",
            "MET Norway": "挪威气象局",
            "IBTrACS": "国际热带气旋最佳路径档案",
            "USGS": "美国地质调查局",
            "NOAA SWPC OVATION": "美国空间天气中心未来 30 分钟极光提示",
            "NOAA OVATION": "美国未来 30 分钟极光提示",
            "API": "接口",
            "cache/live": "已保存/刚刚获取",
            "live/cache": "刚刚获取/已保存",
            "cache": "已保存",
            "live": "刚刚获取",
            "partial": "信息不完整",
            "proxy": "参考提示",
            "fallback": "备用值",
            "Forecast": "预报",
        }
        text = message
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    if "未来预报请求失败" in message or "请求失败" in message or "不可用" in message:
        return "Data request failed or is unavailable; cached or backup reference data may be used."
    if "缓存" in message or "cache" in message.lower():
        return "Loaded data from local cache."
    if "已获取" in message or "live" in message.lower():
        return "Fetched live data and updated cache."
    if "Forecast 只支持" in message:
        return "Forecast supports only a 1-16 day range."
    if any("\u4e00" <= character <= "\u9fff" for character in message):
        return "A data-source status message is available; no additional English detail is available."
    return message


def _days_label(value: int | float, lang: str) -> str:
    return f"{value} {'days' if value != 1 else 'day'}" if lang == LANG_EN else f"{value} 天"


def _disclaimer(lang: str) -> str:
    if lang == LANG_EN:
        return "This is a city climate-comparison tool. Scores compare only the cities selected in this session; they are not probabilities, authoritative ratings, or individualized professional advice. Do not use them for real estate, insurance, medical, emergency, or disaster-safety decisions."
    return DISCLAIMER


def _hazard_disclaimer(lang: str) -> str:
    if lang == LANG_EN:
        return "Historical hazard records show public data and simplified indicators only. Past events do not guarantee future events, and missing records do not imply no risk. Results are not safety ratings, site-selection advice, or hazard predictions."
    return HAZARD_DISCLAIMER.replace("proxy", "参考提示")


def _hazard_note(note: str, lang: str, kind: str) -> str:
    if lang == LANG_ZH:
        return note.replace("proxy", "参考提示")
    notes = {
        "typhoon": "Direct tropical-cyclone track exposure is estimated from coarse seed-table flags unless IBTrACS has been queried. Track distance is not the same as damage or safety.",
        "rain": "Flood and heavy-rain exposure is currently represented by precipitation indicators only. Drainage, terrain, river networks, and land use are not modeled.",
        "landslide": "Landslide exposure is currently represented by terrain and precipitation indicators only. Geology, slope, and event databases are not yet integrated.",
    }
    return notes[kind]


def _hazard_scope_note(kind: str, lang: str) -> str:
    today = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    flood_start = (date.today() - timedelta(days=365)).isoformat()
    if kind == "earthquake":
        return _t(
            lang,
            f"查看范围：{HISTORY_START_DATE} 至 {yesterday}，城市周边 500km 内 4 级及以上地震的总次数；100/200/500km 是这段时间的累计次数，不是每年次数。",
            f"Scope: cumulative M4+ earthquakes from {HISTORY_START_DATE} to {yesterday} within 500 km. 100/200/500 km counts are totals for this period, not annual counts.",
        )
    if kind == "typhoon":
        return _t(
            lang,
            "查看范围：按城市位置选择对应海域的 2000 年以来热带气旋路径资料；100/200/500km 是这段时间内靠近城市的累计次数，不是每年次数。",
            "Scope: NOAA IBTrACS best-track data since 2000 from the basin subset selected for the city. 100/200/500 km counts are cumulative nearby-cyclone totals, not annual counts.",
        )
    if kind == "flood":
        return _t(
            lang,
            f"查看范围：洪涝事件为 {HISTORY_START_DATE} 至 {today} 城市周边 500km 内的公开记录；河流流量为 {flood_start} 至 {today} 近一年的资料。",
            f"Scope: flood events are public event records from {HISTORY_START_DATE} to {today} within 500 km; river discharge covers the last 365 days from {flood_start} to {today}.",
        )
    if kind == "landslide":
        return _t(
            lang,
            f"查看范围：滑坡事件为 {HISTORY_START_DATE} 至 {today} 城市周边 500km 内的公开记录；未查到记录不等于没有风险。",
            f"Scope: landslide events are public event records from {HISTORY_START_DATE} to {today} within 500 km. No matched event does not mean no risk.",
        )
    return ""


def _event_table(events: list[dict[str, object]], lang: str) -> pd.DataFrame:
    if lang == LANG_ZH:
        table = pd.DataFrame(events)
        for column in ["记录", "数据口径", "证据类型", "类型"]:
            if column in table.columns:
                table[column] = table[column].map(_localize_event_value)
        for column in ["日期", "距离"]:
            if column in table.columns:
                table[column] = table[column].map(lambda value: "" if pd.isna(value) else str(value))
        return table
    if not events:
        return pd.DataFrame()
    rows = []
    for event in events:
        rows.append(
            {
                "Type": _hazard_event_type_en(str(event.get("类型", ""))),
                "Record": _hazard_event_record_en(str(event.get("记录", ""))),
                "Data Basis": _hazard_event_basis_en(str(event.get("数据口径", ""))),
                "Evidence Type": _hazard_evidence_type_en(str(event.get("证据类型", ""))),
            }
        )
    return pd.DataFrame(rows)


def _hazard_event_type_en(value: str) -> str:
    mapping = {
        "极端降水提示": "Extreme-rain reminder",
        "强降水提示": "Heavy-rain reminder",
        "洪涝事件": "Flood event",
        "滑坡事件": "Landslide event",
        "河流流量数据": "River-discharge data",
        "未检索到洪涝事件": "No matched flood events",
        "未检索到滑坡事件": "No matched landslide events",
        "暂无重大洪涝事件库": "No flood event database yet",
        "山地或高原提示": "Mountain or plateau reminder",
        "暂无重大滑坡事件库": "No landslide event database yet",
    }
    if value in mapping:
        return mapping[value]
    return "Hazard-related record" if any("\u4e00" <= character <= "\u9fff" for character in value) else value.replace("proxy", "reminder")


def _hazard_evidence_type_en(value: str) -> str:
    """Translate hazard evidence types without presenting them as confidence."""

    return {
        "公开目录记录": "Public catalog record",
        "区域估算数据": "Area estimate",
        "公开目录检索": "Public catalog search",
        "参考提示": "Reference reminder",
        "heuristic": "Heuristic",
        "fallback": "Fallback",
        "cache": "Cache",
        "live": "Live",
        "partial": "Partial",
    }.get(value, "Data-source status" if any("\u4e00" <= character <= "\u9fff" for character in value) else value)


def _localize_event_value(value: object) -> object:
    if not isinstance(value, str):
        return value
    replacements = {
        "China": "中国",
        "Sichuan": "四川",
        "Yunnan": "云南",
        "Guangdong": "广东",
        "Guangxi": "广西",
        "Fujian": "福建",
        "Zhejiang": "浙江",
        "Hunan": "湖南",
        "Hubei": "湖北",
        "Jiangxi": "江西",
        "Henan": "河南",
        "Landslides": "滑坡",
        "Landslide": "滑坡",
        "Floods": "洪涝",
        "Flood": "洪涝",
        ",": "，",
    }
    text = value
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _hazard_event_record_en(value: str) -> str:
    if "未接" in value or "未连接" in value:
        return "No city-level event database is connected yet; only simple reminders are shown."
    if "强降水" in value or "极端降水" in value:
        return "The selected month has elevated precipitation indicators. This is not a recorded damage event."
    if "城市附近区域" in value or "河流流量" in value:
        return "Estimated river discharge for the surrounding area is shown for context only; it does not indicate city flood probability."
    if "未返回" in value or "未检索到" in value:
        return "No matching public event records were returned within 500 km. This does not mean there is no risk."
    if "地形" in value or "区域" in value:
        return "The city's terrain label suggests that slope and event databases should be checked later."
    return value


def _hazard_event_basis_en(value: str) -> str:
    if "美国航天局地球观测自然事件接口" in value:
        return "NASA EONET public event catalog"
    if "开放气象洪水资料" in value:
        return "Open-Meteo flood-data river-discharge estimate"
    if "开放气象历史天气" in value:
        return "Open-Meteo historical weather or bundled seed-city aggregation"
    if "Open-Meteo" in value:
        return "Open-Meteo historical weather or bundled backup aggregation"
    if "日降水" in value:
        return "Daily precipitation threshold"
    if "基础城市" in value:
        return "Basic city-area label plus precipitation information"
    if "后续预留" in value:
        return "Reserved for future EM-DAT/GDIS/NASA COOLR or domestic public sources"
    return "Data-source description available in the Chinese interface only." if any("\u4e00" <= character <= "\u9fff" for character in value) else value


def _source_notes(lang: str) -> list[str]:
    if lang == LANG_EN:
        return [
            "Earthquakes: USGS Earthquake Catalog API, M4+ events within 500 km, summarized by 100/200/500 km bands.",
            "Tropical cyclones: NOAA IBTrACS best-track basin subset selected for each city, summarized by distance to track points.",
            "Floods and landslides: NASA EONET public event records within 500 km; floods also include Open-Meteo river discharge when available.",
        ]
    return [
        "地震：美国地质调查局地震目录，查询 2000 年以来城市周边 500km 内 4 级及以上地震，并分别统计 100/200/500km 内的次数。",
        "热带气旋：按城市坐标选择国际热带气旋最佳路径档案对应海盆子集，并按城市到路径点距离统计 100/200/500km 接近次数。",
        "洪涝/滑坡：美国航天局地球观测自然事件接口，按城市周边 500km 查询公开事件；洪涝还读取开放气象洪水接口河流流量。",
    ]


def _aurora_label(label: str, lang: str) -> str:
    if lang == LANG_ZH:
        return label
    return {
        "高纬相对较高": "Relatively high latitude opportunity",
        "低到中等": "Low to moderate",
        "较低": "Low",
        "极低": "Very low",
        "未来 30 分钟机会较高": "Near-term opportunity elevated",
    }.get(label, label)


def _aurora_explanation(summary, lang: str) -> str:
    if lang == LANG_ZH:
        return summary.explanation
    if summary.nearest_probability is not None:
        return f"NOAA OVATION nearest-grid probability is about {summary.nearest_probability:.1f}. Actual visibility still depends on darkness, cloud cover, light pollution, and geomagnetic activity."
    if summary.opportunity_score >= 50:
        return "This high-latitude city has a low but relatively higher aurora opportunity during strong geomagnetic activity. Visibility still depends on clouds, darkness, and light pollution."
    if summary.opportunity_score >= 20:
        return "Aurora visibility is generally low and would require unusually strong geomagnetic activity."
    return "Aurora visibility is very low under normal conditions; this is only a travel-interest indicator."


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --wf-bg: #f7f8fa;
            --wf-panel: #ffffff;
            --wf-line: #d9dee7;
            --wf-text: #18212f;
            --wf-muted: #667085;
            --wf-purple: #a78bfa;
            --wf-indigo: #0072b2;
            --wf-teal: #009e73;
            --wf-amber: #e69f00;
            --wf-magenta: #a78bfa;
            --wf-slate: #475467;
        }
        .stApp {background: var(--wf-bg);}
        .stApp,
        .stApp label,
        .stApp p,
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6,
        .stApp span,
        [data-testid="stMarkdownContainer"] {
            color: var(--wf-text);
        }
        .block-container {padding-top: .65rem; padding-bottom: 2.5rem; max-width: 1240px;}
        [data-testid="stHorizontalBlock"]:has(#where-fit) {
            flex-wrap: nowrap !important;
            align-items: flex-start !important;
        }
        [data-testid="stHorizontalBlock"]:has(#where-fit) > [data-testid="stColumn"]:first-child {
            flex: 1 1 auto !important;
            min-width: 0 !important;
            width: auto !important;
        }
        [data-testid="stHorizontalBlock"]:has(#where-fit) > [data-testid="stColumn"]:has([data-testid="stPopover"]) {
            flex: 0 0 auto !important;
            min-width: 0 !important;
            width: auto !important;
        }
        [data-testid="stHorizontalBlock"]:has(#where-fit) [data-testid="stPopover"] {
            position: relative;
        }
        [data-testid="stHorizontalBlock"]:has(#where-fit) [data-testid="stPopoverButton"] {
            min-height: 2.1rem !important;
            padding: .25rem .45rem !important;
            border-color: var(--wf-line) !important;
            color: var(--wf-muted) !important;
            font-size: .82rem !important;
            font-weight: 800 !important;
            white-space: nowrap;
        }
        [data-testid="stHorizontalBlock"]:has(#where-fit) [data-testid="stPopover"] > .st-ae {
            z-index: 1000;
        }
        @media (min-width: 901px) {
            [data-testid="stHorizontalBlock"]:has(#where-fit) [data-testid="stPopover"]:hover > .st-ae,
            [data-testid="stHorizontalBlock"]:has(#where-fit) [data-testid="stPopover"]:focus-within > .st-ae {
                position: absolute !important;
                top: calc(100% + .2rem);
                right: 0;
                display: block !important;
                min-width: 7.3rem;
                padding: .35rem;
                border: 1px solid var(--wf-line);
                border-radius: 8px;
                background: var(--wf-panel);
                box-shadow: 0 8px 20px rgba(15, 23, 42, .14);
            }
        }
        @media (max-width: 900px) {
            [data-testid="stHorizontalBlock"]:has(#where-fit) {
                gap: .25rem !important;
            }
            [data-testid="stHorizontalBlock"]:has(#where-fit) h1 {
                font-size: 1.7rem !important;
                line-height: 1.15 !important;
            }
            [data-testid="stHorizontalBlock"]:has(#where-fit) [data-testid="stPopoverButton"] {
                min-height: 1.9rem !important;
                padding: .18rem .28rem !important;
                border: 0 !important;
                background: transparent !important;
            }
        }
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--wf-line);
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarHeader"],
        section[data-testid="stSidebar"] [data-testid="stLogoSpacer"],
        section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
            display: none !important;
            height: 0 !important;
            min-height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] > div,
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] > div > div,
        section[data-testid="stSidebar"] > div:first-child {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            padding-top: 0 !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .wf-sidebar-top-spacer {
            height: 2.5rem;
        }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: var(--wf-text) !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] span,
        section[data-testid="stSidebar"] [role="slider"] span {
            color: inherit !important;
        }
        [data-testid="stMetricValue"] {font-size: 1.45rem;}
        h1 {font-size: 2.15rem !important; letter-spacing: -.02em !important; margin-bottom: .15rem !important;}
        h2, h3 {letter-spacing: 0 !important;}
        [data-testid="stCaptionContainer"] {
            color: var(--wf-muted) !important;
            font-size: .82rem;
            line-height: 1.45;
        }
        section[data-testid="stSidebar"] h2 {
            padding-top: 0 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stHeading"] {
            height: auto !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] > div {
            min-height: 2.35rem;
            border: 1px solid var(--wf-line) !important;
            border-radius: 8px !important;
            background: var(--wf-panel) !important;
            box-shadow: 0 1px 2px rgba(15, 23, 42, .05);
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] > div:hover {
            border-color: var(--wf-slate) !important;
        }
        div[data-testid="stHorizontalBlock"] [role="radiogroup"] {
            gap: .35rem;
        }
        div[data-testid="stHorizontalBlock"] [role="radiogroup"] label {
            background: #f8fafc;
            border: 1px solid var(--wf-line);
            border-radius: 999px;
            padding: .2rem .55rem;
            min-height: 2rem;
        }
        div[data-testid="stHorizontalBlock"] [role="radiogroup"] label:has(input:checked) {
            border-color: #c4b5fd;
            background: #f5f3ff;
        }
        label[data-baseweb="radio"]:has(input:checked) > div:first-child {
            background: #a78bfa !important;
            border-color: #a78bfa !important;
        }
        label[data-baseweb="radio"]:has(input:checked) > div:first-child > div {
            background: #ffffff !important;
        }
        [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
            background: #ede9fe !important;
        }
        [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child > div,
        [data-testid="stSlider"] [data-baseweb="slider"] > div:nth-child(2) {
            background: #a78bfa !important;
        }
        [data-testid="stSlider"] [role="slider"] {
            background: #a78bfa !important;
            border-color: #8b5cf6 !important;
            box-shadow: 0 0 0 2px #f5f3ff !important;
        }
        [data-testid="stSlider"] [role="slider"]:focus-visible {
            box-shadow: 0 0 0 3px #ddd6fe !important;
        }
        div[data-testid="stAlert"] {
            background: #f5f3ff !important;
            border-color: #c4b5fd !important;
            color: #4c1d95 !important;
        }
        div[data-testid="stAlert"] * {
            color: #4c1d95 !important;
            fill: #8b5cf6 !important;
        }
        button[data-testid="stBaseButton-primary"] {
            background: var(--wf-purple) !important;
            border-color: var(--wf-purple) !important;
            color: #33275c !important;
        }
        button[data-testid="stBaseButton-primary"]:hover {
            background: #c4b5fd !important;
            border-color: #c4b5fd !important;
            color: #33275c !important;
        }
        div[data-testid="stTabs"] [role="tablist"] {
            gap: .25rem;
            border-bottom: 1px solid var(--wf-line);
        }
        div[data-testid="stTabs"] [role="tab"] {
            border-radius: 6px 6px 0 0;
            padding: .55rem .9rem;
        }
        .wf-sidebar-section {
            margin: 1rem 0 .4rem;
            padding-top: .8rem;
            border-top: 1px solid var(--wf-line);
            color: var(--wf-slate);
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .02em;
            text-transform: uppercase;
        }
        .wf-decision {
            display: grid;
            grid-template-columns: minmax(280px, 1.25fr) minmax(360px, 1fr);
            gap: 1rem;
            background: var(--wf-panel);
            border: 1px solid var(--wf-line);
            border-radius: 8px;
            padding: 1rem;
            margin: .6rem 0 1rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, .06);
        }
        .wf-eyebrow {
            color: var(--wf-muted);
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .02em;
        }
        .wf-city-title {
            color: var(--wf-text);
            font-size: 2rem;
            line-height: 1.15;
            font-weight: 850;
            margin-top: .2rem;
        }
        .wf-city-subtitle {
            color: var(--wf-slate);
            margin: .45rem 0 .6rem;
            line-height: 1.55;
        }
        .wf-decision-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: .65rem;
        }
        .wf-stat {
            border: 1px solid var(--wf-line);
            border-left-width: 4px;
            border-radius: 8px;
            padding: .75rem .8rem;
            background: #fbfcfe;
        }
        .wf-stat-teal {border-left-color: var(--wf-purple);}
        .wf-stat-blue {border-left-color: var(--wf-indigo);}
        .wf-stat-amber {border-left-color: var(--wf-amber);}
        .wf-stat-slate {border-left-color: var(--wf-slate);}
        .wf-stat-label {color: var(--wf-muted); font-size: .76rem; font-weight: 700;}
        .wf-stat-value {color: var(--wf-text); font-size: 1.32rem; font-weight: 820; line-height: 1.25; margin-top: .18rem;}
        .wf-stat-detail {color: var(--wf-muted); font-size: .78rem; margin-top: .2rem;}
        .wf-watchout {
            grid-column: 1 / -1;
            border-top: 1px solid var(--wf-line);
            padding-top: .75rem;
            color: var(--wf-slate);
            font-size: .9rem;
        }
        .wf-watchout span {
            display: inline-block;
            margin-right: .5rem;
            color: var(--wf-purple);
            font-weight: 800;
        }
        .wf-callout {
            border: 1px solid #bfdbfe;
            background: #eff6ff;
            color: #1e3a8a;
            border-radius: 8px;
            padding: .85rem 1rem;
            margin-bottom: .8rem;
        }
        .wf-callout span {font-weight: 850; margin-right: .5rem;}
        .wf-callout p {display: inline; margin: 0;}
        .wf-candidate-panel {
            background: var(--wf-panel);
            border: 1px solid var(--wf-line);
            border-radius: 8px;
            margin: .8rem 0 1.1rem;
            overflow: hidden;
            box-shadow: 0 8px 24px rgba(15, 23, 42, .05);
        }
        .wf-candidate-head {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: center;
            padding: .95rem 1rem .7rem;
            border-bottom: 1px solid var(--wf-line);
        }
        .wf-candidate-title {
            color: var(--wf-text);
            font-size: 1.15rem;
            font-weight: 850;
            margin-top: .15rem;
        }
        .wf-candidate-count {
            color: var(--wf-purple);
            background: #f5f3ff;
            border: 1px solid #ddd6fe;
            border-radius: 999px;
            padding: .28rem .65rem;
            font-weight: 800;
            white-space: nowrap;
        }
        .wf-candidate-note {
            color: var(--wf-muted);
            padding: .65rem 1rem;
            border-bottom: 1px solid var(--wf-line);
            font-size: .88rem;
        }
        .wf-table-wrap {
            width: 100%;
            overflow-x: auto;
        }
        .wf-table {
            width: 100%;
            border-collapse: collapse;
            font-size: .9rem;
            color: var(--wf-text);
        }
        .wf-table th {
            background: #f8fafc;
            color: var(--wf-slate);
            font-weight: 800;
            text-align: left;
            padding: .62rem .75rem;
            border-bottom: 1px solid var(--wf-line);
            white-space: nowrap;
        }
        .wf-table td {
            padding: .62rem .75rem;
            border-bottom: 1px solid #edf1f6;
            white-space: nowrap;
        }
        .wf-table tr:last-child td {border-bottom: 0;}
        .wf-table tr:hover td {background: #f9fafb;}
        .wf-city-card {
            background: var(--wf-panel);
            border: 1px solid var(--wf-line);
            border-radius: 8px;
            padding: 1rem;
            margin: .8rem 0;
        }
        .wf-city-card-head {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
        }
        .wf-card-rank {
            color: var(--wf-muted);
            font-weight: 800;
            font-size: .78rem;
        }
        .wf-city-card h3 {
            margin: .1rem 0 .15rem !important;
            font-size: 1.35rem !important;
        }
        .wf-city-card p {
            color: var(--wf-muted);
            margin: 0;
        }
        .wf-score-pill {
            border: 1px solid;
            border-radius: 999px;
            padding: .32rem .65rem;
            white-space: nowrap;
            font-weight: 800;
            background: #fff;
        }
        .wf-score-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .75rem;
            margin: .9rem 0;
        }
        .wf-score-row-top {
            display: flex;
            justify-content: space-between;
            gap: .5rem;
            color: var(--wf-slate);
            font-size: .82rem;
            margin-bottom: .25rem;
        }
        .wf-score-track {
            height: .45rem;
            background: #edf1f6;
            border-radius: 999px;
            overflow: hidden;
        }
        .wf-score-fill {height: 100%; border-radius: 999px;}
        .wf-score-teal {background: var(--wf-purple);}
        .wf-score-blue {background: var(--wf-indigo);}
        .wf-score-amber {background: var(--wf-amber);}
        .wf-score-violet {background: var(--wf-purple);}
        .wf-card-body {
            color: var(--wf-text);
            line-height: 1.6;
            margin-top: .2rem;
        }
        .wf-chip-row {display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .65rem;}
        .wf-chip, .wf-status {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: .18rem .5rem;
            font-size: .76rem;
            font-weight: 750;
            line-height: 1.3;
        }
        .wf-chip-good {background: #ecfdf3; color: #027a48 !important;}
        .wf-chip-warn {background: #fffaeb; color: #b54708 !important;}
        .wf-status-good {background: #ecfdf3; color: #027a48 !important;}
        .wf-status-warn {background: #fffaeb; color: #b54708 !important;}
        .wf-status-muted {background: #f2f4f7; color: #475467 !important;}
        .wf-status-bad {background: #f7edff; color: #7e22ce !important;}
        @media (max-width: 900px) {
            .wf-decision {grid-template-columns: 1fr;}
            .wf-decision-grid, .wf-score-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}
            .wf-city-card-head {flex-direction: column;}
        }
        .wf-inline-preference-marker {display: none;}
        @media (min-width: 901px) {
            [data-testid="stExpander"]:has(.wf-inline-preference-marker) {
                display: none !important;
            }
        }
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stHeader"],
        [data-testid="stStatusWidget"],
        header,
        #MainMenu,
        footer {
            display: none !important;
        }
        .mapboxgl-ctrl-attrib,
        .mapboxgl-ctrl-logo,
        [class*="mapboxgl-ctrl-attrib"],
        [class*="mapboxgl-ctrl-logo"] {
            display: none !important;
        }
        [data-testid="stMarkdownContainer"],
        [data-testid="stAlert"],
        [data-testid="stExpander"],
        [data-testid="stText"],
        p, li, h1, h2, h3, h4, h5, h6 {
            -webkit-user-select: text !important;
            user-select: text !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(_theme_override_css(), unsafe_allow_html=True)


def _theme_override_css() -> str:
    mode = str(st.session_state.get("display_mode", "system"))
    if mode == "night":
        return _night_theme_css(wrap=True)
    if mode == "day":
        return _day_theme_css(wrap=True)
    return f"<style>@media (prefers-color-scheme: dark) {{{_night_theme_css(wrap=False)}}}</style>"


def _day_theme_css(wrap: bool) -> str:
    css = """
    :root {
        --wf-bg: #f7f8fa;
        --wf-panel: #ffffff;
        --wf-line: #d9dee7;
        --wf-text: #18212f;
        --wf-muted: #667085;
        --wf-slate: #475467;
        --wf-purple: #a78bfa;
        --wf-indigo: #0072b2;
        --wf-teal: #009e73;
        --wf-amber: #e69f00;
        --wf-magenta: #a78bfa;
    }
    """
    return f"<style>{css}</style>" if wrap else css


def _night_theme_css(wrap: bool) -> str:
    css = """
    :root {
        --wf-bg: #0f172a;
        --wf-panel: #111827;
        --wf-line: #334155;
        --wf-text: #e5e7eb;
        --wf-muted: #a7b0c0;
        --wf-slate: #cbd5e1;
        --wf-purple: #c4b5fd;
        --wf-indigo: #56b4e9;
        --wf-teal: #44aa99;
        --wf-amber: #f0e442;
        --wf-magenta: #c4b5fd;
    }
    .stApp {background: var(--wf-bg) !important;}
    section[data-testid="stSidebar"] {background: #0b1220 !important;}
    [data-testid="stVerticalBlockBorderWrapper"],
    .wf-candidate-panel,
    .wf-city-card,
    .wf-decision,
    .wf-stat {
        background: var(--wf-panel) !important;
        border-color: var(--wf-line) !important;
    }
    .wf-callout {
        background: #102a43 !important;
        border-color: #1d4ed8 !important;
        color: #dbeafe !important;
    }
    .wf-table th {
        background: #172033 !important;
        color: var(--wf-slate) !important;
    }
    .wf-table td {
        border-bottom-color: #253247 !important;
    }
    .wf-table tr:hover td {background: #162033 !important;}
    .wf-candidate-note,
    .wf-candidate-head {
        border-color: var(--wf-line) !important;
    }
    div[data-testid="stHorizontalBlock"] [role="radiogroup"] label {
        background: #111827 !important;
        border-color: #334155 !important;
    }
    div[data-testid="stHorizontalBlock"] [role="radiogroup"] label:has(input:checked) {
        background: #33275c !important;
        border-color: var(--wf-purple) !important;
    }
    label[data-baseweb="radio"]:has(input:checked) > div:first-child {
        background: var(--wf-purple) !important;
        border-color: var(--wf-purple) !important;
    }
    label[data-baseweb="radio"]:has(input:checked) > div:first-child > div {
        background: #111827 !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
        background: #30264f !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child > div,
    [data-testid="stSlider"] [data-baseweb="slider"] > div:nth-child(2) {
        background: var(--wf-purple) !important;
    }
    [data-testid="stSlider"] [role="slider"] {
        background: var(--wf-purple) !important;
        border-color: #ddd6fe !important;
        box-shadow: 0 0 0 2px #111827 !important;
    }
    [data-testid="stSlider"] [role="slider"]:focus-visible {
        box-shadow: 0 0 0 3px #4c1d95 !important;
    }
    div[data-testid="stAlert"] {
        background: #2f2554 !important;
        border-color: var(--wf-purple) !important;
        color: #ede9fe !important;
    }
    div[data-testid="stAlert"] * {
        color: #ede9fe !important;
        fill: var(--wf-purple) !important;
    }
    button[data-testid="stBaseButton-primary"]:hover {
        background: #c4b5fd !important;
        border-color: #c4b5fd !important;
    }
    [data-baseweb="select"] > div,
    input,
    textarea {
        background: #111827 !important;
        color: var(--wf-text) !important;
        border-color: var(--wf-line) !important;
    }
    """
    return f"<style>{css}</style>" if wrap else css


if __name__ == "__main__":
    main()
