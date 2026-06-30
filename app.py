from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wherefit.config import DISCLAIMER, HAZARD_DISCLAIMER, HISTORY_START_DATE, MODE_LABELS
from wherefit.data_loader import load_seed_cities, match_cities, parse_city_input, row_to_location, row_to_metrics
from wherefit.data_sources.open_meteo_forecast import default_forecast_dates, get_forecast_summary
from wherefit.data_sources.open_meteo_history import default_history_end_date, get_history_metrics
from wherefit.hazards.aurora import build_aurora_summary
from wherefit.hazards.summary import build_hazard_summary
from wherefit.models import CityResult, ForecastSummary, HazardSummary, TyphoonSummary, UserPreference
from wherefit.report.template_report import generate_city_report, generate_comparison_report
from wherefit.scoring.overall import evaluate_city, rank_cities
from wherefit.visualization.cards import risk_label, score_label
from wherefit.visualization.charts import make_radar_chart, make_ranking_bar_chart
from wherefit.visualization.maps import make_map


DATA_PATH = ROOT / "data" / "city_seed.csv"
HISTORY_CACHE_DIR = ROOT / "data" / "cache" / "weather" / "history"
FORECAST_CACHE_DIR = ROOT / "data" / "cache" / "weather" / "forecast"
HAZARD_CACHE_DIR = ROOT / "data" / "cache" / "hazards"
AURORA_CACHE_DIR = ROOT / "data" / "cache" / "aurora"
DEFAULT_CITIES = "北京, 上海, 广州, 成都, 昆明, 哈尔滨, 青岛"
DATA_MODE_STATIC = "Demo 静态数据"
DATA_MODE_HISTORY = "历史气候常态（2000 至今）"


def main() -> None:
    st.set_page_config(page_title="WhereFit AI", page_icon="🌏", layout="wide")
    _inject_css()

    st.title("WhereFit AI")
    st.caption("用气候偏好、历史天气和历史灾害档案比较国内城市。")

    pref, data_mode, history_start, history_end, force_refresh = _sidebar_inputs()
    city_input = st.text_input("候选城市", value=DEFAULT_CITIES, help="支持中文、英文、英文逗号、中文逗号和换行。")
    run_clicked = st.button("开始比较", type="primary")

    if not city_input.strip():
        st.info("请输入至少一个城市，例如：北京, 上海, 广州, 成都, 昆明")
        _method_section()
        st.warning(DISCLAIMER)
        return

    if run_clicked or city_input.strip():
        results, missing, messages = _evaluate(city_input, pref, data_mode, history_start, history_end, force_refresh)
        if missing:
            st.warning(f"这些城市暂未收录：{', '.join(missing)}。v01 先聚焦国内主要城市。")
        for message in messages:
            st.caption(message)
        if not results:
            st.error("没有匹配到可评分城市。请尝试输入北京、上海、广州、成都、昆明、青岛等。")
            return
        _render_results(results, pref, data_mode)

    _method_section()
    st.warning(DISCLAIMER)


def _sidebar_inputs() -> tuple[UserPreference, str, str, str, bool]:
    with st.sidebar:
        st.header("你的气候偏好")
        data_mode = st.radio("数据模式", [DATA_MODE_STATIC, DATA_MODE_HISTORY], horizontal=False)
        mode_cn = st.radio("使用场景", ["旅行", "长期居住", "城市对比"], horizontal=False)
        mode = {"旅行": "Travel", "长期居住": "Living", "城市对比": "Compare"}[mode_cn]
        month = st.selectbox("目标月份", list(range(1, 13)), index=6, format_func=lambda m: f"{m} 月")

        history_start = HISTORY_START_DATE
        history_end = default_history_end_date()
        force_refresh = False
        if data_mode == DATA_MODE_HISTORY:
            with st.expander("历史数据设置", expanded=True):
                history_start = st.text_input("历史开始日期", HISTORY_START_DATE)
                history_end = st.text_input("历史结束日期", default_history_end_date())
                force_refresh = st.checkbox("刷新 API 数据，不读缓存", value=False)
                st.caption("默认使用 2000 年至最近可用完整日；首次会按 5 年分块下载并缓存，失败分块会跳过，完全失败才回退静态数据。")

        heat = st.slider("怕热程度", 0, 5, 3)
        cold = st.slider("怕冷程度", 0, 5, 2)
        humidity = st.slider("讨厌潮湿程度", 0, 5, 3)
        rain = st.slider("讨厌下雨程度", 0, 5, 3)
        air = st.slider("关注空气质量程度", 0, 5, 3)
        extreme = st.slider("关注极端天气程度", 0, 5, 3)
        st.caption("0 表示不敏感，5 表示非常敏感。")
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
    return pref, data_mode, history_start, history_end, force_refresh


@st.cache_data
def _load_data() -> pd.DataFrame:
    return load_seed_cities(DATA_PATH)


def _evaluate(
    city_input: str,
    pref: UserPreference,
    data_mode: str = DATA_MODE_STATIC,
    history_start: str = HISTORY_START_DATE,
    history_end: str | None = None,
    force_refresh: bool = False,
) -> tuple[list[CityResult], list[str], list[str]]:
    data = _load_data()
    requested = parse_city_input(city_input)
    matched, missing = match_cities(data, requested)
    results: list[CityResult] = []
    messages: list[str] = []
    for _, row in matched.iterrows():
        location = row_to_location(row)
        fallback_metrics = row_to_metrics(row, pref.month)
        metrics = fallback_metrics
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
                metrics = replace(history.metrics, data_status=history.status)
        results.append(evaluate_city(location, metrics, pref))
    return rank_cities(results), missing, messages


def _render_results(results: list[CityResult], pref: UserPreference, data_mode: str) -> None:
    best = results[0]
    overview_tab, history_tab, forecast_tab, hazard_tab, aurora_tab, report_tab = st.tabs(
        ["城市适配", "历史天气", "未来预报", "历史灾害", "极光机会", "解释报告"]
    )
    with overview_tab:
        _render_overview(results, pref, data_mode)
    with history_tab:
        _render_history_tab(results)
    with forecast_tab:
        _render_forecast_tab(results)
    with hazard_tab:
        _render_hazard_tab(results)
    with aurora_tab:
        _render_aurora_tab(results)
    with report_tab:
        st.subheader("中文解释报告")
        st.success(generate_comparison_report(results, pref))
        for item in results:
            with st.expander(f"{item.location.city} 解释", expanded=item == best):
                st.write(generate_city_report(item, pref))
                st.caption("提示：" + "；".join(item.score.warnings))


def _render_overview(results: list[CityResult], pref: UserPreference, data_mode: str) -> None:
    best = results[0]
    st.subheader("推荐总结")
    st.success(generate_comparison_report(results, pref))

    top_cols = st.columns(4)
    top_cols[0].metric("最佳城市", best.location.city)
    top_cols[1].metric("个性化匹配度", f"{best.score.personal_fit_score:.0f}/100")
    top_cols[2].metric("当前场景", MODE_LABELS.get(pref.mode, "城市对比"))
    top_cols[3].metric("数据模式", "历史" if data_mode == DATA_MODE_HISTORY else "静态")

    st.subheader("城市排名")
    st.dataframe(_ranking_table(results), hide_index=True, use_container_width=True)

    chart_col, radar_col = st.columns([1.25, 1])
    with chart_col:
        st.plotly_chart(make_ranking_bar_chart(results), use_container_width=True)
    with radar_col:
        selected_city = st.selectbox("查看城市画像", [item.location.city for item in results])
        selected = next(item for item in results if item.location.city == selected_city)
        st.plotly_chart(make_radar_chart(selected), use_container_width=True)

    st.subheader("地图")
    st.pydeck_chart(make_map(results), use_container_width=True)

    st.subheader("城市卡片")
    for index, item in enumerate(results, start=1):
        _city_card(index, item, pref)


def _render_history_tab(results: list[CityResult]) -> None:
    st.subheader("历史天气数据概览")
    rows = []
    for item in results:
        metrics = item.metrics
        rows.append(
            {
                "城市": item.location.city,
                "数据源": metrics.data_source,
                "状态": metrics.data_status,
                "样本年数": metrics.sample_years,
                "月均温": round(metrics.temperature_mean, 1),
                "体感温度": round(metrics.apparent_temperature, 1),
                "降水日/年": round(metrics.precipitation_days, 1),
                "强降水日/年": round(metrics.heavy_rain_days, 1),
                "极端降水日/年": round(metrics.precipitation_extreme_days, 1),
                "高温日/年": round(metrics.hot_days, 1),
                "缺失率": round(metrics.missing_rate, 3),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("历史天气模式使用 Open-Meteo Historical Weather API。首次请求会按 5 年分块缓存；状态为 partial 时表示部分年份请求失败但已用可用年份计算。")


def _render_forecast_tab(results: list[CityResult]) -> None:
    st.subheader("未来天气预报")
    start_default, end_default = default_forecast_dates()
    col_a, col_b, col_c = st.columns([1, 1, 1])
    start = col_a.date_input("出行开始日期", value=date.fromisoformat(start_default), key="forecast_start")
    end = col_b.date_input("出行结束日期", value=date.fromisoformat(end_default), key="forecast_end")
    force_refresh = col_c.checkbox("刷新预报缓存", value=False)
    st.caption("Open-Meteo Forecast 当前按 1-16 天出行窗口计算；预报越远，置信度越低。")
    if st.button("查询未来天气预报"):
        summaries = [
            get_forecast_summary(
                item.location,
                cache_dir=FORECAST_CACHE_DIR,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                force_refresh=force_refresh,
            )
            for item in results
        ]
        _render_forecast_summaries(summaries)
    else:
        st.info("点击“查询未来天气预报”后，会为当前候选城市请求或读取未来天气缓存。")


def _render_forecast_summaries(summaries: list[ForecastSummary]) -> None:
    table = pd.DataFrame(
        {
            "城市": [item.city for item in summaries],
            "状态": [item.status for item in summaries],
            "天数": [item.days for item in summaries],
            "最高温均值": [round(item.temp_max_mean, 1) for item in summaries],
            "最高体感均值": [round(item.apparent_temp_max_mean, 1) for item in summaries],
            "降水天数": [item.precipitation_days for item in summaries],
            "最大降水概率": [round(item.precipitation_probability_max, 1) for item in summaries],
            "强降水天数": [item.heavy_rain_days for item in summaries],
            "大风天数": [item.windy_days for item in summaries],
            "置信度": [round(item.confidence, 2) for item in summaries],
            "说明": [item.message for item in summaries],
        }
    )
    st.dataframe(table, hide_index=True, use_container_width=True)


def _render_hazard_tab(results: list[CityResult]) -> None:
    st.subheader("历史灾害档案")
    st.warning(HAZARD_DISCLAIMER)
    selected_city = st.selectbox("选择城市", [item.location.city for item in results], key="hazard_city")
    selected = next(item for item in results if item.location.city == selected_city)
    col_a, col_b = st.columns(2)
    query_earthquake = col_a.checkbox("查询/刷新 USGS 地震历史", value=False, help="按城市周边约 300km 查询 2000 年至今 M4+ 地震。")
    query_typhoon = col_b.checkbox("查询/刷新 IBTrACS 台风路径", value=False, help="下载或读取西北太平洋最佳路径数据，统计城市周边 100/200/500km 接近次数。")
    summary = build_hazard_summary(
        selected.location,
        selected.metrics,
        cache_dir=HAZARD_CACHE_DIR,
        force_refresh=query_earthquake or query_typhoon,
        include_earthquake=query_earthquake,
        include_typhoon=query_typhoon,
    )
    _render_hazard_summary(summary)


def _render_hazard_summary(summary: HazardSummary) -> None:
    eq = summary.earthquake
    cols = st.columns(4)
    cols[0].metric("M4+ 地震", eq.event_count_m4 if eq.status != "not_requested" else "未查询")
    cols[1].metric("M5+ 地震", eq.event_count_m5 if eq.status != "not_requested" else "未查询")
    cols[2].metric("M6+ 地震", eq.event_count_m6 if eq.status != "not_requested" else "未查询")
    cols[3].metric("最大震级", f"{eq.max_magnitude:.1f}" if eq.max_magnitude else "无数据")
    if eq.status == "not_requested":
        st.info("点击“查询/刷新 USGS 地震历史”后，会请求该城市周边历史地震记录并写入本地缓存。")
    elif eq.status == "failed":
        st.warning("USGS 地震数据请求失败；当前只展示静态灾害 proxy。")
    else:
        st.caption(f"地震数据源：{eq.source}；状态：{eq.status}；最近事件：{eq.latest_event_date or '无'}；最近距离：{eq.nearest_distance_km or '无'} km")
    if summary.typhoon is None:
        st.info("勾选“查询/刷新 IBTrACS 台风路径”后，会按城市周边 100/200/500km 统计 2000 年以来台风路径接近次数。")
    else:
        _render_typhoon_summary(summary.typhoon)
    st.markdown(f"**台风摘要：** {summary.typhoon_note}")
    st.markdown(f"**洪涝/强降水摘要：** {summary.rainfall_extreme_note}")
    st.markdown(f"**滑坡摘要：** {summary.landslide_note}")
    with st.expander("数据源说明"):
        for note in summary.source_notes:
            st.write("- " + note)


def _render_typhoon_summary(summary: TyphoonSummary) -> None:
    cols = st.columns(4)
    cols[0].metric("100km 内台风", summary.count_100km if summary.status != "failed" else "失败")
    cols[1].metric("200km 内台风", summary.count_200km if summary.status != "failed" else "失败")
    cols[2].metric("500km 内台风", summary.count_500km if summary.status != "failed" else "失败")
    cols[3].metric("最近路径点", f"{summary.nearest_distance_km:.0f} km" if summary.nearest_distance_km is not None else "无数据")
    if summary.status == "failed":
        st.warning(summary.message)
        return
    st.caption(
        "；".join(
            [
                f"台风数据源：{summary.source}",
                f"状态：{summary.status}",
                f"最强接近：{summary.strongest_name or '无'} {summary.strongest_year or ''}".strip(),
                f"最近年份：{summary.latest_nearby_name or '无'} {summary.latest_nearby_year or ''}".strip(),
            ]
        )
    )
    st.caption(summary.message)


def _render_aurora_tab(results: list[CityResult]) -> None:
    st.subheader("极光机会")
    col_a, col_b = st.columns(2)
    include_live = col_a.checkbox("查询 NOAA SWPC OVATION 短临预报", value=False, help="读取官方 30 分钟极光概率网格，按最近网格点估算当前机会。")
    force_refresh = col_b.checkbox("刷新极光缓存", value=False)
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
            "城市": [item.city for item in summaries],
            "机会等级": [item.opportunity_label for item in summaries],
            "机会分": [item.opportunity_score for item in summaries],
            "最近网格概率": [item.nearest_probability for item in summaries],
            "最近网格距离 km": [item.nearest_distance_km for item in summaries],
            "预报时间": [item.forecast_time for item in summaries],
            "说明": [item.explanation for item in summaries],
            "数据状态": [item.status for item in summaries],
        }
    ).sort_values("机会分", ascending=False)
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("未勾选短临预报时使用纬度启发式；勾选后优先读取 NOAA SWPC OVATION，失败则自动回退。后续还可叠加云量、夜间时段和光污染。")


def _ranking_table(results: list[CityResult]) -> pd.DataFrame:
    rows = []
    for index, item in enumerate(results, start=1):
        rows.append(
            {
                "排名": index,
                "城市": item.location.city,
                "省份/地区": item.location.province,
                "匹配度": item.score.personal_fit_score,
                "旅行舒适": item.score.travel_comfort_score,
                "长期风险": item.score.long_term_risk_score,
                "风险标签": risk_label(item.score.long_term_risk_score),
                "数据源": item.score.data_source,
                "状态": item.score.data_status,
                "主要理由": "、".join(item.score.strengths[:2]),
            }
        )
    return pd.DataFrame(rows)


def _city_card(index: int, item: CityResult, pref: UserPreference) -> None:
    label, color = score_label(item.score.personal_fit_score)
    with st.container(border=True):
        col_a, col_b, col_c, col_d = st.columns([1.2, 1, 1, 1])
        col_a.markdown(f"### #{index} {item.location.city}")
        col_a.caption(f"{item.location.province or item.location.country} | {item.score.data_status} | 置信度 {item.score.confidence:.0%}")
        col_b.metric("匹配度", f"{item.score.personal_fit_score:.0f}/100", label)
        col_c.metric("旅行舒适", f"{item.score.travel_comfort_score:.0f}/100")
        col_d.metric("长期风险", f"{item.score.long_term_risk_score:.0f}/100", risk_label(item.score.long_term_risk_score))
        st.markdown(f"<span style='color:{color}; font-weight:700'>{label}</span>", unsafe_allow_html=True)
        st.write(generate_city_report(item, pref))
        if item.score.weaknesses:
            st.caption("注意：" + "；".join(item.score.weaknesses[:3]))


def _method_section() -> None:
    with st.expander("方法说明"):
        st.markdown(
            """
            v01 增加了真实历史天气数据入口：历史气候常态模式会请求 Open-Meteo Historical Weather API，
            默认时间窗为 2000 年至最近可用完整日，并把结果缓存到 `data/cache/weather/history/`。

            - API 成功：使用历史天气聚合指标评分。
            - API 失败：自动回退到 `city_seed.csv` 静态等级数据。
            - 历史灾害页支持 USGS 地震查询和 IBTrACS 西北太平洋台风路径统计。
            - 未来预报页支持 Open-Meteo Forecast 1-16 天短期窗口。
            - 极光页支持 NOAA SWPC OVATION 短临预报；未请求或请求失败时使用纬度启发式。

            所有灾害和极光信息只用于历史记录与旅行兴趣展示，不构成安全评级或预测。
            """
        )


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        [data-testid="stMetricValue"] {font-size: 1.55rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
