"""Plotly chart builders."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from wherefit.data_loader import display_city_name
from wherefit.models import CityResult


RADAR_KEYS = ["温度舒适", "湿度舒适", "降水友好", "空气质量", "高温较少", "强降雨较少"]
RADAR_LABELS_EN = ["Temperature", "Humidity", "Rain", "Air quality", "Less heat", "Less heavy rain"]


def make_radar_chart(city_result: CityResult, lang: str = "zh") -> go.Figure:
    components = city_result.score.component_scores
    values = [
        components["温度舒适"],
        components["湿度舒适"],
        components["降水友好"],
        components["空气质量"],
        100 - components["高温风险"],
        100 - components["强降水风险"],
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + values[:1],
            theta=_radar_labels(lang) + _radar_labels(lang)[:1],
            mode="lines+markers",
            fill="toself",
            fillcolor="rgba(196, 181, 253, 0.50)",
            line={"color": "#7C3AED", "width": 2.5},
            marker={"color": "#6D28D9", "size": 7},
            name=display_city_name(city_result.location.city, city_result.location.city_en, lang),
        )
    )
    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 100]}},
        showlegend=False,
        margin={"l": 30, "r": 30, "t": 30, "b": 30},
        height=360,
    )
    return fig


def make_ranking_bar_chart(results: list[CityResult], lang: str = "zh") -> go.Figure:
    city_label = "City" if lang == "en" else "城市"
    fit_label = "Your preference match" if lang == "en" else "你的偏好匹配分"
    comfort_label = "Climate comfort" if lang == "en" else "气候舒适"
    exposure_fit_label = "Fewer bad-weather concerns" if lang == "en" else "糟糕天气更少"
    data = pd.DataFrame(
        {
            city_label: [_city_name(item, lang) for item in results],
            fit_label: [item.score.personal_fit_score for item in results],
            comfort_label: [item.score.travel_comfort_score for item in results],
            exposure_fit_label: [100.0 - item.score.long_term_risk_score for item in results],
        }
    )
    fig = px.bar(
        data,
        x=city_label,
        y=[fit_label, comfort_label, exposure_fit_label],
        barmode="group",
        range_y=[0, 100],
        color_discrete_sequence=["#0072B2", "#009E73", "#CC79A7"],
    )
    fig.update_layout(
        legend_title_text="Metric" if lang == "en" else "指标",
        yaxis_title="Score (higher is better)" if lang == "en" else "分数（越高越匹配）",
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        height=380,
    )
    return fig


def _radar_labels(lang: str) -> list[str]:
    return RADAR_LABELS_EN if lang == "en" else RADAR_KEYS


def _city_name(item: CityResult, lang: str) -> str:
    return display_city_name(item.location.city, item.location.city_en, lang)
