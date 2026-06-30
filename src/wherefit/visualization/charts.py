"""Plotly chart builders."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from wherefit.models import CityResult


RADAR_KEYS = ["温度舒适", "湿度舒适", "降水友好", "空气质量", "高温安全", "强降水安全"]


def make_radar_chart(city_result: CityResult) -> go.Figure:
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
            theta=RADAR_KEYS + RADAR_KEYS[:1],
            fill="toself",
            name=city_result.location.city,
        )
    )
    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 100]}},
        showlegend=False,
        margin={"l": 30, "r": 30, "t": 30, "b": 30},
        height=360,
    )
    return fig


def make_ranking_bar_chart(results: list[CityResult]) -> go.Figure:
    data = pd.DataFrame(
        {
            "城市": [item.location.city for item in results],
            "个性化匹配度": [item.score.personal_fit_score for item in results],
            "旅行舒适度": [item.score.travel_comfort_score for item in results],
            "长期风险": [item.score.long_term_risk_score for item in results],
        }
    )
    fig = px.bar(
        data,
        x="城市",
        y=["个性化匹配度", "旅行舒适度", "长期风险"],
        barmode="group",
        range_y=[0, 100],
        color_discrete_sequence=["#2563eb", "#16a34a", "#dc2626"],
    )
    fig.update_layout(
        legend_title_text="指标",
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        height=380,
    )
    return fig
