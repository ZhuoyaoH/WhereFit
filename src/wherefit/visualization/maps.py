"""Pydeck map builder."""

from __future__ import annotations

import pandas as pd
import pydeck as pdk

from wherefit.models import CityResult


def make_map(results: list[CityResult]) -> pdk.Deck:
    data = pd.DataFrame(
        {
            "city": [item.location.city for item in results],
            "country": [item.location.country for item in results],
            "latitude": [item.location.latitude for item in results],
            "longitude": [item.location.longitude for item in results],
            "score": [item.score.personal_fit_score for item in results],
            "risk": [item.score.long_term_risk_score for item in results],
        }
    )
    data["color"] = data["score"].apply(_score_color)
    center_lat = float(data["latitude"].mean()) if not data.empty else 35.0
    center_lon = float(data["longitude"].mean()) if not data.empty else 120.0
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position="[longitude, latitude]",
        get_fill_color="color",
        get_radius=65000,
        pickable=True,
        opacity=0.82,
    )
    return pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=3.1),
        layers=[layer],
        tooltip={
            "html": "<b>{city}</b><br/>匹配度: {score}<br/>长期风险: {risk}",
            "style": {"backgroundColor": "#111827", "color": "white"},
        },
    )


def _score_color(score: float) -> list[int]:
    if score >= 75:
        return [22, 163, 74, 210]
    if score >= 60:
        return [202, 138, 4, 210]
    return [220, 38, 38, 210]
