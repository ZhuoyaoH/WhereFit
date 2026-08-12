"""Pydeck map builder."""

from __future__ import annotations

import math
from urllib.parse import quote

import pandas as pd
import pydeck as pdk

from wherefit.models import CityResult


AMAP_TILE_URL = "https://webrd04.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}"
AMAP_ATTRIBUTION = "@高德"
AMAP_MAP_STYLE = {
    "version": 8,
    "sources": {
        "amap": {
            "type": "raster",
            "tiles": [AMAP_TILE_URL],
            "tileSize": 256,
            "attribution": AMAP_ATTRIBUTION,
        }
    },
    "layers": [{"id": "amap-raster", "type": "raster", "source": "amap"}],
}
AMAP_MAP_STYLE_URL = "data:application/json;charset=utf-8," + quote(str(AMAP_MAP_STYLE).replace("'", '"'))
GCJ_A = 6378245.0
GCJ_EE = 0.00669342162296594323


def make_map(results: list[CityResult]) -> pdk.Deck:
    positions = [_display_position(item.location.latitude, item.location.longitude, item.location.country) for item in results]
    data = pd.DataFrame(
        {
            "city": [item.location.city for item in results],
            "country": [item.location.country for item in results],
            "latitude": [lat for lat, _ in positions],
            "longitude": [lon for _, lon in positions],
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
        map_style=AMAP_MAP_STYLE_URL,
        map_provider="mapbox",
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=3.1),
        layers=[layer],
        tooltip={
            "html": f"<b>{{city}}</b><br/>你的偏好匹配分: {{score}}<br/>糟糕天气指数: {{risk}}<br/><span style='font-size:11px'>{AMAP_ATTRIBUTION}</span>",
            "style": {"backgroundColor": "#111827", "color": "white"},
        },
    )


def make_earthquake_map(location, events: list[dict[str, object]]) -> pdk.Deck:
    city_lat, city_lon = _display_position(location.latitude, location.longitude, location.country)
    event_rows = _display_event_rows(events, location.country)
    city_data = pd.DataFrame(
        {
            "city": [location.city],
            "latitude": [city_lat],
            "longitude": [city_lon],
            "kind": ["城市"],
        }
    )
    layers = []
    if not event_rows.empty:
        event_rows["radius"] = event_rows["magnitude"].apply(lambda value: 18000 + float(value) * 5500)
        event_rows["color"] = event_rows["magnitude"].apply(_magnitude_color)
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=event_rows,
                get_position="[longitude, latitude]",
                get_fill_color="color",
                get_radius="radius",
                pickable=True,
                opacity=0.72,
            )
        )
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=city_data,
            get_position="[longitude, latitude]",
            get_fill_color=[37, 99, 235, 235],
            get_radius=42000,
            pickable=True,
            opacity=0.9,
        )
    )
    return pdk.Deck(
        map_style=AMAP_MAP_STYLE_URL,
        map_provider="mapbox",
        initial_view_state=pdk.ViewState(latitude=city_lat, longitude=city_lon, zoom=5.2),
        layers=layers,
        tooltip={
            "html": "<b>{city}{place}</b><br/>震级: {magnitude}<br/>日期: {date}<br/>距离: {distance_km} km",
            "style": {"backgroundColor": "#111827", "color": "white"},
        },
    )


def make_typhoon_track_map(location, track_points: list[dict[str, object]]) -> pdk.Deck:
    city_lat, city_lon = _display_position(location.latitude, location.longitude, location.country)
    track_data = _display_track_rows(track_points, location.country)
    city_data = pd.DataFrame(
        {
            "city": [location.city],
            "latitude": [city_lat],
            "longitude": [city_lon],
            "name": ["城市"],
            "time": [""],
            "distance_km": [0],
        }
    )
    layers = []
    if not track_data.empty:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=_track_paths(track_data),
                get_path="path",
                get_color=[14, 116, 144, 155],
                width_min_pixels=2,
                pickable=True,
            )
        )
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=track_data,
                get_position="[longitude, latitude]",
                get_fill_color="[245, 158, 11, 170]",
                get_radius=26000,
                pickable=True,
                opacity=0.72,
            )
        )
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=city_data,
            get_position="[longitude, latitude]",
            get_fill_color=[37, 99, 235, 235],
            get_radius=42000,
            pickable=True,
            opacity=0.9,
        )
    )
    return pdk.Deck(
        map_style=AMAP_MAP_STYLE_URL,
        map_provider="mapbox",
        initial_view_state=pdk.ViewState(latitude=city_lat, longitude=city_lon, zoom=4.5),
        layers=layers,
        tooltip={
            "html": "<b>{name}</b><br/>时间: {time}<br/>距离: {distance_km} km<br/>风速: {wind}",
            "style": {"backgroundColor": "#111827", "color": "white"},
        },
    )


def _score_color(score: float) -> list[int]:
    if score >= 75:
        return [22, 163, 74, 210]
    if score >= 60:
        return [202, 138, 4, 210]
    return [220, 38, 38, 210]


def _display_event_rows(events: list[dict[str, object]], country: str) -> pd.DataFrame:
    rows = []
    for event in events:
        if event.get("latitude") is None or event.get("longitude") is None:
            continue
        lat, lon = _display_position(float(event["latitude"]), float(event["longitude"]), country)
        rows.append(
            {
                **event,
                "latitude": lat,
                "longitude": lon,
                "city": "",
                "place": str(event.get("place") or "地震事件"),
                "magnitude": float(event.get("magnitude") or 0.0),
                "distance_km": event.get("distance_km") or "",
            }
        )
    return pd.DataFrame(rows)


def _display_track_rows(track_points: list[dict[str, object]], country: str) -> pd.DataFrame:
    rows = []
    for point in track_points:
        if point.get("latitude") is None or point.get("longitude") is None:
            continue
        lat, lon = _display_position(float(point["latitude"]), float(point["longitude"]), country)
        rows.append(
            {
                **point,
                "latitude": lat,
                "longitude": lon,
                "name": point.get("name") or "未命名",
                "distance_km": point.get("distance_km") or "",
                "wind": point.get("wind") or "",
            }
        )
    return pd.DataFrame(rows)


def _track_paths(track_data: pd.DataFrame) -> pd.DataFrame:
    paths = []
    for _, group in track_data.groupby("sid", dropna=False):
        path = group[["longitude", "latitude"]].values.tolist()
        if len(path) >= 2:
            paths.append({"path": path, "name": group.iloc[0].get("name", "台风路径")})
    return pd.DataFrame(paths)


def _magnitude_color(value: float) -> list[int]:
    if value >= 6:
        return [220, 38, 38, 190]
    if value >= 5:
        return [234, 88, 12, 185]
    return [202, 138, 4, 175]


def _display_position(latitude: float, longitude: float, country: str) -> tuple[float, float]:
    if str(country).lower() == "china" and _inside_china(latitude, longitude):
        return _wgs84_to_gcj02(latitude, longitude)
    return latitude, longitude


def _wgs84_to_gcj02(latitude: float, longitude: float) -> tuple[float, float]:
    dlat = _transform_lat(longitude - 105.0, latitude - 35.0)
    dlon = _transform_lon(longitude - 105.0, latitude - 35.0)
    radlat = latitude / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - GCJ_EE * magic * magic
    sqrt_magic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((GCJ_A * (1 - GCJ_EE)) / (magic * sqrt_magic) * math.pi)
    dlon = (dlon * 180.0) / (GCJ_A / sqrt_magic * math.cos(radlat) * math.pi)
    return latitude + dlat, longitude + dlon


def _inside_china(latitude: float, longitude: float) -> bool:
    return 3.0 <= latitude <= 54.0 and 73.0 <= longitude <= 136.0


def _transform_lat(x: float, y: float) -> float:
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lon(x: float, y: float) -> float:
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret
