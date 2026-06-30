"""Build hazard summaries from available providers and proxies."""

from __future__ import annotations

from pathlib import Path

from wherefit.hazards.earthquake import get_earthquake_summary
from wherefit.hazards.typhoon import get_typhoon_summary
from wherefit.models import EarthquakeSummary, HazardSummary, Location, ClimateMetrics


def build_hazard_summary(
    location: Location,
    metrics: ClimateMetrics,
    cache_dir: Path,
    force_refresh: bool = False,
    include_earthquake: bool = False,
    include_typhoon: bool = False,
) -> HazardSummary:
    if include_earthquake:
        earthquake = get_earthquake_summary(
            location,
            cache_dir=cache_dir / "earthquake",
            force_refresh=force_refresh,
        )
    else:
        earthquake = EarthquakeSummary(0, 0, 0, None, None, None, "USGS Earthquake Catalog API", "not_requested")
    typhoon = None
    if include_typhoon:
        typhoon = get_typhoon_summary(
            location,
            cache_dir=cache_dir / "typhoon",
            force_refresh=force_refresh,
        )
    return HazardSummary(
        city=location.city,
        earthquake=earthquake,
        typhoon_note=_typhoon_note(location),
        rainfall_extreme_note=_rainfall_note(metrics),
        landslide_note=_landslide_note(location, metrics),
        typhoon=typhoon,
        source_notes=[
            "地震：USGS Earthquake Catalog API，按城市周边半径查询历史事件。",
            "台风：IBTrACS Western Pacific，按城市到路径点距离统计 100/200/500km 接近次数。",
            "洪涝/滑坡：当前版本先用强降水和地形区域 proxy，后续接事件库。",
        ],
    )


def _typhoon_note(location: Location) -> str:
    if location.typhoon_region and location.coastal:
        return "沿海且位于台风影响区，后续应接 IBTrACS 统计 100/200/500km 台风路径接近次数。"
    if location.typhoon_region:
        return "可能受台风残余降水影响，直接路径影响需用 IBTrACS 数据复核。"
    return "直接台风路径影响 proxy 较低。"


def _rainfall_note(metrics: ClimateMetrics) -> str:
    if metrics.precipitation_extreme_days >= 1 or metrics.heavy_rain_days >= 4:
        return "历史/静态指标显示强降水暴露偏高，需结合流域、地形和排水能力判断洪涝。"
    if metrics.heavy_rain_days >= 2:
        return "存在一定强降水暴露，当前仅为天气 proxy。"
    return "强降水 proxy 相对较低。"


def _landslide_note(location: Location, metrics: ClimateMetrics) -> str:
    mountainous = location.region_type in {"southwest_mountain", "southwest_plateau", "plateau"}
    if mountainous and metrics.precipitation_extreme_days >= 0.5:
        return "山地/高原区域叠加强降水 proxy，后续应接滑坡事件库和 DEM 坡度数据。"
    if mountainous:
        return "地形 proxy 提示需关注山区滑坡，但当前不代表真实滑坡风险。"
    return "当前地形 proxy 未显示突出滑坡暴露。"
