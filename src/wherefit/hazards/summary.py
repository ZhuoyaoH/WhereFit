"""Build hazard summaries from available providers and proxies."""

from __future__ import annotations

from pathlib import Path

from wherefit.hazards.earthquake import get_earthquake_summary
from wherefit.hazards.hydro_events import get_eonet_events, get_flood_discharge_record
from wherefit.hazards.typhoon import get_typhoon_summary
from wherefit.models import EarthquakeSummary, HazardSummary, Location, ClimateMetrics


def build_hazard_summary(
    location: Location,
    metrics: ClimateMetrics,
    cache_dir: Path,
    force_refresh: bool = False,
    include_earthquake: bool = False,
    include_typhoon: bool = False,
    include_hydro_events: bool = False,
    refresh_hydro_events: bool = False,
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
    if include_hydro_events:
        flood_events, flood_status = _real_flood_events(location, metrics, cache_dir, refresh_hydro_events)
        landslide_events, landslide_status = _real_landslide_events(location, metrics, cache_dir, refresh_hydro_events)
    else:
        flood_events, flood_status = _flood_proxy_events(metrics), "not_requested"
        landslide_events, landslide_status = _landslide_proxy_events(location, metrics), "not_requested"
    return HazardSummary(
        city=location.city,
        earthquake=earthquake,
        typhoon_note=_typhoon_note(location, typhoon),
        rainfall_extreme_note=_rainfall_note(metrics, flood_events, flood_status),
        landslide_note=_landslide_note(location, metrics, landslide_events, landslide_status),
        typhoon=typhoon,
        source_notes=[
            "地震：美国地质调查局地震目录，查询 2000 年以来城市周边 500km 内 4 级及以上地震，并分别统计 100/200/500km 内的次数。",
            "台风/热带气旋：按城市坐标选择国际热带气旋最佳路径档案对应海盆数据，按城市到路径点距离统计 100/200/500km 接近次数。",
            "洪涝：美国航天局地球观测自然事件接口，按城市周边 500km 查询洪涝事件；同时读取开放气象洪水接口的河流流量。",
            "滑坡：美国航天局地球观测自然事件接口，按城市周边 500km 查询滑坡事件。",
        ],
        flood_events=flood_events,
        landslide_events=landslide_events,
        flood_status=flood_status,
        landslide_status=landslide_status,
        hazard_exposure_score=_hazard_exposure_score(location, metrics, earthquake, typhoon),
    )


def _typhoon_note(location: Location, typhoon) -> str:
    """Explain whether the summary uses queried tracks or a coarse seed flag."""

    if typhoon is not None and typhoon.status != "failed":
        return (
            f"已查询 IBTrACS 路径：2000 年以来 100/200/500km 内累计接近次数分别为 "
            f"{typhoon.count_100km}/{typhoon.count_200km}/{typhoon.count_500km}；路径接近不等于损失或安全风险。"
        )
    if location.typhoon_region and location.coastal:
        return "尚未查询路径；基础城市信息显示该城市靠海且可能受台风影响。这只是粗略提示，不是历史次数或事件发生概率。"
    if location.typhoon_region:
        return "尚未查询路径；基础表提示可能受台风影响，直接路径需用 IBTrACS 复核。"
    return "尚未查询路径；基础城市信息没有突出提示台风影响，但不代表过去完全没有影响。"


def _rainfall_note(metrics: ClimateMetrics, flood_events: list[dict[str, object]], status: str) -> str:
    if status == "not_requested":
        return "尚未查询洪涝事件数据；当前只根据降水情况给出粗略提示。"
    real_count = sum(1 for item in flood_events if item.get("证据类型") == "公开目录记录")
    if real_count:
        return f"公开事件数据在城市周边 500km 内检索到 {real_count} 条洪涝事件；仍需结合流域、地形和排水能力判断。"
    if status in {"cache", "live"}:
        return "公开事件数据暂未检索到城市周边 500km 洪涝事件；这不代表没有洪涝风险。"
    if metrics.precipitation_extreme_days >= 1 or metrics.heavy_rain_days >= 4:
        return "洪涝事件数据暂不可用；历史天气显示强降水较多，还需结合河流、地形和排水情况判断。"
    if metrics.heavy_rain_days >= 2:
        return "洪涝事件数据暂不可用；历史天气显示有一定强降水情况。"
    return "洪涝事件数据暂不可用；强降水指标相对较低。"


def _landslide_note(location: Location, metrics: ClimateMetrics, landslide_events: list[dict[str, object]], status: str) -> str:
    if status == "not_requested":
        return "尚未查询滑坡事件数据；当前只根据地形和降水情况给出粗略提示。"
    real_count = sum(1 for item in landslide_events if item.get("证据类型") == "公开目录记录")
    if real_count:
        return f"公开事件数据在城市周边 500km 内检索到 {real_count} 条滑坡事件；仍需结合坡度、地质和降雨条件复核。"
    if status in {"cache", "live"}:
        return "公开事件数据暂未检索到城市周边 500km 滑坡事件；这不代表没有滑坡风险。"
    mountainous = location.region_type in {"southwest_mountain", "southwest_plateau", "plateau"}
    if mountainous and metrics.precipitation_extreme_days >= 0.5:
        return "滑坡事件数据暂不可用；山地或高原地区加上强降水时，建议再查看坡度和地质资料。"
    if mountainous:
        return "滑坡事件数据暂不可用；地形指标提示需关注山区滑坡。"
    return "滑坡事件数据暂不可用；当前地形和降水信息没有特别突出的提示。"


def _real_flood_events(
    location: Location,
    metrics: ClimateMetrics,
    cache_dir: Path,
    force_refresh: bool,
) -> tuple[list[dict[str, object]], str]:
    events, status = get_eonet_events(location, "floods", cache_dir / "flood", force_refresh=force_refresh)
    discharge, discharge_status = get_flood_discharge_record(location, cache_dir / "flood", force_refresh=force_refresh)
    rows = list(events)
    if discharge is not None:
        rows.insert(0, discharge)
    if rows:
        return rows, status if status != "failed" else discharge_status
    if status in {"cache", "live"}:
        return [
            {
                "类型": "未检索到洪涝事件",
                "记录": "公开事件接口未返回城市周边 500km 洪涝事件；不等同于无风险。",
                "日期": "",
                "距离": "",
                "数据口径": "美国航天局地球观测自然事件接口",
                "证据类型": "公开目录检索",
            }
        ], status
    return _flood_proxy_events(metrics), "failed"


def _real_landslide_events(
    location: Location,
    metrics: ClimateMetrics,
    cache_dir: Path,
    force_refresh: bool,
) -> tuple[list[dict[str, object]], str]:
    events, status = get_eonet_events(location, "landslides", cache_dir / "landslide", force_refresh=force_refresh)
    if events:
        return events, status
    if status in {"cache", "live"}:
        return [
            {
                "类型": "未检索到滑坡事件",
                "记录": "公开事件接口未返回城市周边 500km 滑坡事件；不等同于无风险。",
                "日期": "",
                "距离": "",
                "数据口径": "美国航天局地球观测自然事件接口",
                "证据类型": "公开目录检索",
            }
        ], status
    return _landslide_proxy_events(location, metrics), "failed"


def _flood_proxy_events(metrics: ClimateMetrics) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    if metrics.precipitation_extreme_days >= 1:
        events.append(
            {
                "类型": "极端降水提示",
                "记录": f"所选月份平均每年约 {metrics.precipitation_extreme_days:.1f} 天达到极端降水阈值",
                "数据口径": "开放气象历史天气或基础城市数据聚合",
                "证据类型": metrics.data_status,
            }
        )
    if metrics.heavy_rain_days >= 3:
        events.append(
            {
                "类型": "强降水提示",
                "记录": f"所选月份平均每年约 {metrics.heavy_rain_days:.1f} 天达到强降水阈值",
                "数据口径": "日降水 >= 20mm",
                "证据类型": metrics.data_status,
            }
        )
    if not events:
        events.append(
            {
                "类型": "暂无重大洪涝事件库",
                "记录": "当前版本暂未连接城市级洪涝事件资料，只显示降水情况提示。",
                "数据口径": "后续预留公开灾害资料",
                "证据类型": "参考提示",
            }
        )
    return events


def _landslide_proxy_events(location: Location, metrics: ClimateMetrics) -> list[dict[str, object]]:
    mountainous = location.region_type in {"southwest_mountain", "southwest_plateau", "plateau"}
    if mountainous:
        return [
            {
                "类型": "山地或高原提示",
                "记录": f"{location.city} 属于山地或高原地区，建议再查看坡度和历史滑坡记录。",
                "数据口径": "基础城市地区标签 + 强降水情况",
                "证据类型": "参考提示",
            }
        ]
    if metrics.precipitation_extreme_days >= 1:
        return [
            {
                "类型": "强降水提示",
                "记录": "极端降水天数偏高，但缺少地形/地质约束，不能直接判断滑坡风险。",
                "数据口径": "开放气象历史天气或基础城市数据聚合",
                "证据类型": metrics.data_status,
            }
        ]
    return [
        {
            "类型": "暂无重大滑坡事件库",
            "记录": "当前版本暂未连接城市级滑坡事件资料，只显示地形和降水情况提示。",
            "数据口径": "后续预留公开灾害资料",
            "证据类型": "参考提示",
        }
    ]


def _hazard_exposure_score(
    location: Location,
    metrics: ClimateMetrics,
    earthquake: EarthquakeSummary,
    typhoon,
) -> float:
    score = 0.0
    score += min(35.0, metrics.heavy_rain_days * 5.0 + metrics.precipitation_extreme_days * 10.0)
    if location.coastal:
        score += 6.0
    if location.typhoon_region:
        score += 12.0
    if typhoon is not None and typhoon.status != "failed":
        score += min(24.0, typhoon.count_200km * 2.4 + typhoon.count_100km * 2.0)
    if earthquake.status != "not_requested" and earthquake.status != "failed":
        score += min(24.0, earthquake.count_200km * 0.35 + earthquake.event_count_m5 * 1.5 + earthquake.event_count_m6 * 3.0)
    return round(min(100.0, score), 1)
