"""Shared constants for scoring and presentation."""

from __future__ import annotations

from dataclasses import dataclass


DISCLAIMER = (
    "本项目是气候适配推荐原型，评分仅用于信息展示和作品集演示，"
    "不应用于房产、保险、医疗或灾害避险等高风险决策。"
)

HAZARD_DISCLAIMER = (
    "历史灾害档案仅展示公开数据和简化 proxy。历史发生过不代表未来一定发生，"
    "历史未记录也不代表没有风险；结果不构成安全评级、选址建议或灾害预测。"
)

HISTORY_START_DATE = "2000-01-01"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
USGS_EARTHQUAKE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
IBTRACS_WP_CSV_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.WP.list.v04r01.csv"
)
SWPC_AURORA_OVATION_URL = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"

MODE_LABELS = {
    "Travel": "旅行",
    "Living": "长期居住",
    "Compare": "城市对比",
}

MODE_WEIGHTS = {
    "Travel": {"comfort": 0.75, "risk": 0.25},
    "Living": {"comfort": 0.45, "risk": 0.55},
    "Compare": {"comfort": 0.60, "risk": 0.40},
}

SCORE_BANDS = [
    (90, "非常适合", "#16a34a"),
    (75, "比较适合", "#65a30d"),
    (60, "一般适合", "#ca8a04"),
    (40, "不太适合", "#ea580c"),
    (0, "不推荐", "#dc2626"),
]

RISK_BANDS = [
    (80, "很高风险"),
    (60, "较高风险"),
    (40, "中等风险"),
    (20, "中低风险"),
    (0, "低风险"),
]


@dataclass(frozen=True)
class ScoringConfig:
    ideal_temp_lower: float = 18.0
    ideal_temp_upper: float = 25.0
    humidity_comfort_ceiling: float = 65.0
    clean_pm25_threshold: float = 10.0
    moderate_pm25_threshold: float = 35.0
    max_confidence: float = 0.72
    static_data_confidence: float = 0.62
    historical_data_confidence: float = 0.86
    earthquake_history_confidence: float = 0.78


CONFIG = ScoringConfig()
