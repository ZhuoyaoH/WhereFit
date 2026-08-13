"""Shared constants for scoring and presentation."""

from __future__ import annotations

from dataclasses import dataclass


DISCLAIMER = (
    "本项目是帮助比较城市气候的探索工具，分数只用于比较这次选中的城市，不是概率、权威评级或个性化专业建议，"
    "不应用于房产、保险、医疗或灾害避险等高风险决策。"
)

HAZARD_DISCLAIMER = (
    "历史灾害记录只展示公开资料和简单参考提示。历史发生过不代表未来一定发生，"
    "历史未记录也不代表没有风险；结果不构成安全评级、选址建议或灾害预测。"
)

HISTORY_START_DATE = "2000-01-01"
CLIMATE_BASELINE_START_YEAR = 2000
CLIMATE_BASELINE_END_YEAR = 2025
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_HISTORY_MODEL = "era5"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MET_NO_LOCATIONFORECAST_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
USGS_EARTHQUAKE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
IBTRACS_CSV_URL_TEMPLATE = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.{basin}.list.v04r01.csv"
)
SWPC_AURORA_OVATION_URL = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
AURORA_CACHE_TTL_MINUTES = 30

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
    (90, "高度匹配", "#a78bfa"),
    (75, "较高匹配", "#b49afc"),
    (60, "中等匹配", "#c4b5fd"),
    (40, "较低匹配", "#d8b4fe"),
    (0, "低匹配", "#ddd6fe"),
]

RISK_BANDS = [
    (80, "糟糕天气很多"),
    (60, "糟糕天气较多"),
    (40, "糟糕天气中等"),
    (20, "糟糕天气较少"),
    (0, "糟糕天气很少"),
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
