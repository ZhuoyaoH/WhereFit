"""Dataclasses used by the WhereFit scoring pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Location:
    city: str
    country: str
    latitude: float
    longitude: float
    timezone: str
    coastal: bool
    typhoon_region: bool
    province: str = ""
    city_en: str = ""
    region_type: str = ""
    admin_level: str = ""


@dataclass(frozen=True)
class UserPreference:
    mode: str
    month: int
    heat_sensitivity: int
    cold_sensitivity: int
    humidity_sensitivity: int
    rain_sensitivity: int
    air_quality_sensitivity: int
    extreme_weather_sensitivity: int


@dataclass(frozen=True)
class ClimateMetrics:
    temperature_mean: float
    temperature_max: float
    apparent_temperature: float
    relative_humidity_mean: float
    precipitation_days: float
    heavy_rain_days: float
    pm25: float
    hot_days: float
    winter_cold_level: float
    coastal: bool
    typhoon_region: bool
    data_source: str = "静态种子数据"
    data_status: str = "fallback"
    sample_years: int = 0
    missing_rate: float = 0.0
    precipitation_extreme_days: float = 0.0
    cold_days: float = 0.0
    windy_days: float = 0.0
    snow_days: float = 0.0


@dataclass(frozen=True)
class ScoreResult:
    travel_comfort_score: float
    long_term_risk_score: float
    personal_fit_score: float
    component_scores: dict[str, float]
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    confidence: float = 0.0
    data_source: str = "静态种子数据"
    data_status: str = "fallback"


@dataclass(frozen=True)
class CityResult:
    location: Location
    metrics: ClimateMetrics
    score: ScoreResult


@dataclass(frozen=True)
class EarthquakeSummary:
    event_count_m4: int
    event_count_m5: int
    event_count_m6: int
    max_magnitude: float | None
    latest_event_date: str | None
    nearest_distance_km: float | None
    source: str
    status: str


@dataclass(frozen=True)
class HazardSummary:
    city: str
    earthquake: EarthquakeSummary
    typhoon_note: str
    rainfall_extreme_note: str
    landslide_note: str
    typhoon: "TyphoonSummary | None" = None
    source_notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AuroraSummary:
    city: str
    opportunity_label: str
    opportunity_score: float
    explanation: str
    source: str
    status: str
    forecast_time: str | None = None
    nearest_probability: float | None = None
    nearest_distance_km: float | None = None


@dataclass(frozen=True)
class ForecastSummary:
    city: str
    start_date: str
    end_date: str
    days: int
    temp_max_mean: float
    apparent_temp_max_mean: float
    precipitation_days: int
    precipitation_probability_max: float
    heavy_rain_days: int
    windy_days: int
    confidence: float
    source: str
    status: str
    message: str


@dataclass(frozen=True)
class TyphoonSummary:
    city: str
    count_100km: int
    count_200km: int
    count_500km: int
    nearest_distance_km: float | None
    strongest_name: str | None
    strongest_year: int | None
    strongest_wind: float | None
    latest_nearby_name: str | None
    latest_nearby_year: int | None
    source: str
    status: str
    message: str
