from pathlib import Path

import pandas as pd

from wherefit.data_sources.air_quality import summarize_air_quality
from wherefit.hazards.hydro_events import get_eonet_events, get_flood_discharge_record
from wherefit.hazards.earthquake import get_earthquake_summary
from wherefit.models import Location


def test_summarize_air_quality_uses_real_pm25() -> None:
    location = Location("北京", "China", 39.9, 116.4, "Asia/Shanghai", False, False)
    data = pd.DataFrame({"pm2_5": [10.0, 20.0, 30.0], "us_aqi": [40, 60, 80]})

    summary = summarize_air_quality(data, location, "cache", "读取空气质量缓存")

    assert summary.pm25_mean == 20.0
    assert summary.us_aqi_mean == 60.0
    assert summary.status == "cache"
    assert summary.sample_hours == 3


def test_eonet_events_are_filtered_by_radius(tmp_path, monkeypatch) -> None:
    location = Location("成都", "China", 30.67, 104.06, "Asia/Shanghai", False, False)

    def fake_get(url, params, timeout):
        class Response:
            def raise_for_status(self) -> None:
                return None

            def json(self):
                return {
                    "features": [
                        {
                            "id": "E1",
                            "properties": {"title": "Sichuan, China Landslides", "geometries": [{"date": "2024-01-01"}]},
                            "geometry": {"type": "Point", "coordinates": [103.5, 31.0]},
                        }
                    ]
                }

        return Response()

    monkeypatch.setattr("wherefit.hazards.hydro_events.requests.get", fake_get)

    events, status = get_eonet_events(location, "landslides", tmp_path, force_refresh=True)

    assert status == "live"
    assert events[0]["类型"] == "滑坡事件"
    assert events[0]["距离"] < 100


def test_flood_discharge_record_from_open_meteo(tmp_path, monkeypatch) -> None:
    location = Location("北京", "China", 39.9, 116.4, "Asia/Shanghai", False, False)

    def fake_get(url, params, timeout):
        class Response:
            def raise_for_status(self) -> None:
                return None

            def json(self):
                return {"daily": {"time": ["2026-01-01", "2026-01-02"], "river_discharge": [2.0, 4.0], "river_discharge_max": [3.0, 6.0]}}

        return Response()

    monkeypatch.setattr("wherefit.hazards.hydro_events.requests.get", fake_get)

    record, status = get_flood_discharge_record(location, Path(tmp_path), force_refresh=True)

    assert status == "live"
    assert record is not None
    assert "峰值约 6.0" in str(record["记录"])


def test_usgs_query_paginates_below_provider_result_cap(tmp_path, monkeypatch) -> None:
    """A full page must trigger a stable offset query instead of truncating events."""

    location = Location("东京", "Japan", 35.68, 139.69, "Asia/Tokyo", True, True, city_en="Tokyo")
    offsets: list[int] = []
    monkeypatch.setattr("wherefit.hazards.earthquake.USGS_QUERY_LIMIT", 2)

    def fake_get(url, params, timeout):
        """Return two features on page one and one feature on page two."""

        offsets.append(int(params["offset"]))
        page = {
            1: [
                _earthquake_feature("a", 4.2, 1_600_000_000_000),
                _earthquake_feature("b", 5.1, 1_610_000_000_000),
            ],
            3: [_earthquake_feature("c", 6.0, 1_620_000_000_000)],
        }[int(params["offset"])]

        class Response:
            """Minimal requests response used by the provider test."""

            def raise_for_status(self) -> None:
                """Represent a successful HTTP response."""

                return None

            def json(self):
                """Return the selected synthetic GeoJSON page."""

                return {"features": page}

        return Response()

    monkeypatch.setattr("wherefit.hazards.earthquake.requests.get", fake_get)
    summary = get_earthquake_summary(
        location,
        Path(tmp_path),
        start_date="2020-01-01",
        end_date="2020-12-31",
        force_refresh=True,
    )

    assert offsets == [1, 3]
    assert summary.status == "live"
    assert summary.event_count_m4 == 3
    assert summary.event_count_m6 == 1


def _earthquake_feature(event_id: str, magnitude: float, timestamp_ms: int) -> dict[str, object]:
    """Build one synthetic USGS GeoJSON feature near Tokyo."""

    return {
        "id": event_id,
        "properties": {"time": timestamp_ms, "mag": magnitude, "place": event_id},
        "geometry": {"coordinates": [139.8, 35.7, 10.0]},
    }
