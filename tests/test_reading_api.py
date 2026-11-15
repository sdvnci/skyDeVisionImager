from datetime import UTC, datetime
from pathlib import Path

from flask import Flask
from flask.testing import FlaskClient

from server import create_app
from server._utils import HEADERS
from server.db import (
    CameraModel,
    DeviceType,
    ReadingService,
    Station,
    StationService,
    dt2str,
    str2dt,
)
from server.metar import MetarCacheLayer

MAC = "00:1A:2B:3C:4D:5E"
TIMESTAMP = "2026-01-01 12:00:00"
READING = {
    "timestamp": TIMESTAMP,
    "humidity": 40.5,
    "temperature": 21.25,
    "pressure": 1013.2,
    "altitude": 400.0,
    "dewpoint": 7.5,
}


def device_headers(mac: str = MAC, timestamp: str = TIMESTAMP) -> dict[str, str]:
    return {
        HEADERS.TIMESTAMP.value: timestamp,
        HEADERS.MACADDRESS.value: mac,
        HEADERS.FIRMWAREVERSION.value: "1.0.0",
    }


def register_station() -> None:
    StationService.insert(
        Station.model_validate(
            {
                "MAC": MAC,
                "name": "station",
                "device_model": DeviceType.ESP32,
                "camera_model": CameraModel.DSLR,
                "firmware_version": "1.0.0",
                "altitude": 400,
                "latitude": 83.3323,
                "longitude": 82.5546,
            }
        )
    )


def test_post_reading_creates_reading(client: FlaskClient) -> None:
    register_station()

    response = client.post("/api/reading", json=READING, headers=device_headers())
    assert response.status_code == 200, response.get_json()

    stored = ReadingService.get(MAC, str2dt(TIMESTAMP))
    assert stored is not None, "The posted reading was not stored"
    assert stored.temperature == 21.25
    assert stored.pressure == 1013.2
    assert stored.timestamp == datetime(2026, 1, 1, 12, tzinfo=UTC)


def test_post_reading_twice_replaces_it(client: FlaskClient) -> None:
    register_station()

    for temperature in (1.0, 2.0):
        response = client.post(
            "/api/reading",
            json={**READING, "temperature": temperature},
            headers=device_headers(),
        )
        assert response.status_code == 200, response.get_json()

    readings = ReadingService.list(MAC)
    assert len(readings) == 1
    assert readings[0].temperature == 2.0


def test_post_reading_takes_its_time_from_the_header(client: FlaskClient) -> None:
    register_station()
    sent = dt2str(datetime(2026, 2, 2, 8, 30, tzinfo=UTC))

    response = client.post(
        "/api/reading", json=READING, headers=device_headers(timestamp=sent)
    )
    assert response.status_code == 200, response.get_json()
    assert ReadingService.exists(MAC, datetime(2026, 2, 2, 8, 30, tzinfo=UTC))
    assert not ReadingService.exists(MAC, str2dt(TIMESTAMP))


def test_post_reading_unregistered_station(client: FlaskClient) -> None:
    response = client.post("/api/reading", json=READING, headers=device_headers())
    assert response.status_code == 404, response.get_json()
    assert response.get_json()["errors"][0]["title"] == "Not Found"


def test_post_reading_validation_error(client: FlaskClient) -> None:
    register_station()
    response = client.post(
        "/api/reading",
        json={**READING, "temperature": "warm"},
        headers=device_headers(),
    )
    assert response.status_code == 422, response.get_json()
    assert response.get_json()["errors"][0]["title"] == "Validation Error"


def test_post_reading_invalid_headers(client: FlaskClient) -> None:
    for headers in (
        device_headers(mac="not-a-mac"),
        device_headers(timestamp="yesterday-ish"),
    ):
        response = client.post("/api/reading", json=READING, headers=headers)
        assert response.status_code == 400, response.get_json()
        assert response.get_json()["errors"][0]["title"] == "Invalid Headers"


def test_post_body_must_be_a_json_object(client: FlaskClient) -> None:
    response = client.post("/api/reading", json=[1, 2, 3], headers=device_headers())
    assert response.status_code == 400, response.get_json()
    assert response.get_json()["errors"][0]["title"] == "Invalid Input"

    response = client.post(
        "/api/reading",
        data="{not json",
        content_type="application/json",
        headers=device_headers(),
    )
    assert response.status_code == 400
    assert "errors" in response.get_json(), "Malformed JSON should use the error shape"


def test_register_uses_firmware_version_header(client: FlaskClient) -> None:
    response = client.post(
        "/api/register",
        json={
            "name": "station",
            "device_model": "ESP32",
            "altitude": 400,
            "latitude": 83.3323,
            "longitude": 82.5546,
        },
        headers=device_headers(),
    )
    assert response.status_code == 200, response.get_json()

    station = StationService.get(MAC)
    assert station is not None
    assert station.firmware_version == "1.0.0"
    assert station.camera_model is CameraModel.UNKNOWN
    assert station.sensors is not None, "A registered station should have a status"


def test_check(client: FlaskClient) -> None:
    response = client.get("/api/check")
    assert response.status_code == 200


def qnh_app(tmp_path: Path, qnh: dict[str, int]) -> Flask:
    cache = MetarCacheLayer(
        airports=["ESMX", "ESSA"], api_key="", cache_path=tmp_path / "airports.json"
    )
    cache._cache.update(qnh)
    return create_app(cache, refresh_metar=False)


def test_qnh(tmp_path: Path) -> None:
    client = qnh_app(tmp_path, {"ESMX": 1013, "ESSA": 998}).test_client()

    response = client.get("/api/qnh")
    assert response.status_code == 200
    assert response.get_json() == {"qnh": 1013}

    response = client.get("/api/qnh?airport=ESSA")
    assert response.get_json() == {"qnh": 998}


def test_qnh_unavailable(tmp_path: Path) -> None:
    client = qnh_app(tmp_path, {}).test_client()

    response = client.get("/api/qnh")
    assert response.status_code == 404
    assert response.get_json()["errors"][0]["title"] == "QNH Not Found"

    response = client.get("/api/qnh?airport=KJFK")
    assert response.status_code == 404
    assert response.get_json()["errors"][0]["title"] == "Not Found"
