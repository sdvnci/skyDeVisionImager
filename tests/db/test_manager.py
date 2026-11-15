from pathlib import Path
from sqlite3 import Connection
from threading import Thread

import pytest
from flask.testing import FlaskClient

from server.db import (
    CameraModel,
    DeviceType,
    InvalidInputError,
    Manager,
    QueryBuilder,
    Reading,
    ReadingService,
    SQLiteConnectionPool,
    Station,
    StationService,
    StatusService,
    normalize_mac,
    str2dt,
)

MAC = "00:1A:2B:3C:4D:5E"


def make_station(mac: str = MAC) -> Station:
    return Station.model_validate(
        {
            "MAC": mac,
            "name": "station",
            "device_model": DeviceType.ESP32,
            "camera_model": CameraModel.DSLR,
            "firmware_version": "1.0.0",
            "altitude": 400,
            "latitude": 83.3323,
            "longitude": 82.5546,
        }
    )


def make_reading(mac: str = MAC) -> Reading:
    return Reading(
        MAC=normalize_mac(mac),
        timestamp=str2dt("2026-01-01 12:00:00"),
        temperature=1.0,
        humidity=2.0,
        pressure=3.0,
        dewpoint=4.0,
    )


def test_pool_opens_connections_lazily_up_to_size(tmp_path: Path) -> None:
    pool = SQLiteConnectionPool(str(tmp_path / "pool.db"), size=2, timeout=0.1)
    first, second = pool.getconn(), pool.getconn()
    assert first is not second

    with pytest.raises(ConnectionError):
        _ = pool.getconn()

    pool.putconn(first)
    assert pool.getconn() is first
    pool.closeall()
    assert pool.closed


def test_pool_shares_one_connection_for_memory_database() -> None:
    pool = SQLiteConnectionPool(":memory:", size=5)
    assert pool.size == 1
    pool.closeall()


def test_cursor_rolls_back_on_error(client: FlaskClient) -> None:
    with pytest.raises(RuntimeError):
        with Manager.cursor() as cursor:
            StationService.insert(make_station(), cursor=cursor)
            raise RuntimeError("abort the transaction")

    assert not StationService.exists(MAC)
    assert not StatusService.exists(MAC), "Status outlived its rolled back station"


def test_connect_replaces_the_open_pool(client: FlaskClient, tmp_path: Path) -> None:
    StationService.insert(make_station())
    Manager.connect(tmp_path / "other.db")
    assert not StationService.exists(MAC), "Still reading the previous database"


def test_connections_are_returned_under_concurrency(client: FlaskClient) -> None:
    StationService.insert(make_station())
    errors: list[Exception] = []

    def work() -> None:
        try:
            for _ in range(20):
                assert len(StationService.list()) == 1
        except Exception as err:
            errors.append(err)

    threads = [Thread(target=work) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"Concurrent readers failed: {errors[:1]}"


def test_reading_for_unregistered_station_is_rejected(client: FlaskClient) -> None:
    with pytest.raises(InvalidInputError):
        ReadingService.insert(make_reading())


def test_delete_station_with_readings(client: FlaskClient) -> None:
    StationService.insert(make_station())
    ReadingService.insert(make_reading())

    with pytest.raises(InvalidInputError):
        StationService.delete(MAC)
    assert StationService.exists(MAC)
    assert StatusService.exists(MAC), "Failed delete removed the station's status"

    StationService.delete(MAC, readings=True)
    assert not StationService.exists(MAC)
    assert ReadingService.list(MAC) == []


def test_upsert_reading_replaces_existing(client: FlaskClient) -> None:
    StationService.insert(make_station())
    reading = make_reading()
    ReadingService.upsert(reading)
    ReadingService.upsert(reading.model_copy(update={"temperature": 9.5}))

    readings = ReadingService.list(MAC)
    assert len(readings) == 1
    assert readings[0].temperature == 9.5


def test_station_update_rejects_unknown_columns(client: FlaskClient) -> None:
    StationService.insert(make_station())
    with pytest.raises(InvalidInputError):
        StationService.update(MAC, {"nope": 1})  # type: ignore[typeddict-unknown-key]


def test_query_builder() -> None:
    query, params = (
        QueryBuilder("SELECT * FROM Readings")
        .eq("MAC", MAC)
        .eq("filepath", None)
        .gte("timestamp", "2026")
        .order_by("timestamp", desc=True)
        .limit(5)
        .offset(10)
        .build()
    )
    assert query == (
        'SELECT * FROM Readings WHERE "MAC" = ? AND "timestamp" >= ? '
        'ORDER BY "timestamp" DESC LIMIT ? OFFSET ?;'
    )
    assert params == (MAC, "2026", 5, 10)


def test_query_builder_rejects_bad_identifiers() -> None:
    with pytest.raises(ValueError):
        _ = QueryBuilder("SELECT * FROM Readings").eq("MAC = 1 OR 1", 1)


def test_connection_type(client: FlaskClient) -> None:
    with Manager.connection() as conn:
        assert isinstance(conn, Connection)
