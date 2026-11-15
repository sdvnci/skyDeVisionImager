from asyncio import gather, sleep
from pathlib import Path

import pytest
from aiosqlite import Connection
from quart.typing import TestClientProtocol as Client

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


async def test_pool_opens_connections_lazily_up_to_size(tmp_path: Path) -> None:
    pool = SQLiteConnectionPool(str(tmp_path / "pool.db"), size=2, timeout=0.1)
    first, second = await pool.getconn(), await pool.getconn()
    assert first is not second

    with pytest.raises(ConnectionError):
        _ = await pool.getconn()

    await pool.putconn(first)
    assert await pool.getconn() is first
    await pool.closeall()
    assert pool.closed


async def test_pool_never_opens_more_than_its_size(tmp_path: Path) -> None:
    pool = SQLiteConnectionPool(str(tmp_path / "pool.db"), size=2, timeout=5)

    async def borrow() -> Connection:
        conn = await pool.getconn()
        await sleep(0.01)
        await pool.putconn(conn)
        return conn

    borrowed = await gather(*(borrow() for _ in range(10)))
    assert len({id(conn) for conn in borrowed}) <= 2
    await pool.closeall()


async def test_pool_shares_one_connection_for_memory_database() -> None:
    pool = SQLiteConnectionPool(":memory:", size=5)
    assert pool.size == 1
    await pool.closeall()


async def test_cursor_rolls_back_on_error(client: Client) -> None:
    with pytest.raises(RuntimeError):
        async with Manager.cursor() as cursor:
            await StationService.insert(make_station(), cursor=cursor)
            raise RuntimeError("abort the transaction")

    assert not await StationService.exists(MAC)
    assert not await StatusService.exists(
        MAC
    ), "Status outlived its rolled back station"


async def test_connect_replaces_the_open_pool(client: Client, tmp_path: Path) -> None:
    await StationService.insert(make_station())
    await Manager.connect(tmp_path / "other.db")
    assert not await StationService.exists(MAC), "Still reading the previous database"


async def test_connections_are_returned_under_concurrency(
    client: Client,
) -> None:
    await StationService.insert(make_station())

    async def work() -> int:
        count = 0
        for _ in range(20):
            count += len(await StationService.list())
        return count

    assert await gather(*(work() for _ in range(16))) == [20] * 16


async def test_reading_for_unregistered_station_is_rejected(
    client: Client,
) -> None:
    with pytest.raises(InvalidInputError):
        await ReadingService.insert(make_reading())


async def test_delete_station_with_readings(client: Client) -> None:
    await StationService.insert(make_station())
    await ReadingService.insert(make_reading())

    with pytest.raises(InvalidInputError):
        await StationService.delete(MAC)
    assert await StationService.exists(MAC)
    assert await StatusService.exists(MAC), "Failed delete removed the station's status"

    await StationService.delete(MAC, readings=True)
    assert not await StationService.exists(MAC)
    assert await ReadingService.list(MAC) == []


async def test_upsert_reading_replaces_existing(client: Client) -> None:
    await StationService.insert(make_station())
    reading = make_reading()
    await ReadingService.upsert(reading)
    await ReadingService.upsert(reading.model_copy(update={"temperature": 9.5}))

    readings = await ReadingService.list(MAC)
    assert len(readings) == 1
    assert readings[0].temperature == 9.5


async def test_station_update_rejects_unknown_columns(
    client: Client,
) -> None:
    await StationService.insert(make_station())
    with pytest.raises(InvalidInputError):
        await StationService.update(MAC, {"nope": 1})  # type: ignore[typeddict-unknown-key]


async def test_query_builder() -> None:
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


async def test_query_builder_rejects_bad_identifiers() -> None:
    with pytest.raises(ValueError):
        _ = QueryBuilder("SELECT * FROM Readings").eq("MAC = 1 OR 1", 1)


async def test_connection_type(client: Client) -> None:
    async with Manager.connection() as conn:
        assert isinstance(conn, Connection)
