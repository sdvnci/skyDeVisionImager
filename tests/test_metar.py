from asyncio import sleep, wait_for
from json import dumps, loads
from pathlib import Path

import pytest
from httpx import ConnectError, MockTransport, Request, Response

from server.metar import MetarCacheLayer, UnregisteredAirportException


def cache_at(
    tmp_path: Path, transport: MockTransport | None = None, api_key: str = "key"
) -> MetarCacheLayer:
    return MetarCacheLayer(
        airports=["ESMX", "ESSA"],
        api_key=api_key,
        cache_path=tmp_path / "airports.json",
        transport=transport,
    )


def test_nothing_runs_until_started(tmp_path: Path) -> None:
    cache = cache_at(tmp_path)
    assert not cache.running
    assert cache.get_qnh("ESMX") is None
    assert cache.updated is None


async def test_start_without_api_key_does_nothing(tmp_path: Path) -> None:
    cache = cache_at(tmp_path, api_key="")
    assert await cache.start() is False
    assert not cache.running


def test_unregistered_airport(tmp_path: Path) -> None:
    with pytest.raises(UnregisteredAirportException):
        _ = cache_at(tmp_path).get_qnh("KJFK")


async def test_refresh_updates_and_saves(tmp_path: Path) -> None:
    asked: list[str] = []

    def handler(request: Request) -> Response:
        airport = request.url.params["id"]
        asked.append(airport)
        assert request.url.params["api_key"] == "key"
        if airport == "ESSA":
            return Response(500, json={})
        return Response(200, json={"metar": {"qnh": 1009}})

    cache = cache_at(tmp_path, MockTransport(handler))
    await cache.refresh_cache()

    assert sorted(asked) == ["ESMX", "ESSA"]
    assert cache.get_qnh("ESMX") == 1009
    assert cache.get_qnh("ESSA") is None, "A failed fetch must not invent a value"
    assert cache.updated is not None

    saved = loads((tmp_path / "airports.json").read_text())
    assert saved["qnh"] == {"ESMX": 1009, "ESSA": None}
    assert saved["updated"] == cache.updated

    # A new cache over the same file serves the saved values straight away
    reloaded = cache_at(tmp_path)
    assert reloaded.get_qnh("ESMX") == 1009
    assert reloaded.updated == cache.updated


async def test_failed_refresh_keeps_last_known_values(tmp_path: Path) -> None:
    _ = (tmp_path / "airports.json").write_text(dumps({"ESMX": 1001, "KJFK": 990}))

    def handler(request: Request) -> Response:
        raise ConnectError("no route to host", request=request)

    cache = cache_at(tmp_path, MockTransport(handler))
    await cache.refresh_cache()

    assert cache.get_qnh("ESMX") == 1001, "Legacy cache files should still load"
    assert cache.updated is None


async def test_start_and_stop(tmp_path: Path) -> None:
    def handler(request: Request) -> Response:
        return Response(200, json={"metar": {"qnh": 1020}})

    cache = cache_at(tmp_path, MockTransport(handler))

    assert await cache.start() is True
    assert await cache.start() is False, "Already running"
    assert cache.running

    async def refreshed() -> None:
        while cache.updated is None:
            await sleep(0.01)

    await wait_for(refreshed(), timeout=5)
    await cache.stop(timeout=5)

    assert not cache.running
    assert cache.get_qnh("ESMX") == 1020
