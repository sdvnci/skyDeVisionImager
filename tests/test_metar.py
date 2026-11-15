from json import dumps, loads
from pathlib import Path

import pytest
from requests import ConnectionError as RequestsConnectionError

from server.metar import MetarCacheLayer, UnregisteredAirportException
from server.metar import metar as metar_module


class FakeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        return self._payload


def cache_at(tmp_path: Path, api_key: str = "key") -> MetarCacheLayer:
    return MetarCacheLayer(
        airports=["ESMX", "ESSA"],
        api_key=api_key,
        cache_path=tmp_path / "airports.json",
    )


def test_nothing_runs_until_started(tmp_path: Path) -> None:
    cache = cache_at(tmp_path)
    assert not cache.running
    assert cache.get_qnh("ESMX") is None
    assert cache.updated is None


def test_start_without_api_key_does_nothing(tmp_path: Path) -> None:
    cache = cache_at(tmp_path, api_key="")
    assert cache.start() is False
    assert not cache.running


def test_unregistered_airport(tmp_path: Path) -> None:
    with pytest.raises(UnregisteredAirportException):
        _ = cache_at(tmp_path).get_qnh("KJFK")


def test_refresh_updates_and_saves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []

    def fake_get(url: str, **kwargs: dict[str, str]) -> FakeResponse:
        airport = kwargs["params"]["id"]
        asked.append(airport)
        if airport == "ESSA":
            return FakeResponse(500, {})
        return FakeResponse(200, {"metar": {"qnh": 1009}})

    monkeypatch.setattr(metar_module, "get", fake_get)
    cache = cache_at(tmp_path)
    cache.refresh_cache()

    assert asked == ["ESMX", "ESSA"]
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


def test_failed_refresh_keeps_last_known_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "airports.json").write_text(dumps({"ESMX": 1001, "KJFK": 990}))

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        raise RequestsConnectionError("https://example.invalid/?api_key=key")

    monkeypatch.setattr(metar_module, "get", fake_get)
    cache = cache_at(tmp_path)
    cache.refresh_cache()

    assert cache.get_qnh("ESMX") == 1001, "Legacy cache files should still load"
    assert cache.updated is None


def test_start_and_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        return FakeResponse(200, {"metar": {"qnh": 1020}})

    monkeypatch.setattr(metar_module, "get", fake_get)
    cache = cache_at(tmp_path)

    assert cache.start() is True
    assert cache.start() is False, "Already running"
    cache.stop(timeout=5)

    assert not cache.running
    assert cache.get_qnh("ESMX") == 1020
