from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest
from quart import Quart
from quart.typing import TestClientProtocol as Client

from server import Manager, create_app


@pytest.fixture(autouse=True)
def mock_app() -> Generator[Quart, None, None]:
    app: Quart = create_app(refresh_metar=False)
    app.config.update(
        {
            "TESTING": True,
        }
    )
    yield app


@pytest.fixture()
async def client(mock_app: Quart, tmp_path: Path) -> AsyncGenerator[Client, None]:
    # A fresh database file per test, removed with pytest's temporary directory
    try:
        await Manager.connect(tmp_path / "test.db")
        yield mock_app.test_client()
    finally:
        await Manager.close()
