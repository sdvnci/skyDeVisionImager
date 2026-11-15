from collections.abc import Generator
from pathlib import Path

import pytest
from flask import Flask
from flask.testing import FlaskClient

from server import Manager, create_app


@pytest.fixture(autouse=True)
def mock_app() -> Generator[Flask, None, None]:
    app: Flask = create_app(refresh_metar=False)
    app.config.update(
        {
            "TESTING": True,
        }
    )
    yield app


@pytest.fixture()
def client(mock_app: Flask, tmp_path: Path) -> Generator[FlaskClient, None, None]:
    # A fresh database file per test, removed with pytest's temporary directory
    try:
        Manager.connect(tmp_path / "test.db")
        yield mock_app.test_client()
    finally:
        Manager.close()
