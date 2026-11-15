from json import JSONDecodeError, dump, load
from os import getenv
from pathlib import Path
from threading import Event, Lock, Thread
from time import time
from typing import Final, cast

from dotenv import load_dotenv
from pydantic import SecretStr
from requests import RequestException, get

from ..utilities import Loggable
from ._types import MetarResponse, UnregisteredAirportException

METAR_ENDPOINT: Final = "https://api.metar-taf.com/metar"
METAR_API_VERSION: Final = "2.3"
METAR_KEY_ENV: Final = "METAR-TAF"
USER_AGENT: Final = "skyDeVisionImager/1.0.0"
DEFAULT_UPDATE_INTERVAL: Final = 600  # seconds
DEFAULT_REQUEST_TIMEOUT: Final = 30.0  # seconds
DEFAULT_CACHE_PATH: Final = (
    Path(__file__).parent.resolve() / "__metar_cache__" / "airports.json"
)


class MetarCacheLayer(Loggable, log_name="metar", debug=True):
    """
    A cache of the QNH at a set of airports, kept fresh from the Metar-Taf API
    by a background thread and saved to disk, so a restart serves the last
    known values straight away.

    Nothing runs until ``start`` is called. Without an API key (``api_key``,
    or the ``METAR-TAF`` environment variable) the cache never refreshes and
    only serves what was saved.
    """

    __slots__ = (
        "_airports",
        "_api_key",
        "_cache",
        "_cache_path",
        "_interval",
        "_lock",
        "_stop",
        "_thread",
        "_timeout",
        "_updated",
    )

    def __init__(
        self,
        airports: list[str],
        update_interval: int = DEFAULT_UPDATE_INTERVAL,
        api_key: str | None = None,
        cache_path: Path = DEFAULT_CACHE_PATH,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        if update_interval <= 0:
            raise ValueError(
                f"update_interval must be more than 0 seconds, not {update_interval}"
            )

        _ = load_dotenv()
        key = api_key if api_key is not None else getenv(METAR_KEY_ENV, "").strip()
        self._api_key: SecretStr | None = SecretStr(key) if key else None
        self._airports: tuple[str, ...] = tuple(airports)
        self._interval: int = update_interval
        self._timeout: float = timeout
        self._cache_path: Path = cache_path
        self._cache: dict[str, int | None] = {port: None for port in self._airports}
        self._updated: int | None = None
        self._lock: Lock = Lock()
        self._stop: Event = Event()
        self._thread: Thread | None = None
        self._load_cache()

    @property
    def airports(self) -> tuple[str, ...]:
        return self._airports

    @property
    def updated(self) -> int | None:
        """When the cache was last refreshed, in seconds since the epoch."""
        with self._lock:
            return self._updated

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _load_cache(self) -> None:
        if not self._cache_path.exists():
            return

        try:
            with self._cache_path.open("r", encoding="utf-8") as f:
                data = load(f)
        except (OSError, JSONDecodeError) as err:
            self.warning(f"Error loading cache: {err}")
            return

        if not isinstance(data, dict):
            return

        # Files saved before the update time was recorded are a flat
        # {airport: qnh} mapping.
        saved = data.get("qnh", data)
        updated = data.get("updated")
        if not isinstance(saved, dict):
            return

        with self._lock:
            self._cache.update(
                {
                    airport: qnh
                    for airport, qnh in saved.items()
                    if airport in self._cache and isinstance(qnh, (int, float))
                }
            )
            self._updated = updated if isinstance(updated, int) else None

    def _save_cache(self) -> None:
        with self._lock:
            data = {"updated": self._updated, "qnh": dict(self._cache)}

        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._cache_path.open("w", encoding="utf-8") as f:
                dump(data, f, indent=4)
        except OSError as err:
            self.warning(f"Error saving cache: {err}")

    def _fetch_qnh(self, airport: str) -> int | None:
        """
        Fetches the QNH from the METAR data in hPa
        Args:
            airport (str): The airport code for which to fetch the QNH value.
        Returns:
            int | None: The QNH value in hPa if available, otherwise None.
        """
        if self._api_key is None:
            return None

        try:
            response = get(
                METAR_ENDPOINT,
                params={
                    "api_key": self._api_key.get_secret_value(),
                    "v": METAR_API_VERSION,
                    "locale": "en-US",
                    "id": airport,
                },
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                timeout=self._timeout,
            )
        except RequestException as err:
            # Only the kind of failure: the message carries the URL, key included.
            self.warning(f"Request for {airport} METAR failed: {type(err).__name__}")
            return None

        if response.status_code != 200:
            self.warning(f"Error fetching {airport} METAR: {response.status_code}")
            return None

        try:
            data = cast(MetarResponse, response.json())
        except ValueError as err:
            self.warning(f"{airport} METAR response is not JSON: {err}")
            return None

        if not isinstance(data, dict):
            return None
        return data.get("metar", {}).get("qnh", None)

    def refresh_cache(self) -> None:
        """
        Refreshes the QNH cache by fetching the latest QNH value for every
        airport, then saves it. An airport whose fetch fails keeps its last
        known value.
        """
        fetched = {airport: self._fetch_qnh(airport) for airport in self._airports}
        fresh = {airport: qnh for airport, qnh in fetched.items() if qnh is not None}
        if not fresh:
            return

        with self._lock:
            self._cache.update(fresh)
            self._updated = int(time())
        self._save_cache()

    def get_qnh(self, airport: str) -> int | None:
        """
        Retrieves the QNH value for the specified airport from the cache.

        Args:
            airport (str): The airport code for which to retrieve the QNH value.

        Returns:
            int | None: The QNH value in hPa if available, otherwise None.

        Raises:
            UnregisteredAirportException: If the airport is not registered in the cache.
        """
        if airport not in self._airports:
            raise UnregisteredAirportException(
                f"Airport {airport} is not registered in the cache."
            )

        with self._lock:
            return self._cache.get(airport, None)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.refresh_cache()
                self.debug(f"QNH cache refreshed. Next refresh in {self._interval}s.")
            except Exception as err:
                # Never let one bad refresh end the thread.
                self.error(f"Error during scheduled refresh: {type(err).__name__}")
            _ = self._stop.wait(self._interval)

    def start(self) -> bool:
        """
        Start refreshing the cache in the background. Returns False, starting
        nothing, if there is no API key or it is already running.
        """
        if self._api_key is None:
            self.warning(
                f"{METAR_KEY_ENV} API key is not set: the QNH cache will not refresh."
            )
            return False
        if self.running:
            return False

        self.info(f"Starting QNH cache refresh every {self._interval} seconds.")
        self._stop.clear()
        self._thread = Thread(target=self._run, name="metar-refresh", daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout: float | None = None) -> None:
        """Stop the background refresh, waiting up to ``timeout`` seconds for it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
