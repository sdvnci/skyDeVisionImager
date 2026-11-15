from asyncio import Event, Task, create_task, gather, wait_for
from json import JSONDecodeError, dump, load
from os import getenv
from pathlib import Path
from time import time
from typing import Final, cast

from dotenv import load_dotenv
from httpx import AsyncBaseTransport, AsyncClient, HTTPError
from pydantic import SecretStr

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
    by a background task and saved to disk, so a restart serves the last
    known values straight away.

    Nothing runs until ``start`` is awaited, on the event loop that will
    serve requests. Without an API key (``api_key``,
    or the ``METAR-TAF`` environment variable) the cache never refreshes and
    only serves what was saved.
    """

    __slots__ = (
        "_airports",
        "_api_key",
        "_cache",
        "_cache_path",
        "_interval",
        "_stop",
        "_task",
        "_timeout",
        "_transport",
        "_updated",
    )

    def __init__(
        self,
        airports: list[str],
        update_interval: int = DEFAULT_UPDATE_INTERVAL,
        api_key: str | None = None,
        cache_path: Path = DEFAULT_CACHE_PATH,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
        transport: AsyncBaseTransport | None = None,
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
        self._transport: AsyncBaseTransport | None = transport
        self._stop: Event = Event()
        self._task: Task[None] | None = None
        self._load_cache()

    @property
    def airports(self) -> tuple[str, ...]:
        return self._airports

    @property
    def updated(self) -> int | None:
        """When the cache was last refreshed, in seconds since the epoch."""
        return self._updated

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

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

        self._cache.update(
            {
                airport: qnh
                for airport, qnh in saved.items()
                if airport in self._cache and isinstance(qnh, (int, float))
            }
        )
        self._updated = updated if isinstance(updated, int) else None

    def _save_cache(self) -> None:
        data = {"updated": self._updated, "qnh": dict(self._cache)}

        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._cache_path.open("w", encoding="utf-8") as f:
                dump(data, f, indent=4)
        except OSError as err:
            self.warning(f"Error saving cache: {err}")

    async def _fetch_qnh(self, client: AsyncClient, airport: str) -> int | None:
        """
        Fetches the QNH from the METAR data in hPa
        Args:
            client (AsyncClient): The HTTP client to make the request with.
            airport (str): The airport code for which to fetch the QNH value.
        Returns:
            int | None: The QNH value in hPa if available, otherwise None.
        """
        if self._api_key is None:
            return None

        try:
            response = await client.get(
                METAR_ENDPOINT,
                params={
                    "api_key": self._api_key.get_secret_value(),
                    "v": METAR_API_VERSION,
                    "locale": "en-US",
                    "id": airport,
                },
            )
        except HTTPError as err:
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

    async def refresh_cache(self) -> None:
        """
        Refreshes the QNH cache by fetching the latest QNH value for every
        airport at once, then saves it. An airport whose fetch fails keeps its
        last known value.
        """
        async with AsyncClient(
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=self._timeout,
            transport=self._transport,
        ) as client:
            fetched = await gather(
                *(self._fetch_qnh(client, airport) for airport in self._airports)
            )

        fresh = {
            airport: qnh
            for airport, qnh in zip(self._airports, fetched)
            if qnh is not None
        }
        if not fresh:
            return

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

        return self._cache.get(airport, None)

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self.refresh_cache()
                self.debug(f"QNH cache refreshed. Next refresh in {self._interval}s.")
            except Exception as err:
                # Never let one bad refresh end the task.
                self.error(f"Error during scheduled refresh: {type(err).__name__}")

            try:
                _ = await wait_for(self._stop.wait(), timeout=self._interval)
            except TimeoutError:
                pass  # the interval is up: refresh again

    async def start(self) -> bool:
        """
        Start refreshing the cache in the background, on the running event
        loop. Returns False, starting nothing, if there is no API key or it is
        already running.
        """
        if self._api_key is None:
            self.warning(
                f"{METAR_KEY_ENV} API key is not set: the QNH cache will not refresh."
            )
            return False
        if self.running:
            return False

        self.info(f"Starting QNH cache refresh every {self._interval} seconds.")
        self._stop = Event()
        self._task = create_task(self._run(), name="metar-refresh")
        return True

    async def stop(self, timeout: float | None = None) -> None:
        """
        Stop the background refresh, waiting up to ``timeout`` seconds for a
        refresh in progress before cancelling it.
        """
        task, self._task = self._task, None
        if task is None:
            return

        self._stop.set()
        try:
            await wait_for(task, timeout=timeout)
        except TimeoutError:
            pass  # wait_for has cancelled the refresh in progress
