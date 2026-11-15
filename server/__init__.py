from pathlib import Path

from quart import Quart

from .api import METAR_EXTENSION, QNH_AIRPORT, apiRouter
from .db import (
    AlreadyExistsError,
    CameraModel,
    DatabaseError,
    DeviceType,
    InternalDBError,
    InvalidInputError,
    Location,
    LocationJSON,
    Manager,
    NotFoundError,
    Reading,
    ReadingJSON,
    ReadingService,
    Service,
    Station,
    StationJSON,
    StationService,
    StationStatus,
    StationStatusJSON,
    StatusService,
    UnauthorizedError,
    User,
    UserJSON,
    UserRole,
    UserService,
)
from .metar import (
    Airport,
    Metar,
    MetarCacheLayer,
    MetarResponse,
    Runway,
    UnregisteredAirportException,
)
from .utilities import Loggable

DEFAULT_QNH_AIRPORT = "ESMX"


def create_app(
    metar: MetarCacheLayer | None = None,
    *,
    database: str | Path | None = None,
    qnh_airport: str = DEFAULT_QNH_AIRPORT,
    refresh_metar: bool = True,
) -> Quart:
    """
    Build the Quart app serving the station API.

    While the app is serving, it holds the database connection pool open and
    (with ``refresh_metar``) keeps the QNH cache fresh; both are shut down
    with it.

    Args:
        metar: The QNH cache to serve from. By default, one for ``qnh_airport``.
        database: Path to the SQLite database file. By default, the DBFILE
            environment variable.
        qnh_airport: The airport whose QNH is served when a request names none.
        refresh_metar: Keep the QNH cache fresh from the Metar-Taf API in the
            background. Turn off to serve only what the cache already holds.

    Raises:
        ValueError: If DBSECRET is not set in the environment.
    """
    Manager.load(database=database)

    app = Quart(__name__)
    app.config["SECRET_KEY"] = Manager.secret().get_secret_value()
    app.config[QNH_AIRPORT] = qnh_airport

    cache = metar if metar is not None else MetarCacheLayer(airports=[qnh_airport])
    app.extensions[METAR_EXTENSION] = cache

    @app.before_serving
    async def startup() -> None:
        if not Manager.connected():
            await Manager.connect()
        if refresh_metar:
            _ = await cache.start()

    @app.after_serving
    async def shutdown() -> None:
        await cache.stop()
        await Manager.close()

    app.register_blueprint(apiRouter)
    return app


__all__ = (
    "Airport",
    "AlreadyExistsError",
    "CameraModel",
    "DatabaseError",
    "DeviceType",
    "InternalDBError",
    "InvalidInputError",
    "Location",
    "LocationJSON",
    "Loggable",
    "Manager",
    "Metar",
    "MetarCacheLayer",
    "MetarResponse",
    "NotFoundError",
    "Reading",
    "ReadingJSON",
    "ReadingService",
    "Runway",
    "Service",
    "Station",
    "StationJSON",
    "StationService",
    "StationStatus",
    "StationStatusJSON",
    "StatusService",
    "UnauthorizedError",
    "UnregisteredAirportException",
    "User",
    "UserJSON",
    "UserRole",
    "UserService",
    "create_app",
)
