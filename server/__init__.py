from flask import Flask

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
    qnh_airport: str = DEFAULT_QNH_AIRPORT,
    refresh_metar: bool = True,
) -> Flask:
    """
    Build the Flask app serving the station API.

    Args:
        metar: The QNH cache to serve from. By default, one for ``qnh_airport``.
        qnh_airport: The airport whose QNH is served when a request names none.
        refresh_metar: Keep the QNH cache fresh from the Metar-Taf API in the
            background. Turn off to serve only what the cache already holds.

    Raises:
        ValueError: If DBSECRET is not set in the environment.
    """
    Manager.load()

    app = Flask(__name__)
    app.config["SECRET_KEY"] = Manager.secret().get_secret_value()
    app.config[QNH_AIRPORT] = qnh_airport

    cache = metar if metar is not None else MetarCacheLayer(airports=[qnh_airport])
    app.extensions[METAR_EXTENSION] = cache
    if refresh_metar:
        _ = cache.start()

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
