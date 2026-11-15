from datetime import UTC, datetime
from typing import Final, cast

from flask import Blueprint, Response, current_app, jsonify, request
from pydantic import ValidationError
from werkzeug.exceptions import HTTPException

from ._types import ErrorDict, ErrorResponse
from ._utils import InvalidHeadersError, MissingHeadersError, device_headers
from .db import (
    AlreadyExistsError,
    CameraModel,
    DatabaseError,
    InvalidInputError,
    Manager,
    NotFoundError,
    Reading,
    ReadingService,
    Station,
    StationService,
    StationStatus,
    StatusService,
    UnauthorizedError,
)
from .metar import MetarCacheLayer, UnregisteredAirportException

#: Where the app keeps its MetarCacheLayer, in ``app.extensions``.
METAR_EXTENSION: Final = "metar"
#: Config key for the airport whose QNH is served when none is asked for.
QNH_AIRPORT: Final = "QNH_AIRPORT"

apiRouter = Blueprint("api", __name__, url_prefix="/api")
"""
API endpoints for handling status and environmental reading data.
"""

# The title and HTTP status each error is reported with. Flask picks the entry
# for the closest base class, so subclasses are matched before DatabaseError.
_CLIENT_ERRORS: Final[dict[type[Exception], tuple[str, int]]] = {
    MissingHeadersError: ("Missing Headers", 400),
    InvalidHeadersError: ("Invalid Headers", 400),
    InvalidInputError: ("Invalid Input", 400),
    UnauthorizedError: ("Unauthorized", 401),
    NotFoundError: ("Not Found", 404),
    UnregisteredAirportException: ("Not Found", 404),
    AlreadyExistsError: ("Already Exists", 409),
    ValidationError: ("Validation Error", 422),
}


def _error_response(title: str, details: str, status: int) -> tuple[Response, int]:
    return (
        jsonify(
            ErrorResponse(
                errors=[
                    ErrorDict(
                        title=title,
                        details=details,
                        code=str(status),
                        source=request.path,
                    )
                ],
                timestamp=datetime.now(tz=UTC).isoformat(),
            )
        ),
        status,
    )


def _handle_client_error(err: Exception) -> tuple[Response, int]:
    title, status = next(
        entry for kind, entry in _CLIENT_ERRORS.items() if isinstance(err, kind)
    )
    return _error_response(title, str(err), status)


for _kind in _CLIENT_ERRORS:
    apiRouter.register_error_handler(_kind, _handle_client_error)


@apiRouter.errorhandler(HTTPException)
def _handle_http_error(err: HTTPException) -> tuple[Response, int]:
    """Malformed requests (bad JSON, wrong method, ...) in the API's error shape."""
    return _error_response(err.name, err.description or err.name, err.code or 500)


@apiRouter.errorhandler(DatabaseError)
@apiRouter.errorhandler(ConnectionError)
def _handle_server_error(err: Exception) -> tuple[Response, int]:
    """Failures on our side. The cause is logged, not sent to the client."""
    Manager.error(f"{request.path} failed: {err}")
    return _error_response(
        "Internal Database Error", "The database could not serve this request.", 500
    )


def _payload() -> dict[str, object]:
    """The request's JSON body, which must be an object."""
    body = request.get_json()
    if not isinstance(body, dict):
        raise InvalidInputError("The request body must be a JSON object.")
    return cast(dict[str, object], body)


def _metar() -> MetarCacheLayer:
    return cast(MetarCacheLayer, current_app.extensions[METAR_EXTENSION])


@apiRouter.route("/status", methods=["POST", "PUT"])
def status() -> tuple[Response, int]:
    """
    Endpoint to handle status updates.
    """
    device = device_headers(request.headers)
    status = StationStatus.model_validate(
        {**_payload(), "MAC": device.mac, "timestamp": device.timestamp}
    )

    StatusService.update(device.mac, status)
    return jsonify({"status": "success"}), 200


@apiRouter.route("/register", methods=["POST", "PUT"])
def register() -> tuple[Response, int]:
    """
    Endpoint to handle station registration.
    """
    device = device_headers(request.headers)

    stationdict = _payload()
    _ = stationdict.pop("sensors", None)
    _ = stationdict.setdefault("camera_model", CameraModel.UNKNOWN.value)
    _ = stationdict.setdefault("firmware_version", device.firmware_version)
    station = Station.model_validate({**stationdict, "MAC": device.mac})

    StationService.insert(station)
    return jsonify({"status": "success"}), 200


@apiRouter.route("/reading", methods=["POST", "PUT"])
def reading() -> tuple[Response, int]:
    """
    Endpoint to handle environmental readings. The reading's time is taken
    from the timestamp header; a station resending a reading replaces it.
    """
    device = device_headers(request.headers)
    reading = Reading.model_validate(
        {**_payload(), "MAC": device.mac, "timestamp": device.timestamp}
    )

    with Manager.cursor() as cursor:
        if not StationService.exists(device.mac, cursor=cursor):
            raise NotFoundError(f"Station with MAC {device.mac} is not registered.")
        ReadingService.upsert(reading, cursor=cursor)

    return jsonify({"status": "success"}), 200


@apiRouter.route("/qnh", methods=["GET"])
def qnh() -> tuple[Response, int]:
    """
    Endpoint to get the QNH at an airport: the ``airport`` query parameter,
    or the configured default.
    """
    airport = request.args.get("airport") or current_app.config[QNH_AIRPORT]
    qnh = _metar().get_qnh(airport)

    if qnh is None:
        return _error_response(
            "QNH Not Found",
            "QNH value for the requested airport is not available.",
            404,
        )

    return jsonify({"qnh": qnh}), 200


@apiRouter.route("/version", methods=["GET"])
def version() -> tuple[Response, int]:
    """
    Endpoint to handle firmware version updates.
    """
    return jsonify({"status": "success"}), 200


@apiRouter.route("/check", methods=["GET"])
def check() -> tuple[Response, int]:
    """
    Endpoint to check if the API is running.
    """
    return jsonify({"status": "API is running"}), 200
