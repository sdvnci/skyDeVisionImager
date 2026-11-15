from collections.abc import Iterable
from datetime import datetime
from enum import Enum
from typing import Final, NamedTuple

from pydantic import ValidationError
from pydantic_extra_types.mac_address import MacAddress
from werkzeug.datastructures import Headers

from .db import normalize_mac, str2dt


class MissingHeadersError(Exception):
    """
    Exception raised when a required header is missing.
    """

    def __init__(self: "MissingHeadersError", headers: Iterable[str]) -> None:
        super().__init__(f"Missing required headers: {list(headers)}")


class InvalidHeadersError(Exception):
    """
    Exception raised when a required header's value is unusable.
    """

    def __init__(self: "InvalidHeadersError", header: str, reason: str) -> None:
        super().__init__(f"Invalid {header} header: {reason}")


class HEADERS(str, Enum):
    """
    Expected Headers to the /api route from esp devices.
    """

    MACADDRESS = "X-MAC-Address"
    TIMESTAMP = "X-Timestamp"
    FIRMWAREVERSION = "X-Firmware-Version"
    CONTENTTYPE = "Content-Type"
    CONTENTLENGTH = "Content-Length"
    USERAGENT = "User-Agent"


REQUIRED: Final[tuple[str, str, str]] = (
    HEADERS.MACADDRESS.value,
    HEADERS.TIMESTAMP.value,
    HEADERS.FIRMWAREVERSION.value,
)


class DeviceHeaders(NamedTuple):
    """The validated headers every station sends with a request."""

    mac: MacAddress
    timestamp: datetime
    firmware_version: str


def headercheck(headers: Headers, required: Iterable[str] | None = None) -> None:
    """
    Check if all required headers are present in the request.

    Args:
        headers (Headers): Headers from the request.
        required (Iterable[str]): List of required header names.
    Raises:
        MissingHeadersError: If any required headers are missing.
    """
    if required is None:
        required = REQUIRED
    missing = [head for head in required if not headers.get(head)]
    if missing:
        raise MissingHeadersError(missing)


def device_headers(headers: Headers) -> DeviceHeaders:
    """
    Validate and parse the headers a station identifies itself with.

    Raises:
        MissingHeadersError: If any required headers are missing.
        InvalidHeadersError: If the MAC address or timestamp cannot be parsed.
    """
    headercheck(headers)

    try:
        mac = normalize_mac(headers[HEADERS.MACADDRESS.value])
    except ValidationError as err:
        reason = "; ".join(error["msg"] for error in err.errors())
        raise InvalidHeadersError(HEADERS.MACADDRESS.value, reason) from err

    try:
        timestamp = str2dt(headers[HEADERS.TIMESTAMP.value])
    except ValueError as err:
        raise InvalidHeadersError(HEADERS.TIMESTAMP.value, str(err)) from err

    return DeviceHeaders(mac, timestamp, headers[HEADERS.FIRMWAREVERSION.value])
