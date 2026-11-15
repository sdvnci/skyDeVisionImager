from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Final, NotRequired, Self, TypedDict, TypeVar

from pydantic import (
    AfterValidator,
    BaseModel,
    EmailStr,
    SecretStr,
    TypeAdapter,
    model_validator,
)
from pydantic_extra_types.coordinate import Latitude, Longitude
from pydantic_extra_types.mac_address import MacAddress

#: The format timestamps are stored in. Always UTC, so it sorts chronologically.
TIMESTAMP_FORMAT: Final = "%Y-%m-%d %H:%M:%S.%f"


def as_utc(dt: datetime) -> datetime:
    """A timezone-aware UTC datetime. A naive datetime is taken to be UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def dt2str(dt: datetime | None) -> str:
    """The stored form of a timestamp, or an empty string for None."""
    if dt is None:
        return ""
    return as_utc(dt).strftime(TIMESTAMP_FORMAT)


def str2dt(s: str) -> datetime:
    """Parse an ISO 8601 or stored timestamp into a UTC datetime."""
    try:
        return as_utc(datetime.fromisoformat(s))
    except ValueError:
        # fallback for legacy format without tz
        return datetime.strptime(s, TIMESTAMP_FORMAT).replace(tzinfo=UTC)


_MAC: Final[TypeAdapter[MacAddress]] = TypeAdapter(MacAddress)


def normalize_mac(mac: MacAddress | str) -> MacAddress:
    """
    A MAC address in the one form it is stored and compared in, so
    ``00:1A:2B:3C:4D:5E`` and ``00-1a-2b-3c-4d-5e`` name the same station.
    Raises a pydantic ValidationError if it is not a MAC address.
    """
    return _MAC.validate_python(mac)


#: Shared timestamp annotation: always timezone-aware and in UTC, whether it
#: came from a device, an API caller or a database row.
Timestamp = Annotated[datetime, AfterValidator(as_utc)]


class MatchEnum(str, Enum):
    """
    A string Enum that falls back to its ``UNKNOWN`` member rather than
    raising, matching member names and values case-insensitively. Every
    subclass must define ``UNKNOWN``.
    """

    @classmethod
    def match(cls: type[Self], value: str | None) -> Self:
        """
        Match input string to a member by name or value, ignoring case.
        Returns the ``UNKNOWN`` member if there is no match.
        """
        if value is not None:
            needle = value.strip().casefold()
            for member in cls:
                if needle in (member.name.casefold(), str(member.value).casefold()):
                    return member
        return cls["UNKNOWN"]

    @classmethod
    def _missing_(cls: type[Self], value: object) -> Self:
        return cls.match(value if isinstance(value, str) else None)

    @classmethod
    def is_valid(cls: type[Self], value: str | None) -> bool:
        """Check if the input string matches a member other than ``UNKNOWN``."""
        return cls.match(value) is not cls["UNKNOWN"]

    @classmethod
    def members(cls: type[Self]) -> tuple[Self, ...]:
        return tuple(cls)

    @classmethod
    def names(cls: type[Self]) -> tuple[str, ...]:
        return tuple(member.name for member in cls)


class CameraModel(MatchEnum):
    """
    The camera models supported by the system.
    """

    UNKNOWN = "UNKNOWN"
    OV5640 = "OV5640"
    DSLR = "DSLR"


class DeviceType(MatchEnum):
    """
    The device types supported by the system.
    """

    UNKNOWN = "UNKNOWN"
    ESP32S3 = "ESP32S3"
    ESP32 = "ESP32"
    ESP8266 = "ESP8266"


class UserRole(MatchEnum):
    ADMIN = "admin"
    MEMBER = "member"
    VISITOR = "visitor"
    UNKNOWN = "unknown"


class DBEntity(BaseModel):
    """Base for every model that maps to a database row."""


M = TypeVar("M", bound=DBEntity)


class StationStatus(DBEntity):
    """
    A representation of the state of the sensor onboard a given weather station
    at a given timestamp. Equivalent to a single row in the Status table.

    Attributes:
        MAC (MacAddress): The MAC address of the station.
        timestamp (datetime): The timestamp of the status.
        SHT (bool): Status of the SHT sensor.
        BMP (bool): Status of the BMP sensor.
        CAM (bool): Status of the camera.
        WIFI (bool): Status of the WiFi connection.
    """

    MAC: MacAddress
    timestamp: Timestamp | None = None
    SHT: bool = False
    BMP: bool = False
    CAM: bool = False
    WIFI: bool = False


class StationStatusJSON(TypedDict, total=False):
    """
    JSON representation of the StationStatus.
    """

    MAC: MacAddress | str
    timestamp: datetime | str | None
    SHT: bool
    BMP: bool
    CAM: bool
    WIFI: bool


class Station(DBEntity):
    """
    Represents a station in the database.

    Attributes:
        MAC (MacAddress): The MAC address of the station.
        name (str): The name of the station.
        device_model (DeviceType): The type of device used for the station.
        camera_model (CameraModel): The model of the camera used in the station.
        firmware_version (str): The version of the firmware running on the station.
        altitude (float): The altitude of the station in meters.
        latitude (Latitude): The latitude of the station.
        longitude (Longitude): The longitude of the station.
        sensors (StationStatus | None): The status of the sensors onboard the station.
    """

    MAC: MacAddress
    name: str
    device_model: DeviceType
    camera_model: CameraModel
    firmware_version: str
    altitude: float
    latitude: Latitude
    longitude: Longitude
    sensors: StationStatus | None = None

    @model_validator(mode="after")
    def check_device_model(self: Self) -> Self:
        if self.device_model is DeviceType.UNKNOWN:
            raise ValueError("device_model cannot be DeviceType.UNKNOWN")
        return self


class StationJSON(TypedDict, total=False):
    """
    JSON representation of the Station. Also the shape of a partial update:
    only the keys present are changed.
    """

    MAC: MacAddress | str
    name: str
    device_model: DeviceType | str
    camera_model: CameraModel | str
    firmware_version: str
    altitude: float
    latitude: Latitude | float
    longitude: Longitude | float
    sensors: NotRequired[StationStatusJSON | None]


class Reading(DBEntity):
    """
    Represents a reading from a device in the database.

    Attributes:
        MAC (MacAddress): The MAC address of the station.
        timestamp (datetime): The timestamp of the reading.
        temperature (float): The temperature reading in degrees Celsius.
        humidity (float): The humidity reading in percentage.
        pressure (float): The pressure reading in hPa.
        dewpoint (float): The dew point reading in degrees Celsius.
        filepath (str | None): The file path to the reading data, if applicable.
    """

    MAC: MacAddress
    timestamp: Timestamp
    temperature: float
    humidity: float
    pressure: float
    dewpoint: float
    filepath: str | None = None


class ReadingJSON(TypedDict, total=False):
    """
    JSON representation of the Reading.
    """

    MAC: MacAddress | str
    timestamp: datetime | str
    temperature: float
    humidity: float
    pressure: float
    dewpoint: float
    filepath: str | None


class Location(DBEntity):
    """
    Represents a location in the database.

    Attributes:
        country (str): The country of the location.
        region (str): The region of the location.
        city (str): The city of the location.
        latitude (Latitude): The latitude of the location.
        longitude (Longitude): The longitude of the location.
    """

    country: str
    region: str
    city: str
    latitude: Latitude
    longitude: Longitude


class LocationJSON(TypedDict, total=False):
    """
    JSON representation of the Location.
    """

    country: str
    region: str
    city: str
    latitude: Latitude | float
    longitude: Longitude | float


class User(DBEntity):
    """
    Represents a user in the database. ``password`` is the plain password on
    the way in, and the stored hash on the way out.
    """

    ID: str
    name: str
    email: EmailStr
    password: SecretStr
    role: UserRole = UserRole.VISITOR


class UserJSON(TypedDict, total=False):
    """
    Represents a user in JSON format. Also the shape of a partial update:
    only the keys present are changed.
    """

    ID: str
    name: str
    email: EmailStr
    password: SecretStr | str
    role: UserRole | str
