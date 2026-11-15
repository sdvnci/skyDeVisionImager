from typing import Any, TypedDict


class Airport(TypedDict, total=False):
    id: str
    iata: str
    name: str
    name_translated: str
    city_name: str
    admin1: str
    admin2: str
    country_id: str
    country_name: str
    lat: float
    lng: float
    metar: bool
    taf: bool
    timezone: int
    fir: str
    elevation: int
    type: int
    last_notam: int


class MetarCloud(TypedDict, total=False):
    id: int
    height: int
    report: str
    amount: str


class RunwayVisibility(TypedDict, total=False):
    id: str
    value: int
    prefix: str | None
    min: int | None
    max: int | None
    trend: str


class Metar(TypedDict, total=False):
    cavok: bool
    ceiling: int
    ceiling_color: str
    clouds: list[MetarCloud]
    code: str
    code_colour: str
    colour_state: str | None
    dewpoint: int
    dewpoint_exact: float | None
    humidity: int
    is_day: bool
    observed: int
    qnh: int
    raw: str
    recent_weather_report: str | None
    remarks: str | None
    runway_condition: list[Any] | None
    runway_visibility: list[RunwayVisibility]
    snoclo: bool
    station_id: str
    sunrise: int
    sunset: int
    temperature: int
    temperature_exact: float | None
    trends: list[Any]
    vertical_visibility: int | None
    visibility: int
    visibility_sign: str | None
    visibility_color: str
    visibility_min: int | None
    warnings: list[str]
    weather: str | None
    weather_image: str
    weather_report: str | None
    wind_color: str
    wind_dir: int
    wind_dir_max: int | None
    wind_dir_min: int | None
    wind_gust: int | None
    wind_speed: int
    ws_all: str
    ws_runways: str | None
    id: int


class Runway(TypedDict, total=False):
    id_l: str
    id_h: str
    hdg_l: int
    hdg_h: int
    in_use: int
    xwnd: int
    hwnd: float


class Station(TypedDict, total=False):
    id: str
    name: str
    taf: bool


class MetarResponse(TypedDict, total=False):
    status: bool
    credits: int
    airport: Airport
    metar: Metar
    runways: list[Runway]
    stations: list[Station]


class UnregisteredAirportException(Exception): ...
