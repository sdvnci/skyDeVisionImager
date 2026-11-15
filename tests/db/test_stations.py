from typing import no_type_check
from quart.typing import TestClientProtocol as Client
from server.db import CameraModel, DeviceType, Station, StationService, NotFoundError


@no_type_check
async def test_get_station_not_found(client: Client) -> None:
    result = await StationService.get(MAC="00:1A:2B:3C:4D:5E")
    assert (
        result is None
    ), f"MAC address is not registered, should be None but got {result}"


@no_type_check
async def test_insert_and_get_station_sucess(client: Client) -> None:
    station = Station(
        MAC="00:1A:2B:3C:4D:5E",
        name="station",
        device_model=DeviceType.ESP32,
        camera_model=CameraModel.DSLR,
        firmware_version="1.0.0",
        altitude=400,
        latitude=83.3323,
        longitude=82.5546,
        sensors=None,
    )
    await StationService.insert(station=station)
    newstation = await StationService.get(MAC=station.MAC)
    assert newstation is not None, "Could not retrieve newly inserted station"
    assert newstation == station


@no_type_check
async def test_list_stations(client: Client) -> None:
    station = Station(
        MAC="00:1A:2B:3C:4D:5E",
        name="station",
        device_model=DeviceType.ESP32S3,
        camera_model=CameraModel.DSLR,
        firmware_version="1.0.0",
        altitude=400,
        latitude=83.3323,
        longitude=82.5546,
        sensors=None,
    )
    await StationService.insert(station=station)
    stations = await StationService.list()
    assert isinstance(stations, list), "Expected a list of stations"
    assert len(stations) > 0, "Expected at least one station in the database"
    for station in stations:
        assert isinstance(
            station, Station
        ), f"Expected Station instance, got {type(station)}"


@no_type_check
async def test_list_stations_unsuccessful(client: Client) -> None:
    stations = await StationService.list()
    assert isinstance(stations, list), "Expected a list of stations"
    assert len(stations) == 0, f"Expected 0 stations, got {len(stations)}"


@no_type_check
async def test_list_stations_multiple(client: Client) -> None:
    station1 = Station(
        MAC="00:1A:2B:3C:4D:5E",
        name="station1",
        device_model=DeviceType.ESP32,
        camera_model=CameraModel.DSLR,
        firmware_version="1.0.0",
        altitude=400,
        latitude=83.3323,
        longitude=82.5546,
        sensors=None,
    )
    station2 = Station(
        MAC="00:1A:2B:3C:4D:5F",
        name="station2",
        device_model=DeviceType.ESP8266,
        camera_model=CameraModel.OV5640,
        firmware_version="1.0.1",
        altitude=500,
        latitude=84.3323,
        longitude=83.5546,
        sensors=None,
    )
    await StationService.insert(station=station1)
    await StationService.insert(station=station2)

    stations = await StationService.list()
    assert len(stations) == 2, "Expected at least two stations in the database"


@no_type_check
async def test_list_station_offset(client: Client) -> None:
    stations = [
        Station(
            MAC=f"00:1A:2B:3C:4D:{i:02X}",
            name=f"station{i}",
            device_model=DeviceType.ESP32,
            camera_model=CameraModel.DSLR,
            firmware_version="1.0.0",
            altitude=400 + i * 10,
            latitude=83.3323 + i * 0.01,
            longitude=82.5546 + i * 0.01,
            sensors=None,
        )
        for i in range(10)
    ]

    for station in stations:
        await StationService.insert(station=station)

    # Test with offset
    limit = 5
    page = 1
    offset = limit * page
    limited_stations = await StationService.list(limit=limit, page=page)
    assert isinstance(limited_stations, list), "Expected a list of stations"
    assert (
        len(limited_stations) == len(stations) - offset
    ), f"Expected {len(stations) - offset} stations, got {len(limited_stations)}"


@no_type_check
async def test_update_station_successful(client: Client) -> None:
    station = Station(
        MAC="00:1A:2B:3C:4D:5E",
        name="station",
        device_model=DeviceType.ESP32,
        camera_model=CameraModel.DSLR,
        firmware_version="1.0.0",
        altitude=400,
        latitude=83.3323,
        longitude=82.5546,
        sensors=None,
    )
    await StationService.insert(station=station)

    # Update the station's name
    updates = {"name": "updated_station", "device_model": DeviceType.ESP8266}
    await StationService.update(MAC=station.MAC, station=updates)

    updated_station = await StationService.get(MAC=station.MAC)
    assert updated_station is not None, "Updated station could not be retrieved"
    assert (
        updated_station.name == "updated_station"
    ), "Station name was not updated correctly"
    assert (
        updated_station.device_model == DeviceType.ESP8266
    ), "Device model was not updated correctly"


@no_type_check
async def test_update_station_nonexistant(client: Client) -> None:
    MAC = "00:1A:2B:3C:4D:5E"
    updates = {"name": "updated_station", "device_model": DeviceType.ESP8266}

    try:
        await StationService.update(MAC=MAC, station=updates)
        raise AssertionError("updating nonexistant station should raise NotFOundError")
    except NotFoundError as nfr:
        pass


@no_type_check
async def test_delete_station_successful(client: Client) -> None:
    station = Station(
        MAC="00:1A:2B:3C:4D:5E",
        name="station",
        device_model=DeviceType.ESP32,
        camera_model=CameraModel.DSLR,
        firmware_version="1.0.0",
        altitude=400,
        latitude=83.3323,
        longitude=82.5546,
        sensors=None,
    )
    await StationService.insert(station=station)
    tempsta = await StationService.get(MAC=station.MAC)
    assert tempsta is not None, "Station to be deleted could not be retrieved"
    assert tempsta == station, "Retrieved station does not match inserted station"

    # Delete the station
    await StationService.delete(MAC=station.MAC)

    # Try to retrieve the deleted station
    deleted_station = await StationService.get(MAC=station.MAC)
    assert deleted_station is None, "Deleted station should not be retrievable"


@no_type_check
async def test_delete_station_nonexistant(client: Client) -> None:
    MAC = "00:1A:2B:3C:4D:5E"
    station = await StationService.get(MAC=MAC)
    assert station is None, f"Station should not exist, should be None, got {station}"

    try:
        await StationService.delete(MAC=MAC)
        raise AssertionError(
            "Deleting nonexstent sation should have raised NotFoundError"
        )
    except NotFoundError as err:
        pass


@no_type_check
async def test_station_exists_true(client: Client) -> None:
    station = Station(
        MAC="00:1A:2B:3C:4D:5E",
        name="station",
        device_model=DeviceType.ESP32,
        camera_model=CameraModel.DSLR,
        firmware_version="1.0.0",
        altitude=400,
        latitude=83.3323,
        longitude=82.5546,
        sensors=None,
    )
    await StationService.insert(station=station)

    exists = await StationService.exists(MAC=station.MAC)
    assert exists, "Station should exist after insertion"


@no_type_check
async def test_station_exists_false(client: Client) -> None:
    MAC = "00:1A:2B:3C:4D:5E"
    exists = await StationService.exists(MAC=MAC)
    assert exists == False, "Station should exist after insertion"


@no_type_check
async def test_station_exists_after_delete(client: Client) -> None:
    station = Station(
        MAC="00:1A:2B:3C:4D:5E",
        name="station",
        device_model=DeviceType.ESP32,
        camera_model=CameraModel.DSLR,
        firmware_version="1.0.0",
        altitude=400,
        latitude=83.3323,
        longitude=82.5546,
        sensors=None,
    )
    await StationService.insert(station=station)

    exists = await StationService.exists(MAC=station.MAC)
    assert exists, "Station should exist after insertion"

    # Delete the station
    await StationService.delete(MAC=station.MAC)

    exists_after_deletion = await StationService.exists(MAC=station.MAC)
    assert not exists_after_deletion, "Station should not exist after deletion"
