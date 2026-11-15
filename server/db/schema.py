from typing import Final

#: One statement per table, each safe to run against an existing database.
SCHEMA: Final[tuple[str, ...]] = (
    """
    CREATE TABLE IF NOT EXISTS Stations(
        MAC TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        device_model TEXT NOT NULL,
        camera_model TEXT NOT NULL,
        firmware_version TEXT NOT NULL,
        altitude REAL NOT NULL,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Readings(
        MAC TEXT,
        timestamp TEXT,
        temperature REAL,
        humidity REAL,
        pressure REAL,
        dewpoint REAL,
        filepath TEXT,
        PRIMARY KEY (MAC, timestamp),
        FOREIGN KEY (MAC) REFERENCES Stations(MAC)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Status(
        MAC TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        SHT INTEGER NOT NULL,
        BMP INTEGER NOT NULL,
        CAM INTEGER NOT NULL,
        WIFI INTEGER NOT NULL,
        FOREIGN KEY (MAC) REFERENCES Stations(MAC)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Locations(
        country TEXT,
        region TEXT,
        city TEXT,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        PRIMARY KEY (latitude, longitude)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Users(
        ID TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL
    );
    """,
)
