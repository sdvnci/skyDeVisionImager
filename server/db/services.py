import builtins
from abc import ABC
from asyncio import to_thread
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from enum import Enum
from sqlite3 import Error as SQLError
from sqlite3 import IntegrityError, Row
from typing import Any, ClassVar, Final, Generic, cast, final, override

from aiosqlite import Cursor
from pydantic import SecretStr, ValidationError
from pydantic_extra_types.mac_address import MacAddress

from ._query import QueryBuilder, quote
from .entities import (
    M,
    Reading,
    Station,
    StationJSON,
    StationStatus,
    User,
    UserJSON,
    dt2str,
    normalize_mac,
)
from .manager import Manager

DEFAULT_PAGE_SIZE: Final = 20

# sqlite3 extended result codes for a row that is already there.
_SQLITE_CONSTRAINT_PRIMARYKEY: Final = 1555
_SQLITE_CONSTRAINT_UNIQUE: Final = 2067


class DatabaseError(Exception):
    """
    Base for every error a Service raises.
    """


class NotFoundError(DatabaseError): ...


class AlreadyExistsError(DatabaseError): ...


class InvalidInputError(DatabaseError): ...


class InternalDBError(DatabaseError): ...


class UnauthorizedError(DatabaseError): ...


def _db_value(value: object) -> object:
    """A model field's value in the form it is stored in."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return dt2str(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return float(value)  # Latitude and Longitude are float subclasses
    if isinstance(value, str):
        return str(value)  # likewise MacAddress
    return value


def _mac(mac: MacAddress | str, purpose: str) -> str:
    """The stored form of a MAC address needed for ``purpose``."""
    if not mac:
        raise InvalidInputError(f"MAC must be provided to {purpose}.")
    try:
        return str(normalize_mac(mac))
    except ValidationError as err:
        raise InvalidInputError(f"{mac!r} is not a MAC address.") from err


def _page(limit: int, page: int) -> tuple[int, int]:
    """Validate a page request, returning its (limit, offset)."""
    if limit < 1:
        raise InvalidInputError(f"limit must be at least 1, not {limit}.")
    if page < 0:
        raise InvalidInputError(f"page must be at least 0, not {page}.")
    return limit, page * limit


class Service(ABC, Generic[M]):
    """
    Abstract base class for a service: the in-between class for transactions
    on one database table. Do not instantiate; every method is a classmethod,
    and every one that touches the database is a coroutine.

    Subclasses name their table and the model its rows deserialize into:

        class StationService(
            Service[Station], table_name="Stations", table_row_type=Station
        ): ...

    Every method takes an optional ``cursor``. Without one, the call is its
    own transaction; with one, it joins that cursor's transaction, which is
    how operations spanning tables stay atomic:

        async with Manager.cursor() as cur:
            await StationService.insert(station, cursor=cur)
            await ReadingService.insert(reading, cursor=cur)

    Methods:
        - get: Get an entity from the database.
        - list: Get a slice of entities from the database.
        - insert: Add an entity to the database.
        - update: Update an entity in the database.
        - delete: Delete an entity from the database.
        - exists: Check if an entity exists in the database.

    Raises (any method):
        InvalidInputError: If the arguments, or what they refer to, are unusable.
        AlreadyExistsError: If a row with the same key is already there.
        NotFoundError: If the row to update or delete is not there.
        InternalDBError: If the database itself fails.
    """

    __slots__: Final[tuple[str, ...]] = ()
    _table_name: ClassVar[str]
    _model_cls: type[M]

    def __init__(self) -> None:
        raise NotImplementedError("Do not make instance objects of Service types.")

    def __init_subclass__(cls, table_name: str, table_row_type: type[M]) -> None:
        super().__init_subclass__()
        cls._table_name = quote(table_name)
        cls._model_cls = table_row_type

    @classmethod
    def table_name(cls: type["Service[M]"]) -> str:
        """
        Returns the quoted name of the database table associated with this class.
        """
        return cls._table_name

    @classmethod
    def model_cls(cls: type["Service[M]"]) -> type[M]:
        """
        Returns the Pydantic model used to deserialize rows for this table.
        """
        return cls._model_cls

    @classmethod
    def select(cls: type["Service[M]"]) -> QueryBuilder:
        """
        Return a QueryBuilder seeded with ``SELECT * FROM {table}`` for this
        table. Chain filters, ordering and limits, then hand ``build()``'s
        output to ``cursor.execute()``.
        """
        return QueryBuilder(f"SELECT * FROM {cls._table_name}")

    @classmethod
    @asynccontextmanager
    async def _cursor(
        cls: type["Service[M]"], cursor: Cursor | None = None
    ) -> AsyncGenerator[Cursor, None]:
        """
        The given cursor, or a new one in its own transaction when there is
        none. sqlite3 errors raised inside the block become DatabaseErrors.
        """
        try:
            if cursor is not None:
                yield cursor
            else:
                async with Manager.cursor() as cur:
                    yield cur
        except IntegrityError as err:
            if err.sqlite_errorcode in (
                _SQLITE_CONSTRAINT_PRIMARYKEY,
                _SQLITE_CONSTRAINT_UNIQUE,
            ):
                raise AlreadyExistsError(
                    f"Already exists in {cls._table_name}: {err}"
                ) from err
            raise InvalidInputError(
                f"Constraint failed on {cls._table_name}: {err}"
            ) from err
        except SQLError as err:
            Manager.error(f"{cls.__name__} database failure: {err}")
            raise InternalDBError(f"Database failure: {err}") from err

    @classmethod
    def _model(cls: type["Service[M]"], row: Row | Mapping[str, object]) -> M:
        """Deserialize one result row into this table's model."""
        try:
            return cls._model_cls.model_validate(dict(row))
        except ValidationError as err:
            raise InternalDBError(
                f"Failed to deserialize row into {cls._model_cls.__name__}: {err}"
            ) from err

    @classmethod
    async def _fetch(
        cls: type["Service[M]"],
        builder: QueryBuilder,
        cursor: Cursor | None = None,
    ) -> builtins.list[M]:
        query, params = builder.build()
        async with cls._cursor(cursor) as cur:
            rows = await (await cur.execute(query, params)).fetchall()
        return [cls._model(row) for row in rows]

    @classmethod
    async def _has(
        cls: type["Service[M]"],
        builder: QueryBuilder,
        cursor: Cursor | None = None,
    ) -> bool:
        query, params = builder.limit(1).build()
        async with cls._cursor(cursor) as cur:
            row = await (await cur.execute(query, params)).fetchone()
        return row is not None

    @classmethod
    def _assignments(
        cls: type["Service[M]"],
        values: Mapping[str, object],
        *,
        exclude: frozenset[str] = frozenset(),
    ) -> tuple[str, builtins.list[object]]:
        """
        The ``SET`` clause and parameters for a partial update. Keys must be
        columns of this table's model; None values are left unchanged.
        """
        columns = frozenset(cls._model_cls.model_fields) - exclude
        unknown = sorted(set(values) - columns)
        if unknown:
            raise InvalidInputError(
                f"Cannot update {', '.join(unknown)} on {cls._table_name}."
            )

        changes = {key: value for key, value in values.items() if value is not None}
        if not changes:
            raise InvalidInputError("No fields to update.")

        clause = ", ".join(f"{quote(column)} = ?" for column in changes)
        return clause, [_db_value(value) for value in changes.values()]

    @classmethod
    async def get(cls: type["Service[M]"], *args: Any, **kwargs: Any) -> M | None:
        raise NotImplementedError(f".get() method not implemented for {cls.__name__}")

    @classmethod
    async def list(
        cls: type["Service[M]"], *args: Any, **kwargs: Any
    ) -> builtins.list[M]:
        raise NotImplementedError(f".list() method not implemented for {cls.__name__}")

    @classmethod
    async def insert(cls: type["Service[M]"], *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            f".insert() method not implemented for {cls.__name__}"
        )

    @classmethod
    async def update(cls: type["Service[M]"], *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            f".update() method not implemented for {cls.__name__}"
        )

    @classmethod
    async def delete(cls: type["Service[M]"], *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            f".delete() method not implemented for {cls.__name__}"
        )

    @classmethod
    async def exists(cls: type["Service[M]"], *args: Any, **kwargs: Any) -> bool:
        raise NotImplementedError(
            f".exists() method not implemented for {cls.__name__}"
        )

    @classmethod
    async def insert_batch(cls: type["Service[M]"], *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            f".insert_batch() method not implemented for {cls.__name__}"
        )

    @classmethod
    async def update_batch(cls: type["Service[M]"], *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            f".update_batch() method not implemented for {cls.__name__}"
        )

    @classmethod
    async def delete_batch(cls: type["Service[M]"], *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            f".delete_batch() method not implemented for {cls.__name__}"
        )


@final
class UserService(Service[User], table_name="Users", table_row_type=User):
    """
    Service class for managing User entities in the database. Users are
    looked up by ``id`` or, failing that, ``email``. Passwords are hashed on
    the way in; a User read back carries the hash, never the password.
    """

    @classmethod
    def _identify(
        cls: type["UserService"], id: str | None, email: str | None
    ) -> tuple[str, str]:
        """The (column, value) that singles out a user."""
        if id:
            return "ID", id
        if email:
            return "email", email
        raise InvalidInputError("Either 'id' or 'email' must be provided.")

    @classmethod
    async def _params(cls: type["UserService"], user: User) -> tuple[object, ...]:
        return (
            user.ID,
            user.name,
            user.email,
            await to_thread(Manager.hash_password, user.password.get_secret_value()),
            user.role.value,
        )

    @classmethod
    async def _stored(cls: type["UserService"], values: UserJSON) -> dict[str, object]:
        """A partial update with its password, if any, replaced by its hash."""
        stored: dict[str, object] = dict(values)
        password = stored.get("password")
        if isinstance(password, SecretStr):
            password = password.get_secret_value()
        if isinstance(password, str):
            stored["password"] = await to_thread(Manager.hash_password, password)
        return stored

    @override
    @classmethod
    async def get(
        cls: type["UserService"],
        id: str | None = None,
        email: str | None = None,
        *,
        cursor: Cursor | None = None,
    ) -> User | None:
        """
        Get a user from the database.
        Args:
            id (str): The ID of the user to retrieve.
            email (str): The email of the user to retrieve.

        Returns:
            User | None: The user object if found, None otherwise.
        """
        column, value = cls._identify(id, email)
        users = await cls._fetch(cls.select().eq(column, value).limit(1), cursor)
        return users[0] if users else None

    @override
    @classmethod
    async def list(
        cls: type["UserService"],
        limit: int = DEFAULT_PAGE_SIZE,
        page: int = 0,
        *,
        cursor: Cursor | None = None,
    ) -> builtins.list[User]:
        """
        Get a slice of users from the database, ordered by name.
        Args:
            limit (int): The maximum number of users to retrieve.
            page (int): The page number to retrieve (0-indexed).
        """
        limit, offset = _page(limit, page)
        return await cls._fetch(
            cls.select().order_by("name").limit(limit).offset(offset), cursor
        )

    @override
    @classmethod
    async def insert(
        cls: type["UserService"], user: User, *, cursor: Cursor | None = None
    ) -> None:
        """
        Insert a user into the database.
        Args:
            user (User): The user to insert.
        """
        await cls.insert_batch([user], cursor=cursor)

    @override
    @classmethod
    async def insert_batch(
        cls: type["UserService"],
        users: Sequence[User],
        *,
        cursor: Cursor | None = None,
    ) -> None:
        """
        Insert a batch of users into the database, all or none.
        Args:
            users (Sequence[User]): The users to insert.
        """
        if not users:
            raise InvalidInputError("No users provided for insertion.")

        Manager.debug(f"Inserting users :: {[user.ID for user in users]}")
        query = f"INSERT INTO {cls._table_name} VALUES (?, ?, ?, ?, ?);"
        async with cls._cursor(cursor) as cur:
            _ = await cur.executemany(
                query, [await cls._params(user) for user in users]
            )

    @override
    @classmethod
    async def update(
        cls: type["UserService"],
        user: UserJSON,
        id: str | None = None,
        email: str | None = None,
        *,
        cursor: Cursor | None = None,
    ) -> None:
        """
        Update a user in the database.
        Args:
            user (UserJSON): The fields to change. A password is hashed.
            id (str): The ID of the user to update.
            email (str): The email of the user to update.
        Raises:
            NotFoundError: If no user is found to update.
        """
        column, value = cls._identify(id, email)
        if not user:
            raise InvalidInputError("No user provided for update.")

        clause, params = cls._assignments(await cls._stored(user))
        query, params_tup = (
            QueryBuilder(f"UPDATE {cls._table_name} SET {clause}", *params)
            .eq(column, value, skip_none=False)
            .build()
        )
        async with cls._cursor(cursor) as cur:
            _ = await cur.execute(query, params_tup)
            if cur.rowcount == 0:
                raise NotFoundError("No user found to update.")

    @override
    @classmethod
    async def update_batch(
        cls: type["UserService"],
        users: Sequence[tuple[str | None, str | None, UserJSON]],
        *,
        cursor: Cursor | None = None,
    ) -> None:
        """
        Update multiple users in the database, all or none.

        Args:
            users (Sequence[tuple[str | None, str | None, UserJSON]]): For each user, their ID, their email, and the fields to change.
        Raises:
            NotFoundError: If any of the users is not found.
        """
        if not users:
            raise InvalidInputError("No user data provided.")

        async with cls._cursor(cursor) as cur:
            for id, email, user in users:
                await cls.update(user, id=id, email=email, cursor=cur)

    @override
    @classmethod
    async def delete(
        cls: type["UserService"],
        id: str | None = None,
        email: str | None = None,
        *,
        cursor: Cursor | None = None,
    ) -> None:
        """
        Delete a user from the database.
        Args:
            id (str): The ID of the user to delete.
            email (str): The email of the user to delete.
        Raises:
            NotFoundError: If no user is found to delete.
        """
        column, value = cls._identify(id, email)
        query, params = (
            QueryBuilder(f"DELETE FROM {cls._table_name}")
            .eq(column, value, skip_none=False)
            .build()
        )
        async with cls._cursor(cursor) as cur:
            _ = await cur.execute(query, params)
            if cur.rowcount == 0:
                raise NotFoundError("No user found to delete.")

    @override
    @classmethod
    async def exists(
        cls: type["UserService"],
        id: str | None = None,
        email: str | None = None,
        *,
        cursor: Cursor | None = None,
    ) -> bool:
        column, value = cls._identify(id, email)
        return await cls._has(cls.select().eq(column, value), cursor)


@final
class StatusService(
    Service[StationStatus], table_name="Status", table_row_type=StationStatus
):
    """
    Service class for managing StationStatus entities in the database.
    """

    _UPDATE: ClassVar[str] = (
        "UPDATE Status SET timestamp = ?, SHT = ?, BMP = ?, CAM = ?, WIFI = ? "
        "WHERE MAC = ?;"
    )

    @classmethod
    def _flags(cls: type["StatusService"], status: StationStatus) -> tuple[object, ...]:
        """A status's timestamp (now, if it has none) and sensor flags."""
        return (
            dt2str(status.timestamp or datetime.now(tz=UTC)),
            int(status.SHT),
            int(status.BMP),
            int(status.CAM),
            int(status.WIFI),
        )

    @override
    @classmethod
    async def get(
        cls: type["StatusService"],
        MAC: MacAddress | str,
        *,
        cursor: Cursor | None = None,
    ) -> StationStatus | None:
        """
        Get a station status from the database.
        Args:
            MAC (MacAddress): The MAC address of the station to retrieve.

        Returns:
            StationStatus | None: The station status object if found, None otherwise.
        """
        statuses = await cls._fetch(
            cls.select().eq("MAC", _mac(MAC, "retrieve a station status")).limit(1),
            cursor,
        )
        return statuses[0] if statuses else None

    @override
    @classmethod
    async def list(
        cls: type["StatusService"],
        limit: int = DEFAULT_PAGE_SIZE,
        page: int = 0,
        *,
        cursor: Cursor | None = None,
    ) -> builtins.list[StationStatus]:
        """
        Get a slice of station statuses from the database, oldest first.
        Args:
            limit (int): The maximum number of station statuses to retrieve.
            page (int): The page number to retrieve (0-indexed).
        """
        limit, offset = _page(limit, page)
        return await cls._fetch(
            cls.select().order_by("timestamp").limit(limit).offset(offset), cursor
        )

    @override
    @classmethod
    async def insert(
        cls: type["StatusService"],
        status: StationStatus,
        *,
        cursor: Cursor | None = None,
    ) -> None:
        """
        Insert a station status into the database.
        Args:
            status (StationStatus): The station status to insert.
        """
        await cls.insert_batch([status], cursor=cursor)

    @override
    @classmethod
    async def insert_batch(
        cls: type["StatusService"],
        statuses: Sequence[StationStatus],
        *,
        cursor: Cursor | None = None,
    ) -> None:
        """
        Insert a batch of station statuses into the database, all or none.
        Args:
            statuses (Sequence[StationStatus]): The station statuses to insert.
        """
        if not statuses:
            raise InvalidInputError("No station statuses provided for insertion.")

        Manager.debug(f"Inserting station statuses :: {statuses}")
        query = f"INSERT INTO {cls._table_name} VALUES (?, ?, ?, ?, ?, ?);"
        params = [(str(status.MAC), *cls._flags(status)) for status in statuses]
        async with cls._cursor(cursor) as cur:
            _ = await cur.executemany(query, params)

    @override
    @classmethod
    async def update(
        cls: type["StatusService"],
        MAC: MacAddress | str,
        status: StationStatus,
        *,
        cursor: Cursor | None = None,
    ) -> None:
        """
        Update a station status in the database.
        Args:
            MAC (MacAddress): The MAC address of the station to update.
            status (StationStatus): The station status data to update.
        Raises:
            NotFoundError: If no station status is found to update.
        """
        await cls.update_batch([(MAC, status)], cursor=cursor)

    @override
    @classmethod
    async def update_batch(
        cls: type["StatusService"],
        statuses: Sequence[tuple[MacAddress | str, StationStatus]],
        *,
        cursor: Cursor | None = None,
    ) -> None:
        """
        Update multiple station statuses in the database, all or none.

        Args:
            statuses (Sequence[tuple[MacAddress, StationStatus]]): For each status, the MAC address of its station and the status data.
        Raises:
            NotFoundError: If any of the station statuses is not found.
        """
        if not statuses:
            raise InvalidInputError("No station statuses provided for update.")
        params = [
            (*cls._flags(status), _mac(mac, "update a station status"))
            for mac, status in statuses
        ]
        async with cls._cursor(cursor) as cur:
            _ = await cur.executemany(cls._UPDATE, params)
            if cur.rowcount != len(params):
                raise NotFoundError("No station status found to update.")

    @override
    @classmethod
    async def delete(
        cls: type["StatusService"],
        MAC: MacAddress | str,
        *,
        cursor: Cursor | None = None,
    ) -> None:
        """
        Delete a station status from the database.
        Args:
            MAC (MacAddress): The MAC address of the station status to delete.
        Raises:
            NotFoundError: If no station status is found to delete.
        """
        await cls.delete_batch([MAC], cursor=cursor)

    @override
    @classmethod
    async def delete_batch(
        cls: type["StatusService"],
        MACS: Sequence[MacAddress | str],
        *,
        cursor: Cursor | None = None,
    ) -> None:
        """
        Delete a list of station statuses from the database, all or none.
        Args:
            MACS (Sequence[MacAddress]): The MAC addresses of the station statuses to delete.
        Raises:
            NotFoundError: If any of the station statuses is not found.
        """
        if not MACS:
            raise InvalidInputError("MAC must be provided to delete a station status.")

        query = f"DELETE FROM {cls._table_name} WHERE MAC = ?;"
        params = [(_mac(mac, "delete a station status"),) for mac in MACS]
        async with cls._cursor(cursor) as cur:
            _ = await cur.executemany(query, params)
            if cur.rowcount != len(params):
                raise NotFoundError("No station status found to delete.")

    @override
    @classmethod
    async def exists(
        cls: type["StatusService"],
        MAC: MacAddress | str,
        *,
        cursor: Cursor | None = None,
    ) -> bool:
        """
        Check if a station status exists in the database.
        Args:
            MAC (MacAddress): The MAC address of the station status to check.
        """
        return await cls._has(
            cls.select().eq("MAC", _mac(MAC, "check station status existence")), cursor
        )

    @classmethod
    async def exists_batch(
        cls: type["StatusService"],
        MACS: Sequence[MacAddress | str],
        *,
        cursor: Cursor | None = None,
    ) -> builtins.list[bool]:
        """
        Check if a set of station statuses exist in the database.
        Args:
            MACS (Sequence[MacAddress]): The MAC addresses of the station statuses to check.
        Returns:
            list[bool]: For each MAC address, in order, whether its status exists.
        """
        if not MACS:
            raise InvalidInputError(
                "MAC must be provided to check station status existence."
            )

        macs: list[object] = [
            _mac(mac, "check station status existence") for mac in MACS
        ]
        query, params = (
            QueryBuilder(f"SELECT MAC FROM {cls._table_name}").in_("MAC", macs).build()
        )
        async with cls._cursor(cursor) as cur:
            rows = await (await cur.execute(query, params)).fetchall()
        found = {row["MAC"] for row in rows}
        return [mac in found for mac in macs]


@final
class StationService(Service[Station], table_name="Stations", table_row_type=Station):
    """
    Service class for managing Station entities in the database. A station
    is read together with its sensor status, in one query.
    """

    _STATUS_FLAGS: ClassVar[tuple[str, ...]] = ("SHT", "BMP", "CAM", "WIFI")

    @override
    @classmethod
    def select(cls: type["StationService"]) -> QueryBuilder:
        return QueryBuilder(
            "SELECT s.*, st.timestamp AS status_timestamp, "
            "st.SHT, st.BMP, st.CAM, st.WIFI "
            "FROM Stations AS s LEFT JOIN Status AS st ON st.MAC = s.MAC"
        )

    @override
    @classmethod
    def _model(cls: type["StationService"], row: Row | Mapping[str, object]) -> Station:
        record = dict(row)
        timestamp = record.pop("status_timestamp", None)
        flags = {flag: record.pop(flag, None) for flag in cls._STATUS_FLAGS}
        if timestamp is not None:
            record["sensors"] = {
                "MAC": record["MAC"],
                "timestamp": timestamp,
                **flags,
            }
        return super()._model(record)

    @override
    @classmethod
    async def get(
        cls: type["StationService"],
        MAC: MacAddress | str,
        *,
        cursor: Cursor | None = None,
    ) -> Station | None:
        """
        Get a station from the database.
        Args:
            MAC (MacAddress): The MAC address of the station to retrieve.

        Returns:
            Station | None: The station object if found, None otherwise.
        """
        stations = await cls._fetch(
            cls.select().eq("s.MAC", _mac(MAC, "retrieve a station")).limit(1), cursor
        )
        return stations[0] if stations else None

    @override
    @classmethod
    async def list(
        cls: type["StationService"],
        limit: int = DEFAULT_PAGE_SIZE,
        page: int = 0,
        *,
        cursor: Cursor | None = None,
    ) -> builtins.list[Station]:
        """
        Get a slice of stations from the database, ordered by name.
        Args:
            limit (int): The maximum number of stations to retrieve.
            page (int): The page number to retrieve (0-indexed).
        """
        limit, offset = _page(limit, page)
        return await cls._fetch(
            cls.select().order_by("s.name").limit(limit).offset(offset), cursor
        )

    @override
    @classmethod
    async def insert(
        cls: type["StationService"],
        station: Station,
        *,
        cursor: Cursor | None = None,
    ) -> None:
        """
        Insert a station and its sensor status into the database, both or
        neither. A station without ``sensors`` is given a status with every
        sensor down, stamped now.
        Args:
            station (Station): The station to insert.
        Raises:
            AlreadyExistsError: If a station with this MAC is already registered.
        """
        Manager.debug(f"Inserting station :: {station}")
        status = station.sensors or StationStatus(
            MAC=station.MAC, timestamp=datetime.now(tz=UTC)
        )

        query = f"INSERT INTO {cls._table_name} VALUES (?, ?, ?, ?, ?, ?, ?, ?);"
        async with cls._cursor(cursor) as cur:
            _ = await cur.execute(
                query,
                (
                    str(station.MAC),
                    station.name,
                    station.device_model.value,
                    station.camera_model.value,
                    station.firmware_version,
                    station.altitude,
                    float(station.latitude),
                    float(station.longitude),
                ),
            )
            await StatusService.insert(status, cursor=cur)

        station.sensors = status

    @override
    @classmethod
    async def update(
        cls: type["StationService"],
        MAC: MacAddress | str,
        station: StationJSON,
        *,
        cursor: Cursor | None = None,
    ) -> None:
        """
        Update a station in the database.
        Args:
            MAC (MacAddress): The MAC address of the station to update.
            station (StationJSON): The fields to change. ``sensors`` is ignored; use StatusService.
        Raises:
            NotFoundError: If no station is found to update.
        """
        if not station:
            raise InvalidInputError("No station provided for update.")

        values = {key: value for key, value in station.items() if key != "sensors"}
        clause, params = cls._assignments(values, exclude=frozenset({"sensors"}))
        query, params_tup = (
            QueryBuilder(f"UPDATE {cls._table_name} SET {clause}", *params)
            .eq("MAC", _mac(MAC, "update a station"), skip_none=False)
            .build()
        )
        async with cls._cursor(cursor) as cur:
            _ = await cur.execute(query, params_tup)
            if cur.rowcount == 0:
                raise NotFoundError("No station found to update.")

    @override
    @classmethod
    async def delete(
        cls: type["StationService"],
        MAC: MacAddress | str,
        *,
        readings: bool = False,
        cursor: Cursor | None = None,
    ) -> None:
        """
        Delete a station and its sensor status from the database, both or
        neither.
        Args:
            MAC (MacAddress): The MAC address of the station to delete.
            readings (bool): Also delete the station's readings. Without this, a station that has readings cannot be deleted.
        Raises:
            NotFoundError: If no station is found to delete.
            InvalidInputError: If the station still has readings.
        """

        mac = (_mac(MAC, "delete a station"),)
        async with cls._cursor(cursor) as cur:
            if readings:
                _ = await cur.execute("DELETE FROM Readings WHERE MAC = ?;", mac)
            _ = await cur.execute("DELETE FROM Status WHERE MAC = ?;", mac)
            _ = await cur.execute(f"DELETE FROM {cls._table_name} WHERE MAC = ?;", mac)
            if cur.rowcount == 0:
                raise NotFoundError("No station found to delete.")

    @override
    @classmethod
    async def exists(
        cls: type["StationService"],
        MAC: MacAddress | str,
        *,
        cursor: Cursor | None = None,
    ) -> bool:
        """
        Check if a station exists in the database.
        Args:
            MAC (MacAddress): The MAC address of the station to check.
        """
        return await cls._has(
            cls.select().eq("s.MAC", _mac(MAC, "check if a station exists")), cursor
        )


@final
class ReadingService(Service[Reading], table_name="Readings", table_row_type=Reading):
    """
    Service class for managing Reading entities in the database. A reading
    is keyed by its station's MAC address and its timestamp.
    """

    @classmethod
    def _key(
        cls: type["ReadingService"],
        MAC: MacAddress | str,
        timestamp: datetime,
    ) -> QueryBuilder:
        """A SELECT for the reading with this key."""
        if not timestamp:
            raise InvalidInputError("timestamp must be provided to identify a reading.")
        return (
            cls.select()
            .eq("MAC", _mac(MAC, "identify a reading"))
            .eq("timestamp", dt2str(timestamp))
        )

    @classmethod
    def _params(cls: type["ReadingService"], reading: Reading) -> tuple[object, ...]:
        return (
            str(reading.MAC),
            dt2str(reading.timestamp),
            reading.temperature,
            reading.humidity,
            reading.pressure,
            reading.dewpoint,
            reading.filepath,
        )

    @override
    @classmethod
    async def get(
        cls: type["ReadingService"],
        MAC: MacAddress | str,
        timestamp: datetime,
        *,
        cursor: Cursor | None = None,
    ) -> Reading | None:
        """
        Get a reading from the database.
        Args:
            MAC (MacAddress): The MAC address of the station the reading belongs to.
            timestamp (datetime): The timestamp of the reading to retrieve.

        Returns:
            Reading | None: The reading object if found, None otherwise.
        """
        readings = await cls._fetch(cls._key(MAC, timestamp).limit(1), cursor)
        return readings[0] if readings else None

    @override
    @classmethod
    async def list(
        cls: type["ReadingService"],
        MAC: MacAddress | str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
        page: int = 0,
        *,
        cursor: Cursor | None = None,
    ) -> builtins.list[Reading]:
        """
        Get a slice of readings from the database for a given station,
        newest first.
        Args:
            MAC (MacAddress): The MAC address of the station to filter readings by.
            start (datetime): The start timestamp to filter readings by, inclusive.
            end (datetime): The end timestamp to filter readings by, inclusive.
            limit (int): The maximum number of readings to retrieve.
            page (int): The page number to retrieve (0-indexed).
        """

        limit, offset = _page(limit, page)
        return await cls._fetch(
            cls.select()
            .eq("MAC", _mac(MAC, "list readings"))
            .gte("timestamp", dt2str(start) if start else None)
            .lte("timestamp", dt2str(end) if end else None)
            .order_by("timestamp", desc=True)
            .limit(limit)
            .offset(offset),
            cursor,
        )

    @override
    @classmethod
    async def insert(
        cls: type["ReadingService"],
        reading: Reading,
        *,
        cursor: Cursor | None = None,
    ) -> None:
        """
        Insert a reading into the database.
        Args:
            reading (Reading): The reading to insert.
        Raises:
            AlreadyExistsError: If the station already has a reading at this timestamp.
            InvalidInputError: If the reading's station is not registered.
        """
        await cls.insert_batch([reading], cursor=cursor)

    @override
    @classmethod
    async def insert_batch(
        cls: type["ReadingService"],
        readings: Sequence[Reading],
        *,
        cursor: Cursor | None = None,
    ) -> None:
        """
        Insert a batch of readings into the database, all or none.
        Args:
            readings (Sequence[Reading]): The readings to insert.
        """
        if not readings:
            raise InvalidInputError("No readings provided for insertion.")

        Manager.debug(f"Inserting readings :: {readings}")
        query = f"INSERT INTO {cls._table_name} VALUES (?, ?, ?, ?, ?, ?, ?);"
        async with cls._cursor(cursor) as cur:
            _ = await cur.executemany(
                query, [cls._params(reading) for reading in readings]
            )

    @classmethod
    async def upsert(
        cls: type["ReadingService"],
        reading: Reading,
        *,
        cursor: Cursor | None = None,
    ) -> None:
        """
        Insert a reading, or replace the one its station already has at the
        same timestamp. This makes a device resending a reading harmless.
        Args:
            reading (Reading): The reading to store.
        Raises:
            InvalidInputError: If the reading's station is not registered.
        """
        query = (
            f"INSERT INTO {cls._table_name} VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (MAC, timestamp) DO UPDATE SET "
            "temperature = excluded.temperature, humidity = excluded.humidity, "
            "pressure = excluded.pressure, dewpoint = excluded.dewpoint, "
            "filepath = excluded.filepath;"
        )
        async with cls._cursor(cursor) as cur:
            _ = await cur.execute(query, cls._params(reading))

    @override
    @classmethod
    async def update(
        cls: type["ReadingService"],
        MAC: MacAddress | str,
        timestamp: datetime,
        reading: Reading,
        *,
        cursor: Cursor | None = None,
    ) -> None:
        """
        Update a reading in the database.
        Args:
            MAC (MacAddress): The MAC address of the station the reading belongs to.
            timestamp (datetime): The timestamp of the reading to update.
            reading (Reading): The reading data to update.
        Raises:
            NotFoundError: If no reading is found to update.
        """
        if not timestamp:
            raise InvalidInputError("timestamp must be provided to update a reading.")

        query = (
            f"UPDATE {cls._table_name} SET temperature = ?, humidity = ?, "
            "pressure = ?, dewpoint = ?, filepath = ? WHERE MAC = ? AND timestamp = ?;"
        )
        params = (
            *cls._params(reading)[2:],
            _mac(MAC, "update a reading"),
            dt2str(timestamp),
        )
        async with cls._cursor(cursor) as cur:
            _ = await cur.execute(query, params)
            if cur.rowcount == 0:
                raise NotFoundError("No reading found to update.")

    @override
    @classmethod
    async def delete(
        cls: type["ReadingService"],
        MAC: MacAddress | str,
        timestamp: datetime,
        *,
        cursor: Cursor | None = None,
    ) -> None:
        """
        Delete a reading from the database.
        Args:
            MAC (MacAddress): The MAC address of the station the reading belongs to.
            timestamp (datetime): The timestamp of the reading to delete.
        Raises:
            NotFoundError: If no reading is found to delete.
        """
        if not timestamp:
            raise InvalidInputError("timestamp must be provided to delete a reading.")

        query = f"DELETE FROM {cls._table_name} WHERE MAC = ? AND timestamp = ?;"
        async with cls._cursor(cursor) as cur:
            _ = await cur.execute(
                query, (_mac(MAC, "delete a reading"), dt2str(timestamp))
            )
            if cur.rowcount == 0:
                raise NotFoundError("No reading found to delete.")

    @override
    @classmethod
    async def exists(
        cls: type["ReadingService"],
        MAC: MacAddress | str,
        timestamp: datetime,
        *,
        cursor: Cursor | None = None,
    ) -> bool:
        """
        Check if a reading exists in the database.
        Args:
            MAC (MacAddress): The MAC address of the station the reading belongs to.
            timestamp (datetime): The timestamp of the reading to check.
        """
        return await cls._has(cls._key(MAC, timestamp), cursor)
