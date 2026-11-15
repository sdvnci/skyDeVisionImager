from asyncio import Lock, Queue, QueueEmpty, wait_for
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from hashlib import sha256
from hmac import new as hmac_new
from os import getenv
from pathlib import Path
from sqlite3 import Error as SQLError
from sqlite3 import Row
from typing import ClassVar, Final, final

from aiosqlite import Connection, Cursor, connect
from bcrypt import checkpw, gensalt, hashpw
from dotenv import load_dotenv
from pydantic import SecretStr

from ..utilities import Loggable
from .schema import SCHEMA

DEFAULT_DBFILE: Final = Path("database.db")  # DBFILE
DEFAULT_POOL_SIZE: Final = 5  # DB_POOL_SIZE
DEFAULT_POOL_TIMEOUT: Final = 10.0  # DB_POOL_TIMEOUT, seconds


def _setting(name: str, default: str) -> str:
    """The environment variable ``name``, stripped, or ``default`` if unset or
    blank."""
    return getenv(name, "").strip() or default


def _int_setting(name: str, given: int | None, default: int) -> int:
    if given is not None:
        return given
    raw = _setting(name, str(default))
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, not {raw!r}") from None


def _seconds_setting(name: str, given: float | None, default: float) -> float:
    if given is not None:
        return given
    raw = _setting(name, str(default))
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number of seconds, not {raw!r}") from None


class SQLiteConnectionPool:
    """
    An async connection pool for SQLite database connections.

    Connections are opened as they are needed, up to ``size``. When all of
    them are in use, ``getconn`` waits up to ``timeout`` seconds for one to be
    returned, then raises a ConnectionError.

    A plain ``:memory:`` database is private to its connection, so its pool
    holds a single connection, which every caller shares in turn.

    A pool belongs to the event loop it was first used on.
    """

    __slots__ = (
        "_closed",
        "_connections",
        "_idle",
        "_opened",
        "database",
        "size",
        "timeout",
        "uri",
    )

    def __init__(
        self,
        database: str,
        size: int = DEFAULT_POOL_SIZE,
        timeout: float = DEFAULT_POOL_TIMEOUT,
        uri: bool = False,
    ) -> None:
        if size < 1:
            raise ValueError(f"The pool needs room for a connection: size is {size}")
        if timeout <= 0:
            raise ValueError(f"timeout must be more than 0 seconds, not {timeout}")

        self.database: str = database
        self.size: int = 1 if database == ":memory:" else size
        self.timeout: float = timeout
        self.uri: bool = uri
        self._idle: Queue[Connection] = Queue()
        self._connections: list[Connection] = []
        self._opened: int = 0  # connections open or being opened
        self._closed: bool = False

    async def _connect(self) -> Connection:
        conn = await connect(
            database=self.database,
            timeout=self.timeout,
            uri=self.uri,
        )
        try:
            conn.row_factory = Row
            _ = await conn.execute("PRAGMA foreign_keys = ON")
            _ = await conn.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
        except BaseException:
            await conn.close()
            raise
        return conn

    @property
    def closed(self) -> bool:
        return self._closed

    async def getconn(self) -> Connection:
        """Take a connection from the pool, opening one if there is room."""
        if self._closed:
            raise ConnectionError("The connection pool is closed.")

        try:
            return self._idle.get_nowait()
        except QueueEmpty:
            pass

        if self._opened < self.size:
            # Claim the slot before awaiting, so concurrent callers can't
            # open more connections than the pool has room for.
            self._opened += 1
            try:
                conn = await self._connect()
            except BaseException:
                self._opened -= 1
                raise
            self._connections.append(conn)
            return conn

        try:
            return await wait_for(self._idle.get(), timeout=self.timeout)
        except TimeoutError:
            raise ConnectionError(
                f"No database connection became free within {self.timeout:g}s"
            ) from None

    async def putconn(self, connection: Connection) -> None:
        """Return a connection to the pool."""
        if self._closed:
            await connection.close()
            return
        self._idle.put_nowait(connection)

    async def closeall(self) -> None:
        """Close every connection the pool has opened, in use or not."""
        self._closed = True
        connections, self._connections = self._connections, []
        self._opened = 0
        for conn in connections:
            await conn.close()


@final
class Manager(Loggable, log_name="dbmanager", debug=True):
    """
    Async management class for SQLite config and pooled connections.
    Uses aiosqlite for async connections.
    """

    _pool: ClassVar[SQLiteConnectionPool | None] = None
    _pool_lock: ClassVar[Lock] = Lock()
    _pool_size: ClassVar[int] = DEFAULT_POOL_SIZE
    _timeout: ClassVar[float] = DEFAULT_POOL_TIMEOUT
    _dbfile: ClassVar[Path] = DEFAULT_DBFILE
    _uri: ClassVar[bool] = False
    _secret: ClassVar[SecretStr | None] = None

    @classmethod
    def load(
        cls: type["Manager"],
        secret: str | None = None,
        database: str | Path | None = None,
        pool_size: int | None = None,
        timeout: float | None = None,
    ) -> None:
        """
        Load the database configuration from environment variables or provided parameters.
        If parameters are provided, they will override environment variables.

        - ``secret`` (DBSECRET): the key passwords are peppered with. Required.
        - ``database`` (DBFILE): path to the SQLite database file.
        - ``pool_size`` (DB_POOL_SIZE): the most connections open at once.
        - ``timeout`` (DB_POOL_TIMEOUT): seconds to wait for a free connection
          when all are in use, and for a locked database, before giving up.
        """
        _ = load_dotenv()

        secret = secret if secret is not None else _setting("DBSECRET", "")
        if not secret:
            cls.error("DBSECRET not found in environment variables. Cannot continue.")
            raise ValueError("DBSECRET not found in environment variables.")

        pool_size = _int_setting("DB_POOL_SIZE", pool_size, DEFAULT_POOL_SIZE)
        timeout = _seconds_setting("DB_POOL_TIMEOUT", timeout, DEFAULT_POOL_TIMEOUT)
        if pool_size < 1:
            raise ValueError(
                f"The pool needs room for a connection: DB_POOL_SIZE is {pool_size}"
            )
        if timeout <= 0:
            raise ValueError(
                f"DB_POOL_TIMEOUT must be more than 0 seconds, not {timeout}"
            )

        cls._secret = SecretStr(secret)
        cls._pool_size = pool_size
        cls._timeout = timeout
        cls._dbfile = Path(
            database
            if database is not None
            else _setting("DBFILE", str(DEFAULT_DBFILE))
        )
        cls.info(f"Database configuration loaded. Database file: {cls._dbfile}")

    @classmethod
    def loaded(cls: type["Manager"]) -> bool:
        return cls._secret is not None

    @classmethod
    def secret(cls: type["Manager"]) -> SecretStr:
        """The database secret. Loads the configuration if that hasn't happened."""
        if cls._secret is None:
            cls.load()
        if cls._secret is None:
            raise ValueError("Database secret is not loaded.")
        return cls._secret

    @classmethod
    def database_file(cls: type["Manager"]) -> Path:
        return cls._dbfile

    @classmethod
    def _peppered(cls: type["Manager"], password: str) -> bytes:
        """
        The password keyed with the database secret. A fixed 64 bytes, so a
        long password is never cut off at bcrypt's 72 byte limit.
        """
        key = cls.secret().get_secret_value().encode("utf-8")
        return hmac_new(key, password.encode("utf-8"), sha256).hexdigest().encode()

    @classmethod
    def hash_password(cls: type["Manager"], password: str) -> str:
        """
        Hash + salt a password via bcrypt, peppered with the database secret.

        Args:
            - password(str): The password to hash

        Returns:
            - str: The hashed password
        """
        return hashpw(cls._peppered(password), gensalt()).decode("utf-8")

    @classmethod
    def password_match(
        cls: type["Manager"], stored_hash: str, provided_password: str
    ) -> bool:
        """
        Check if a password matches the stored hash.

        Args:
            - stored_hash(str): The stored password hash
            - provided_password(str): The password to check

        Returns:
            - bool: True if the password matches, False otherwise
        """
        hashed = stored_hash.encode("utf-8")
        try:
            if checkpw(cls._peppered(provided_password), hashed):
                return True
            # Hashes stored before passwords were keyed with HMAC.
            legacy = f"{provided_password}{cls.secret().get_secret_value()}"
            return checkpw(legacy.encode("utf-8"), hashed)
        except ValueError as err:
            # A malformed hash, or a legacy password over bcrypt's length limit.
            cls.warning(f"Could not check password: {err}")
            return False

    @classmethod
    def connected(cls: type["Manager"]) -> bool:
        """Check if the database is connected."""
        return cls._pool is not None and not cls._pool.closed

    @classmethod
    async def _connect_unsafe(
        cls: type["Manager"], db_path: str | Path | None = None, uri: bool = False
    ) -> SQLiteConnectionPool:
        """Open a new pool and apply the schema, without acquiring the lock."""
        await cls._close_pool_unsafe()

        if db_path is not None:
            cls._dbfile = Path(db_path)
            cls._uri = uri

        cls.info(f"Connecting to database: {cls._dbfile}")
        pool = SQLiteConnectionPool(
            database=str(cls._dbfile),
            size=cls._pool_size,
            timeout=cls._timeout,
            uri=cls._uri,
        )

        try:
            conn = await pool.getconn()
            try:
                for statement in SCHEMA:
                    _ = await conn.execute(statement)
                await conn.commit()
            finally:
                await pool.putconn(conn)
        except SQLError as err:
            cls.error(f"Error connecting to the database: {err}")
            await pool.closeall()
            raise

        cls._pool = pool
        return pool

    @classmethod
    async def connect(
        cls: type["Manager"], db_path: str | Path | None = None, uri: bool = False
    ) -> None:
        """
        Open the connection pool and apply the schema. Any pool that is
        already open is closed first.

        Args:
            db_path: Optional path to the database file. If not provided, uses the default or previously set path.
            uri: Whether ``db_path`` is an SQLite URI.

        Raises:
            sqlite3.Error: If the database cannot be opened or the schema applied.
        """
        if not cls.loaded():
            cls.load()

        async with cls._pool_lock:
            _ = await cls._connect_unsafe(db_path, uri)

    @classmethod
    async def get_pool(cls: type["Manager"]) -> SQLiteConnectionPool:
        """The open connection pool, connecting first if there isn't one."""
        pool = cls._pool
        if pool is not None and not pool.closed:
            return pool

        if not cls.loaded():
            cls.load()

        async with cls._pool_lock:
            if cls._pool is not None and not cls._pool.closed:
                return cls._pool
            return await cls._connect_unsafe()

    @classmethod
    @asynccontextmanager
    async def connection(cls: type["Manager"]) -> AsyncGenerator[Connection, None]:
        """
        Async context manager to acquire a connection from the pool. The
        transaction is committed when the block exits cleanly, and rolled back
        if it raises.

        Usage:
        ------
            async with Manager.connection() as conn:
                await conn.execute("SELECT * FROM table")

        Raises:
            ConnectionError: If no connection could be acquired.
        """
        pool = await cls.get_pool()
        conn = await pool.getconn()
        try:
            yield conn
            await conn.commit()
        except BaseException:
            await conn.rollback()
            raise
        finally:
            await pool.putconn(conn)

    @classmethod
    @asynccontextmanager
    async def cursor(cls: type["Manager"]) -> AsyncGenerator[Cursor, None]:
        """
        Async context manager to acquire a cursor from a pooled connection.
        Everything executed on it is a single transaction (see ``connection``).

        Usage:
        ------
            async with Manager.cursor() as cursor:
                await cursor.execute("SELECT * FROM table")
                results = await cursor.fetchall()
        """
        async with cls.connection() as conn:
            cursor = await conn.cursor()
            try:
                yield cursor
            finally:
                await cursor.close()

    @classmethod
    async def _close_pool_unsafe(cls: type["Manager"]) -> None:
        """Close the pool without acquiring the lock."""
        if cls._pool is not None:
            pool, cls._pool = cls._pool, None
            await pool.closeall()

    @classmethod
    async def close(cls: type["Manager"]) -> None:
        """Close all connections in the pool."""
        async with cls._pool_lock:
            await cls._close_pool_unsafe()
