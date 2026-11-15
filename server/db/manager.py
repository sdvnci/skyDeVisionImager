from collections.abc import Generator
from contextlib import contextmanager
from hashlib import sha256
from hmac import new as hmac_new
from os import getenv
from pathlib import Path
from queue import Empty, SimpleQueue
from sqlite3 import Connection, Cursor, Row, connect
from sqlite3 import Error as SQLError
from threading import Lock, RLock
from typing import ClassVar, Final, final

from bcrypt import checkpw, gensalt, hashpw
from dotenv import load_dotenv
from pydantic import SecretStr

from ..utilities import Loggable
from .schema import apply_schema

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
    A thread-safe connection pool for SQLite database connections.

    Connections are opened as they are needed, up to ``size``. When all of
    them are in use, ``getconn`` waits up to ``timeout`` seconds for one to be
    returned, then raises a ConnectionError.

    A plain ``:memory:`` database is private to its connection, so its pool
    holds a single connection, which every caller shares in turn.
    """

    __slots__ = (
        "_closed",
        "_connections",
        "_idle",
        "_lock",
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
        self._idle: SimpleQueue[Connection] = SimpleQueue()
        self._connections: list[Connection] = []
        self._lock: Lock = Lock()
        self._closed: bool = False

    def _connect(self) -> Connection:
        conn = connect(
            database=self.database,
            timeout=self.timeout,
            check_same_thread=False,  # Required for multi-threaded access
            uri=self.uri,
        )
        conn.row_factory = Row
        _ = conn.execute("PRAGMA foreign_keys = ON")
        _ = conn.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
        return conn

    @property
    def closed(self) -> bool:
        return self._closed

    def getconn(self) -> Connection:
        """Take a connection from the pool, opening one if there is room."""
        try:
            return self._idle.get_nowait()
        except Empty:
            pass

        with self._lock:
            if self._closed:
                raise ConnectionError("The connection pool is closed.")
            if len(self._connections) < self.size:
                conn = self._connect()
                self._connections.append(conn)
                return conn

        try:
            return self._idle.get(timeout=self.timeout)
        except Empty:
            raise ConnectionError(
                f"No database connection became free within {self.timeout:g}s"
            ) from None

    def putconn(self, connection: Connection) -> None:
        """Return a connection to the pool."""
        if self._closed:
            connection.close()
            return
        self._idle.put(connection)

    def closeall(self) -> None:
        """Close every connection the pool has opened, in use or not."""
        with self._lock:
            self._closed = True
            for conn in self._connections:
                conn.close()
            self._connections.clear()


@final
class Manager(Loggable, log_name="dbmanager", debug=True):
    """
    Management class for SQLite config and pooled connections.
    """

    _pool: ClassVar[SQLiteConnectionPool | None] = None
    _pool_lock: ClassVar[RLock] = RLock()
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
    def connect(
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

        with cls._pool_lock:
            cls._close_pool_unsafe()

            if db_path is not None:
                cls._dbfile = Path(db_path)
                cls._uri = uri

            cls.info(f"Connecting to database: {cls._dbfile}")
            cls._pool = SQLiteConnectionPool(
                database=str(cls._dbfile),
                size=cls._pool_size,
                timeout=cls._timeout,
                uri=cls._uri,
            )

            try:
                with cls.cursor() as cursor:
                    apply_schema(cursor)
            except SQLError as err:
                cls.error(f"Error connecting to the database: {err}")
                cls._close_pool_unsafe()
                raise

    @classmethod
    def get_pool(cls: type["Manager"]) -> SQLiteConnectionPool:
        """The open connection pool, connecting first if there isn't one."""
        pool = cls._pool
        if pool is not None and not pool.closed:
            return pool

        with cls._pool_lock:
            if cls._pool is None or cls._pool.closed:
                cls.connect()
            if cls._pool is None:
                raise ConnectionError("Could not open the database connection pool.")
            return cls._pool

    @classmethod
    @contextmanager
    def connection(cls: type["Manager"]) -> Generator[Connection, None, None]:
        """
        Context manager to acquire a connection from the pool. The transaction
        is committed when the block exits cleanly, and rolled back if it raises.

        Usage:
        ------
            with Manager.connection() as conn:
                conn.execute("SELECT * FROM table")

        Raises:
            ConnectionError: If no connection could be acquired.
        """
        pool = cls.get_pool()
        conn = pool.getconn()
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            pool.putconn(conn)

    @classmethod
    @contextmanager
    def cursor(cls: type["Manager"]) -> Generator[Cursor, None, None]:
        """
        Context manager to acquire a cursor from a pooled connection. Everything
        executed on it is a single transaction (see ``connection``).

        Usage:
        ------
            with Manager.cursor() as cursor:
                cursor.execute("SELECT * FROM table")
                results = cursor.fetchall()
        """
        with cls.connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
            finally:
                cursor.close()

    @classmethod
    def _close_pool_unsafe(cls: type["Manager"]) -> None:
        """Close the pool without acquiring the lock."""
        if cls._pool is not None:
            cls._pool.closeall()
            cls._pool = None

    @classmethod
    def close(cls: type["Manager"]) -> None:
        """Close all connections in the pool."""
        with cls._pool_lock:
            cls._close_pool_unsafe()
