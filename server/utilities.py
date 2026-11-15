import sys
from logging import (
    DEBUG,
    ERROR,
    INFO,
    WARNING,
    FileHandler,
    Formatter,
    Handler,
    Logger,
    StreamHandler,
    getLogger,
)
from pathlib import Path
from typing import Any, ClassVar, Literal


class Loggable:
    """Mixin that adds class-level structured logging to any hierarchy.

    Inherit from ``Loggable`` (alongside ``ABC`` or any other base) and every
    subclass automatically gets a per-class ``Logger`` and a uniform
    ``ClassName::function::LEVEL::message`` format.  All log methods are
    ``@classmethod`` so callers do not need an instance.

    WARNING and above are always emitted.  DEBUG and INFO are **disabled by
    default** -- set ``debug=True`` at class definition time or call
    ``cls.debug_on()`` at runtime:

    .. code-block:: python

        class MyProcessor(Loggable, log_name="processor", debug=True):
            __slots__ = ()

            @classmethod
            def process(cls, item: str) -> None:
                cls.debug(f"starting on {item}")

    ``__init_subclass__`` accepts these optional keyword arguments that are
    consumed by ``Loggable`` and forwarded to the rest of the MRO:

    - ``log_name: str`` -- override the logger name (default ``cls.__name__``)
    - ``debug: bool`` -- enable debug output immediately (default ``False``)
    - ``handlers: list[Handler]`` -- attach extra logging handlers
    - ``log_path: Path`` -- the log file (default ``logs/<log_name>.log``
      beside the package)

    The log file is only created when the first message is written to it.
    """

    __slots__: tuple[str, ...] = ()
    _debug: ClassVar[bool] = False
    _logger: ClassVar[Logger | None] = None
    _formatter: ClassVar[Formatter | None] = None
    _DEFAULT_DIR: ClassVar[Path] = Path(__file__).parent.parent.resolve() / "logs"

    def __init_subclass__(
        cls: type["Loggable"],
        log_name: str | None = None,
        *,
        debug: bool = False,
        handlers: list[Handler] | None = None,
        log_path: Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        cls._logger = getLogger(log_name) if log_name else getLogger(cls.__name__)
        cls._formatter = Formatter(
            f"%(asctime)s::{cls.__name__}::%(funcName)s::%(levelname)s::%(message)s"
        )

        log_path = (
            log_path if log_path is not None else cls._DEFAULT_DIR / cls._logger.name
        )
        cls._ensure_log_handlers(log_path)

        if handlers is not None:
            for handler in handlers:
                cls.add_handler(handler)

        cls._debug = debug

    @classmethod
    def _ensure_log_handlers(cls: type["Loggable"], logpath: Path) -> None:
        """Attach file + console handlers (idempotent per process)."""

        if cls._logger is None:
            return

        if cls._logger.handlers:
            return

        if logpath.suffix != ".log":
            logpath = logpath.with_suffix(".log")

        logpath.parent.mkdir(parents=True, exist_ok=True)

        # delay=True: the file is opened on the first record, not on import.
        _file_handler = FileHandler(logpath, encoding="utf-8", delay=True)
        _file_handler.setLevel(DEBUG)
        cls.add_handler(_file_handler)

        _console_handler = StreamHandler(sys.stdout)
        _console_handler.setLevel(INFO)
        cls.add_handler(_console_handler)

        # Let messages through to handlers; _log() gates DEBUG/INFO behind _debug.
        cls._logger.setLevel(DEBUG)

    @classmethod
    def add_handler(cls: type["Loggable"], handler: Handler) -> None:
        """Attach *handler* formatted with this class's default formatter.

        Sets ``propagate = False`` so messages are not duplicated by the root
        logger.
        """
        if cls._logger is None:
            return

        if cls._formatter is None:
            return

        handler.setFormatter(cls._formatter)
        cls._logger.addHandler(handler)
        cls._logger.propagate = False

    @classmethod
    def clear_handlers(cls: type["Loggable"]) -> None:
        if cls._logger is None:
            return

        for handler in tuple(cls._logger.handlers):
            cls._logger.removeHandler(handler)
            handler.close()

    @classmethod
    def set_formatter(cls: type["Loggable"], formatter: Formatter) -> None:
        """Replace the formatter on every attached handler."""
        if cls._logger is None:
            return

        for handler in cls._logger.handlers:
            handler.setFormatter(formatter)

        cls._formatter = formatter

    @classmethod
    def debug_on(cls: type["Loggable"]) -> bool:
        """Enable debug logging.  Returns ``True``."""
        cls._debug = True
        return cls._debug

    @classmethod
    def debug_off(cls: type["Loggable"]) -> bool:
        """Disable debug logging.  Returns ``False``."""
        cls._debug = False
        return cls._debug

    @classmethod
    def _log(
        cls: type["Loggable"],
        message: str,
        level: Literal[10, 20, 30, 40] = INFO,
    ) -> None:
        """Emit *message* at *level*.  WARNING and above always pass;
        DEBUG and INFO require ``debug=True``."""
        if cls._logger is None:
            return
        if level < WARNING and not cls._debug:
            return
        cls._logger.log(level, message, stacklevel=3)

    @classmethod
    def debug(cls: type["Loggable"], message: str) -> None:
        """Log at ``DEBUG`` level (10)."""
        cls._log(message, level=DEBUG)

    @classmethod
    def info(cls: type["Loggable"], message: str) -> None:
        """Log at ``INFO`` level (20)."""
        cls._log(message, level=INFO)

    @classmethod
    def warning(cls: type["Loggable"], message: str) -> None:
        """Log at ``WARNING`` level (30)."""
        cls._log(message, level=WARNING)

    @classmethod
    def error(cls: type["Loggable"], message: str) -> None:
        """Log at ``ERROR`` level (40)."""
        cls._log(message, level=ERROR)
