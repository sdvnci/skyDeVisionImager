from re import compile as re_compile
from typing import Final

_IDENTIFIER: Final = re_compile(r"[A-Za-z_][A-Za-z0-9_]*")


def quote(identifier: str) -> str:
    """
    Quote a table or column name, supporting qualified names like ``s.MAC``.
    Raises a ValueError for anything that is not a plain identifier, so names
    taken from caller-supplied mappings can never carry SQL.
    """
    parts = identifier.split(".")
    for part in parts:
        if _IDENTIFIER.fullmatch(part) is None:
            raise ValueError(f"Not a valid SQL identifier: {identifier!r}")
    return ".".join(f'"{part}"' for part in parts)


class QueryBuilder:
    """
    Assembles a SQLite query from a static base plus dynamic clauses.

    Columns are quoted via `quote` and values are passed as query parameters;
    `build()` returns the query and its parameter tuple, ready for
    `cursor.execute()`.
    """

    __slots__: tuple[str, ...] = (
        "_base",
        "_limit_val",
        "_offset_val",
        "_order",
        "_params",
        "_wheres",
    )

    def __init__(self, base: str, *params: object) -> None:
        self._base: str = base
        self._wheres: list[str] = []
        self._params: list[object] = list(params)
        self._order: list[str] = []
        self._limit_val: int | None = None
        self._offset_val: int | None = None

    def _compare(
        self, column: str, operator: str, value: object, skip_none: bool
    ) -> "QueryBuilder":
        if value is not None or not skip_none:
            self._wheres.append(f"{quote(column)} {operator} ?")
            self._params.append(value)
        return self

    def where_raw(self, condition: str, *params: object) -> "QueryBuilder":
        """
        Append a raw SQL condition (e.g. ``"a = ? OR b = ?"``). Spliced in
        verbatim; columns are NOT identifier-quoted, so this is for conditions
        the helpers below can't express.
        """
        self._wheres.append(condition)
        self._params.extend(params)
        return self

    def eq(
        self, column: str, value: object, *, skip_none: bool = True
    ) -> "QueryBuilder":
        """
        Append ``column = value``. With the default `skip_none=True` a None
        value drops the condition entirely (optional-filter semantics); pass
        `skip_none=False` to keep it (e.g. for destructive queries, where
        ``column = NULL`` matches nothing rather than everything).
        """
        return self._compare(column, "=", value, skip_none)

    def not_eq(self, column: str, value: object) -> "QueryBuilder":
        return self._compare(column, "<>", value, True)

    def gte(self, column: str, value: object) -> "QueryBuilder":
        return self._compare(column, ">=", value, True)

    def lte(self, column: str, value: object) -> "QueryBuilder":
        return self._compare(column, "<=", value, True)

    def like(self, column: str, value: object) -> "QueryBuilder":
        return self._compare(column, "LIKE", value, True)

    def in_(self, column: str, values: list[object] | None) -> "QueryBuilder":
        if values:
            placeholders = ", ".join("?" for _ in values)
            self._wheres.append(f"{quote(column)} IN ({placeholders})")
            self._params.extend(values)
        return self

    def order_by(self, column: str, *, desc: bool = False) -> "QueryBuilder":
        self._order.append(f"{quote(column)} DESC" if desc else quote(column))
        return self

    def limit(self, n: int | None) -> "QueryBuilder":
        self._limit_val = n
        return self

    def offset(self, n: int | None) -> "QueryBuilder":
        self._offset_val = n
        return self

    def build(self) -> tuple[str, tuple[object, ...]]:
        parts: list[str] = [self._base]
        params = list(self._params)
        if self._wheres:
            parts.append("WHERE " + " AND ".join(self._wheres))
        if self._order:
            parts.append("ORDER BY " + ", ".join(self._order))
        if self._limit_val is not None:
            parts.append("LIMIT ?")
            params.append(self._limit_val)
        if self._offset_val is not None:
            if self._limit_val is None:
                # SQLite only accepts OFFSET after a LIMIT; -1 means no limit.
                parts.append("LIMIT -1")
            parts.append("OFFSET ?")
            params.append(self._offset_val)
        return " ".join(parts) + ";", tuple(params)
