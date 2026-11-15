from typing import Any, TypedDict


class ErrorDict(TypedDict, total=False):
    """
    Schema for a single error in a response.

    ``title`` names the kind of error, ``details`` says what went wrong,
    ``code`` is the HTTP status as a string and ``source`` is the route that
    raised it.
    """

    title: str
    details: str
    code: str
    source: str
    target: str
    meta: dict[str, Any]


class ErrorResponse(TypedDict, total=False):
    """
    Errors and warning response schema.
    It contains a list of errors and warnings encountered during the request.
    The `timestamp` field is used to indicate when the response was generated.
    """

    errors: list[ErrorDict]
    warnings: list[ErrorDict]
    timestamp: str
