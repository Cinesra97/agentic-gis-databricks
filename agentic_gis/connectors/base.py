"""Shared connector types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd


class ConnectorError(RuntimeError):
    """Base class for data access errors that are safe to show to users."""


class PermissionDeniedError(ConnectorError):
    """The credentials do not grant access to the requested object."""


class QueryTimeoutError(ConnectorError):
    """The query exceeded the configured timeout."""


class ConnectionNotConfiguredError(ConnectorError):
    """Required connection settings are missing."""


@dataclass
class QueryResult:
    df: pd.DataFrame
    sql: str
    backend: str
    elapsed_ms: int
    row_limit: int | None = None
    truncated: bool = False
    geometry_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return len(self.df)


@dataclass
class ConnectionStatus:
    ok: bool
    backend: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ColumnInfo:
    name: str
    type: str
    comment: str = ""


class QueryEngine(Protocol):
    backend: str
    dialect: str

    def execute(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        max_rows: int | None = None,
    ) -> QueryResult: ...

    def describe(self, table: str) -> list[ColumnInfo]: ...

    def test_connection(self) -> ConnectionStatus: ...
