"""Databricks SQL warehouse connector (official ``databricks-sql-connector``).

Read-only use is enforced by the SQL guard *and* should be enforced by Unity Catalog grants:
connect with a principal that only has ``USE CATALOG``, ``USE SCHEMA`` and ``SELECT`` on the
registered tables (see docs/DATABRICKS.md). This module never bypasses access controls; it
reports permission failures to the user instead.

Authentication (``DATABRICKS_AUTH_TYPE``):

* ``pat``       -- personal access token in ``DATABRICKS_TOKEN`` (from the environment / .env)
* ``oauth-u2m`` -- interactive browser login (``auth_type="databricks-oauth"``)
* ``oauth-m2m`` -- service principal via ``databricks-sdk`` (``DATABRICKS_CLIENT_ID/SECRET``),
  which is also how Databricks Apps authenticate.
"""

from __future__ import annotations

import contextlib
import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Any

import pandas as pd

from agentic_gis.config.settings import Settings
from agentic_gis.connectors.base import (
    ColumnInfo,
    ConnectionNotConfiguredError,
    ConnectionStatus,
    ConnectorError,
    PermissionDeniedError,
    QueryResult,
    QueryTimeoutError,
)
from agentic_gis.security.audit import redact_text

log = logging.getLogger(__name__)

_PERMISSION_MARKERS = (
    "PERMISSION_DENIED",
    "INSUFFICIENT_PERMISSIONS",
    "does not have",
    "not authorized",
    "Unauthorized",
    "403",
    "User does not have",
)
_AUTH_MARKERS = ("Invalid access token", "401", "invalid_client", "Token is expired", "Authentication")
_NOT_FOUND_MARKERS = ("TABLE_OR_VIEW_NOT_FOUND", "SCHEMA_NOT_FOUND", "CATALOG_NOT_FOUND", "cannot be found")


def translate_error(exc: Exception, secrets: list[str]) -> ConnectorError:
    msg = redact_text(str(exc), secrets)
    first = msg.strip().splitlines()[0][:400] if msg.strip() else type(exc).__name__
    if any(m in msg for m in _AUTH_MARKERS):
        return PermissionDeniedError(
            "Databricks authentication failed. Check DATABRICKS_TOKEN (or OAuth settings) and that the "
            f"token has not expired. Details: {first}"
        )
    if any(m in msg for m in _PERMISSION_MARKERS):
        return PermissionDeniedError(
            "Your Databricks identity does not have permission for this object. Ask a workspace administrator "
            f"for USE CATALOG / USE SCHEMA / SELECT grants. Details: {first}"
        )
    if any(m in msg for m in _NOT_FOUND_MARKERS):
        return ConnectorError(f"Table or schema not found (or not visible to you): {first}")
    if "TIMEOUT" in msg.upper() or "cancel" in msg.lower():
        return QueryTimeoutError(f"The Databricks query timed out or was cancelled: {first}")
    return ConnectorError(f"Databricks error: {first}")


class DatabricksConnector:
    backend = "databricks"
    dialect = "databricks"

    def __init__(self, settings: Settings, connect_fn: Callable[..., Any] | None = None):
        self.settings = settings
        self._connect_fn = connect_fn
        self._conn: Any = None
        self._lock = threading.Lock()

    # ---- connection ---------------------------------------------------------------------------
    def connect_kwargs(self) -> dict[str, Any]:
        s = self.settings
        if not s.databricks_server_hostname or not s.databricks_http_path:
            raise ConnectionNotConfiguredError(
                "Set DATABRICKS_SERVER_HOSTNAME and DATABRICKS_HTTP_PATH (see .env.example)."
            )
        kw: dict[str, Any] = {
            "server_hostname": s.databricks_server_hostname,
            "http_path": s.databricks_http_path,
            "session_configuration": {"STATEMENT_TIMEOUT": str(int(s.query_timeout_s))},
            "user_agent_entry": "agentic-gis-databricks",
        }
        if s.databricks_catalog:
            kw["catalog"] = s.databricks_catalog
        if s.databricks_schema:
            kw["schema"] = s.databricks_schema
        if s.databricks_auth_type == "pat":
            if s.databricks_token is None or not s.databricks_token.get_secret_value():
                raise ConnectionNotConfiguredError("DATABRICKS_TOKEN is not set.")
            kw["access_token"] = s.databricks_token.get_secret_value()
        elif s.databricks_auth_type == "oauth-u2m":
            kw["auth_type"] = "databricks-oauth"
        else:
            if not (s.databricks_client_id and s.databricks_client_secret):
                raise ConnectionNotConfiguredError(
                    "DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET are required."
                )
            try:
                from databricks.sdk.core import Config, oauth_service_principal  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ConnectionNotConfiguredError(
                    "OAuth M2M needs databricks-sdk: uv sync --extra databricks-auth"
                ) from exc
            host = f"https://{s.databricks_server_hostname}"
            client_id = s.databricks_client_id
            secret = s.databricks_client_secret.get_secret_value()

            def provider() -> Any:
                return oauth_service_principal(Config(host=host, client_id=client_id, client_secret=secret))

            kw["credentials_provider"] = provider
        return kw

    def connection(self) -> Any:
        with self._lock:
            if self._conn is None:
                kw = self.connect_kwargs()
                try:
                    if self._connect_fn is not None:
                        self._conn = self._connect_fn(**kw)
                    else:
                        from databricks import sql as dbsql

                        self._conn = dbsql.connect(**kw)
                except ConnectorError:
                    raise
                except Exception as exc:
                    raise translate_error(exc, self.settings.secret_values()) from exc
            return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    # ---- queries -------------------------------------------------------------------------------
    def execute(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        max_rows: int | None = None,
    ) -> QueryResult:
        """Execute already-validated SQL with native named parameters (``:p0``)."""
        conn = self.connection()
        cursor = conn.cursor()
        timer: threading.Timer | None = None
        timed_out = threading.Event()
        timeout = timeout_s or self.settings.query_timeout_s
        if timeout:

            def _cancel() -> None:
                timed_out.set()
                try:
                    cursor.cancel()
                except Exception:  # pragma: no cover - best effort
                    log.warning("Could not cancel Databricks query")

            timer = threading.Timer(timeout + 5, _cancel)
            timer.daemon = True
            timer.start()
        t0 = time.perf_counter()
        try:
            cursor.execute(sql, parameters=params or None)
            df = self._fetch(cursor, max_rows)
        except Exception as exc:
            if timed_out.is_set():
                raise QueryTimeoutError(f"Query exceeded {timeout:.0f}s and was cancelled") from exc
            raise translate_error(exc, self.settings.secret_values()) from exc
        finally:
            if timer:
                timer.cancel()
            with contextlib.suppress(Exception):  # closing a cursor must not mask the real error
                cursor.close()
        elapsed = int((time.perf_counter() - t0) * 1000)
        truncated = max_rows is not None and len(df) >= max_rows
        return QueryResult(
            df=df, sql=sql, backend="databricks", elapsed_ms=elapsed, row_limit=max_rows, truncated=truncated
        )

    @staticmethod
    def _fetch(cursor: Any, max_rows: int | None) -> pd.DataFrame:
        if hasattr(cursor, "fetchall_arrow"):
            try:
                table = cursor.fetchall_arrow()
                df = table.to_pandas()
                return df.head(max_rows) if max_rows else df
            except (ImportError, AttributeError, NotImplementedError):
                pass
        rows = cursor.fetchmany(max_rows) if max_rows else cursor.fetchall()
        cols = [d[0] for d in (cursor.description or [])]
        return pd.DataFrame([tuple(r) for r in rows], columns=cols)

    def describe(self, table: str) -> list[ColumnInfo]:
        parts = table.split(".")
        if len(parts) != 3:
            raise ConnectorError("Use a three-part name catalog.schema.table")
        cursor = self.connection().cursor()
        try:
            cursor.columns(catalog_name=parts[0], schema_name=parts[1], table_name=parts[2])
            rows = cursor.fetchall()
            cols = [d[0].upper() for d in (cursor.description or [])]
        except Exception as exc:
            raise translate_error(exc, self.settings.secret_values()) from exc
        finally:
            cursor.close()
        out = []
        for r in rows:
            rec = dict(zip(cols, tuple(r), strict=False))
            out.append(
                ColumnInfo(
                    name=str(rec.get("COLUMN_NAME")),
                    type=str(rec.get("TYPE_NAME", "")),
                    comment=str(rec.get("REMARKS") or ""),
                )
            )
        if not out:
            raise ConnectorError(f"No columns visible for {table} (missing table or no SELECT permission).")
        return out

    def list_tables(self, catalog: str | None = None, schema: str | None = None) -> list[dict[str, Any]]:
        catalog = catalog or self.settings.databricks_catalog
        schema = schema or self.settings.databricks_schema
        for v in (catalog, schema):
            if v and not re.fullmatch(r"[A-Za-z0-9_\-]+", v):
                raise ConnectorError("Invalid catalog or schema name")
        cursor = self.connection().cursor()
        try:
            cursor.tables(catalog_name=catalog, schema_name=schema)
            rows = cursor.fetchall()
            cols = [d[0].upper() for d in (cursor.description or [])]
        except Exception as exc:
            raise translate_error(exc, self.settings.secret_values()) from exc
        finally:
            cursor.close()
        out = []
        for r in rows[:2000]:
            rec = dict(zip(cols, tuple(r), strict=False))
            out.append(
                {
                    "catalog": rec.get("TABLE_CAT"),
                    "schema": rec.get("TABLE_SCHEM"),
                    "table": rec.get("TABLE_NAME"),
                    "type": rec.get("TABLE_TYPE"),
                    "comment": rec.get("REMARKS"),
                }
            )
        return out

    def test_connection(self) -> ConnectionStatus:
        try:
            cursor = self.connection().cursor()
            try:
                # Fixed diagnostic query (not user or LLM supplied).
                cursor.execute(
                    "SELECT current_user() AS user, current_catalog() AS catalog, current_schema() AS schema"
                )
                row = cursor.fetchone()
            finally:
                cursor.close()
            user, catalog, schema = (tuple(row) + (None, None, None))[:3] if row else (None, None, None)
            return ConnectionStatus(
                True,
                "databricks",
                f"Connected to {self.settings.databricks_server_hostname} as {user}",
                {
                    "user": user,
                    "catalog": catalog,
                    "schema": schema,
                    "http_path": self.settings.databricks_http_path,
                },
            )
        except ConnectorError as exc:
            return ConnectionStatus(False, "databricks", str(exc))
        except Exception as exc:
            return ConnectionStatus(
                False, "databricks", str(translate_error(exc, self.settings.secret_values()))
            )
