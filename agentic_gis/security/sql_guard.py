"""AST-based SQL validation for read-only, allowlisted, row-limited queries.

Every SQL statement -- whether written by the LLM or rendered from a tool template -- passes
through :class:`SQLGuard` before execution. The guard parses SQL with SQLGlot (no regular
expressions over raw SQL) and enforces:

* exactly one statement, and it must be a query (SELECT / set operation / CTE query);
* no DDL, DML, session commands, file IO, extension loading, or other side effects anywhere
  in the tree (including inside CTEs and subqueries);
* every referenced table resolves to an allowlisted dataset (CTE names excepted);
* no table functions (``read_csv(...)``, ``glob(...)``, ``IDENTIFIER(...)``...) and no denied
  scalar functions (file readers, secrets, external AI/HTTP calls, reflection, system catalogs);
* no qualified (catalog.schema.fn) function calls, which could invoke arbitrary UDFs;
* no references to columns registered as sensitive;
* a LIMIT no larger than the configured maximum.

Database-side controls (read-only credentials, Unity Catalog grants, DuckDB external access
disabled) remain the primary enforcement; this guard is defense in depth and gives users a
clear explanation before anything runs.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import sqlglot
from pydantic import BaseModel, Field
from sqlglot import exp
from sqlglot.errors import ErrorLevel, ParseError

MAX_SQL_LENGTH = 20_000

_FORBIDDEN_NODE_NAMES = [
    "Insert",
    "Update",
    "Delete",
    "Drop",
    "Create",
    "Alter",
    "Merge",
    "Command",
    "Set",
    "SetItem",
    "Copy",
    "Attach",
    "Detach",
    "Pragma",
    "Install",
    "Use",
    "Grant",
    "Revoke",
    "Transaction",
    "Commit",
    "Rollback",
    "LoadData",
    "Into",
    "Describe",
    "Export",
    "Refresh",
    "Analyze",
    "Cache",
    "Uncache",
    "TruncateTable",
    "Summarize",
    "Show",
    "Kill",
    "Declare",
    "Comment",
    "Lock",
]
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, n) for n in _FORBIDDEN_NODE_NAMES if hasattr(exp, n)
)

DENIED_FUNCTION_PATTERNS = [
    re.compile(p)
    for p in (
        r"^read_",  # read_csv, read_parquet, read_json, read_files (Databricks), read_kafka ...
        r"_scan$",  # parquet_scan, delta_scan, iceberg_scan, sqlite_scan ...
        r"^scan_",
        r"^glob$",
        r"^getenv$",
        r"^getvariable$",
        r"^st_read",  # DuckDB spatial file readers
        r"^load_",
        r"^install",
        r"^sniff_csv$",
        r"^query$",
        r"^query_table$",
        r"^parquet_",
        r"^duckdb_",  # DuckDB system catalog functions
        r"^pragma_",
        r"^current_setting$",
        r"^which_secret$",
        r"^(try_)?secret$",  # Databricks secret()
        r"^ai_",  # Databricks AI functions call external models
        r"^http_",  # http_request / http_get ...
        r"^(try_)?reflect$",  # Spark reflect()
        r"^java_method$",
        r"^vector_search$",
        r"^remote_query$",
        r"^cloud_files",
        r"^input_file_",
        r"^sql_auto_complete$",
        r"^(enable|disable)_",
        r"^json_execute_serialized_sql$",
        r"^from_substrait",
        r"^(postgres|mysql|sqlite)_",
        r"^(pg_)?sleep$",
        r"^system$",
        r"^shell$",
        r"^export_",
        r"^(force_)?checkpoint$",
    )
]


class SQLValidationResult(BaseModel):
    ok: bool
    sql: str = Field(description="Normalized SQL that will be executed (LIMIT enforced, comments removed).")
    original_sql: str
    dialect: str
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    tables: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    limit: int | None = None
    fingerprint: str = ""


@dataclass
class SQLPolicy:
    """What a given backend/session is allowed to query."""

    dialect: str  # "duckdb" | "databricks"
    allowed_tables: dict[str, str]  # normalized identifier -> dataset id
    max_rows: int
    sensitive_columns: dict[str, set[str]] = field(default_factory=dict)  # dataset id -> column names
    default_catalog: str | None = None
    default_schema: str | None = None


def fingerprint_sql(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()[:16]


def _function_name(fn: exp.Func) -> str:
    if isinstance(fn, exp.Anonymous):
        name = fn.this if isinstance(fn.this, str) else fn.name
        return str(name).lower()
    try:
        return fn.sql_name().lower()
    except Exception:  # pragma: no cover - defensive
        return type(fn).__name__.lower()


class SQLGuard:
    """Validate SQL against a :class:`SQLPolicy`."""

    def validate(self, sql: str, policy: SQLPolicy) -> SQLValidationResult:
        original = sql or ""
        errors: list[str] = []
        warnings: list[str] = []

        def fail(msg: str) -> SQLValidationResult:
            errors.append(msg)
            return SQLValidationResult(
                ok=False,
                sql="",
                original_sql=original,
                dialect=policy.dialect,
                errors=errors,
                warnings=warnings,
            )

        text = original.strip()
        if not text:
            return fail("Empty SQL.")
        if len(text) > MAX_SQL_LENGTH:
            return fail(f"SQL exceeds the maximum length of {MAX_SQL_LENGTH} characters.")
        if "\x00" in text:
            return fail("SQL contains a NUL byte.")

        try:
            statements = [
                s for s in sqlglot.parse(text, read=policy.dialect, error_level=ErrorLevel.RAISE) if s
            ]
        except ParseError as exc:
            return fail(f"SQL could not be parsed: {str(exc).splitlines()[0][:300]}")
        except Exception as exc:  # sqlglot can raise TokenError and others
            return fail(f"SQL could not be parsed: {type(exc).__name__}: {str(exc)[:300]}")

        if len(statements) != 1:
            return fail(f"Exactly one SQL statement is allowed; found {len(statements)}.")
        root = statements[0]
        while isinstance(root, exp.Subquery) and isinstance(root.this, exp.Query):
            root = root.this
        if not isinstance(root, exp.Query):
            return fail(f"Only read-only queries (SELECT) are allowed; got {type(root).__name__.upper()}.")

        for node in root.walk():
            if isinstance(node, FORBIDDEN_NODES):
                errors.append(
                    f"Statement type {type(node).__name__.upper()} is not allowed in a read-only query."
                )
        if errors:
            return fail(errors.pop())

        cte_names = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
        tables: list[str] = []
        datasets: list[str] = []
        for table in root.find_all(exp.Table):
            if not isinstance(table.this, exp.Identifier):
                inner = table.this
                name = _function_name(inner) if isinstance(inner, exp.Func) else type(inner).__name__
                errors.append(f"Table functions are not allowed (found '{name}').")
                continue
            catalog, db, name = (
                (table.catalog or "").lower(),
                (table.db or "").lower(),
                (table.name or "").lower(),
            )
            if not catalog and not db and name in cte_names:
                continue
            if re.search(r"[/\\:]|\.(csv|parquet|json|tsv|txt|db|duckdb|xlsx?)$", name) or name.startswith(
                "."
            ):
                errors.append(f"File paths are not allowed as tables ('{table.name}').")
                continue
            resolved = self._resolve(policy, catalog, db, name)
            if resolved not in policy.allowed_tables:
                errors.append(
                    f"Table '{'.'.join(p for p in (catalog, db, name) if p)}' is not in the dataset allowlist."
                )
                continue
            tables.append(resolved)
            datasets.append(policy.allowed_tables[resolved])
            if policy.dialect == "databricks":
                parts = resolved.split(".")
                if len(parts) == 3:  # execute with fully-qualified names, independent of session defaults
                    table.set("catalog", exp.to_identifier(parts[0]))
                    table.set("db", exp.to_identifier(parts[1]))

        for fn in root.find_all(exp.Func):
            fname = _function_name(fn)
            if isinstance(fn.parent, exp.Dot):
                errors.append(
                    f"Qualified function calls (user-defined functions) are not allowed ('{fname}')."
                )
                continue
            if any(p.search(fname) for p in DENIED_FUNCTION_PATTERNS):
                errors.append(f"Function '{fname}' is not allowed.")

        sensitive = set()
        for did in set(datasets):
            sensitive |= {c.lower() for c in policy.sensitive_columns.get(did, set())}
        if sensitive:
            for col in root.find_all(exp.Column):
                if col.name.lower() in sensitive:
                    errors.append(f"Column '{col.name}' is registered as sensitive and cannot be queried.")
            if any(isinstance(s, exp.Star) for s in root.find_all(exp.Star)):
                errors.append(
                    "SELECT * is not allowed on datasets with sensitive columns; list columns explicitly."
                )
        if not tables and not errors:
            warnings.append("Query does not read any registered dataset.")
        if errors:
            # de-duplicate while keeping order
            seen: set[str] = set()
            uniq = [e for e in errors if not (e in seen or seen.add(e))]  # type: ignore[func-returns-value]
            return SQLValidationResult(
                ok=False,
                sql="",
                original_sql=original,
                dialect=policy.dialect,
                errors=uniq,
                warnings=warnings,
                tables=sorted(set(tables)),
                datasets=sorted(set(datasets)),
            )

        limit = policy.max_rows
        existing = root.args.get("limit")
        if existing is not None:
            value = existing.expression if isinstance(existing, exp.Limit) else None
            if isinstance(value, exp.Literal) and value.is_int:
                current = int(value.this)
                if current <= policy.max_rows:
                    limit = current
                else:
                    warnings.append(f"LIMIT {current} reduced to the maximum of {policy.max_rows} rows.")
            else:
                warnings.append(f"Non-literal LIMIT replaced with {policy.max_rows}.")
        root = root.limit(limit)
        normalized = root.sql(dialect=policy.dialect, comments=False)
        return SQLValidationResult(
            ok=True,
            sql=normalized,
            original_sql=original,
            dialect=policy.dialect,
            warnings=warnings,
            tables=sorted(set(tables)),
            datasets=sorted(set(datasets)),
            limit=limit,
            fingerprint=fingerprint_sql(normalized),
        )

    @staticmethod
    def _resolve(policy: SQLPolicy, catalog: str, db: str, name: str) -> str:
        if policy.dialect == "databricks":
            cat = catalog or (policy.default_catalog or "").lower()
            sch = db or (policy.default_schema or "").lower()
            return ".".join(p for p in (cat, sch, name) if p)
        # DuckDB: tables live in the in-memory "main" schema
        if db in ("", "main") and catalog in ("", "memory"):
            return name
        return ".".join(p for p in (catalog, db, name) if p)
