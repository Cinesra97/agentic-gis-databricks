"""Dataset registry: the allowlist of datasets the agent may discover and query.

Datasets are declared by administrators in ``config/datasets.yaml`` (committed) and an
optional ``config/datasets.local.yaml`` overlay (git-ignored, for workspace-specific or
sensitive entries). Only *enabled* datasets that are visible in the active mode can be
listed, described, or queried; the SQL guard uses this registry as its table allowlist.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from agentic_gis.config.settings import PROJECT_ROOT, SENSITIVITY_ORDER, AppMode, Sensitivity

DatasetRole = Literal[
    "events", "infrastructure", "parcels", "streets", "boundary", "reference", "imagery_index", "other"
]
GeometryType = Literal["point", "line", "polygon", "mixed"]
GeometryFormat = Literal["wkb", "wkt", "geojson", "latlon", "native"]

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")
_SAFE_ENV_VALUE = re.compile(r"^[A-Za-z0-9_.\-]*$")
_ID_PATTERN = r"^[a-z][a-z0-9_]{1,62}$"


class RegistryError(ValueError):
    """Invalid registry configuration."""


class UnknownDatasetError(KeyError):
    """The dataset id is not registered, not enabled, or not visible in this mode."""


class ColumnSpec(BaseModel):
    name: str
    type: str = "string"
    description: str = ""
    known_values: list[str] = Field(default_factory=list)
    sensitive: bool = Field(
        False, description="Sensitive columns are never shown to the LLM and are blocked in SQL."
    )

    @field_validator("known_values", mode="before")
    @classmethod
    def _str_values(cls, v: Any) -> Any:
        return [str(x) for x in v] if isinstance(v, list) else v


class SpatialSpec(BaseModel):
    geometry_column: str | None = None
    geometry_type: GeometryType = "point"
    format: GeometryFormat = "wkb"
    crs: str = "EPSG:4326"
    lat_column: str | None = None
    lon_column: str | None = None

    @model_validator(mode="after")
    def _check(self) -> SpatialSpec:
        if self.format == "latlon" and not (self.lat_column and self.lon_column):
            raise ValueError("format 'latlon' requires lat_column and lon_column")
        if self.format != "latlon" and not self.geometry_column:
            raise ValueError(f"format '{self.format}' requires geometry_column")
        return self


class TemporalSpec(BaseModel):
    column: str
    granularity: Literal["day", "timestamp", "month", "year"] = "timestamp"


class SocrataSourceSpec(BaseModel):
    dataset_id: str = Field(pattern=r"^[a-z0-9]{4}-[a-z0-9]{4}$")
    select: str | None = None
    where: str | None = None
    order: str | None = None
    limit: int = Field(50_000, ge=1, le=2_000_000)
    geojson: bool = False
    verified_id: bool = Field(False, description="Whether the dataset id was verified against the portal.")


class SourceSpec(BaseModel):
    type: Literal["file", "socrata", "databricks_table"]
    path: str | None = None
    format: Literal["geoparquet", "parquet", "geojson", "csv"] | None = None
    table: str | None = None
    socrata: SocrataSourceSpec | None = None

    @model_validator(mode="after")
    def _check(self) -> SourceSpec:
        if self.type in ("file", "socrata") and not self.path:
            raise ValueError(f"source type '{self.type}' requires a local 'path'")
        if self.type == "socrata" and self.socrata is None:
            raise ValueError("source type 'socrata' requires a 'socrata' block")
        if self.type == "databricks_table":
            if not self.table:
                raise ValueError("source type 'databricks_table' requires 'table'")
            parts = self.table.split(".")
            if len(parts) != 3 or not all(re.fullmatch(r"[A-Za-z0-9_\-]+", p) for p in parts):
                raise ValueError("Databricks table must be a three-part name catalog.schema.table")
        return self

    def resolved_path(self) -> Path | None:
        if not self.path:
            return None
        p = Path(self.path)
        return p if p.is_absolute() else PROJECT_ROOT / p


class ProvenanceSpec(BaseModel):
    publisher: str = "Unknown"
    source_url: str | None = None
    license: str | None = None
    retrieved: str | None = None
    notes: str | None = None
    citation: str | None = None


class DatasetSpec(BaseModel):
    id: str = Field(pattern=_ID_PATTERN)
    name: str
    short_name: str | None = Field(
        None, description="Short label used in result titles, e.g. '311 complaints'"
    )
    description: str
    backend: Literal["local", "databricks"]
    modes: list[AppMode]
    enabled: bool = True
    synthetic: bool = False
    sensitivity: Sensitivity = "public"
    role: DatasetRole = "other"
    geography_role: str | None = Field(
        None, description="For boundary datasets: borough | neighborhood | zip | community_district ..."
    )
    name_column: str | None = None
    tags: list[str] = Field(default_factory=list)
    synonyms: dict[str, str] = Field(default_factory=dict)
    source: SourceSpec
    table: str | None = Field(None, description="DuckDB table name (local backend). Defaults to id.")
    spatial: SpatialSpec | None = None
    temporal: TemporalSpec | None = None
    id_column: str | None = None
    category_column: str | None = None
    join_keys: dict[str, str] = Field(default_factory=dict)
    columns: list[ColumnSpec] = Field(default_factory=list)
    schema_discovery: bool = False
    verification_source: bool = Field(
        False,
        description="True only when records represent verified outcomes (e.g. confirmed inspections).",
    )
    provenance: ProvenanceSpec = Field(default_factory=ProvenanceSpec)
    max_rows: int | None = None

    @field_validator("tags", mode="before")
    @classmethod
    def _str_tags(cls, v: Any) -> Any:
        return [str(x) for x in v] if isinstance(v, list) else v

    @field_validator("table")
    @classmethod
    def _table_ident(cls, v: str | None) -> str | None:
        if v is not None and not re.fullmatch(r"[a-z][a-z0-9_]{1,62}", v):
            raise ValueError("local table names must be lowercase identifiers")
        return v

    @model_validator(mode="after")
    def _check(self) -> DatasetSpec:
        if self.backend == "databricks" and self.source.type != "databricks_table":
            raise ValueError(f"{self.id}: databricks datasets need source.type 'databricks_table'")
        if self.backend == "local" and self.source.type == "databricks_table":
            raise ValueError(f"{self.id}: local datasets cannot use a databricks_table source")
        return self

    # ---- convenience ---------------------------------------------------------------------
    @property
    def sql_table(self) -> str:
        """Identifier used in SQL (DuckDB table name or Databricks three-part name)."""
        if self.backend == "databricks":
            assert self.source.table
            return self.source.table
        return self.table or self.id

    @property
    def label(self) -> str:
        return self.short_name or self.name

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def sensitive_columns(self) -> set[str]:
        return {c.name.lower() for c in self.columns if c.sensitive}

    def column(self, name: str) -> ColumnSpec | None:
        for c in self.columns:
            if c.name.lower() == name.lower():
                return c
        return None

    def known_categories(self) -> list[str]:
        if not self.category_column:
            return []
        col = self.column(self.category_column)
        return list(col.known_values) if col else []

    def search_text(self) -> str:
        parts = [self.id, self.name, self.description, " ".join(self.tags), self.role]
        parts += [c.name + " " + c.description for c in self.columns]
        parts += self.known_categories()
        parts += list(self.synonyms)
        return " ".join(parts).lower()


class DatasetStatus(BaseModel):
    dataset_id: str
    state: Literal["ready", "missing_data", "disabled", "needs_configuration"]
    message: str = ""


def _substitute_env(text: str, env: dict[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        name, default = m.group(1), m.group(2)
        value = env.get(name)
        if value is None or value == "":
            if default is None:
                raise RegistryError(f"Environment variable {name} is required by the dataset registry")
            value = default
        if not _SAFE_ENV_VALUE.match(value):
            raise RegistryError(f"Environment variable {name} contains characters not allowed in identifiers")
        return value

    return _ENV_PATTERN.sub(repl, text)


def _read_yaml(path: Path, env: dict[str, str]) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(_substitute_env(raw, env)) or {}
    if not isinstance(data, dict):
        raise RegistryError(f"{path} must contain a mapping at the top level")
    return data


class DatasetRegistry:
    """In-memory registry of dataset specifications and geography roles."""

    def __init__(self, datasets: Iterable[DatasetSpec], geographies: dict[str, Any] | None = None):
        self._datasets: dict[str, DatasetSpec] = {}
        for ds in datasets:
            self._datasets[ds.id] = ds
        self.geographies: dict[str, Any] = geographies or {}
        self._validate_unique_tables()

    # ---- construction -------------------------------------------------------------------
    @classmethod
    def load(
        cls, path: Path, overlay: Path | None = None, env: dict[str, str] | None = None
    ) -> DatasetRegistry:
        env = dict(os.environ) if env is None else env
        base = _read_yaml(path, env)
        merged: dict[str, dict[str, Any]] = {}
        for item in base.get("datasets", []) or []:
            merged[item["id"]] = item
        geographies = dict(base.get("geographies", {}) or {})
        if overlay is not None and overlay.exists():
            extra = _read_yaml(overlay, env)
            for item in extra.get("datasets", []) or []:
                if item.get("id") in merged and item.get("_merge", True):
                    merged[item["id"]] = {**merged[item["id"]], **item}
                else:
                    merged[item["id"]] = item
            geographies.update(extra.get("geographies", {}) or {})
        specs = []
        for item in merged.values():
            item = {k: v for k, v in item.items() if not k.startswith("_")}
            try:
                specs.append(DatasetSpec.model_validate(item))
            except Exception as exc:  # pragma: no cover - message path tested via RegistryError
                raise RegistryError(f"Invalid dataset '{item.get('id')}': {exc}") from exc
        return cls(specs, geographies)

    def _validate_unique_tables(self) -> None:
        seen: dict[str, str] = {}
        for ds in self._datasets.values():
            key = f"{ds.backend}:{ds.sql_table.lower()}"
            if key in seen:
                raise RegistryError(f"Datasets {seen[key]} and {ds.id} map to the same table {ds.sql_table}")
            seen[key] = ds.id

    # ---- lookup ----------------------------------------------------------------------------
    def all(self) -> list[DatasetSpec]:
        return list(self._datasets.values())

    def get_any(self, dataset_id: str) -> DatasetSpec:
        try:
            return self._datasets[dataset_id]
        except KeyError as exc:
            raise UnknownDatasetError(dataset_id) from exc

    def get(self, dataset_id: str, mode: str) -> DatasetSpec:
        ds = self._datasets.get(dataset_id)
        if ds is None or not ds.enabled or mode not in ds.modes:
            raise UnknownDatasetError(
                f"Dataset '{dataset_id}' is not registered, not enabled, or not available in '{mode}' mode"
            )
        return ds

    def status(self, ds: DatasetSpec) -> DatasetStatus:
        if not ds.enabled:
            return DatasetStatus(dataset_id=ds.id, state="disabled", message="Disabled by administrator")
        if ds.backend == "local":
            p = ds.source.resolved_path()
            if p is None or not p.exists():
                hint = (
                    f"Run: agentic-gis ingest {ds.id}"
                    if ds.source.type == "socrata"
                    else "Data file not found"
                )
                return DatasetStatus(dataset_id=ds.id, state="missing_data", message=hint)
        if ds.backend == "databricks" and ds.schema_discovery and not ds.columns:
            return DatasetStatus(
                dataset_id=ds.id,
                state="ready",
                message="Columns will be discovered from Unity Catalog at query time",
            )
        return DatasetStatus(dataset_id=ds.id, state="ready")

    def available(self, mode: str, backend: str | None = None, ready_only: bool = True) -> list[DatasetSpec]:
        out = []
        for ds in self._datasets.values():
            if not ds.enabled or mode not in ds.modes:
                continue
            if backend and ds.backend != backend:
                continue
            if ready_only and self.status(ds).state != "ready":
                continue
            out.append(ds)
        return out

    def geography(self, role: str, mode: str) -> DatasetSpec | None:
        """Resolve a geography role (borough, neighborhood, zip ...) to a boundary dataset."""
        entry = self.geographies.get(role)
        candidates: list[str] = []
        if isinstance(entry, str):
            candidates = [entry]
        elif isinstance(entry, dict):
            v = entry.get(mode) or entry.get("default")
            if isinstance(v, list):
                candidates = list(v)
            elif v:
                candidates = [v]
        elif isinstance(entry, list):
            candidates = list(entry)
        ready = {d.id for d in self.available(mode)}
        for cid in candidates:
            if cid in ready:
                return self._datasets[cid]
        for ds in self.available(mode):
            if ds.geography_role == role:
                return ds
        return None

    def geography_roles(self, mode: str) -> list[str]:
        roles = set(self.geographies)
        roles |= {d.geography_role for d in self.available(mode) if d.geography_role}
        return sorted(r for r in roles if self.geography(r, mode) is not None)

    def allowed_tables(self, mode: str, backend: str) -> dict[str, str]:
        """Map normalized SQL table identifiers -> dataset id for the SQL guard allowlist."""
        out: dict[str, str] = {}
        for ds in self.available(mode, backend=backend):
            out[ds.sql_table.lower()] = ds.id
        return out

    def search(self, text: str, mode: str, limit: int = 8) -> list[tuple[DatasetSpec, float]]:
        tokens = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2]
        scored: list[tuple[DatasetSpec, float]] = []
        for ds in self.available(mode):
            hay = ds.search_text()
            score = 0.0
            for t in tokens:
                if t in hay:
                    score += 1.0
                else:
                    stem = t.rstrip("s")
                    if len(stem) > 3 and stem in hay:
                        score += 0.7
            score += 0.3 * SequenceMatcher(None, text.lower(), ds.name.lower()).ratio()
            if ds.role == "events":
                score += 0.2
            scored.append((ds, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def max_sensitivity(self, dataset_ids: Iterable[str]) -> str:
        level = 0
        for did in dataset_ids:
            ds = self._datasets.get(did)
            if ds is not None:
                level = max(level, SENSITIVITY_ORDER[ds.sensitivity])
        return [k for k, v in SENSITIVITY_ORDER.items() if v == level][0]

    # ---- administration -------------------------------------------------------------------
    def register(self, spec: DatasetSpec, overlay_path: Path) -> None:
        """Add or replace a dataset in the git-ignored overlay file (admin action)."""
        existing: dict[str, Any] = {}
        if overlay_path.exists():
            existing = yaml.safe_load(overlay_path.read_text(encoding="utf-8")) or {}
        items = [d for d in existing.get("datasets", []) or [] if d.get("id") != spec.id]
        items.append({"_merge": False, **spec.model_dump(mode="json", exclude_none=True)})
        existing["datasets"] = items
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.write_text(yaml.safe_dump(existing, sort_keys=False), encoding="utf-8")
        self._datasets[spec.id] = spec
        self._validate_unique_tables()


def load_registry(settings: Any) -> DatasetRegistry:
    return DatasetRegistry.load(settings.datasets_config, settings.datasets_local_config)
