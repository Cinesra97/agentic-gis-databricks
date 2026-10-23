"""Extensible tool registry and the built-in GIS tools.

Design rules:

* Every tool has a Pydantic input model. The LLM (or rule planner) supplies JSON arguments;
  they are validated before the tool runs. Unknown tools/arguments are rejected.
* Tools receive data only *by reference* (artifact ids such as ``s1`` / ``t2.s3`` or
  registered dataset ids) -- never raw records or code from the LLM.
* SQL-bearing tools expose a ``sql_renderer`` so the graph can validate and (when required)
  obtain user approval for the exact SQL before execution.
* There is no tool for arbitrary Python, shell, file or network access.

Add a tool by calling :meth:`ToolRegistry.register` (see ``docs/EXTENDING.md``).
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from difflib import get_close_matches
from typing import Any, Literal

import geopandas as gpd
import httpx
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlglot import exp

from agentic_gis.agent.workspace import Artifact, ResultStore
from agentic_gis.config.registry import DatasetRegistry, DatasetSpec, UnknownDatasetError
from agentic_gis.config.settings import Settings
from agentic_gis.connectors.base import ColumnInfo, ConnectorError, QueryEngine
from agentic_gis.security.audit import AuditLogger
from agentic_gis.security.sanitize import clean_text
from agentic_gis.security.sql_guard import SQLGuard, SQLPolicy, SQLValidationResult
from agentic_gis.spatial import analysis as A
from agentic_gis.spatial import operations as O
from agentic_gis.spatial.geometry import SpatialError, ensure_gdf, to_wgs84
from agentic_gis.ui.layers import build_layer_spec

log = logging.getLogger(__name__)

Unit = Literal["ft", "m", "km", "mi", "yd", "block"]
REF: Any = {"x-ref": True}  # marks reference arguments in tool input models

BOROUGH_ALIASES = {
    "manhattan": "MANHATTAN",
    "new york county": "MANHATTAN",
    "mn": "MANHATTAN",
    "bronx": "BRONX",
    "the bronx": "BRONX",
    "bx": "BRONX",
    "brooklyn": "BROOKLYN",
    "kings": "BROOKLYN",
    "bk": "BROOKLYN",
    "queens": "QUEENS",
    "qn": "QUEENS",
    "qns": "QUEENS",
    "staten island": "STATEN ISLAND",
    "richmond": "STATEN ISLAND",
    "si": "STATEN ISLAND",
}


def normalize_borough(name: str) -> str | None:
    key = re.sub(r"\s+", " ", name.strip().lower())
    if key in BOROUGH_ALIASES:
        return BOROUGH_ALIASES[key]
    hit = get_close_matches(key, list(BOROUGH_ALIASES), n=1, cutoff=0.8)
    return BOROUGH_ALIASES[hit[0]] if hit else None


# =========================================================================================
# Registry infrastructure
# =========================================================================================
class ToolOutput(BaseModel):
    status: Literal["ok", "error", "empty", "needs_approval", "unavailable"] = "ok"
    summary: str = ""
    headline: str | None = Field(None, description="One-line takeaway when this step is the main result")
    highlights: list[str] = Field(default_factory=list)
    artifact_id: str | None = None
    rows: int | None = None
    sql: str | None = None
    data: Any = None
    layer: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    elapsed_ms: int = 0


@dataclass
class RenderedSQL:
    sql: str
    params: dict[str, Any]
    backend: str
    kind: Literal["template", "llm_sql"]
    datasets: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class NeedsApprovalError(RuntimeError):
    def __init__(self, review: SQLValidationResult):
        super().__init__("SQL requires approval")
        self.review = review


@dataclass
class ToolContext:
    settings: Settings
    registry: DatasetRegistry
    mode: str
    store: ResultStore
    engines: dict[str, QueryEngine]
    guard: SQLGuard
    audit: AuditLogger
    session_id: str = "local"
    turn: int = 1
    step_id: str = "s0"
    approved: set[str] = field(default_factory=set)
    layers: list[dict[str, Any]] = field(default_factory=list)
    schema_cache: dict[str, list[ColumnInfo]] = field(default_factory=dict)
    reference_cache: dict[str, gpd.GeoDataFrame] = field(default_factory=dict)

    # ---- datasets -----------------------------------------------------------------------
    def dataset(self, dataset_id: str) -> DatasetSpec:
        return self.registry.get(dataset_id, self.mode)

    def engine_for(self, ds: DatasetSpec) -> QueryEngine:
        engine = self.engines.get(ds.backend)
        if engine is None:
            raise ConnectorError(
                f"The {ds.backend} engine is not connected; dataset '{ds.id}' cannot be queried in this session."
            )
        return engine

    def policy_for(self, backend: str) -> SQLPolicy:
        engine = self.engines.get(backend)
        dialect = getattr(engine, "dialect", "duckdb" if backend == "local" else "databricks")
        sensitive = {d.id: d.sensitive_columns for d in self.registry.available(self.mode, backend=backend)}
        return SQLPolicy(
            dialect=dialect,
            allowed_tables=self.registry.allowed_tables(self.mode, backend),
            max_rows=self.settings.max_result_rows,
            sensitive_columns=sensitive,
            default_catalog=self.settings.databricks_catalog,
            default_schema=self.settings.databricks_schema,
        )

    def columns_for(self, ds: DatasetSpec) -> list[ColumnInfo]:
        if ds.columns and not ds.schema_discovery:
            return [ColumnInfo(c.name, c.type, c.description) for c in ds.columns]
        if ds.id not in self.schema_cache:
            engine = self.engine_for(ds)
            live = engine.describe(ds.sql_table)
            reg = {c.name.lower(): c for c in ds.columns}
            merged = [
                ColumnInfo(
                    c.name,
                    c.type,
                    reg[c.name.lower()].description
                    if c.name.lower() in reg
                    else clean_text(c.comment, 160)[0],
                )
                for c in live
            ]
            self.schema_cache[ds.id] = merged
        return self.schema_cache[ds.id]

    def requires_approval(self, kind: str, backend: str) -> bool:
        mode = self.settings.sql_approval
        if mode == "never":
            return False
        if mode == "always":
            return True
        return kind == "llm_sql" or backend == "databricks"

    def validate_sql(self, rendered: RenderedSQL) -> SQLValidationResult:
        return self.guard.validate(rendered.sql, self.policy_for(rendered.backend))

    def run_sql(self, rendered: RenderedSQL) -> tuple[pd.DataFrame, SQLValidationResult, Any]:
        review = self.validate_sql(rendered)
        if not review.ok:
            raise ConnectorError("SQL rejected by policy: " + "; ".join(review.errors))
        if (
            self.requires_approval(rendered.kind, rendered.backend)
            and review.fingerprint not in self.approved
        ):
            raise NeedsApprovalError(review)
        engine = self.engines.get(rendered.backend)
        if engine is None:
            raise ConnectorError(f"The {rendered.backend} engine is not connected.")
        self.audit.log(
            "sql_execute",
            self.session_id,
            turn=self.turn,
            step=self.step_id,
            backend=rendered.backend,
            sql=review.sql,
            fingerprint=review.fingerprint,
            datasets=review.datasets,
        )
        result = engine.execute(
            review.sql, rendered.params, timeout_s=self.settings.query_timeout_s, max_rows=review.limit
        )
        self.audit.log(
            "sql_result",
            self.session_id,
            turn=self.turn,
            step=self.step_id,
            rows=result.rows,
            elapsed_ms=result.elapsed_ms,
            truncated=result.truncated,
        )
        return result.df, review, result

    # ---- artifacts ----------------------------------------------------------------------
    def resolve(self, ref: str) -> Artifact:
        art = self.store.resolve(ref, self.turn)
        if art is None:
            raise SpatialError(
                f"Unknown input reference '{ref}'. Use a previous step id (e.g. s1) or 'last'."
            )
        return art

    def resolve_spatial(self, ref: str) -> tuple[Artifact, gpd.GeoDataFrame]:
        art = self.resolve(ref)
        if not art.is_spatial:
            raise SpatialError(f"Input '{ref}' ({art.title}) has no geometry")
        return art, art.data

    def save(
        self,
        title: str,
        data: pd.DataFrame,
        tool: str,
        inputs: list[Artifact] | None = None,
        datasets: list[str] | None = None,
        description: str = "",
        sql: str | None = None,
        notes: list[str] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> Artifact:
        inputs = inputs or []
        ds_ids = list(dict.fromkeys((datasets or []) + [d for a in inputs for d in a.datasets]))
        synthetic = any(a.synthetic for a in inputs) or any(
            self.registry.get_any(d).synthetic for d in ds_ids if d in {x.id for x in self.registry.all()}
        )
        art = Artifact(
            id=f"t{self.turn}.{self.step_id}",
            turn=self.turn,
            step_id=self.step_id,
            tool=tool,
            title=title,
            data=data,
            description=description,
            datasets=ds_ids,
            synthetic=synthetic,
            sql=sql,
            notes=notes or [],
            extras=extras or {},
        )
        return self.store.add(art)

    def primary_spec(self, art: Artifact) -> DatasetSpec | None:
        for did in art.datasets:
            try:
                ds = self.registry.get_any(did)
            except UnknownDatasetError:
                continue
            if ds.role in ("events", "infrastructure", "parcels", "streets"):
                return ds
        return None

    def load_reference(self, ds: DatasetSpec) -> gpd.GeoDataFrame:
        """Load a (bounded) local boundary/reference dataset in full."""
        if ds.backend != "local":
            raise SpatialError(f"Geography dataset '{ds.id}' must be registered with the local backend")
        if ds.id not in self.reference_cache:
            rendered = render_filter_sql(self, FilterDatasetArgs(dataset=ds.id))
            df, _, _ = self.run_sql(rendered)
            gdf = to_gdf(df, ds)
            if gdf is None:
                raise SpatialError(f"Dataset '{ds.id}' has no geometry")
            self.reference_cache[ds.id] = gdf
        return self.reference_cache[ds.id]

    def geography(self, name: str) -> tuple[DatasetSpec, gpd.GeoDataFrame]:
        ds = self.registry.geography(name, self.mode)
        if ds is None:
            roles = ", ".join(self.registry.geography_roles(self.mode))
            raise SpatialError(f"Unknown geography '{name}'. Available: {roles}, grid")
        return ds, self.load_reference(ds)


@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[[ToolContext, Any], ToolOutput]
    category: str = "analysis"
    sql_renderer: Callable[[ToolContext, Any], RenderedSQL] | None = None
    available: Callable[[ToolContext], bool] | None = None

    @property
    def ref_fields(self) -> list[str]:
        out = []
        for name, f in self.args_model.model_fields.items():
            extra = f.json_schema_extra if isinstance(f.json_schema_extra, dict) else {}
            if extra.get("x-ref"):
                out.append(name)
        return out


class UnknownToolError(KeyError):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,40}", spec.name):
            raise ValueError(f"Invalid tool name {spec.name}")
        self._tools[spec.name] = spec
        return spec

    def tool(
        self,
        name: str,
        description: str,
        args_model: type[BaseModel],
        category: str = "analysis",
        sql_renderer: Callable[[ToolContext, Any], RenderedSQL] | None = None,
        available: Callable[[ToolContext], bool] | None = None,
    ) -> Callable[[Callable], Callable]:
        def deco(fn: Callable[[ToolContext, Any], ToolOutput]) -> Callable:
            self.register(ToolSpec(name, description, args_model, fn, category, sql_renderer, available))
            return fn

        return deco

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise UnknownToolError(f"Unknown tool '{name}'")
        return self._tools[name]

    def specs(self, ctx: ToolContext | None = None) -> list[ToolSpec]:
        return [t for t in self._tools.values() if ctx is None or t.available is None or t.available(ctx)]

    def names(self, ctx: ToolContext | None = None) -> list[str]:
        return [t.name for t in self.specs(ctx)]

    def validate_args(self, name: str, args: dict[str, Any]) -> BaseModel:
        spec = self.get(name)
        return spec.args_model.model_validate(args or {})

    def catalog_for_prompt(self, ctx: ToolContext | None = None) -> str:
        lines = []
        for t in self.specs(ctx):
            params = []
            for fname, f in t.args_model.model_fields.items():
                ann = getattr(f.annotation, "__name__", None) or str(f.annotation).replace("typing.", "")
                ann = re.sub(r"<class '([^']+)'>", r"\1", ann)
                default = "" if f.is_required() else f"={f.default!r}" if f.default is not None else "=None"
                desc = f" -- {f.description}" if f.description else ""
                params.append(f"    {fname}: {ann}{default}{desc}")
            lines.append(f"- {t.name}: {t.description}\n" + "\n".join(params))
        return "\n".join(lines)

    def invoke(self, name: str, raw_args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        t0 = time.perf_counter()
        try:
            spec = self.get(name)
            if spec.available is not None and not spec.available(ctx):
                out = ToolOutput(
                    status="unavailable", summary=f"Tool '{name}' is not available in this session."
                )
            else:
                args = spec.args_model.model_validate(raw_args or {})
                out = spec.func(ctx, args)
        except UnknownToolError as exc:
            out = ToolOutput(status="error", summary=str(exc))
        except ValidationError as exc:
            msgs = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5])
            out = ToolOutput(status="error", summary=f"Invalid arguments for {name}: {msgs}")
        except NeedsApprovalError as exc:
            out = ToolOutput(
                status="needs_approval",
                summary="This query needs user approval before it runs.",
                sql=exc.review.sql,
            )
        except (SpatialError, ConnectorError, UnknownDatasetError, ValueError) as exc:
            out = ToolOutput(status="error", summary=str(exc).strip("'\""))
        except Exception as exc:  # unexpected: log details, show a generic message
            log.exception("Tool %s failed", name)
            out = ToolOutput(status="error", summary=f"{name} failed unexpectedly ({type(exc).__name__}).")
        out.elapsed_ms = int((time.perf_counter() - t0) * 1000)
        ctx.audit.log(
            "tool_call",
            ctx.session_id,
            turn=ctx.turn,
            step=ctx.step_id,
            tool=name,
            args=_safe_args(raw_args),
            status=out.status,
            rows=out.rows,
            elapsed_ms=out.elapsed_ms,
            summary=out.summary[:300],
        )
        return out


def _empty_input(art: Artifact, what: str) -> ToolOutput | None:
    if art.rows == 0:
        return ToolOutput(
            status="empty",
            summary=f"Skipped {what}: the input '{art.title}' has no records.",
            artifact_id=None,
            rows=0,
            datasets=art.datasets,
        )
    return None


def _measure_label(measure: str, value_column: str | None) -> str:
    vc = (value_column or "").replace("_", " ")
    return {
        "count": "count",
        "count_distinct": f"distinct {vc} values",
        "sum": f"total {vc}",
        "mean": f"average {vc}",
    }.get(measure, measure)


def _safe_args(args: dict[str, Any] | None) -> dict[str, Any]:
    return {
        k: (v if not isinstance(v, str) or len(v) < 2000 else v[:2000] + "…") for k, v in (args or {}).items()
    }


# =========================================================================================
# Helpers
# =========================================================================================
def to_gdf(df: pd.DataFrame, ds: DatasetSpec | None) -> gpd.GeoDataFrame | None:
    sp = ds.spatial if ds else None
    if sp is None:
        return ensure_gdf(df)
    if sp.format == "latlon":
        return ensure_gdf(df, None, sp.lat_column, sp.lon_column, sp.crs)
    gdf = ensure_gdf(df, sp.geometry_column, sp.lat_column, sp.lon_column, sp.crs)
    if gdf is not None and gdf.geometry.name != "geometry":
        gdf = gdf.rename_geometry("geometry")
    return gdf


def eval_list(text: str) -> list[str]:
    """Parse the "['a', 'b']" list rendered in filter notes (no eval)."""
    return [x.strip().strip("'\"") for x in text.strip("[]").split(",") if x.strip()]


def restrict_areas_to_input(areas: gpd.GeoDataFrame, points: pd.DataFrame) -> gpd.GeoDataFrame:
    """When the input was filtered to some boroughs, keep only areas in those boroughs."""
    if "borough" in areas.columns and "borough" in points.columns:
        present = set(points["borough"].dropna().astype(str).str.upper())
        if 0 < len(present) < 5:
            sub = areas[areas["borough"].astype(str).str.upper().isin(present)]
            if len(sub):
                return sub
    return areas


def resolve_categories(ds: DatasetSpec, terms: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Map user terms to known category values. Returns (exact_values, like_terms, notes)."""
    known = ds.known_categories()
    syn = {k.lower(): v for k, v in ds.synonyms.items()}
    exact: list[str] = []
    like: list[str] = []
    notes: list[str] = []
    for term in terms:
        t = term.strip().lower()
        if not t:
            continue
        if t in syn and (not known or syn[t] in known):
            exact.append(syn[t])
            continue
        hits = [k for k in known if k.lower() == t]
        if not hits:
            stem = t.rstrip("s")
            hits = [k for k in known if stem in k.lower() or k.lower().rstrip("s") in t]
        if not hits and known:
            hits = get_close_matches(term, known, n=2, cutoff=0.75)
        if hits:
            exact.extend(hits)
        else:
            like.append(t)
            if known:
                notes.append(f"'{term}' is not a known {ds.category_column} value; matched as text.")
    return list(dict.fromkeys(exact)), like, notes


# =========================================================================================
# Tool argument models
# =========================================================================================
class ListDatasetsArgs(BaseModel):
    query: str | None = Field(None, description="Optional keywords to rank datasets")
    role: str | None = Field(
        None, description="Optional role filter: events, infrastructure, parcels, streets, boundary"
    )


class DatasetArg(BaseModel):
    dataset: str = Field(description="Registered dataset id")


class AttributeFilter(BaseModel):
    column: str
    op: Literal[
        "=",
        "!=",
        ">",
        ">=",
        "<",
        "<=",
        "in",
        "not_in",
        "contains",
        "starts_with",
        "is_null",
        "not_null",
        "between",
    ] = "="
    value: Any = None


class FilterDatasetArgs(BaseModel):
    dataset: str = Field(description="Registered dataset id to read")
    categories: list[str] | None = Field(
        None, description="Values/keywords for the dataset's category column"
    )
    boroughs: list[str] | None = Field(None, description="Borough names to keep")
    date_from: str | None = Field(None, description="Inclusive start date YYYY-MM-DD (temporal column)")
    date_to: str | None = Field(None, description="Inclusive end date YYYY-MM-DD")
    filters: list[AttributeFilter] = Field(default_factory=list, description="Extra column filters")
    columns: list[str] | None = Field(None, description="Columns to return (default: all non-sensitive)")
    limit: int | None = Field(None, ge=1, description="Row limit (capped by configuration)")

    @field_validator("date_from", "date_to")
    @classmethod
    def _iso(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        try:
            return date.fromisoformat(str(v)[:10]).isoformat()
        except ValueError as exc:
            raise ValueError("dates must be YYYY-MM-DD") from exc


class QuerySQLArgs(BaseModel):
    sql: str = Field(description="A single read-only SELECT over registered tables")
    purpose: str = Field("", description="What the query is for (shown to the user)")


class SpatialJoinArgs(BaseModel):
    left: str = Field(
        description="Reference to features to keep (step id, 'last', or dataset id)", json_schema_extra=REF
    )
    right: str = Field(description="Reference to features to test against", json_schema_extra=REF)
    predicate: Literal["intersects", "within", "contains", "dwithin"] = "intersects"
    distance: float | None = Field(None, ge=0, le=52800, description="Distance for predicate 'dwithin'")
    units: Unit = "ft"
    how: Literal["inner", "left"] = "inner"
    summarize: bool = Field(
        True, description="One row per left feature with match_count (instead of all pairs)"
    )


class BufferArgs(BaseModel):
    input: str = Field(description="Reference to features to buffer", json_schema_extra=REF)
    distance: float = Field(gt=0, le=52800)
    units: Unit = "ft"
    dissolve: bool = True


class NearestArgs(BaseModel):
    input: str = Field(description="Features that need a nearest neighbor", json_schema_extra=REF)
    target: str = Field(description="Candidate neighbor features", json_schema_extra=REF)
    max_distance: float | None = Field(None, ge=0, le=52800)
    units: Unit = "ft"


class FilterByAreaArgs(BaseModel):
    input: str = Field(description="Features to filter", json_schema_extra=REF)
    area_type: str = Field(
        "neighborhood", description="Geography role: borough, neighborhood, community_district, zip"
    )
    names: list[str] = Field(min_length=1, description="Area names or codes to keep")


class AggregateArgs(BaseModel):
    input: str = Field(description="Point features to aggregate", json_schema_extra=REF)
    geography: str = Field(
        "neighborhood",
        description="borough | neighborhood | community_district | zip | grid, "
        "or a reference to line/polygon features (e.g. streets)",
        json_schema_extra=REF,
    )
    measure: Literal["count", "count_distinct", "sum", "mean"] = "count"
    value_column: str | None = Field(None, description="Column for count_distinct/sum/mean")
    category_column: str | None = Field(None, description="Add per-category counts (top 6)")
    cell_size: float = Field(1000, gt=50, le=26400, description="Grid cell size when geography='grid'")
    units: Unit = "ft"
    max_distance: float = Field(
        100, gt=0, le=2000, description="Max snap distance (ft) when aggregating to lines"
    )
    min_value: float | None = Field(None, description="Keep areas with value >= min_value")
    top_n: int | None = Field(None, ge=1, le=5000)


class HotspotArgs(BaseModel):
    input: str = Field(description="Point features", json_schema_extra=REF)
    method: Literal["clusters", "gi_star"] = Field(
        "clusters",
        description="clusters = repeated reports at/near the same place (DBSCAN); "
        "gi_star = statistically significant concentrations on a grid",
    )
    radius: float = Field(300, gt=0, le=5280, description="Neighborhood radius for clusters")
    min_points: int = Field(5, ge=2, le=1000, description="Minimum reports per cluster")
    cell_size: float = Field(1000, gt=100, le=10560, description="Grid cell size for gi_star")
    units: Unit = "ft"


class TemporalArgs(BaseModel):
    input: str = Field(description="Records with a date column", json_schema_extra=REF)
    date_column: str | None = Field(None, description="Defaults to the dataset's temporal column")
    freq: Literal["day", "week", "month", "quarter", "year"] = "month"
    group_by: str | None = Field(
        None, description="Optional column to compare, e.g. borough or complaint_type"
    )
    top_groups: int = Field(8, ge=1, le=20)


class ComparePeriodsArgs(BaseModel):
    input: str = Field(description="Point records with dates", json_schema_extra=REF)
    geography: str = Field("neighborhood", description="borough | neighborhood | community_district | zip")
    date_column: str | None = None
    period_a: list[str] | None = Field(
        None, description="[start, end] dates YYYY-MM-DD; default earliest full year"
    )
    period_b: list[str] | None = Field(None, description="[start, end] dates; default latest full year")
    min_baseline: int = Field(5, ge=0)
    top_n: int | None = Field(None, ge=1, le=500)


class GeocodeArgs(BaseModel):
    query: str = Field(min_length=2, max_length=200, description="Place name, street, ZIP code, or address")
    kind: Literal["auto", "borough", "neighborhood", "community_district", "zip", "street", "address"] = (
        "auto"
    )


class MapLayerArgs(BaseModel):
    input: str = Field(description="Spatial result to draw", json_schema_extra=REF)
    title: str | None = None
    style: Literal["auto", "points", "cluster", "heatmap", "choropleth", "lines", "polygons"] = "auto"
    color_by: str | None = None
    tooltip: list[str] | None = None
    context: bool = Field(False, description="Draw as a muted reference layer (e.g. streets, buffers)")


class SummarizeArgs(BaseModel):
    input: str = Field(description="Result to describe", json_schema_extra=REF)
    group_by: str | None = None
    top_n: int = Field(8, ge=1, le=50)


class ImagerySearchArgs(BaseModel):
    query: str | None = Field(None, description="Text or example description of what to look for")
    area: str | None = Field(None, description="Optional area reference", json_schema_extra=REF)
    k: int = Field(10, ge=1, le=100)


# =========================================================================================
# SQL rendering (template queries)
# =========================================================================================
def render_filter_sql(ctx: ToolContext, a: FilterDatasetArgs) -> RenderedSQL:
    ds = ctx.dataset(a.dataset)
    engine = ctx.engine_for(ds)
    dialect = getattr(engine, "dialect", "duckdb")
    cols = ctx.columns_for(ds)
    names = {c.name.lower(): c.name for c in cols}
    sensitive = ds.sensitive_columns
    notes: list[str] = []

    def col(name: str) -> exp.Column:
        key = name.lower()
        if key not in names:
            raise SpatialError(
                f"Column '{name}' does not exist in {ds.id}. Columns: {', '.join(list(names)[:25])}"
            )
        if key in sensitive:
            raise SpatialError(f"Column '{name}' is sensitive and cannot be queried")
        return exp.column(names[key], quoted=True)

    params: dict[str, Any] = {}

    def param(value: Any) -> exp.Placeholder:
        key = f"p{len(params)}"
        params[key] = value
        return exp.Placeholder(this=key)

    must = set()
    sp = ds.spatial
    if sp:
        for gc in (sp.geometry_column, sp.lat_column, sp.lon_column):
            if gc and gc.lower() in names:
                must.add(names[gc.lower()])
    for kc in (ds.id_column, ds.category_column, ds.temporal.column if ds.temporal else None, ds.name_column):
        if kc and kc.lower() in names:
            must.add(names[kc.lower()])
    if a.columns:
        chosen = [col(name).name for name in a.columns]
        selected = list(dict.fromkeys(chosen + sorted(must)))
    else:
        selected = [ci.name for ci in cols if ci.name.lower() not in sensitive]

    projections: list[Any] = []
    for name in selected:
        c: Any = exp.column(name, quoted=True)
        if (
            sp
            and sp.geometry_column
            and name.lower() == sp.geometry_column.lower()
            and dialect == "databricks"
            and sp.format == "native"
        ):
            projections.append(exp.alias_(exp.Anonymous(this="ST_AsWKB", expressions=[c]), name, quoted=True))
        else:
            projections.append(c)

    conds: list[Any] = []
    if a.categories:
        if not ds.category_column:
            notes.append(f"{ds.id} has no category column; 'categories' ignored.")
        else:
            exact, like, cnotes = resolve_categories(ds, a.categories)
            notes += cnotes
            cc = col(ds.category_column)
            ors: list[Any] = []
            cond: Any
            if exact:
                ors.append(exp.In(this=cc, expressions=[param(v) for v in exact]))
            for t in like:
                ors.append(exp.Like(this=exp.Lower(this=cc.copy()), expression=param(f"%{t}%")))
            if ors:
                cond = ors[0]
                for o in ors[1:]:
                    cond = exp.or_(cond, o)
                conds.append(exp.paren(cond))
            if exact:
                notes.append(f"{ds.category_column} in {exact}")
    if a.boroughs:
        boros = [b for b in (normalize_borough(x) for x in a.boroughs) if b]
        bcol = ds.join_keys.get("borough") or ("borough" if "borough" in names else None)
        if not boros:
            notes.append(f"Unrecognized borough names {a.boroughs}; no borough filter applied.")
        elif bcol and bcol.lower() in names:
            conds.append(exp.In(this=exp.Upper(this=col(bcol)), expressions=[param(b) for b in boros]))
        else:
            notes.append("POST_FILTER_BOROUGHS:" + ",".join(boros))
    if a.date_from or a.date_to:
        if not ds.temporal:
            notes.append(f"{ds.id} has no temporal column; date filters ignored.")
        else:
            tc = col(ds.temporal.column)
            if a.date_from:
                conds.append(
                    exp.GTE(
                        this=tc,
                        expression=exp.Cast(this=param(a.date_from), to=exp.DataType.build("TIMESTAMP")),
                    )
                )
            if a.date_to:
                conds.append(
                    exp.LT(
                        this=tc.copy(),
                        expression=exp.Add(
                            this=exp.Cast(this=param(a.date_to), to=exp.DataType.build("TIMESTAMP")),
                            expression=exp.Interval(this=exp.Literal.string("1"), unit=exp.Var(this="DAY")),
                        ),
                    )
                )
    for f in a.filters:
        fc: Any = col(f.column)
        if f.op == "is_null":
            conds.append(exp.Is(this=fc, expression=exp.Null()))
        elif f.op == "not_null":
            conds.append(exp.Not(this=exp.Is(this=fc, expression=exp.Null())))
        elif f.op in ("in", "not_in"):
            vals = f.value if isinstance(f.value, list) else [f.value]
            e = exp.In(this=fc, expressions=[param(v) for v in vals])
            conds.append(exp.Not(this=e) if f.op == "not_in" else e)
        elif f.op == "contains":
            conds.append(
                exp.Like(
                    this=exp.Lower(this=exp.Cast(this=fc, to=exp.DataType.build("VARCHAR"))),
                    expression=param(f"%{str(f.value).lower()}%"),
                )
            )
        elif f.op == "starts_with":
            conds.append(
                exp.Like(
                    this=exp.Lower(this=exp.Cast(this=fc, to=exp.DataType.build("VARCHAR"))),
                    expression=param(f"{str(f.value).lower()}%"),
                )
            )
        elif f.op == "between":
            vals = f.value if isinstance(f.value, list) else []
            if len(vals) != 2:
                raise SpatialError("'between' needs a [low, high] value")
            conds.append(exp.Between(this=fc, low=param(vals[0]), high=param(vals[1])))
        else:
            cls = {"=": exp.EQ, "!=": exp.NEQ, ">": exp.GT, ">=": exp.GTE, "<": exp.LT, "<=": exp.LTE}[f.op]
            conds.append(cls(this=fc, expression=param(f.value)))

    query = exp.select(*projections).from_(exp.to_table(ds.sql_table, dialect=dialect))
    for cnd in conds:
        query = query.where(cnd)
    limit = min(a.limit or ctx.settings.max_result_rows, ctx.settings.max_result_rows, ds.max_rows or 10**9)
    query = query.limit(limit)
    return RenderedSQL(
        sql=query.sql(dialect=dialect),
        params=params,
        backend=ds.backend,
        kind="template",
        datasets=[ds.id],
        notes=notes,
    )


def render_query_local(ctx: ToolContext, a: QuerySQLArgs) -> RenderedSQL:
    return RenderedSQL(sql=a.sql, params={}, backend="local", kind="llm_sql")


def render_query_databricks(ctx: ToolContext, a: QuerySQLArgs) -> RenderedSQL:
    return RenderedSQL(sql=a.sql, params={}, backend="databricks", kind="llm_sql")


# =========================================================================================
# Built-in tools
# =========================================================================================
def build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()

    # ---- discovery ------------------------------------------------------------------------
    @reg.tool(
        "list_datasets",
        "List authorized datasets available in the current mode.",
        ListDatasetsArgs,
        category="discovery",
    )
    def list_datasets(ctx: ToolContext, a: ListDatasetsArgs) -> ToolOutput:
        items = (
            ctx.registry.search(a.query, ctx.mode, limit=50)
            if a.query
            else [(d, 0.0) for d in ctx.registry.available(ctx.mode)]
        )
        rows = []
        for ds, score in items:
            if a.role and ds.role != a.role:
                continue
            rows.append(
                {
                    "id": ds.id,
                    "name": ds.name,
                    "role": ds.role,
                    "backend": ds.backend,
                    "geometry": ds.spatial.geometry_type if ds.spatial else None,
                    "temporal": ds.temporal.column if ds.temporal else None,
                    "synthetic": ds.synthetic,
                    "sensitivity": ds.sensitivity,
                    "score": round(score, 2),
                }
            )
        return ToolOutput(
            summary=f"{len(rows)} datasets available in {ctx.mode} mode.", data=rows, rows=len(rows)
        )

    @reg.tool(
        "describe_dataset",
        "Describe a dataset: purpose, columns, spatial/temporal fields, provenance.",
        DatasetArg,
        category="discovery",
    )
    def describe_dataset(ctx: ToolContext, a: DatasetArg) -> ToolOutput:
        ds = ctx.dataset(a.dataset)
        info = {
            "id": ds.id,
            "name": ds.name,
            "description": ds.description,
            "role": ds.role,
            "backend": ds.backend,
            "table": ds.sql_table,
            "synthetic": ds.synthetic,
            "sensitivity": ds.sensitivity,
            "spatial": ds.spatial.model_dump() if ds.spatial else None,
            "temporal": ds.temporal.model_dump() if ds.temporal else None,
            "category_column": ds.category_column,
            "categories": ds.known_categories(),
            "columns": [
                {"name": c.name, "type": c.type, "description": c.description}
                for c in ds.columns
                if not c.sensitive
            ],
            "provenance": ds.provenance.model_dump(),
        }
        return ToolOutput(summary=f"{ds.name}: {ds.description[:200]}", data=info, datasets=[ds.id])

    @reg.tool(
        "get_schema",
        "Inspect the live column names and types of a dataset table.",
        DatasetArg,
        category="discovery",
    )
    def get_schema(ctx: ToolContext, a: DatasetArg) -> ToolOutput:
        ds = ctx.dataset(a.dataset)
        engine = ctx.engine_for(ds)
        live = engine.describe(ds.sql_table)
        reg_cols = {c.name.lower(): c for c in ds.columns}
        rows = []
        for c in live:
            if c.name.lower() in ds.sensitive_columns:
                continue
            desc = (
                reg_cols[c.name.lower()].description
                if c.name.lower() in reg_cols
                else clean_text(c.comment)[0]
            )
            rows.append({"name": c.name, "type": c.type, "description": desc})
        ctx.schema_cache[ds.id] = [ColumnInfo(r["name"], r["type"], r["description"]) for r in rows]
        return ToolOutput(summary=f"{ds.sql_table}: {len(rows)} columns", data=rows, datasets=[ds.id])

    # ---- data access ------------------------------------------------------------------------
    @reg.tool(
        "filter_dataset",
        "Read records from a registered dataset with structured filters (category, borough, dates, columns). "
        "Preferred over raw SQL.",
        FilterDatasetArgs,
        category="query",
        sql_renderer=render_filter_sql,
    )
    def filter_dataset(ctx: ToolContext, a: FilterDatasetArgs) -> ToolOutput:
        ds = ctx.dataset(a.dataset)
        rendered = render_filter_sql(ctx, a)
        df, review, result = ctx.run_sql(rendered)
        gdf = to_gdf(df, ds)
        data: pd.DataFrame = gdf if gdf is not None else df
        notes = [n for n in rendered.notes if not n.startswith("POST_FILTER_BOROUGHS:")]
        post = [n for n in rendered.notes if n.startswith("POST_FILTER_BOROUGHS:")]
        if post and gdf is not None:
            boros = post[0].split(":", 1)[1].split(",")
            bds, bgdf = ctx.geography("borough")
            sel = bgdf[bgdf["borough"].isin(boros)] if "borough" in bgdf else bgdf
            data = O.filter_by_area(gdf, sel)
            notes.append(f"Filtered to {', '.join(boros)} by point-in-polygon ({bds.id}).")
        warnings = list(review.warnings)
        if result.truncated:
            warnings.append(
                f"Result reached the {review.limit:,}-row limit; narrow filters for complete results."
            )
        cats = [n.split(" in ", 1)[1] for n in rendered.notes if n.startswith(f"{ds.category_column} in ")]
        cat_label = (
            ", ".join(eval_list(cats[0])) if cats else (", ".join(a.categories) if a.categories else "")
        )
        title = f"{cat_label} {ds.label}".strip() if cat_label else ds.label.capitalize()
        name_vals = [
            v
            for f in a.filters
            if ds.name_column and f.column.lower() == ds.name_column.lower() and f.op in ("=", "in")
            for v in (f.value if isinstance(f.value, list) else [f.value])
        ]
        if name_vals:
            title = f"{', '.join(str(v).title() for v in name_vals[:4])} {ds.label}"
        if a.boroughs:
            title += " in " + ", ".join(b.title() for b in a.boroughs)
        art = ctx.save(title, data, "filter_dataset", datasets=[ds.id], sql=review.sql, notes=notes)
        status = "ok" if len(data) else "empty"
        return ToolOutput(
            status=status,
            summary=f"Read {len(data):,} rows from {ds.id}.",
            headline=f"{len(data):,} {title}",
            artifact_id=art.id,
            rows=len(data),
            sql=review.sql,
            warnings=warnings,
            notes=notes,
            datasets=[ds.id],
        )

    def _query(ctx: ToolContext, a: QuerySQLArgs, backend: str) -> ToolOutput:
        rendered = RenderedSQL(sql=a.sql, params={}, backend=backend, kind="llm_sql")
        df, review, result = ctx.run_sql(rendered)
        spec = None
        if len(review.datasets) == 1:
            spec = ctx.registry.get_any(review.datasets[0])
        gdf = to_gdf(df, spec)
        data: pd.DataFrame = gdf if gdf is not None else df
        warnings = list(review.warnings)
        if result.truncated:
            warnings.append(f"Result reached the {review.limit:,}-row limit.")
        art = ctx.save(
            a.purpose or "SQL query result",
            data,
            f"query_{backend}",
            datasets=review.datasets,
            sql=review.sql,
        )
        return ToolOutput(
            status="ok" if len(data) else "empty",
            summary=f"Query returned {len(data):,} rows.",
            artifact_id=art.id,
            rows=len(data),
            sql=review.sql,
            warnings=warnings,
            datasets=review.datasets,
        )

    @reg.tool(
        "query_local_data",
        "Run one read-only SELECT on local DuckDB tables (DuckDB SQL, Spatial functions "
        "ST_* when available). Requires user approval.",
        QuerySQLArgs,
        category="query",
        sql_renderer=render_query_local,
        available=lambda c: "local" in c.engines,
    )
    def query_local_data(ctx: ToolContext, a: QuerySQLArgs) -> ToolOutput:
        return _query(ctx, a, "local")

    @reg.tool(
        "query_databricks",
        "Run one read-only SELECT on authorized Unity Catalog tables (Databricks SQL). "
        "Requires user approval.",
        QuerySQLArgs,
        category="query",
        sql_renderer=render_query_databricks,
        available=lambda c: "databricks" in c.engines,
    )
    def query_databricks(ctx: ToolContext, a: QuerySQLArgs) -> ToolOutput:
        return _query(ctx, a, "databricks")

    # ---- spatial operations ----------------------------------------------------------------
    @reg.tool(
        "spatial_join",
        "Relate two feature sets (intersects/within/contains, or dwithin a distance). "
        "With summarize=true returns left features with match_count.",
        SpatialJoinArgs,
        category="spatial",
    )
    def spatial_join(ctx: ToolContext, a: SpatialJoinArgs) -> ToolOutput:
        la, lg = ctx.resolve_spatial(a.left)
        ra, rg = ctx.resolve_spatial(a.right)
        empty = _empty_input(la, "spatial join") or _empty_input(ra, "spatial join")
        if empty:
            return empty
        out = O.spatial_join(
            lg, rg, a.predicate, ctx.settings.projected_crs, a.distance, a.units, a.how, a.summarize
        )
        dist = f" within {a.distance:g} {a.units}" if a.predicate == "dwithin" else f" ({a.predicate})"
        if a.predicate == "dwithin":
            rel = f"within {a.distance:g} {a.units} of"
        else:
            rel = {"within": "within", "intersects": "intersecting", "contains": "containing"}[a.predicate]
        title = f"{la.title} {rel} {ra.title}"
        art = ctx.save(
            title,
            out,
            "spatial_join",
            [la, ra],
            notes=[f"Spatial join{dist} computed in {ctx.settings.projected_crs}."],
        )
        hl = []
        if a.summarize and len(out) and "match_count" in out:
            hl.append(f"{len(out):,} of {len(lg):,} '{la.title}' features have at least one match{dist}.")
        return ToolOutput(
            status="ok" if len(out) else "empty",
            summary=f"{len(out):,} {la.title} features matched{dist} of {ra.title}.",
            headline=f"{len(out):,} of {len(lg):,} {la.title} are {rel} {ra.title}",
            highlights=hl,
            artifact_id=art.id,
            rows=len(out),
            notes=art.notes,
            datasets=art.datasets,
        )

    @reg.tool(
        "buffer_features",
        "Create buffer zones around features at a distance.",
        BufferArgs,
        category="spatial",
    )
    def buffer_features(ctx: ToolContext, a: BufferArgs) -> ToolOutput:
        art_in, g = ctx.resolve_spatial(a.input)
        empty = _empty_input(art_in, "buffer")
        if empty:
            return empty
        out = O.buffer_features(g, a.distance, a.units, ctx.settings.projected_crs, a.dissolve)
        art = ctx.save(
            f"{a.distance:g} {a.units} buffer around {art_in.title}",
            out,
            "buffer_features",
            [art_in],
            notes=[f"Buffer of {a.distance:g} {a.units} computed in {ctx.settings.projected_crs}."],
        )
        return ToolOutput(
            summary=f"Buffered {len(g):,} features by {a.distance:g} {a.units}.",
            artifact_id=art.id,
            rows=len(out),
            notes=art.notes,
            datasets=art.datasets,
        )

    @reg.tool(
        "nearest_features",
        "Attach the nearest target feature (and distance in feet) to each input feature.",
        NearestArgs,
        category="spatial",
    )
    def nearest_features(ctx: ToolContext, a: NearestArgs) -> ToolOutput:
        ia, ig = ctx.resolve_spatial(a.input)
        ta, tg = ctx.resolve_spatial(a.target)
        empty = _empty_input(ia, "nearest-feature search") or _empty_input(ta, "nearest-feature search")
        if empty:
            return empty
        out = O.nearest_features(ig, tg, ctx.settings.projected_crs, a.max_distance, a.units)
        lim = f" within {a.max_distance:g} {a.units}" if a.max_distance else ""
        art = ctx.save(
            f"{ia.title} with nearest {ta.title}",
            out,
            "nearest_features",
            [ia, ta],
            notes=[f"Nearest-neighbor distances computed in {ctx.settings.projected_crs}{lim}."],
        )
        hl = []
        if len(out):
            hl.append(f"Median distance to nearest {ta.title}: {out['distance_ft'].median():,.0f} ft.")
        return ToolOutput(
            status="ok" if len(out) else "empty",
            summary=f"{len(out):,} of {len(ig):,} features have a nearest {ta.title}{lim}.",
            highlights=hl,
            artifact_id=art.id,
            rows=len(out),
            notes=art.notes,
            datasets=art.datasets,
        )

    @reg.tool(
        "filter_by_area",
        "Keep features inside named areas (boroughs, neighborhoods, ZIP codes).",
        FilterByAreaArgs,
        category="spatial",
    )
    def filter_by_area(ctx: ToolContext, a: FilterByAreaArgs) -> ToolOutput:
        ia, ig = ctx.resolve_spatial(a.input)
        empty = _empty_input(ia, "area filter")
        if empty:
            return empty
        gds, gg = ctx.geography(a.area_type)
        label_cols = [
            c
            for c in (
                gds.name_column,
                "neighborhood",
                "label",
                "boro_name",
                "borough",
                "cd_code",
                "modzcta",
                "zip_codes",
                "ntaname",
            )
            if c and c in gg.columns
        ]
        matched_idx: set[int] = set()
        unmatched = []
        for name in a.names:
            hits: set[int] = set()
            for c in label_cols:
                m = O.match_names([name], gg[c].astype(str).tolist())[name]
                hits |= set(gg.index[gg[c].astype(str).isin(m)])
            if hits:
                matched_idx |= hits
            else:
                unmatched.append(name)
        if not matched_idx:
            raise SpatialError(f"No {a.area_type} matched {a.names}")
        areas = gg.loc[sorted(matched_idx)]
        out = O.filter_by_area(ig, areas)
        label = gds.name_column or label_cols[0]
        names = ", ".join(areas[label].astype(str).head(6))
        art = ctx.save(
            f"{ia.title} in {names}",
            out,
            "filter_by_area",
            [ia],
            datasets=[gds.id],
            notes=[f"Filtered to {len(areas)} {a.area_type} area(s): {names}."],
        )
        warnings = [f"No match for: {', '.join(unmatched)}"] if unmatched else []
        return ToolOutput(
            status="ok" if len(out) else "empty",
            summary=f"{len(out):,} of {len(ig):,} features fall in {names}.",
            artifact_id=art.id,
            rows=len(out),
            warnings=warnings,
            notes=art.notes,
            datasets=art.datasets,
        )

    # ---- analysis --------------------------------------------------------------------------
    @reg.tool(
        "aggregate_by_geography",
        "Count or summarize features by borough, neighborhood, ZIP, a square grid, "
        "or by nearest line (e.g. street segments).",
        AggregateArgs,
        category="analysis",
    )
    def aggregate_by_geography(ctx: ToolContext, a: AggregateArgs) -> ToolOutput:
        ia, ig = ctx.resolve_spatial(a.input)
        empty = _empty_input(ia, "aggregation")
        if empty:
            return empty
        spec = ctx.primary_spec(ia)
        cat = a.category_column or (spec.category_column if spec else None)
        cat = cat if cat in ig.columns else None
        inputs = [ia]
        roles = set(ctx.registry.geography_roles(ctx.mode))
        label = None
        if a.geography == "grid":
            res = A.grid_aggregate(
                ig,
                ctx.settings.projected_crs,
                a.cell_size,
                a.units,
                a.measure,
                a.value_column,
                cat,
                a.min_value,
                a.top_n,
            )
            what = f"{a.cell_size:g} {a.units} grid cells"
            datasets: list[str] = []
        elif a.geography in roles:
            gds, gg = ctx.geography(a.geography)
            gg = restrict_areas_to_input(gg, ig)
            label = gds.name_column
            res = A.aggregate_to_polygons(ig, gg, label, a.measure, a.value_column, cat, a.min_value, a.top_n)
            what = a.geography.replace("_", " ") + " areas"
            datasets = [gds.id]
        else:
            ga, gg = ctx.resolve_spatial(a.geography)
            inputs.append(ga)
            datasets = []
            if ga.geometry_kind == "line":
                res = A.aggregate_to_lines(
                    ig,
                    gg,
                    ctx.settings.projected_crs,
                    a.max_distance,
                    a.measure,
                    a.value_column,
                    cat,
                    a.min_value,
                    a.top_n,
                )
                what = (
                    ga.title.lower()
                    if ga.title.lower().endswith("segments")
                    else f"{ga.title.lower()} segments"
                )
                label = "street_name" if "street_name" in gg else None
            else:
                gspec = ctx.primary_spec(ga)
                label = gspec.name_column if gspec else None
                res = A.aggregate_to_polygons(
                    ig, gg, label, a.measure, a.value_column, cat, a.min_value, a.top_n
                )
                what = f"{ga.title} areas"
        out = res.data
        measure = _measure_label(a.measure, a.value_column)
        art = ctx.save(
            f"{ia.title}: {measure} by {what}",
            out,
            "aggregate_by_geography",
            inputs,
            datasets=datasets,
            notes=res.notes,
        )
        hl = []
        headline = None
        if len(out):
            top = out.iloc[0]
            if label and label in out:
                name = str(top.get(label))
            else:
                c = top.geometry.centroid
                name = f"the cell at {c.y:.4f}, {c.x:.4f}"
            if a.min_value is not None:
                headline = f"{len(out):,} {what} have {measure} ≥ {a.min_value:g}; highest: {name} ({top['value']:,.0f})"
            else:
                headline = f"Highest {measure}: {name} ({top['value']:,.0f})"
            hl.append(
                f"Highest: {name} with {top['value']:,.0f}"
                + (f" ({top['share_pct']}% of total)" if "share_pct" in out else "")
                + "."
            )
            if len(out) > 1 and label and label in out:
                hl.append(
                    "Next: "
                    + ", ".join(f"{r[label]} ({r['value']:,.0f})" for _, r in out.iloc[1:4].iterrows())
                    + "."
                )
        return ToolOutput(
            status="ok" if len(out) else "empty",
            summary=f"Aggregated {len(ig):,} features into {len(out):,} {what} ({measure}).",
            headline=headline,
            highlights=hl,
            artifact_id=art.id,
            rows=len(out),
            notes=res.notes,
            datasets=art.datasets,
        )

    @reg.tool(
        "find_hotspots",
        "Find concentrations: 'clusters' for repeated reports at/near the same place, "
        "'gi_star' for statistically significant hot spots on a grid.",
        HotspotArgs,
        category="analysis",
    )
    def find_hotspots(ctx: ToolContext, a: HotspotArgs) -> ToolOutput:
        ia, ig = ctx.resolve_spatial(a.input)
        empty = _empty_input(ia, "hot spot analysis")
        if empty:
            return empty
        spec = ctx.primary_spec(ia)
        if a.method == "gi_star":
            res = A.gi_star_hotspots(ig, ctx.settings.projected_crs, a.cell_size, a.units)
            out = res.data
            hot = out[out["hotspot_class"].str.startswith("Hot")]
            art = ctx.save(f"Hot spot analysis of {ia.title}", out, "find_hotspots", [ia], notes=res.notes)
            hl = [
                f"{len(hot):,} grid cells are hot spots at ≥90% confidence "
                f"({res.stats['hot_cells_95']:,} at ≥95%)."
            ]
            if len(hot):
                hl.append(
                    f"Strongest cell has {int(hot.iloc[0]['count'])} reports (Gi* z = {hot.iloc[0]['gi_z']})."
                )
            return ToolOutput(
                summary=f"Gi* hot spot analysis on {a.cell_size:g} {a.units} cells.",
                highlights=hl,
                headline=f"{res.stats['hot_cells_95']:,} grid cells ({a.cell_size:g} {a.units}) are "
                f"statistically significant hot spots of {ia.title} (95%+)",
                artifact_id=art.id,
                rows=len(out),
                notes=res.notes,
                datasets=art.datasets,
            )
        cat = spec.category_column if spec and spec.category_column in ig.columns else None
        dcol = spec.temporal.column if spec and spec.temporal and spec.temporal.column in ig.columns else None
        labels = [
            c
            for c in ("incident_address", "street_name", "borough", "cd_code", "community_board")
            if c in ig.columns
        ]
        res = A.dbscan_clusters(
            ig, ctx.settings.projected_crs, a.radius, a.units, a.min_points, cat, dcol, labels
        )
        out = res.data
        art = ctx.save(
            f"Clusters of {ia.title} (≥{a.min_points} within {a.radius:g} {a.units})",
            out,
            "find_hotspots",
            [ia],
            notes=res.notes,
        )
        hl = [
            f"{res.stats['clusters']} clusters contain {res.stats['clustered_points']:,} of {len(ig):,} reports."
        ]
        if len(out):
            top = out.iloc[0]
            where = top.get("incident_address") or top.get("street_name") or top.get("cd_code") or ""
            boro = top.get("borough") or ""
            span = (
                f" over {top['span_days']} days"
                if "span_days" in out and pd.notna(top.get("span_days"))
                else ""
            )
            hl.append(f"Largest cluster: {int(top['count'])} reports{span} near {where} {boro}".strip() + ".")
        return ToolOutput(
            status="ok" if len(out) else "empty",
            summary=f"Found {len(out)} clusters of ≥{a.min_points} reports within {a.radius:g} {a.units}.",
            highlights=hl,
            headline=f"{len(out)} locations with {a.min_points}+ {ia.title} within "
            f"{a.radius:g} {a.units} of each other",
            artifact_id=art.id,
            rows=len(out),
            notes=res.notes,
            datasets=art.datasets,
        )

    @reg.tool(
        "analyze_temporal_patterns",
        "Count records over time (day/week/month/quarter/year), optionally "
        "grouped (e.g. by borough), with trend statistics.",
        TemporalArgs,
        category="analysis",
    )
    def analyze_temporal_patterns(ctx: ToolContext, a: TemporalArgs) -> ToolOutput:
        ia = ctx.resolve(a.input)
        empty = _empty_input(ia, "temporal analysis")
        if empty:
            return empty
        spec = ctx.primary_spec(ia)
        dcol = a.date_column or (spec.temporal.column if spec and spec.temporal else None)
        if not dcol:
            raise SpatialError("No date column; specify date_column")
        res = A.temporal_patterns(ia.data, dcol, a.freq, a.group_by, a.top_groups)
        summary_df: pd.DataFrame = res.stats["summary"]
        art = ctx.save(
            f"{ia.title} by {a.freq}" + (f" and {a.group_by}" if a.group_by else ""),
            res.data,
            "analyze_temporal_patterns",
            [ia],
            notes=res.notes,
            extras={
                "chart": {"type": "line", "x": "period", "y": "count", "color": "group"},
                "summary": summary_df.to_dict("records"),
            },
        )
        hl = []
        pct_col = "pct_change_full_years"
        for _, r in summary_df.head(6).iterrows():
            s = f"{r['group']}: {int(r['total']):,} total"
            if pct_col in summary_df and pd.notna(r.get(pct_col)):
                fy = res.stats["full_years"]
                s += f", {r[pct_col]:+.1f}% from {fy[0]} to {fy[-1]}"
            hl.append(s)
        headline = None
        if len(summary_df) and pct_col in summary_df and summary_df[pct_col].notna().any():
            mv = summary_df.loc[summary_df[pct_col].abs().idxmax()]
            fy = res.stats["full_years"]
            headline = (
                f"{ia.title}: largest change in {mv['group']} ({mv[pct_col]:+.1f}% from {fy[0]} to {fy[-1]})"
            )
        return ToolOutput(
            headline=headline,
            summary=f"Temporal pattern of {ia.title} ({res.stats['date_min'][:10]} to "
            f"{res.stats['date_max'][:10]}).",
            highlights=hl,
            artifact_id=art.id,
            rows=len(res.data),
            data=summary_df.to_dict("records"),
            notes=res.notes,
            datasets=art.datasets,
        )

    @reg.tool(
        "compare_periods",
        "Compare counts per area between two periods (default: earliest vs latest full "
        "year) to find the largest increases/decreases.",
        ComparePeriodsArgs,
        category="analysis",
    )
    def compare_periods(ctx: ToolContext, a: ComparePeriodsArgs) -> ToolOutput:
        ia, ig = ctx.resolve_spatial(a.input)
        empty = _empty_input(ia, "period comparison")
        if empty:
            return empty
        spec = ctx.primary_spec(ia)
        dcol = a.date_column or (spec.temporal.column if spec and spec.temporal else None)
        if not dcol:
            raise SpatialError("No date column; specify date_column")
        gds, gg = ctx.geography(a.geography)
        if "is_joint_interest_area" in gg:
            gg = gg[~gg["is_joint_interest_area"].astype(bool)]
        gg = restrict_areas_to_input(gg, ig)
        pa = (a.period_a[0], a.period_a[1]) if a.period_a and len(a.period_a) == 2 else None
        pb = (a.period_b[0], a.period_b[1]) if a.period_b and len(a.period_b) == 2 else None
        res = A.compare_periods(ig, gg, dcol, gds.name_column, pa, pb, a.min_baseline, a.top_n)
        out = res.data
        label = gds.name_column or "rank"
        art = ctx.save(
            f"Change in {ia.title} by {a.geography}",
            out,
            "compare_periods",
            [ia],
            datasets=[gds.id],
            notes=res.notes,
        )
        hl = [
            f"Totals: {res.stats['total_a']:,} ({res.stats['period_a']}) → {res.stats['total_b']:,} "
            f"({res.stats['period_b']})."
        ]
        inc = out[out["change"] > 0].head(5)
        headline = None
        if len(inc):
            r0 = inc.iloc[0]
            headline = (
                f"Largest increase: {r0[label]} ({int(r0['count_a'])} → {int(r0['count_b'])} reports, "
                f"{res.stats['period_a'][:4]} vs {res.stats['period_b'][:4]})"
            )
        if len(inc):
            hl.append(
                "Largest increases: "
                + "; ".join(
                    f"{r[label]} {int(r['count_a'])}→{int(r['count_b'])}"
                    + (f" ({r['pct_change']:+.0f}%)" if pd.notna(r["pct_change"]) else "")
                    for _, r in inc.iterrows()
                )
                + "."
            )
        dec = out[out["change"] < 0].sort_values("change").head(3)
        if len(dec):
            hl.append(
                "Largest decreases: "
                + "; ".join(f"{r[label]} {int(r['count_a'])}→{int(r['count_b'])}" for _, r in dec.iterrows())
                + "."
            )
        return ToolOutput(
            summary=f"Compared {res.stats['period_a']} with {res.stats['period_b']} across "
            f"{len(out)} {a.geography} areas.",
            highlights=hl,
            artifact_id=art.id,
            headline=headline,
            rows=len(out),
            notes=res.notes,
            datasets=art.datasets,
        )

    @reg.tool(
        "geocode_place",
        "Find a place (borough, neighborhood, ZIP, street) in reference datasets.",
        GeocodeArgs,
        category="spatial",
    )
    def geocode_place(ctx: ToolContext, a: GeocodeArgs) -> ToolOutput:
        frames = []
        kinds = (
            ["borough", "neighborhood", "community_district", "zip"]
            if a.kind in ("auto", "address")
            else [a.kind]
        )
        for role in kinds:
            if role == "street" or role not in ctx.registry.geography_roles(ctx.mode):
                continue
            gds, gg = ctx.geography(role)
            for c in [
                x
                for x in (gds.name_column, "neighborhood", "boro_name", "cd_code", "modzcta", "zip_codes")
                if x and x in gg.columns
            ]:
                m = O.match_names([a.query], gg[c].astype(str).tolist())[a.query]
                if m:
                    hit = gg[gg[c].astype(str).isin(m)].copy()
                    hit["matched_kind"] = role
                    hit["matched_name"] = hit[c].astype(str)
                    frames.append(hit[["matched_kind", "matched_name", "geometry"]])
                    break
        if a.kind in ("auto", "street"):
            for ds in ctx.registry.available(ctx.mode):
                if ds.role == "streets" and ds.backend == "local":
                    sg = ctx.load_reference(ds)
                    m = O.match_names([a.query], sg["street_name"].astype(str).unique().tolist())[a.query]
                    if m:
                        hit = sg[sg["street_name"].isin(m)].dissolve(by="street_name").reset_index()
                        hit["matched_kind"] = "street"
                        hit["matched_name"] = hit["street_name"]
                        frames.append(hit[["matched_kind", "matched_name", "geometry"]])
        notes = []
        if not frames and ctx.settings.geocoder == "nyc_geosearch" and a.kind in ("auto", "address"):
            frames.append(_geosearch(a.query))
            notes.append("Geocoded with NYC Planning Labs GeoSearch (only the query text was sent).")
        if not frames or all(len(f) == 0 for f in frames):
            return ToolOutput(status="empty", summary=f"No reference feature matched '{a.query}'.")
        out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs="EPSG:4326")
        art = ctx.save(f"Place: {a.query}", out, "geocode_place", notes=notes)
        return ToolOutput(
            summary=f"Matched '{a.query}' to {len(out)} feature(s): "
            f"{', '.join(out['matched_name'].astype(str).head(5))}.",
            artifact_id=art.id,
            rows=len(out),
            notes=notes,
        )

    # ---- output ------------------------------------------------------------------------------
    @reg.tool(
        "generate_map_layer",
        "Add a spatial result to the map (auto-styled; choropleth for area values).",
        MapLayerArgs,
        category="output",
    )
    def generate_map_layer(ctx: ToolContext, a: MapLayerArgs) -> ToolOutput:
        art = ctx.resolve(a.input)
        if art.rows == 0 or not art.is_spatial:
            return ToolOutput(
                status="empty", summary=f"Nothing to map: '{art.title}' has no spatial features."
            )
        spec = build_layer_spec(
            art,
            layer_id=f"layer-{art.id}-{len(ctx.layers) + 1}",
            title=a.title,
            style=a.style,
            color_by=a.color_by,
            tooltip=a.tooltip,
            context=a.context,
            z_index=len(ctx.layers),
            max_render_features=ctx.settings.max_render_features,
        )
        layer = spec.model_dump()
        ctx.layers.append(layer)
        return ToolOutput(
            summary=f"Map layer '{spec.title}' ({spec.render}, {spec.feature_count:,} features).",
            layer=layer,
            artifact_id=art.id,
            rows=art.rows,
            datasets=art.datasets,
        )

    @reg.tool(
        "summarize_results",
        "Describe a result: row count, top categories, date range, numeric ranges.",
        SummarizeArgs,
        category="output",
    )
    def summarize_results(ctx: ToolContext, a: SummarizeArgs) -> ToolOutput:
        art = ctx.resolve(a.input)
        s = A.summarize_frame(art.data, a.group_by, a.top_n)
        hl = []
        for c, vc in list(s["categorical"].items())[:3]:
            if c in ("geometry", "is_synthetic") or len(vc) < 1:
                continue
            hl.append(f"Top {c}: " + ", ".join(f"{k} ({v:,})" for k, v in list(vc.items())[:4]))
        for c, rng in list(s["temporal"].items())[:1]:
            hl.append(f"{c} ranges {rng['min'][:10]} to {rng['max'][:10]}")
        return ToolOutput(
            summary=f"{art.title}: {s['rows']:,} rows.",
            highlights=hl,
            data=s,
            rows=s["rows"],
            artifact_id=art.id,
            datasets=art.datasets,
        )

    @reg.tool(
        "search_imagery",
        "Search imagery embeddings for visually similar tiles (Phase 2; returns "
        "'unavailable' unless an imagery index is configured).",
        ImagerySearchArgs,
        category="imagery",
    )
    def search_imagery(ctx: ToolContext, a: ImagerySearchArgs) -> ToolOutput:
        from agentic_gis.imagery.search import get_default_search_service

        svc = get_default_search_service(ctx.settings)
        status = svc.status()
        if not status.available:
            return ToolOutput(
                status="unavailable",
                summary=status.message,
                notes=["No imagery results were generated; nothing was fabricated."],
            )
        return ToolOutput(
            status="unavailable", summary="Imagery search requires a query tile; not yet supported from chat."
        )

    return reg


def _geosearch(query: str) -> gpd.GeoDataFrame:
    r = httpx.get(
        "https://geosearch.planninglabs.nyc/v2/search", params={"text": query, "size": 3}, timeout=10
    )
    r.raise_for_status()
    feats = r.json().get("features", [])
    if not feats:
        return gpd.GeoDataFrame({"matched_kind": [], "matched_name": []}, geometry=[], crs="EPSG:4326")
    g = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
    g["matched_kind"] = "address"
    g["matched_name"] = g.get("label", g.get("name", query))
    return to_wgs84(g[["matched_kind", "matched_name", "geometry"]])
