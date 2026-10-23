"""Local analytical engine: in-memory DuckDB (+ Spatial extension when available).

Registered local datasets (GeoParquet/Parquet/GeoJSON/CSV) are loaded into in-memory
tables at startup. Afterwards external access is disabled and the configuration locked, so
even a query that slipped past the SQL guard could not read files, install extensions, or
re-enable access. Without the Spatial extension, geometry is stored as WKB blobs and
spatial work happens in GeoPandas.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import geopandas as gpd
import pandas as pd
import shapely

from agentic_gis.config.registry import DatasetRegistry, DatasetSpec
from agentic_gis.connectors.base import (
    ColumnInfo,
    ConnectionStatus,
    ConnectorError,
    QueryResult,
    QueryTimeoutError,
)

log = logging.getLogger(__name__)


@dataclass
class LoadedTable:
    dataset_id: str
    table: str
    rows: int
    geometry_column: str | None


def read_dataset_file(ds: DatasetSpec) -> pd.DataFrame | gpd.GeoDataFrame:
    """Read a local dataset file into (Geo)DataFrame in EPSG:4326 where spatial."""
    path = ds.source.resolved_path()
    if path is None or not path.exists():
        raise FileNotFoundError(f"Data file for {ds.id} not found: {path}")
    fmt = ds.source.format or path.suffix.lstrip(".").lower()
    if fmt in ("geoparquet", "parquet"):
        try:
            df: pd.DataFrame = gpd.read_parquet(path)
        except (ValueError, KeyError):
            df = pd.read_parquet(path)
    elif fmt in ("geojson", "json", "gpkg", "shp"):
        df = gpd.read_file(path)
    elif fmt == "csv":
        df = pd.read_csv(path)
    else:
        raise ConnectorError(f"Unsupported file format '{fmt}' for {ds.id}")
    if not isinstance(df, gpd.GeoDataFrame) and ds.spatial is not None:
        sp = ds.spatial
        if sp.format == "latlon" and sp.lat_column in df and sp.lon_column in df:
            df = gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df[sp.lon_column], df[sp.lat_column]), crs=sp.crs
            )
        elif sp.geometry_column and sp.geometry_column in df:
            col = df[sp.geometry_column]
            geoms = shapely.from_wkt(col) if sp.format == "wkt" else shapely.from_wkb(col)
            df = gpd.GeoDataFrame(df.drop(columns=[sp.geometry_column]), geometry=geoms, crs=sp.crs)
    if isinstance(df, gpd.GeoDataFrame):
        if df.crs is None:
            df = df.set_crs(ds.spatial.crs if ds.spatial else "EPSG:4326")
        if df.crs.to_epsg() != 4326:
            df = df.to_crs("EPSG:4326")
        if df.geometry.name != "geometry":
            df = df.rename_geometry("geometry")
    return df


class LocalDuckDBEngine:
    backend = "local"
    dialect = "duckdb"

    def __init__(
        self,
        registry: DatasetRegistry,
        mode: str,
        install_extensions: bool = True,
        threads: int | None = None,
    ):
        self.registry = registry
        self.mode = mode
        self.con = duckdb.connect(database=":memory:")
        if threads:
            self.con.execute(f"SET threads = {int(threads)}")
        self.spatial_available, self.spatial_message = self._load_spatial(install_extensions)
        self.tables: dict[str, LoadedTable] = {}
        self.load_errors: dict[str, str] = {}
        for ds in registry.available(mode, backend="local"):
            try:
                self._load_dataset(ds)
            except Exception as exc:  # keep going; surface the error in the UI
                log.warning("Could not load dataset %s: %s", ds.id, exc)
                self.load_errors[ds.id] = str(exc)
        self._lock_down()

    # ---- setup ---------------------------------------------------------------------------
    def _load_spatial(self, install_extensions: bool) -> tuple[bool, str]:
        try:
            self.con.execute("LOAD spatial")
            return True, "DuckDB Spatial loaded"
        except duckdb.Error:
            pass
        try:  # optional offline wheel (extra: duckdb-spatial-wheel)
            import duckdb_extension_spatial as pkg  # type: ignore[import-not-found]

            files = sorted(Path(pkg.__path__[0]).rglob("spatial.duckdb_extension"))
            if files:
                path = str(files[-1]).replace("'", "''")
                self.con.execute(f"INSTALL '{path}'")
                self.con.execute("LOAD spatial")
                return True, "DuckDB Spatial loaded from the duckdb-extension-spatial wheel"
        except ImportError:
            pass
        except duckdb.Error as exc:
            log.info("Offline spatial wheel present but could not be loaded: %s", exc)
        if install_extensions:
            try:
                self.con.execute("INSTALL spatial")
                self.con.execute("LOAD spatial")
                return True, "DuckDB Spatial installed from the official extension repository"
            except duckdb.Error as exc:
                return (
                    False,
                    f"DuckDB Spatial unavailable ({str(exc).splitlines()[0][:160]}); using GeoPandas",
                )
        return False, "DuckDB Spatial not installed; spatial work uses GeoPandas"

    def _load_dataset(self, ds: DatasetSpec) -> None:
        df = read_dataset_file(ds)
        table = ds.sql_table
        geom_col = None
        if isinstance(df, gpd.GeoDataFrame):
            geom_col = "geometry"
            plain = pd.DataFrame(df.drop(columns="geometry"))
            plain["geometry"] = shapely.to_wkb(df.geometry.values, hex=False)
        else:
            plain = pd.DataFrame(df)
        tmp = f"_tmp_{table}"
        self.con.register(tmp, plain)
        try:
            if geom_col and self.spatial_available:
                self.con.execute(
                    f'CREATE TABLE "{table}" AS SELECT * REPLACE (ST_GeomFromWKB(geometry) AS geometry) FROM "{tmp}"'
                )
            else:
                self.con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM "{tmp}"')
        finally:
            self.con.unregister(tmp)
        self.tables[ds.id] = LoadedTable(ds.id, table, len(plain), geom_col)

    def _lock_down(self) -> None:
        self.con.execute("SET enable_external_access = false")
        self.con.execute("SET lock_configuration = true")

    # ---- queries -------------------------------------------------------------------------
    def describe(self, table: str) -> list[ColumnInfo]:
        cur = self.con.cursor()
        rows = cur.execute(f'DESCRIBE "{table}"').fetchall()
        return [ColumnInfo(name=r[0], type=str(r[1])) for r in rows]

    def execute(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        max_rows: int | None = None,
    ) -> QueryResult:
        """Execute already-validated SQL. Geometry columns are returned as shapely objects."""
        cur = self.con.cursor()
        timer: threading.Timer | None = None
        timed_out = threading.Event()
        if timeout_s:

            def _cancel() -> None:
                timed_out.set()
                cur.interrupt()

            timer = threading.Timer(timeout_s, _cancel)
            timer.daemon = True
            timer.start()
        t0 = time.perf_counter()
        try:
            cur.execute(f"SELECT * FROM ({sql}) AS _q LIMIT 0", params or {})
            geom_cols = [d[0] for d in cur.description or [] if str(d[1]).upper().startswith("GEOMETRY")]
            blob_cols = [d[0] for d in cur.description or [] if str(d[1]).upper() == "BLOB"]
            wrapped = sql
            if geom_cols:
                repl = ", ".join(f'ST_AsWKB("{c}") AS "{c}"' for c in geom_cols)
                wrapped = f"SELECT * REPLACE ({repl}) FROM ({sql}) AS _q"
            cur.execute(wrapped, params or {})
            df = cur.fetchdf()
        except duckdb.InterruptException as exc:
            raise QueryTimeoutError(f"Query exceeded the {timeout_s:.0f}s timeout and was cancelled") from exc
        except duckdb.Error as exc:
            if timed_out.is_set():
                raise QueryTimeoutError(
                    f"Query exceeded the {timeout_s:.0f}s timeout and was cancelled"
                ) from exc
            raise ConnectorError(f"DuckDB error: {str(exc).splitlines()[0][:400]}") from exc
        finally:
            if timer:
                timer.cancel()
        elapsed = int((time.perf_counter() - t0) * 1000)
        out_geom = []
        for c in geom_cols + [b for b in blob_cols if b == "geometry"]:
            if c in df:
                df[c] = shapely.from_wkb([bytes(v) if v is not None else None for v in df[c]])
                out_geom.append(c)
        truncated = max_rows is not None and len(df) >= max_rows
        return QueryResult(
            df=df,
            sql=sql,
            backend=self.backend,
            elapsed_ms=elapsed,
            row_limit=max_rows,
            truncated=truncated,
            geometry_columns=out_geom,
        )

    def test_connection(self) -> ConnectionStatus:
        try:
            self.con.cursor().execute("SELECT 1").fetchone()
            return ConnectionStatus(
                ok=True,
                backend="local",
                message=f"DuckDB {duckdb.__version__} ready with {len(self.tables)} tables. {self.spatial_message}",
                details={
                    "tables": {k: v.rows for k, v in self.tables.items()},
                    "load_errors": self.load_errors,
                },
            )
        except Exception as exc:  # pragma: no cover
            return ConnectionStatus(ok=False, backend="local", message=str(exc))
