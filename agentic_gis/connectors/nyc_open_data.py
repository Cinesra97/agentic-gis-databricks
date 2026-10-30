"""NYC Open Data (Socrata SODA API) ingestion.

Public datasets are ingested *ahead of time* by an operator (CLI or sidebar), stored as
GeoParquet under ``data/ingested/`` and then queried locally through DuckDB. The agent never
issues live web requests on its own.

    uv run agentic-gis ingest nyc_311_sanitation --since 2024-01-01 --until 2024-12-31 --limit 400000

Downloads are ordered by the dataset's ``order`` (newest first for 311) and stop at the row limit,
so ``--since`` alone does not reach older records once the limit is hit: it keeps the newest rows.
For a historical period, download a bounded window (``--since`` *and* ``--until``) with a limit at
least as large as the number of matching records, which is counted before downloading.

An optional free app token (``SOCRATA_APP_TOKEN``) raises rate limits.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import httpx
import pandas as pd

from agentic_gis.config.registry import DatasetSpec
from agentic_gis.connectors.base import ConnectorError

log = logging.getLogger(__name__)


class SocrataClient:
    def __init__(
        self,
        domain: str = "data.cityofnewyork.us",
        app_token: str | None = None,
        timeout: float = 120,
        transport: httpx.BaseTransport | None = None,
    ):
        if not re.fullmatch(r"[a-z0-9.\-]+", domain):
            raise ValueError("Invalid Socrata domain")
        headers = {"Accept": "application/json", "User-Agent": "agentic-gis-databricks"}
        if app_token:
            headers["X-App-Token"] = app_token
        self.domain = domain
        self._client = httpx.Client(
            base_url=f"https://{domain}",
            headers=headers,
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        try:
            r = self._client.get(path, params={k: v for k, v in params.items() if v is not None})
        except httpx.HTTPError as exc:
            raise ConnectorError(f"Could not reach {self.domain}: {type(exc).__name__}") from exc
        if r.status_code == 404:
            raise ConnectorError(f"Dataset not found on {self.domain} ({path}). Check the dataset id.")
        if r.status_code == 429:
            raise ConnectorError("Rate limited by the Socrata API; set SOCRATA_APP_TOKEN or retry later.")
        if r.status_code >= 400:
            raise ConnectorError(f"Socrata API error {r.status_code}: {r.text[:200]}")
        return r.json()

    def query(
        self,
        dataset_id: str,
        select: str | None = None,
        where: str | None = None,
        order: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        _check_id(dataset_id)
        return self._get(
            f"/resource/{dataset_id}.json",
            {"$select": select, "$where": where, "$order": order, "$limit": limit, "$offset": offset},
        )

    def count(self, dataset_id: str, where: str | None = None) -> int | None:
        """Number of records matching ``where`` (None if the server does not answer with a count)."""
        _check_id(dataset_id)
        try:
            rows = self._get(f"/resource/{dataset_id}.json", {"$select": "count(*) AS n", "$where": where})
            return int(rows[0]["n"]) if rows and "n" in rows[0] else None
        except (ConnectorError, KeyError, TypeError, ValueError, IndexError):
            return None

    def query_geojson(self, dataset_id: str, where: str | None = None, limit: int = 5000) -> dict[str, Any]:
        _check_id(dataset_id)
        return self._get(f"/resource/{dataset_id}.geojson", {"$where": where, "$limit": limit})

    def fetch_all(
        self,
        dataset_id: str,
        select: str | None,
        where: str | None,
        order: str | None,
        max_rows: int,
        page_size: int = 50_000,
    ) -> pd.DataFrame:
        frames = []
        offset = 0
        # Offset paging needs a total order: ties in e.g. created_date could repeat or skip records
        # across pages, so the row id is always added as a tie-breaker.
        order = order or ":id"
        if ":id" not in order:
            order = f"{order}, :id"
        while offset < max_rows:
            n = min(page_size, max_rows - offset)
            rows = self.query(dataset_id, select, where, order, n, offset)
            if not rows:
                break
            frames.append(pd.DataFrame(rows))
            offset += len(rows)
            log.info("Fetched %s rows from %s", offset, dataset_id)
            if len(rows) < n:
                break
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _check_id(dataset_id: str) -> None:
    if not re.fullmatch(r"[a-z0-9]{4}-[a-z0-9]{4}", dataset_id):
        raise ValueError(f"Invalid Socrata dataset id '{dataset_id}'")


@dataclass
class IngestResult:
    dataset_id: str
    path: Path
    rows: int
    retrieved: str
    where: str | None
    limit_reached: bool = False
    matching_rows: int | None = None  # records matching the filter on the portal (counted before download)
    limit: int | None = None
    date_min: str | None = None
    date_max: str | None = None


def _coerce_types(df: pd.DataFrame, ds: DatasetSpec) -> pd.DataFrame:
    df.columns = [str(c).lower() for c in df.columns]
    for col in ds.columns:
        name = col.name.lower()
        if name not in df.columns:
            continue
        if col.type in ("timestamp", "date"):
            df[name] = pd.to_datetime(df[name], errors="coerce")
        elif col.type in ("double", "float"):
            df[name] = pd.to_numeric(df[name], errors="coerce")
        elif col.type == "integer":
            df[name] = pd.to_numeric(df[name], errors="coerce").astype("Int64")
    return df


def ingest_dataset(
    ds: DatasetSpec,
    client: SocrataClient,
    since: str | None = None,
    limit: int | None = None,
    extra_where: str | None = None,
    until: str | None = None,
) -> IngestResult:
    """Download a registered Socrata dataset to its configured GeoParquet path.

    ``since``/``until`` (inclusive dates) bound the window on the temporal column. The number of
    matching records is counted first, so a window larger than ``limit`` is reported before the
    (newest-first) download silently keeps only part of it.
    """
    if ds.source.type != "socrata" or ds.source.socrata is None:
        raise ConnectorError(f"{ds.id} is not a Socrata dataset")
    sc = ds.source.socrata
    where_parts = [sc.where] if sc.where else []
    if since:
        date.fromisoformat(since)  # validates format
        if not ds.temporal:
            raise ConnectorError(f"{ds.id} has no temporal column for --since")
        where_parts = [w for w in where_parts if ds.temporal.column not in w]
        where_parts.append(f"{ds.temporal.column} >= '{since}T00:00:00'")
    if until:
        end = date.fromisoformat(until)  # validates format
        if not ds.temporal:
            raise ConnectorError(f"{ds.id} has no temporal column for --until")
        if since and date.fromisoformat(since) > end:
            raise ValueError("--since must not be after --until")
        nxt = date.fromordinal(end.toordinal() + 1).isoformat()
        where_parts.append(f"{ds.temporal.column} < '{nxt}T00:00:00'")
    if extra_where:
        where_parts.append(extra_where)
    where = " AND ".join(f"({w})" for w in where_parts) if where_parts else None
    max_rows = limit or sc.limit
    matching = client.count(sc.dataset_id, where)
    if matching is not None and matching > max_rows:
        log.warning(
            "%s: %s records match %s, but the limit is %s; only part of the window will be stored.",
            ds.id,
            f"{matching:,}",
            where,
            f"{max_rows:,}",
        )
    if sc.geojson:
        gj = client.query_geojson(sc.dataset_id, where, max_rows)
        gdf = gpd.GeoDataFrame.from_features(gj.get("features", []), crs="EPSG:4326")
        gdf.columns = [str(c).lower() if c != "geometry" else c for c in gdf.columns]
        fetched = len(gdf)
    else:
        df = client.fetch_all(sc.dataset_id, sc.select, where, sc.order, max_rows)
        if df.empty:
            raise ConnectorError(f"No rows returned for {ds.id} (where: {where})")
        fetched = len(df)
        df = _coerce_types(df, ds)
        sp = ds.spatial
        lat = (sp.lat_column or "").lower() if sp else ""
        lon = (sp.lon_column or "").lower() if sp else ""
        if sp and lat in df.columns and lon in df.columns:
            df[lat] = pd.to_numeric(df[lat], errors="coerce")
            df[lon] = pd.to_numeric(df[lon], errors="coerce")
            df = df[df[lat].between(40.4, 41.0) & df[lon].between(-74.3, -73.6)]
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs="EPSG:4326")
        else:
            gdf = gpd.GeoDataFrame(df)
    path = ds.source.resolved_path()
    assert path is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(path, index=False)
    retrieved = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta: dict[str, Any] = {
        "dataset_id": ds.id,
        "socrata_id": sc.dataset_id,
        "domain": client.domain,
        "where": where,
        "order": sc.order,
        "limit": max_rows,
        "fetched_rows": fetched,
        "matching_rows": matching,
        "limit_reached": fetched >= max_rows or (matching is not None and matching > fetched),
        "rows": len(gdf),
        "retrieved_utc": retrieved,
    }
    if ds.temporal and ds.temporal.column in gdf.columns:
        dates = pd.to_datetime(gdf[ds.temporal.column], errors="coerce").dropna()
        if len(dates):
            meta["date_column"] = ds.temporal.column
            meta["date_min"] = dates.min().date().isoformat()
            meta["date_max"] = dates.max().date().isoformat()
    if fetched >= max_rows:
        log.warning(
            "%s: the download stopped at the %s-row limit; the stored copy is a subset (order: %s).",
            ds.id,
            f"{max_rows:,}",
            sc.order,
        )
    path.with_suffix(".provenance.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return IngestResult(
        ds.id,
        path,
        len(gdf),
        retrieved,
        where,
        limit_reached=bool(meta["limit_reached"]),
        date_min=meta.get("date_min"),
        date_max=meta.get("date_max"),
        matching_rows=matching,
        limit=max_rows,
    )
