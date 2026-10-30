"""What a stored dataset actually contains: row count, date range, source and completeness.

The configured ingestion window (``config/datasets.yaml``) is not the stored range: a download
can stop at its row limit, and with ``order: created_date DESC`` that keeps only the most
recent records. Coverage is therefore computed from the loaded data itself (once, from the
in-memory frame at startup; no extra file scans) and combined with the ingestion sidecar
(``<file>.provenance.json``) written by :func:`agentic_gis.connectors.nyc_open_data.ingest_dataset`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from agentic_gis.config.registry import DatasetSpec

log = logging.getLogger(__name__)


@dataclass
class DatasetCoverage:
    dataset_id: str
    rows: int | None = None
    date_column: str | None = None
    date_min: str | None = None  # ISO date of the earliest record
    date_max: str | None = None
    missing_dates: int = 0
    source: str | None = None
    retrieved_utc: str | None = None
    ingest_where: str | None = None
    ingest_order: str | None = None
    ingest_limit: int | None = None
    limit_reached: bool = False
    is_subset: bool = False
    synthetic: bool = False
    computed_by: str = "unknown"  # "load" = measured from the loaded data; "unknown" = not measured
    notes: list[str] = field(default_factory=list)

    @property
    def known(self) -> bool:
        return self.computed_by != "unknown"

    def overlap(self, date_from: str | None, date_to: str | None) -> str:
        """How a requested window relates to the stored dates.

        ``all`` (no window requested), ``unknown`` (coverage not measured), ``covered`` (the
        data spans the whole window), ``partial`` (the window extends beyond the data) or
        ``none`` (no overlap at all). Open ends are not counted as extending beyond the data.
        """
        if not date_from and not date_to:
            return "all"
        if not self.date_min or not self.date_max:
            return "unknown"
        if (date_to and date_to < self.date_min) or (date_from and date_from > self.date_max):
            return "none"
        if (date_from and date_from < self.date_min) or (date_to and date_to > self.date_max):
            return "partial"
        return "covered"

    def describe(self) -> str:
        parts = []
        if self.rows is not None:
            parts.append(f"{self.rows:,} records")
        if self.date_min and self.date_max:
            parts.append(f"{self.date_column or 'dates'} {self.date_min} to {self.date_max}")
        text = ", ".join(parts) or "coverage not measured"
        if self.source:
            text += f" ({self.source}"
            text += f", retrieved {self.retrieved_utc[:10]})" if self.retrieved_utc else ")"
        if self.is_subset:
            text += "; a subset of the source dataset"
        return text

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["description"] = self.describe()
        return d


def _read_sidecar(ds: DatasetSpec) -> dict[str, Any]:
    path = ds.source.resolved_path()
    if path is None:
        return {}
    side = path.with_suffix(".provenance.json")
    if not side.exists():
        return {}
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as exc:
        log.info("Could not read %s: %s", side, exc)
        return {}


def _lower_bound(where: str | None, column: str | None) -> str | None:
    if not where or not column:
        return None
    m = re.search(rf"{re.escape(column)}\s*>=?\s*'(\d{{4}}-\d{{2}}-\d{{2}})", where, re.I)
    return m.group(1) if m else None


def profile_frame(ds: DatasetSpec, df: pd.DataFrame) -> DatasetCoverage:
    """Coverage of a dataset from its loaded (in-memory) frame plus the ingestion sidecar."""
    cov = DatasetCoverage(dataset_id=ds.id, rows=len(df), synthetic=ds.synthetic, computed_by="load")
    if ds.temporal and ds.temporal.column in df.columns:
        cov.date_column = ds.temporal.column
        s = pd.to_datetime(df[ds.temporal.column], errors="coerce")
        cov.missing_dates = int(s.isna().sum())
        if s.notna().any():
            cov.date_min = s.min().date().isoformat()
            cov.date_max = s.max().date().isoformat()
    side = _read_sidecar(ds)
    sc = ds.source.socrata if ds.source.type == "socrata" else None
    if sc is not None:
        cov.source = f"NYC Open Data {sc.dataset_id}"
        cov.ingest_where = side.get("where", sc.where)
        cov.ingest_order = side.get("order", sc.order)
        cov.retrieved_utc = side.get("retrieved_utc")
        limit = side.get("limit")
        assumed = limit is None
        cov.ingest_limit = int(limit) if limit is not None else sc.limit
        fetched = int(side.get("fetched_rows", side.get("rows", cov.rows or 0)) or 0)
        cov.limit_reached = bool(side.get("limit_reached", fetched >= (cov.ingest_limit or 10**12)))
        cov.is_subset = bool(cov.ingest_where) or cov.limit_reached
        if not side:
            cov.notes.append("No ingestion record (.provenance.json) was found next to the data file.")
        if cov.limit_reached:
            order = (cov.ingest_order or "").lower()
            col = (cov.date_column or "").lower()
            which = (
                "the most recent records were kept, so older records are missing"
                if col and col in order and "desc" in order
                else "the oldest records were kept, so newer records are missing"
                if col and col in order
                else "the copy may be incomplete"
            )
            cov.notes.append(
                f"The download stopped at its {cov.ingest_limit:,}-row limit"
                + (" (limit taken from the configuration)" if assumed else "")
                + f"; {which}."
            )
        lower = _lower_bound(cov.ingest_where, cov.date_column)
        if lower and cov.date_min and cov.date_min > lower:
            cov.notes.append(
                f"The ingestion filter starts at {lower}, but the stored records start at {cov.date_min}."
            )
    elif ds.source.type == "file":
        cov.source = ds.source.path
    if cov.missing_dates:
        cov.notes.append(f"{cov.missing_dates:,} records have no {cov.date_column}.")
    return cov


def unknown_coverage(ds: DatasetSpec, reason: str) -> DatasetCoverage:
    return DatasetCoverage(
        dataset_id=ds.id,
        date_column=ds.temporal.column if ds.temporal else None,
        synthetic=ds.synthetic,
        notes=[reason],
    )
