"""Per-session store of analytical artifacts (DataFrames / GeoDataFrames)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
import pandas as pd

from agentic_gis.spatial.geometry import bounds_wgs84, geometry_kind


@dataclass
class Artifact:
    id: str
    turn: int
    step_id: str
    tool: str
    title: str
    data: pd.DataFrame
    description: str = ""
    datasets: list[str] = field(default_factory=list)
    synthetic: bool = False
    sql: str | None = None
    notes: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def is_spatial(self) -> bool:
        return isinstance(self.data, gpd.GeoDataFrame) and self.data.geometry.name in self.data

    @property
    def rows(self) -> int:
        return len(self.data)

    @property
    def geometry_kind(self) -> str | None:
        return geometry_kind(self.data) if self.is_spatial else None

    @property
    def bounds(self) -> list[float] | None:
        return bounds_wgs84(self.data) if self.is_spatial else None

    def columns(self) -> list[str]:
        return [str(c) for c in self.data.columns if not (self.is_spatial and c == self.data.geometry.name)]

    def digest(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "tool": self.tool,
            "rows": self.rows,
            "geometry": self.geometry_kind,
            "columns": self.columns()[:30],
            "datasets": self.datasets,
            "synthetic": self.synthetic,
        }


class ResultStore:
    """Thread-safe artifact store with simple reference resolution."""

    def __init__(self, max_artifacts: int = 300):
        self._items: dict[str, Artifact] = {}
        self._primary: dict[int, str] = {}
        self._lock = threading.Lock()
        self.max_artifacts = max_artifacts

    def add(self, artifact: Artifact) -> Artifact:
        with self._lock:
            self._items[artifact.id] = artifact
            if len(self._items) > self.max_artifacts:
                oldest = sorted(self._items.values(), key=lambda a: (a.turn, a.id))[0]
                self._items.pop(oldest.id, None)
        return artifact

    def get(self, artifact_id: str) -> Artifact | None:
        return self._items.get(artifact_id)

    def all(self) -> list[Artifact]:
        return list(self._items.values())

    def for_turn(self, turn: int) -> list[Artifact]:
        return [a for a in self._items.values() if a.turn == turn]

    def set_primary(self, turn: int, artifact_id: str) -> None:
        self._primary[turn] = artifact_id

    def primary(self, turn: int) -> Artifact | None:
        aid = self._primary.get(turn)
        return self._items.get(aid) if aid else None

    def last_primary(self, before_turn: int) -> Artifact | None:
        for t in sorted((t for t in self._primary if t < before_turn), reverse=True):
            art = self.primary(t)
            if art is not None:
                return art
        return None

    def resolve(self, ref: str, turn: int) -> Artifact | None:
        """Resolve 's3' (this turn), 't2.s1' (any turn), or 'last'/'previous' (prior turn's main result)."""
        ref = ref.strip()
        if ref in ("last", "previous", "prev"):
            return self.last_primary(turn)
        if "." in ref and ref.startswith("t"):
            return self._items.get(ref)
        return self._items.get(f"t{turn}.{ref}")

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._primary.clear()
