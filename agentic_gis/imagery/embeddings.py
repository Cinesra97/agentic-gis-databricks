"""Image-embedding models and embedding stores (Phase 2 interfaces).

Embeddings are stored *separately* from imagery, each with its footprint geometry, source
imagery id, acquisition date, and model name/version so that results remain traceable.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, Field


class NotConfiguredError(RuntimeError):
    """The imagery component is not installed or configured."""


class EmbeddingRecord(BaseModel):
    tile_id: str
    source_id: str
    footprint_wkt: str
    acquisition_date: date | None = None
    model_name: str
    model_version: str
    vector: list[float] = Field(repr=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def dim(self) -> int:
        return len(self.vector)


class EmbeddingModel(Protocol):
    name: str
    version: str
    dim: int

    def embed(self, tiles: list[np.ndarray], metadata: list[dict[str, Any]]) -> np.ndarray: ...


class ClayEmbeddingModel:
    """Placeholder for the Clay Foundation Model (https://github.com/Clay-foundation/model).

    Not bundled: Clay requires PyTorch, model weights, and per-sensor normalization metadata
    (band wavelengths, GSD, acquisition time/location). Configure it in a dedicated imagery
    environment and implement :meth:`embed` there. This class never returns fake vectors.
    """

    name = "clay"

    def __init__(self, checkpoint_path: str | None = None, version: str = "v1.5", device: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.version = version
        self.device = device
        self.dim = 1024

    def embed(self, tiles: list[np.ndarray], metadata: list[dict[str, Any]]) -> np.ndarray:
        raise NotConfiguredError(
            "Clay embeddings are not configured. See docs/IMAGERY_ROADMAP.md for setup steps "
            "(PyTorch, weights, sensor metadata)."
        )


class EmbeddingStore(Protocol):
    def upsert(self, records: list[EmbeddingRecord]) -> int: ...

    def search(
        self,
        vector: list[float],
        k: int = 10,
        bbox: list[float] | None = None,
        model_version: str | None = None,
    ) -> list[tuple[EmbeddingRecord, float]]: ...

    def count(self) -> int: ...


class InMemoryEmbeddingStore:
    """Cosine-similarity store for tests and small experiments."""

    def __init__(self) -> None:
        self._records: dict[str, EmbeddingRecord] = {}

    def upsert(self, records: list[EmbeddingRecord]) -> int:
        for r in records:
            self._records[f"{r.tile_id}|{r.model_name}|{r.model_version}"] = r
        return len(records)

    def count(self) -> int:
        return len(self._records)

    def search(
        self,
        vector: list[float],
        k: int = 10,
        bbox: list[float] | None = None,
        model_version: str | None = None,
    ) -> list[tuple[EmbeddingRecord, float]]:
        import shapely

        recs = [
            r for r in self._records.values() if model_version is None or r.model_version == model_version
        ]
        if bbox is not None:
            box = shapely.box(*bbox)
            recs = [r for r in recs if shapely.from_wkt(r.footprint_wkt).intersects(box)]
        recs = [r for r in recs if r.dim == len(vector)]
        if not recs:
            return []
        q = np.asarray(vector, dtype=float)
        q = q / (np.linalg.norm(q) or 1.0)
        m = np.asarray([r.vector for r in recs], dtype=float)
        m = m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)
        scores = m @ q
        order = np.argsort(-scores)[:k]
        return [(recs[i], float(scores[i])) for i in order]


class QdrantEmbeddingStore:
    """Qdrant-backed store (optional extra ``vector``). Payload carries all provenance fields."""

    def __init__(self, url: str, collection: str, dim: int, api_key: str | None = None):
        try:
            from qdrant_client import QdrantClient  # type: ignore[import-not-found]
            from qdrant_client.http import models as qm  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise NotConfiguredError("Install the vector extra: uv sync --extra vector") from exc
        self._qm = qm
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection, vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE)
            )

    def upsert(self, records: list[EmbeddingRecord]) -> int:  # pragma: no cover - needs a server
        import uuid

        points = [
            self._qm.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{r.tile_id}|{r.model_name}|{r.model_version}")),
                vector=r.vector,
                payload=r.model_dump(mode="json", exclude={"vector"}),
            )
            for r in records
        ]
        self.client.upsert(self.collection, points=points)
        return len(points)

    def count(self) -> int:  # pragma: no cover - needs a server
        return int(self.client.count(self.collection).count)

    def search(
        self,
        vector: list[float],
        k: int = 10,
        bbox: list[float] | None = None,
        model_version: str | None = None,
    ) -> list[tuple[EmbeddingRecord, float]]:  # pragma: no cover
        flt = None
        if model_version:
            flt = self._qm.Filter(
                must=[
                    self._qm.FieldCondition(
                        key="model_version", match=self._qm.MatchValue(value=model_version)
                    )
                ]
            )
        hits = self.client.query_points(self.collection, query=vector, limit=k, query_filter=flt).points
        out = []
        for h in hits:
            payload = dict(h.payload or {})
            payload["vector"] = []
            out.append((EmbeddingRecord.model_validate(payload), float(h.score)))
        return out
