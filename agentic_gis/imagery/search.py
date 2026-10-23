"""Imagery similarity search and candidate detections (Phase 2).

Imagery-derived results are always *candidates*. A candidate never becomes a confirmed
sanitation violation without an authoritative verification record (e.g. a field inspection).
Small dumping piles and individual vehicles are generally not resolvable in 10 m Sentinel-2
imagery; sub-meter orthophotography is required for those use cases.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentic_gis.imagery.embeddings import EmbeddingModel, EmbeddingStore

CANDIDATE_NOTE = "Imagery-based candidate for review; not a confirmed sanitation violation."


class ImagerySearchStatus(BaseModel):
    available: bool
    message: str


class CandidateDetection(BaseModel):
    tile_id: str
    source_id: str
    footprint_wkt: str
    score: float
    model_name: str
    model_version: str
    acquisition_date: date | None = None
    status: Literal["candidate"] = "candidate"
    verification: Literal["unverified", "field_verified", "rejected"] = "unverified"
    note: str = Field(default=CANDIDATE_NOTE)


class ImagerySearchService:
    def __init__(self, store: EmbeddingStore | None = None, model: EmbeddingModel | None = None):
        self.store = store
        self.model = model

    def status(self) -> ImagerySearchStatus:
        if self.store is None:
            return ImagerySearchStatus(
                available=False,
                message="Imagery search is not configured (no embedding index). See docs/IMAGERY_ROADMAP.md.",
            )
        n = self.store.count()
        if n == 0:
            return ImagerySearchStatus(available=False, message="The imagery embedding index is empty.")
        return ImagerySearchStatus(available=True, message=f"Imagery index ready ({n:,} embeddings).")

    def similar_to_vector(
        self,
        vector: list[float],
        k: int = 10,
        bbox: list[float] | None = None,
        model_version: str | None = None,
    ) -> list[CandidateDetection]:
        if self.store is None:
            return []
        return [
            CandidateDetection(
                tile_id=r.tile_id,
                source_id=r.source_id,
                footprint_wkt=r.footprint_wkt,
                score=s,
                model_name=r.model_name,
                model_version=r.model_version,
                acquisition_date=r.acquisition_date,
            )
            for r, s in self.store.search(vector, k=k, bbox=bbox, model_version=model_version)
        ]


def get_default_search_service(settings: Any) -> ImagerySearchService:
    """The MVP ships without an imagery index; returns an unconfigured service."""
    return ImagerySearchService(store=None, model=None)
