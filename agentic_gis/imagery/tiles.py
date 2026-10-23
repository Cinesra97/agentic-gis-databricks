"""Imagery sources, tile footprints and (optional) windowed reads from Cloud-Optimized GeoTIFFs.

Phase 2 module. Original imagery is never modified or copied into the embedding store: tiles
are referenced by ``source_id`` + footprint, and pixels are read on demand.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import shapely
from pydantic import BaseModel, Field

from agentic_gis.spatial.geometry import UNIT_TO_METERS, to_crs_units


class ImagerySource(BaseModel):
    """A registered imagery product (e.g. NYC orthophotography COGs for one year)."""

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,62}$")
    name: str
    kind: Literal["cog", "xyz", "wms", "stac_collection"] = "cog"
    uri: str = Field(description="COG path/URL, XYZ template, WMS endpoint, or STAC collection URL")
    acquisition_date: date | None = None
    resolution_m: float | None = Field(None, gt=0)
    crs: str = "EPSG:2263"
    bands: list[str] = Field(default_factory=lambda: ["red", "green", "blue"])
    license: str | None = None
    attribution: str | None = None
    notes: str | None = None


class TileRef(BaseModel):
    tile_id: str
    source_id: str
    footprint_wkt: str = Field(description="Tile footprint polygon in EPSG:4326")
    bounds_native: list[float] = Field(description="minx, miny, maxx, maxy in the tiling CRS")
    crs: str
    acquisition_date: date | None = None
    size_m: float


def tile_grid(
    bounds_wgs84: list[float],
    tile_size: float,
    units: str = "m",
    source_id: str = "unregistered",
    tiling_crs: str = "EPSG:2263",
    acquisition_date: date | None = None,
    max_tiles: int = 200_000,
) -> gpd.GeoDataFrame:
    """Square tile footprints covering ``bounds_wgs84`` (minx, miny, maxx, maxy)."""
    if len(bounds_wgs84) != 4:
        raise ValueError("bounds must be [minx, miny, maxx, maxy]")
    area = gpd.GeoSeries([shapely.box(*bounds_wgs84)], crs="EPSG:4326").to_crs(tiling_crs)
    minx, miny, maxx, maxy = area.total_bounds
    size = to_crs_units(tile_size, units, tiling_crs)
    nx, ny = math.ceil((maxx - minx) / size), math.ceil((maxy - miny) / size)
    if nx * ny > max_tiles:
        raise ValueError(
            f"{nx * ny:,} tiles exceed the limit of {max_tiles:,}; use a larger tile size or smaller area"
        )
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    ix, iy = ix.ravel(), iy.ravel()
    x0 = minx + ix * size
    y0 = miny + iy * size
    geoms = shapely.box(x0, y0, x0 + size, y0 + size)
    gdf = gpd.GeoDataFrame(
        {
            "tile_id": [f"{source_id}:{tile_size:g}{units}:{a}_{b}" for a, b in zip(ix, iy, strict=True)],
            "source_id": source_id,
            "acquisition_date": acquisition_date,
            "size_m": tile_size * UNIT_TO_METERS[units],
            "minx": x0,
            "miny": y0,
            "maxx": x0 + size,
            "maxy": y0 + size,
        },
        geometry=geoms,
        crs=tiling_crs,
    )
    return gdf.to_crs("EPSG:4326")


def read_tile(source: ImagerySource, bounds_native: list[float], out_size: int = 256) -> np.ndarray:
    """Read a tile window from a COG (requires the optional ``imagery`` extra: rasterio)."""
    if source.kind != "cog":
        raise NotImplementedError("Only COG sources support direct pixel reads in this version")
    try:
        import rasterio  # type: ignore[import-not-found]
        from rasterio.enums import Resampling  # type: ignore[import-not-found]
        from rasterio.windows import from_bounds  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the imagery extra: uv sync --extra imagery") from exc
    with rasterio.open(source.uri) as src:
        window = from_bounds(*bounds_native, transform=src.transform)
        data: Any = src.read(
            window=window,
            out_shape=(src.count, out_size, out_size),
            resampling=Resampling.bilinear,
            boundless=True,
        )
    return np.asarray(data)
