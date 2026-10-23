"""CRS, unit and conversion helpers.

All analytical results are stored in WGS84 (EPSG:4326) for mapping and export. Distance,
area and buffer computations are done in a projected CRS -- by default EPSG:2263
(NAD83 / New York Long Island, US survey feet), the standard for NYC city data.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import CRS

WGS84 = "EPSG:4326"
DistanceUnit = Literal["ft", "m", "km", "mi", "yd", "block"]

# Meters per unit. "block" approximates a short Manhattan north-south block (~264 ft = 1/20 mile).
UNIT_TO_METERS: dict[str, float] = {
    "ft": 0.3048,
    "m": 1.0,
    "km": 1000.0,
    "mi": 1609.344,
    "yd": 0.9144,
    "block": 80.4672,
}


class SpatialError(ValueError):
    """User-facing spatial processing error."""


@lru_cache(maxsize=16)
def _crs_meters_per_unit(crs: str) -> float:
    c = CRS.from_user_input(crs)
    if not c.is_projected:
        raise SpatialError(f"{crs} is not a projected CRS; distance operations need projected coordinates")
    return float(c.axis_info[0].unit_conversion_factor)


def to_crs_units(distance: float, units: str, crs: str) -> float:
    if units not in UNIT_TO_METERS:
        raise SpatialError(f"Unknown distance unit '{units}'. Use one of {sorted(UNIT_TO_METERS)}")
    if distance < 0:
        raise SpatialError("Distance must be non-negative")
    return distance * UNIT_TO_METERS[units] / _crs_meters_per_unit(crs)


def crs_units_to_feet(value: float | np.ndarray, crs: str) -> Any:
    return value * _crs_meters_per_unit(crs) / UNIT_TO_METERS["ft"]


def is_geodataframe(df: Any) -> bool:
    return isinstance(df, gpd.GeoDataFrame) and df.geometry is not None and df.geometry.name in df


def ensure_gdf(
    df: pd.DataFrame,
    geometry_column: str | None = None,
    lat_column: str | None = None,
    lon_column: str | None = None,
    crs: str = WGS84,
) -> gpd.GeoDataFrame | None:
    """Best-effort conversion of a query result into a GeoDataFrame in EPSG:4326."""
    if isinstance(df, gpd.GeoDataFrame):
        g = df if df.crs is not None else df.set_crs(crs)
        return g.to_crs(WGS84) if g.crs.to_epsg() != 4326 else g
    candidates = [geometry_column] if geometry_column else []
    candidates += [c for c in ("geometry", "geom", "the_geom", "shape", "wkt", "geom_wkt") if c in df.columns]
    for col in candidates:
        if not col or col not in df.columns:
            continue
        series = df[col]
        sample = series.dropna().head(1)
        if sample.empty:
            continue
        v = sample.iloc[0]
        try:
            if isinstance(v, shapely.Geometry):
                geoms = series.values
            elif isinstance(v, bytes | bytearray | memoryview):
                geoms = shapely.from_wkb([bytes(x) if x is not None else None for x in series])
            elif isinstance(v, str) and v.lstrip().startswith("{"):
                geoms = shapely.from_geojson(series.where(series.notna(), None))
            elif isinstance(v, str):
                geoms = shapely.from_wkt(series.where(series.notna(), None))
            else:
                continue
        except Exception:  # noqa: S112 - try the next candidate column
            continue
        out = gpd.GeoDataFrame(df.drop(columns=[col]), geometry=list(geoms), crs=crs)
        return out.to_crs(WGS84) if out.crs.to_epsg() != 4326 else out
    lat = (
        lat_column
        if lat_column in df.columns
        else next((c for c in ("latitude", "lat", "y") if c in df), None)
    )
    lon = (
        lon_column
        if lon_column in df.columns
        else next((c for c in ("longitude", "lon", "lng", "long", "x") if c in df), None)
    )
    if lat and lon and pd.api.types.is_numeric_dtype(df[lat]) and pd.api.types.is_numeric_dtype(df[lon]):
        valid = df[lat].notna() & df[lon].notna()
        sub = df[valid]
        return gpd.GeoDataFrame(sub, geometry=gpd.points_from_xy(sub[lon], sub[lat]), crs=WGS84)
    return None


def to_projected(gdf: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84)
    return gdf.to_crs(crs)


def to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(WGS84)
    return gdf if gdf.crs.to_epsg() == 4326 else gdf.to_crs(WGS84)


def geometry_kind(gdf: gpd.GeoDataFrame) -> str | None:
    if gdf is None or len(gdf) == 0:
        return None
    types = set(gdf.geometry.dropna().geom_type.unique())
    kinds = set()
    for t in types:
        if "Point" in t:
            kinds.add("point")
        elif "Line" in t:
            kinds.add("line")
        elif "Polygon" in t:
            kinds.add("polygon")
        else:
            kinds.add("mixed")
    if not kinds:
        return None
    return kinds.pop() if len(kinds) == 1 else "mixed"


def bounds_wgs84(gdf: gpd.GeoDataFrame) -> list[float] | None:
    if gdf is None or len(gdf) == 0:
        return None
    b = to_wgs84(gdf).total_bounds
    if not np.all(np.isfinite(b)):
        return None
    return [round(float(x), 6) for x in b]


def for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Tabular copy with geometry as WKT (for CSV export / display)."""
    if isinstance(df, gpd.GeoDataFrame):
        out = pd.DataFrame(df.drop(columns=df.geometry.name))
        out["geometry_wkt"] = shapely.to_wkt(df.geometry.values, rounding_precision=6)
        return out
    return df


def to_geojson_text(gdf: gpd.GeoDataFrame) -> str:
    g = to_wgs84(gdf).copy()
    for c in g.columns:
        if c == g.geometry.name:
            continue
        if pd.api.types.is_datetime64_any_dtype(g[c]):
            g[c] = g[c].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return g.to_json(drop_id=True, na="null")
