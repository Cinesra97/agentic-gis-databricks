"""Core vector operations (buffer, joins, nearest, area filters) on bounded GeoDataFrames.

Inputs and outputs are EPSG:4326; distances are computed in the configured projected CRS.
"""

from __future__ import annotations

import re
from difflib import get_close_matches
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from agentic_gis.spatial.geometry import SpatialError, crs_units_to_feet, to_crs_units, to_projected, to_wgs84

Predicate = Literal["intersects", "within", "contains", "dwithin"]


def _clean_right(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in ("index_right", "index_left") if c in df.columns])


def buffer_features(
    gdf: gpd.GeoDataFrame, distance: float, units: str, projected_crs: str, dissolve: bool = False
) -> gpd.GeoDataFrame:
    if len(gdf) == 0:
        return gdf
    d = to_crs_units(distance, units, projected_crs)
    proj = to_projected(gdf, projected_crs)
    if dissolve:
        geom = shapely.union_all(proj.geometry.buffer(d).values)
        out = gpd.GeoDataFrame(
            {"buffer_distance": [distance], "buffer_units": [units], "source_features": [len(gdf)]},
            geometry=[geom],
            crs=projected_crs,
        )
    else:
        out = proj.copy()
        out["geometry"] = proj.geometry.buffer(d)
        out["buffer_distance"] = distance
        out["buffer_units"] = units
    return to_wgs84(out)


def spatial_join(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    predicate: Predicate,
    projected_crs: str,
    distance: float | None = None,
    units: str = "ft",
    how: Literal["inner", "left"] = "inner",
    summarize: bool = False,
    right_columns: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Join left features to right features.

    ``summarize=True`` keeps one row per left feature with ``match_count`` (and the nearest
    match distance for ``dwithin``); otherwise one row per matching pair is returned.
    """
    if predicate == "dwithin" and distance is None:
        raise SpatialError("predicate 'dwithin' requires a distance")
    lp = to_projected(left, projected_crs).reset_index(drop=True)
    rp = to_projected(right, projected_crs).reset_index(drop=True)
    rcols = [c for c in (right_columns or list(rp.columns)) if c in rp.columns and c != rp.geometry.name]
    rsub = rp[rcols + [rp.geometry.name]]
    rsub = rsub.rename(columns={c: (f"{c}_right" if c in lp.columns else c) for c in rcols})
    kwargs = {}
    if predicate == "dwithin":
        kwargs["distance"] = to_crs_units(distance or 0.0, units, projected_crs)
    joined = gpd.sjoin(lp, rsub, how="inner", predicate=predicate, **kwargs)
    if summarize:
        counts = joined.groupby(level=0).size()
        out = lp.copy()
        out["match_count"] = counts.reindex(out.index).fillna(0).astype(int)
        if predicate == "dwithin" and len(joined):
            rgeom = rp.geometry.values
            d = shapely.distance(lp.geometry.values[joined.index.values], rgeom[joined["index_right"].values])
            mind = pd.Series(d, index=joined.index).groupby(level=0).min()
            out["nearest_match_ft"] = crs_units_to_feet(mind.reindex(out.index), projected_crs).round(1)
        if how == "inner":
            out = out[out["match_count"] > 0]
        return to_wgs84(out)
    if predicate == "dwithin" and len(joined):
        rgeom = rp.geometry.values
        d = shapely.distance(joined.geometry.values, rgeom[joined["index_right"].values])
        joined["distance_ft"] = np.round(crs_units_to_feet(d, projected_crs), 1)
    if how == "left":
        missing = lp.loc[~lp.index.isin(joined.index)]
        joined = pd.concat([joined, missing], axis=0)
    out = gpd.GeoDataFrame(_clean_right(joined), geometry=lp.geometry.name, crs=projected_crs)
    return to_wgs84(out.reset_index(drop=True))


def nearest_features(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    projected_crs: str,
    max_distance: float | None = None,
    units: str = "ft",
    right_columns: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Attach the attributes of (and distance to) the nearest right feature."""
    lp = to_projected(left, projected_crs).reset_index(drop=True)
    rp = to_projected(right, projected_crs).reset_index(drop=True)
    rcols = [c for c in (right_columns or list(rp.columns)) if c in rp.columns and c != rp.geometry.name]
    rsub = rp[rcols + [rp.geometry.name]].rename(
        columns={c: (f"{c}_nearest" if c in lp.columns else c) for c in rcols}
    )
    md = to_crs_units(max_distance, units, projected_crs) if max_distance else None
    joined = gpd.sjoin_nearest(lp, rsub, how="inner", max_distance=md, distance_col="_dist")
    joined = joined[~joined.index.duplicated(keep="first")]
    joined["distance_ft"] = np.round(crs_units_to_feet(joined["_dist"].values, projected_crs), 1)
    joined = joined.drop(columns=["_dist"])
    out = gpd.GeoDataFrame(_clean_right(joined), geometry=lp.geometry.name, crs=projected_crs)
    return to_wgs84(out.reset_index(drop=True))


def normalize_name(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    repl = {
        r"\bAVE\b": "AVENUE",
        r"\bAV\b": "AVENUE",
        r"\bST\b": "STREET",
        r"\bBLVD\b": "BOULEVARD",
        r"\bRD\b": "ROAD",
        r"\bPKWY\b": "PARKWAY",
        r"\bPL\b": "PLACE",
        r"\bDR\b": "DRIVE",
        r"\bEXPWY\b": "EXPRESSWAY",
        r"\bE\b": "EAST",
        r"\bW\b": "WEST",
        r"\bN\b": "NORTH",
        r"\bS\b": "SOUTH",
        r"\b(\d+)(ST|ND|RD|TH)\b": r"\1",
    }
    for pat, r in repl.items():
        s = re.sub(pat, r, s)
    return re.sub(r"\s+", " ", s).strip()


def match_names(names: list[str], candidates: list[str], cutoff: float = 0.78) -> dict[str, list[str]]:
    """Fuzzy-match requested names (streets, neighborhoods ...) to candidate labels."""
    norm_map: dict[str, list[str]] = {}
    for c in candidates:
        if c is None or (isinstance(c, float) and np.isnan(c)):
            continue
        norm_map.setdefault(normalize_name(str(c)), []).append(str(c))
    keys = list(norm_map)
    out: dict[str, list[str]] = {}
    for name in names:
        n = normalize_name(name)
        hits = [k for k in keys if k == n]
        if not hits:
            hits = [k for k in keys if re.search(rf"\b{re.escape(n)}\b", k)]
        if not hits:
            hits = get_close_matches(n, keys, n=3, cutoff=cutoff)
        out[name] = sorted({orig for k in hits for orig in norm_map[k]})
    return out


def filter_by_area(
    gdf: gpd.GeoDataFrame, areas: gpd.GeoDataFrame, predicate: Literal["intersects", "within"] = "intersects"
) -> gpd.GeoDataFrame:
    if len(areas) == 0:
        raise SpatialError("No matching areas to filter by")
    area = shapely.union_all(to_wgs84(areas).geometry.values)
    g = to_wgs84(gdf)
    tree_mask = g.geometry.intersects(area) if predicate == "intersects" else g.geometry.within(area)
    return g[tree_mask.values]
