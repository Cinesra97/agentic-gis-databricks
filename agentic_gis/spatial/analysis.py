"""Analytical spatial methods: aggregation, hotspots, clustering, temporal trends.

Methods are deliberately transparent (documented formulas, deterministic output) so that
results can be explained and reproduced. Each returns data plus a ``notes`` list describing
method parameters and caveats, which the agent cites in its methodology section.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from agentic_gis.spatial.geometry import (
    SpatialError,
    crs_units_to_feet,
    to_crs_units,
    to_projected,
    to_wgs84,
)

Measure = Literal["count", "count_distinct", "sum", "mean"]


@dataclass
class AnalysisResult:
    data: pd.DataFrame
    notes: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def _measure(grouped: Any, measure: Measure, value_column: str | None) -> pd.Series:
    if measure == "count":
        return grouped.size()
    if not value_column:
        raise SpatialError(f"measure '{measure}' requires value_column")
    if measure == "count_distinct":
        return grouped[value_column].nunique()
    if measure == "sum":
        return grouped[value_column].sum()
    return grouped[value_column].mean()


def _category_breakdown(df: pd.DataFrame, key: str, category_column: str, top: int = 6) -> pd.DataFrame:
    cats = df[category_column].astype("string").fillna("(none)")
    top_cats = cats.value_counts().head(top).index.tolist()
    pivot = pd.crosstab(df[key], cats.where(cats.isin(top_cats), "Other"))
    pivot.columns = [f"n_{str(c).lower().replace(' ', '_').replace('/', '_')[:40]}" for c in pivot.columns]
    return pivot


def aggregate_to_polygons(
    points: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    label_column: str | None,
    measure: Measure = "count",
    value_column: str | None = None,
    category_column: str | None = None,
    min_value: float | None = None,
    top_n: int | None = None,
) -> AnalysisResult:
    pts = to_wgs84(points)
    polys = to_wgs84(polygons).reset_index(drop=True)
    polys["_area_id"] = np.arange(len(polys))
    if pts.geometry.geom_type.isin(["Point", "MultiPoint"]).all():
        joined = gpd.sjoin(pts, polys[["_area_id", "geometry"]], how="inner", predicate="within")
    else:  # lines/polygons: assign by representative point
        rep = pts.copy()
        rep["geometry"] = pts.geometry.representative_point()
        joined = gpd.sjoin(rep, polys[["_area_id", "geometry"]], how="inner", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    values = _measure(joined.groupby("_area_id"), measure, value_column)
    out = polys.copy()
    out["value"] = values.reindex(out["_area_id"]).fillna(0).values
    if measure in ("count", "count_distinct"):
        out["value"] = out["value"].astype(int)
    total = float(out["value"].sum()) if measure in ("count", "sum") else None
    if total:
        out["share_pct"] = (out["value"] / total * 100).round(2)
    if category_column and category_column in joined:
        bd = _category_breakdown(joined, "_area_id", category_column)
        out = out.merge(bd, left_on="_area_id", right_index=True, how="left")
        for c in bd.columns:
            out[c] = out[c].fillna(0).astype(int)
    out = out.sort_values("value", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    unassigned = len(pts) - len(joined)
    if min_value is not None:
        out = out[out["value"] >= min_value]
    if top_n:
        out = out.head(top_n)
    notes = [f"Points assigned to areas by point-in-polygon ({len(joined):,} of {len(pts):,} matched)."]
    if unassigned:
        notes.append(
            f"{unassigned:,} features fell outside all areas (e.g. water, parks, or outside the boundary set)."
        )
    return AnalysisResult(
        gpd.GeoDataFrame(out.drop(columns=["_area_id"]), geometry="geometry", crs=polys.crs),
        notes,
        {"assigned": len(joined), "unassigned": unassigned, "label_column": label_column},
    )


def aggregate_to_lines(
    points: gpd.GeoDataFrame,
    lines: gpd.GeoDataFrame,
    projected_crs: str,
    max_distance_ft: float = 100.0,
    measure: Measure = "count",
    value_column: str | None = None,
    category_column: str | None = None,
    min_value: float | None = None,
    top_n: int | None = None,
) -> AnalysisResult:
    lp = to_projected(lines, projected_crs).reset_index(drop=True)
    lp["_line_id"] = np.arange(len(lp))
    pp = to_projected(points, projected_crs)
    md = to_crs_units(max_distance_ft, "ft", projected_crs)
    joined = gpd.sjoin_nearest(pp, lp[["_line_id", "geometry"]], how="inner", max_distance=md)
    joined = joined[~joined.index.duplicated(keep="first")]
    values = _measure(joined.groupby("_line_id"), measure, value_column)
    out = lp.copy()
    out["value"] = values.reindex(out["_line_id"]).fillna(0).values
    if measure in ("count", "count_distinct"):
        out["value"] = out["value"].astype(int)
    if "length_ft" not in out:
        out["length_ft"] = crs_units_to_feet(out.geometry.length.values, projected_crs).round(1)
    out["per_1000ft"] = (out["value"] / out["length_ft"].clip(lower=1) * 1000).round(2)
    if category_column and category_column in joined:
        bd = _category_breakdown(joined, "_line_id", category_column)
        out = out.merge(bd, left_on="_line_id", right_index=True, how="left")
        for c in bd.columns:
            out[c] = out[c].fillna(0).astype(int)
    out = out.sort_values("value", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    if min_value is not None:
        out = out[out["value"] >= min_value]
    if top_n:
        out = out.head(top_n)
    notes = [
        f"Each point was assigned to the nearest line within {max_distance_ft:g} ft "
        f"({len(joined):,} of {len(points):,} points matched)."
    ]
    return AnalysisResult(
        to_wgs84(gpd.GeoDataFrame(out.drop(columns=["_line_id"]), geometry="geometry", crs=projected_crs)),
        notes,
    )


def _grid_index(pp: gpd.GeoDataFrame, cell: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    xy = shapely.get_coordinates(pp.geometry.representative_point().values)
    x0, y0 = math.floor(xy[:, 0].min() / cell) * cell, math.floor(xy[:, 1].min() / cell) * cell
    ix = ((xy[:, 0] - x0) // cell).astype(int)
    iy = ((xy[:, 1] - y0) // cell).astype(int)
    return ix, iy, x0, y0


def _cell_polygons(ix: np.ndarray, iy: np.ndarray, x0: float, y0: float, cell: float) -> np.ndarray:
    return shapely.box(x0 + ix * cell, y0 + iy * cell, x0 + (ix + 1) * cell, y0 + (iy + 1) * cell)


def grid_aggregate(
    points: gpd.GeoDataFrame,
    projected_crs: str,
    cell_size: float = 1000.0,
    units: str = "ft",
    measure: Measure = "count",
    value_column: str | None = None,
    category_column: str | None = None,
    min_value: float | None = None,
    top_n: int | None = None,
) -> AnalysisResult:
    if len(points) == 0:
        raise SpatialError("No features to aggregate")
    cell = to_crs_units(cell_size, units, projected_crs)
    pp = to_projected(points, projected_crs).copy()
    ix, iy, x0, y0 = _grid_index(pp, cell)
    pp["_cell"] = ix.astype(np.int64) * 1_000_000 + iy
    grouped = pp.groupby("_cell")
    values = _measure(grouped, measure, value_column)
    cells = values.index.values
    cix, ciy = cells // 1_000_000, cells % 1_000_000
    out = gpd.GeoDataFrame(
        {
            "cell_id": [f"C{a}_{b}" for a, b in zip(cix, ciy, strict=True)],
            "value": values.values,
            "points": grouped.size().reindex(values.index).values,
        },
        geometry=_cell_polygons(cix, ciy, x0, y0, cell),
        crs=projected_crs,
    )
    if category_column and category_column in pp:
        bd = _category_breakdown(pp, "_cell", category_column)
        out["_cell"] = values.index.values
        out = out.merge(bd, left_on="_cell", right_index=True, how="left").drop(columns=["_cell"])
    out = out.sort_values("value", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    if min_value is not None:
        out = out[out["value"] >= min_value]
    if top_n:
        out = out.head(top_n)
    notes = [
        f"Square grid of {cell_size:g} {units} cells in {projected_crs}; only non-empty cells are returned."
    ]
    return AnalysisResult(to_wgs84(out), notes, {"cells": len(values)})


def gi_star_hotspots(
    points: gpd.GeoDataFrame,
    projected_crs: str,
    cell_size: float = 1000.0,
    units: str = "ft",
    weight_column: str | None = None,
) -> AnalysisResult:
    """Getis-Ord Gi* on a regular grid with queen (3x3, self-inclusive) binary weights.

    The study area is the convex hull of the input points buffered by one cell, so open water
    and areas far from any observation do not inflate significance.
    """
    if len(points) < 10:
        raise SpatialError("Hotspot analysis needs at least 10 features")
    cell = to_crs_units(cell_size, units, projected_crs)
    pp = to_projected(points, projected_crs)
    ix, iy, x0, y0 = _grid_index(pp, cell)
    nx, ny = ix.max() + 1, iy.max() + 1
    if nx * ny > 4_000_000:
        raise SpatialError("Grid too large; increase cell_size")
    w = pp[weight_column].astype(float).values if weight_column else np.ones(len(pp))
    x = np.zeros((nx + 2, ny + 2))
    np.add.at(x, (ix + 1, iy + 1), w)
    gx, gy = np.meshgrid(np.arange(-1, nx + 1), np.arange(-1, ny + 1), indexing="ij")
    centers_x = x0 + (gx + 0.5) * cell
    centers_y = y0 + (gy + 0.5) * cell
    hull = shapely.convex_hull(shapely.union_all(pp.geometry.values)).buffer(cell)
    mask = shapely.contains_xy(hull, centers_x, centers_y)
    xm = np.where(mask, x, 0.0)
    n = int(mask.sum())
    mean = xm[mask].mean()
    s = math.sqrt(max((xm[mask] ** 2).mean() - mean**2, 1e-12))

    def window_sum(a: np.ndarray) -> np.ndarray:
        p = np.pad(a, 1)
        return sum(  # type: ignore[return-value]
            p[1 + dx : p.shape[0] - 1 + dx, 1 + dy : p.shape[1] - 1 + dy]
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
        )

    lag = window_sum(xm)
    wsum = window_sum(mask.astype(float))
    denom = s * np.sqrt(np.maximum((n * wsum - wsum**2) / (n - 1), 1e-12))
    z = np.where(mask, (lag - mean * wsum) / denom, np.nan)
    p = np.vectorize(lambda v: math.erfc(abs(v) / math.sqrt(2)) if np.isfinite(v) else np.nan)(z)

    keep = mask & ((x > 0) | (np.abs(np.nan_to_num(z)) >= 1.645))
    kx, ky = np.nonzero(keep)
    zz, pv, cnt = z[kx, ky], p[kx, ky], x[kx, ky]
    cls = np.select(
        [zz >= 2.576, zz >= 1.960, zz >= 1.645, zz <= -2.576, zz <= -1.960, zz <= -1.645],
        [
            "Hot spot (99%)",
            "Hot spot (95%)",
            "Hot spot (90%)",
            "Cold spot (99%)",
            "Cold spot (95%)",
            "Cold spot (90%)",
        ],
        default="Not significant",
    )
    gx_cells, gy_cells = kx - 1, ky - 1
    out = gpd.GeoDataFrame(
        {
            "cell_id": [f"C{a}_{b}" for a, b in zip(gx_cells, gy_cells, strict=True)],
            "count": cnt.astype(int),
            "gi_z": np.round(zz, 3),
            "p_value": np.round(pv, 4),
            "hotspot_class": cls,
        },
        geometry=_cell_polygons(gx_cells, gy_cells, x0, y0, cell),
        crs=projected_crs,
    ).sort_values("gi_z", ascending=False)
    hot = int((out["gi_z"] >= 1.960).sum())
    notes = [
        f"Getis-Ord Gi* on {cell_size:g} {units} grid cells ({n:,} cells in the study area), queen contiguity "
        "including the cell itself.",
        "Significance uses a normal approximation without multiple-comparison correction; treat 90% "
        "results as exploratory.",
        "Hot spots indicate statistically unusual concentrations of reports, not confirmed conditions.",
    ]
    return AnalysisResult(to_wgs84(out), notes, {"hot_cells_95": hot, "study_cells": n})


def dbscan_clusters(
    points: gpd.GeoDataFrame,
    projected_crs: str,
    radius: float = 300.0,
    units: str = "ft",
    min_points: int = 5,
    category_column: str | None = None,
    date_column: str | None = None,
    label_columns: list[str] | None = None,
    max_pairs: int = 8_000_000,
) -> AnalysisResult:
    """Density-based clusters (DBSCAN semantics) using an STRtree neighbor search.

    Returns one polygon per cluster (convex hull of members buffered by radius/2) with
    member counts, distinct locations, dominant categories and the date span.
    """
    if len(points) == 0:
        raise SpatialError("No features to cluster")
    pp = to_projected(points, projected_crs).reset_index(drop=True)
    geoms = pp.geometry.representative_point().values
    eps = to_crs_units(radius, units, projected_crs)
    tree = STRtree(geoms)
    pairs = tree.query(geoms, predicate="dwithin", distance=eps)
    if pairs.shape[1] > max_pairs:
        raise SpatialError("Too many neighbor pairs; reduce the radius or filter the data first")
    n = len(geoms)
    counts = np.bincount(pairs[0], minlength=n)
    core = counts >= min_points
    edge_mask = core[pairs[0]] & core[pairs[1]]
    a, b = pairs[0][edge_mask], pairs[1][edge_mask]
    labels = np.where(core, np.arange(n), -1)
    while len(a):
        new = labels.copy()
        np.minimum.at(new, a, labels[b])
        np.minimum.at(new, b, labels[a])
        if np.array_equal(new, labels):
            break
        labels = new
    border_mask = (~core[pairs[0]]) & core[pairs[1]]
    bi, bc = pairs[0][border_mask], pairs[1][border_mask]
    order = np.argsort(bi, kind="stable")
    bi, bc = bi[order], bc[order]
    first = np.unique(bi, return_index=True)[1]
    labels[bi[first]] = labels[bc[first]]
    uniq = [u for u in np.unique(labels) if u >= 0]
    remap = {u: i + 1 for i, u in enumerate(uniq)}
    cluster_ids = np.array([remap.get(v, 0) for v in labels])
    pp["cluster_id"] = cluster_ids

    rows = []
    xy = shapely.get_coordinates(geoms)
    for cid in range(1, len(uniq) + 1):
        idx = np.nonzero(cluster_ids == cid)[0]
        member_geoms = geoms[idx]
        hull = shapely.convex_hull(shapely.multipoints(member_geoms)).buffer(eps / 2)
        locs = len({(round(x / 3), round(y / 3)) for x, y in xy[idx]})  # ~3 ft precision
        row: dict[str, Any] = {
            "cluster_id": cid,
            "count": len(idx),
            "distinct_locations": locs,
            "geometry": hull,
        }
        sub = pp.iloc[idx]
        if category_column and category_column in sub:
            vc = sub[category_column].astype("string").value_counts()
            row["top_category"] = str(vc.index[0]) if len(vc) else None
            row["categories"] = "; ".join(f"{k} ({v})" for k, v in vc.head(4).items())
        if date_column and date_column in sub:
            d = pd.to_datetime(sub[date_column], errors="coerce")
            if d.notna().any():
                row["first_date"] = d.min().strftime("%Y-%m-%d")
                row["last_date"] = d.max().strftime("%Y-%m-%d")
                row["span_days"] = int((d.max() - d.min()).days)
        for lc in label_columns or []:
            if lc in sub:
                vc = sub[lc].astype("string").dropna().value_counts()
                row[lc] = str(vc.index[0]) if len(vc) else None
        rows.append(row)
    if rows:
        clusters = gpd.GeoDataFrame(rows, geometry="geometry", crs=projected_crs)
        clusters = clusters.sort_values("count", ascending=False).reset_index(drop=True)
        clusters["rank"] = np.arange(1, len(clusters) + 1)
    else:
        clusters = gpd.GeoDataFrame({"cluster_id": [], "count": []}, geometry=[], crs=projected_crs)
    clustered = int((cluster_ids > 0).sum())
    notes = [
        f"DBSCAN-style clustering: a cluster needs at least {min_points} reports within {radius:g} {units} of a "
        f"core report ({len(uniq)} clusters, {clustered:,} of {n:,} reports clustered).",
        "Cluster outlines are convex hulls buffered by half the search radius; they summarize report locations "
        "and are not verified site boundaries.",
    ]
    return AnalysisResult(
        to_wgs84(clusters),
        notes,
        {"clusters": len(uniq), "clustered_points": clustered, "labeled_points": to_wgs84(pp)},
    )


FREQ = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}


def temporal_patterns(
    df: pd.DataFrame,
    date_column: str,
    freq: Literal["day", "week", "month", "quarter", "year"] = "month",
    group_by: str | None = None,
    top_groups: int = 8,
) -> AnalysisResult:
    if date_column not in df:
        raise SpatialError(f"Date column '{date_column}' not found")
    d = pd.to_datetime(df[date_column], errors="coerce")
    valid = d.notna()
    if not valid.any():
        raise SpatialError("No valid dates to analyze")
    frame = pd.DataFrame({"period": d[valid].dt.to_period(FREQ[freq]).dt.start_time})
    if group_by:
        if group_by not in df:
            raise SpatialError(f"Group column '{group_by}' not found")
        g = df.loc[valid, group_by].astype("string").fillna("(none)")
        top = g.value_counts().head(top_groups).index
        frame["group"] = g.where(g.isin(top), "Other").values
    else:
        frame["group"] = "All"
    wide = frame.groupby(["period", "group"]).size().unstack(fill_value=0).sort_index()
    full_index = pd.period_range(wide.index.min(), wide.index.max(), freq=FREQ[freq]).start_time
    wide = wide.reindex(full_index, fill_value=0)
    wide.index.name = "period"
    long = wide.reset_index().melt(id_vars="period", var_name="group", value_name="count")

    dmin, dmax = d[valid].min(), d[valid].max()
    years = d[valid].dt.year
    full_years = _covered_years(d[valid])
    stats_rows = []
    for grp in wide.columns:
        series = wide[grp].values.astype(float)
        slope = float(np.polyfit(np.arange(len(series)), series, 1)[0]) if len(series) > 1 else 0.0
        row = {
            "group": grp,
            "total": int(series.sum()),
            "mean_per_period": round(float(series.mean()), 2),
            "trend_per_period": round(slope, 3),
            "peak_period": str(wide[grp].idxmax())[:10],
        }
        if len(full_years) >= 2:
            gmask = (frame["group"] == grp).values
            yc = years[valid][gmask].value_counts()
            a, b = int(yc.get(full_years[0], 0)), int(yc.get(full_years[-1], 0))
            row[f"count_{full_years[0]}"] = a
            row[f"count_{full_years[-1]}"] = b
            row["pct_change_full_years"] = round((b - a) / a * 100, 1) if a else None
        stats_rows.append(row)
    stats = pd.DataFrame(stats_rows).sort_values("total", ascending=False)
    notes = [f"Counts per {freq} by {group_by or 'all records'} from {dmin:%Y-%m-%d} to {dmax:%Y-%m-%d}."]
    if group_by and frame["group"].eq("Other").any():
        notes.append(f"Groups beyond the top {top_groups} are combined as 'Other'.")
    if dmax < pd.Timestamp(year=dmax.year, month=12, day=1):
        notes.append(f"The final period ({dmax.year}) is partial; compare full periods only.")
    notes.append("Trend is the least-squares slope of counts per period (reports, not verified incidents).")
    return AnalysisResult(
        long,
        notes,
        {"summary": stats, "full_years": full_years, "date_min": str(dmin), "date_max": str(dmax)},
    )


def _covered_years(d: pd.Series) -> list[int]:
    """Years whose January and December both contain data (robust for sparse, filtered subsets)."""
    years = []
    for y in sorted(d.dt.year.unique()):
        dy = d[d.dt.year == y]
        if dy.min() <= pd.Timestamp(year=int(y), month=1, day=31, hour=23) and dy.max() >= pd.Timestamp(
            year=int(y), month=12, day=1
        ):
            years.append(int(y))
    return years


def default_periods(
    dates: pd.Series,
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], tuple[pd.Timestamp, pd.Timestamp]]:
    d = pd.to_datetime(dates, errors="coerce").dropna()
    if d.empty:
        raise SpatialError("No valid dates")
    dmin, dmax = d.min(), d.max()
    full = _covered_years(d)
    if len(full) >= 2:
        a, b = full[0], full[-1]
        return (
            (
                pd.Timestamp(year=a, month=1, day=1),
                pd.Timestamp(year=a, month=12, day=31, hour=23, minute=59),
            ),
            (
                pd.Timestamp(year=b, month=1, day=1),
                pd.Timestamp(year=b, month=12, day=31, hour=23, minute=59),
            ),
        )
    mid = dmin + (dmax - dmin) / 2
    return (dmin, mid), (mid, dmax)


def compare_periods(
    points: gpd.GeoDataFrame,
    areas: gpd.GeoDataFrame,
    date_column: str,
    label_column: str | None,
    period_a: tuple[Any, Any] | None = None,
    period_b: tuple[Any, Any] | None = None,
    min_baseline: int = 5,
    top_n: int | None = None,
) -> AnalysisResult:
    if date_column not in points:
        raise SpatialError(f"Date column '{date_column}' not found")
    if period_a is None or period_b is None:
        period_a, period_b = default_periods(points[date_column])
    a0, a1 = (pd.Timestamp(x) for x in period_a)
    b0, b1 = (pd.Timestamp(x) for x in period_b)
    pts = to_wgs84(points)
    polys = to_wgs84(areas).reset_index(drop=True)
    polys["_area_id"] = np.arange(len(polys))
    joined = gpd.sjoin(pts, polys[["_area_id", "geometry"]], how="inner", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    d = pd.to_datetime(joined[date_column], errors="coerce")
    ca = joined[(d >= a0) & (d <= a1)].groupby("_area_id").size()
    cb = joined[(d >= b0) & (d <= b1)].groupby("_area_id").size()
    out = polys.copy()
    out["count_a"] = ca.reindex(out["_area_id"]).fillna(0).astype(int).values
    out["count_b"] = cb.reindex(out["_area_id"]).fillna(0).astype(int).values
    out["change"] = out["count_b"] - out["count_a"]
    out["pct_change"] = np.where(
        out["count_a"] >= min_baseline,
        ((out["count_b"] - out["count_a"]) / out["count_a"].clip(lower=1) * 100).round(1),
        np.nan,
    )
    out["low_baseline"] = out["count_a"] < min_baseline
    out = out.sort_values(["change", "count_b"], ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    if top_n:
        out = out.head(top_n)
    la = f"{a0:%Y-%m-%d} to {a1:%Y-%m-%d}"
    lb = f"{b0:%Y-%m-%d} to {b1:%Y-%m-%d}"
    notes = [
        f"Period A: {la}; Period B: {lb}. Change = count B - count A per area.",
        f"Percent change is omitted for areas with fewer than {min_baseline} reports in period A.",
        "Changes in report counts can reflect reporting behavior, outreach, or service changes, not only "
        "conditions on the ground.",
    ]
    return AnalysisResult(
        gpd.GeoDataFrame(out.drop(columns=["_area_id"]), geometry="geometry", crs=polys.crs),
        notes,
        {
            "period_a": la,
            "period_b": lb,
            "total_a": int(out["count_a"].sum()),
            "total_b": int(out["count_b"].sum()),
        },
    )


def summarize_frame(df: pd.DataFrame, group_by: str | None = None, top_n: int = 8) -> dict[str, Any]:
    """Compact, JSON-friendly description of a result set (used for narratives and the UI)."""
    out: dict[str, Any] = {"rows": int(len(df)), "columns": [str(c) for c in df.columns if c != "geometry"]}
    if isinstance(df, gpd.GeoDataFrame) and len(df):
        types = df.geometry.geom_type.value_counts().to_dict()
        out["geometry_types"] = {str(k): int(v) for k, v in types.items()}
        b = to_wgs84(df).total_bounds
        if np.all(np.isfinite(b)):
            out["bounds"] = [round(float(v), 5) for v in b]
    cats: dict[str, Any] = {}
    nums: dict[str, Any] = {}
    dates: dict[str, Any] = {}
    for c in df.columns:
        if c == "geometry":
            continue
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            if s.notna().any():
                dates[str(c)] = {"min": str(s.min())[:19], "max": str(s.max())[:19]}
        elif pd.api.types.is_bool_dtype(s):
            continue
        elif pd.api.types.is_numeric_dtype(s):
            if s.notna().any():
                nums[str(c)] = {
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "mean": round(float(s.mean()), 3),
                    "sum": float(s.sum()),
                }
        else:
            try:
                nunique = s.nunique(dropna=True)
            except TypeError:
                continue
            if 0 < nunique <= 60:
                vc = s.astype("string").value_counts().head(top_n)
                cats[str(c)] = {str(k): int(v) for k, v in vc.items()}
    out["categorical"] = cats
    out["numeric"] = nums
    out["temporal"] = dates
    if group_by and group_by in df:
        out["group_counts"] = {
            str(k): int(v) for k, v in df[group_by].astype("string").value_counts().head(top_n).items()
        }
    return out
