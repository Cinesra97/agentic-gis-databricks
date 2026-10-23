"""Standardized map-layer schema and layer builder.

The agent emits :class:`MapLayerSpec` objects (pure JSON data) that reference result
artifacts. Renderers (Folium, PyDeck) turn specs into maps, so new questions never need
custom UI code. Colors are stored as *roles* (sequential / diverging / categorical / single)
and resolved per basemap theme at render time, using palettes validated for color-vision
deficiency and contrast against light and dark map surfaces.

This module must not import Streamlit (it is used by the agent and tests).
"""

from __future__ import annotations

import json
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

RenderKind = Literal["points", "cluster", "heatmap", "choropleth", "lines", "polygons"]
Theme = Literal["light", "dark"]

# ---- Palettes (validated with the dataviz skill's validate_palette.js) -------------------
# Categorical: first three slots validate all-pairs (maps) in both modes; further values fold to "Other".
CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a"],
    "dark": ["#3987e5", "#d95926", "#199e70"],
}
OTHER_GRAY = "#898781"
# Sequential (blue), low -> high. On dark basemaps low values recede toward the dark surface.
SEQUENTIAL = {
    "light": ["#b7d3f6", "#6da7ec", "#3987e5", "#1c5cab", "#0d366b"],
    "dark": ["#104281", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"],
}
# Diverging: blue (negative / cold) <- neutral gray -> red (positive / hot); 3 steps per arm.
DIVERGING = {
    "light": {
        "neg": ["#104281", "#3987e5", "#86b6ef"],
        "mid": "#d6d5ce",
        "pos": ["#f2b8b5", "#e66767", "#c22f2f"],
    },
    "dark": {
        "neg": ["#b7d3f6", "#3987e5", "#184f95"],
        "mid": "#4a4a46",
        "pos": ["#9e2a2a", "#e34948", "#f2a19d"],
    },
}
SINGLE = {"light": "#2a78d6", "dark": "#3987e5"}
CONTEXT = {"light": "#52514e", "dark": "#c3c2b7"}

HOTSPOT_ORDER = [
    "Hot spot (99%)",
    "Hot spot (95%)",
    "Hot spot (90%)",
    "Not significant",
    "Cold spot (90%)",
    "Cold spot (95%)",
    "Cold spot (99%)",
]


class LegendItem(BaseModel):
    label: str
    role: str = Field(
        description="Palette role key resolved at render time, e.g. 'seq:2', 'cat:0', 'div:pos:2'"
    )


class LayerStyle(BaseModel):
    color_role: Literal["single", "context", "categorical", "sequential", "diverging", "hotspot"] = "single"
    color_by: str | None = None
    breaks: list[float] = Field(default_factory=list, description="Upper class bounds for numeric classes")
    categories: list[str] = Field(default_factory=list, description="Ordered category values (max 3 + Other)")
    radius: float = 5.0
    weight: float = 2.0
    opacity: float = 0.9
    fill_opacity: float = 0.6


class MapLayerSpec(BaseModel):
    """Renderer-independent description of one map layer (schema version 1.0)."""

    schema_version: Literal["1.0"] = "1.0"
    id: str
    title: str
    artifact_id: str
    render: RenderKind
    geometry_kind: Literal["point", "line", "polygon", "mixed"]
    style: LayerStyle = Field(default_factory=LayerStyle)
    tooltip: list[str] = Field(default_factory=list)
    legend_title: str | None = None
    legend: list[LegendItem] = Field(default_factory=list)
    visible: bool = True
    feature_count: int = 0
    bounds: list[float] | None = None
    synthetic: bool = False
    provenance: list[str] = Field(default_factory=list)
    z_index: int = 0
    note: str | None = None


def layer_json_schema() -> dict[str, Any]:
    return MapLayerSpec.model_json_schema()


# ---- color resolution ------------------------------------------------------------------
def resolve_role(role: str, theme: Theme) -> str:
    kind, *rest = role.split(":")
    if kind == "single":
        return SINGLE[theme]
    if kind == "context":
        return CONTEXT[theme]
    if kind == "other":
        return OTHER_GRAY
    if kind == "cat":
        i = int(rest[0])
        pal = CATEGORICAL[theme]
        return pal[i] if i < len(pal) else OTHER_GRAY
    if kind == "seq":
        pal = SEQUENTIAL[theme]
        return pal[min(int(rest[0]), len(pal) - 1)]
    if kind == "div":
        side, i = rest[0], int(rest[1]) if len(rest) > 1 else 0
        d = DIVERGING[theme]
        return d["mid"] if side == "mid" else d[side][min(i, 2)]
    return SINGLE[theme]


def _quantile_breaks(values: pd.Series, k: int = 5) -> list[float]:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return []
    qs = np.quantile(v, np.linspace(0, 1, k + 1)[1:])
    if np.allclose(v, np.round(v)):
        qs = np.round(qs)
    return [float(x) for x in np.unique(qs)]


def _fmt(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 10 or float(x).is_integer() else f"{x:,.2f}"


def feature_role(spec: MapLayerSpec, value: Any) -> str:
    """Palette role for a feature attribute value under this layer's style."""
    st = spec.style
    if st.color_role in ("single", "context"):
        return st.color_role
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "other"
    if st.color_role == "categorical":
        try:
            return f"cat:{st.categories.index(str(value))}"
        except ValueError:
            return "other"
    if st.color_role == "hotspot":
        s = str(value)
        if s.startswith("Hot"):
            return "div:pos:" + {"99": "2", "95": "1"}.get(s[10:12], "0")
        if s.startswith("Cold"):
            return "div:neg:" + {"99": "0", "95": "1"}.get(s[11:13], "2")
        return "div:mid"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "other"
    if st.color_role == "sequential":
        for i, b in enumerate(st.breaks):
            if x <= b:
                return f"seq:{i}"
        return f"seq:{max(len(st.breaks) - 1, 0)}"
    if st.color_role == "diverging":
        if x == 0:
            return "div:mid"
        neg = [b for b in st.breaks if b < 0]
        pos = [b for b in st.breaks if b > 0]
        if x > 0:
            for i, b in enumerate(pos):
                if x <= b:
                    return f"div:pos:{i}"
            return "div:pos:2"
        for i, b in enumerate(sorted(neg)):
            if x <= b:
                return f"div:neg:{i}"
        return "div:neg:2"
    return "single"


def _diverging_breaks(values: pd.Series) -> list[float]:
    v = pd.to_numeric(values, errors="coerce").dropna()
    pos, neg = v[v > 0], v[v < 0]
    out: list[float] = []
    if len(neg):
        qn = np.unique(np.quantile(neg, [1 / 3, 2 / 3, 1.0]))
        out += [float(x) for x in qn]  # ascending; most negative class first
    if len(pos):
        qp = np.unique(np.quantile(pos, [1 / 3, 2 / 3, 1.0]))
        out += [float(x) for x in qp]
    return out


def build_layer_spec(
    artifact: Any,
    layer_id: str,
    title: str | None = None,
    style: str = "auto",
    color_by: str | None = None,
    tooltip: list[str] | None = None,
    context: bool = False,
    visible: bool = True,
    z_index: int = 0,
    max_render_features: int = 6000,
) -> MapLayerSpec:
    """Choose a sensible rendering for an artifact and compute its legend/classes."""
    if not artifact.is_spatial or artifact.rows == 0:
        raise ValueError(f"Artifact {artifact.id} has no spatial features to map")
    gdf = artifact.data
    kind = artifact.geometry_kind or "mixed"
    cols = [c for c in artifact.columns() if not c.startswith("_")]
    if style == "auto":
        if kind == "point":
            style = "cluster" if artifact.rows > 1500 else "points"
        elif kind == "line":
            style = "lines"
        else:
            style = "choropleth" if (color_by or _default_value_column(cols)) and not context else "polygons"
    if style == "heatmap" and kind != "point":
        style = "polygons" if kind == "polygon" else "lines"

    st = LayerStyle()
    legend: list[LegendItem] = []
    legend_title = None
    if context:
        st.color_role = "context"
        st.fill_opacity = 0.12
        st.weight = 1.5 if kind != "line" else 2.5
        legend = [LegendItem(label=title or artifact.title, role="context")]
    else:
        if color_by is None and style in ("choropleth", "polygons", "lines", "points"):
            color_by = _default_value_column(cols) if style in ("choropleth", "lines") else None
        if color_by and color_by in gdf.columns:
            series = gdf[color_by]
            legend_title = color_by.replace("_", " ")
            if color_by == "hotspot_class":
                st.color_role = "hotspot"
                present = [c for c in HOTSPOT_ORDER if c in set(series.astype(str))]
                legend = [
                    LegendItem(
                        label=c,
                        role=feature_role(
                            MapLayerSpec(
                                id="x",
                                title="x",
                                artifact_id="x",
                                render="polygons",
                                geometry_kind="polygon",
                                style=st,
                            ),
                            c,
                        ),
                    )
                    for c in present
                ]
                st.fill_opacity = 0.7
            elif pd.api.types.is_numeric_dtype(series) and series.nunique() > 3:
                vals = pd.to_numeric(series, errors="coerce")
                if color_by in ("change", "pct_change", "gi_z") or (vals.min() < 0 < vals.max()):
                    st.color_role = "diverging"
                    st.breaks = _diverging_breaks(vals)
                    neg = [b for b in st.breaks if b < 0]
                    pos = [b for b in st.breaks if b > 0]
                    prev = float(vals.min())
                    for i, b in enumerate(neg):
                        legend.append(LegendItem(label=f"{_fmt(prev)} to {_fmt(b)}", role=f"div:neg:{i}"))
                        prev = b
                    legend.append(LegendItem(label="0 (no change)", role="div:mid"))
                    prev = 0.0
                    for i, b in enumerate(pos):
                        legend.append(LegendItem(label=f"{_fmt(prev)} to {_fmt(b)}", role=f"div:pos:{i}"))
                        prev = b
                else:
                    st.color_role = "sequential"
                    st.breaks = _quantile_breaks(vals)
                    lo = float(vals.min())
                    for i, b in enumerate(st.breaks):
                        legend.append(LegendItem(label=f"{_fmt(lo)} – {_fmt(b)}", role=f"seq:{i}"))
                        lo = b
            else:
                st.color_role = "categorical"
                vc = series.astype("string").value_counts()
                st.categories = [str(c) for c in vc.index[:3]]
                legend = [LegendItem(label=c, role=f"cat:{i}") for i, c in enumerate(st.categories)]
                if len(vc) > 3:
                    legend.append(LegendItem(label="Other", role="other"))
        else:
            color_by = None
            legend = [LegendItem(label=title or artifact.title, role="single")]
        if style in ("choropleth", "polygons"):
            st.fill_opacity = 0.65 if st.color_role != "single" else 0.35
    st.color_by = color_by
    note = None
    if style in ("points", "cluster") and artifact.rows > max_render_features:
        note = (
            f"{artifact.rows:,} features; the map draws a {max_render_features:,}-feature sample. "
            "The results table and exports contain all rows."
        )
    if tooltip is None:
        tooltip = _default_tooltip(cols, color_by)
    else:
        tooltip = [c for c in tooltip if c in cols][:8]
    return MapLayerSpec(
        id=layer_id,
        title=(title or artifact.title) + (" (synthetic)" if artifact.synthetic else ""),
        artifact_id=artifact.id,
        render=style,  # type: ignore[arg-type]
        geometry_kind=kind,  # type: ignore[arg-type]
        style=st,
        tooltip=tooltip,
        legend_title=legend_title,
        legend=legend,
        visible=visible,
        feature_count=artifact.rows,
        bounds=artifact.bounds,
        synthetic=artifact.synthetic,
        provenance=list(artifact.datasets),
        z_index=z_index,
        note=note,
    )


_VALUE_PREFERENCE = ["change", "value", "count", "match_count", "gi_z", "per_1000ft"]
_TOOLTIP_PREFERENCE = [
    "label",
    "neighborhood",
    "boro_name",
    "street_name",
    "complaint_type",
    "descriptor",
    "created_date",
    "incident_address",
    "borough",
    "cd_code",
    "modzcta",
    "value",
    "count",
    "count_a",
    "count_b",
    "change",
    "pct_change",
    "hotspot_class",
    "gi_z",
    "p_value",
    "distinct_locations",
    "first_date",
    "last_date",
    "top_category",
    "categories",
    "condition",
    "basin_id",
    "lot_id",
    "industrial_area",
    "distance_ft",
    "nearest_match_ft",
    "match_count",
    "per_1000ft",
    "segment_id",
    "status",
    "unique_key",
]


def _default_value_column(cols: list[str]) -> str | None:
    for c in _VALUE_PREFERENCE:
        if c in cols:
            return c
    return None


def _default_tooltip(cols: list[str], color_by: str | None) -> list[str]:
    out = [c for c in _TOOLTIP_PREFERENCE if c in cols]
    if color_by and color_by in cols and color_by not in out:
        out.insert(0, color_by)
    if not out:
        out = cols[:6]
    return out[:8]


def schema_json() -> str:
    return json.dumps(layer_json_schema(), indent=2)
