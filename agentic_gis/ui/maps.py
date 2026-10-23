"""Map renderers driven by :class:`~agentic_gis.ui.layers.MapLayerSpec`.

* :func:`render_folium` -- Leaflet via Folium (default): layer control, clustering, heat maps,
  tooltips/popups, legends, zoom-to-results. Works with any XYZ raster basemap.
* :func:`render_pydeck` -- deck.gl via PyDeck (MapLibre basemap style built from the same XYZ list).

Basemaps are free XYZ services that need no API key. Respect each provider's usage policy
(see docs/LICENSES.md); organizations can point ``AGIS_CUSTOM_BASEMAP_URL`` at an internal
tile server instead.
"""

from __future__ import annotations

import html
import json
import math
from typing import Any

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
from branca.element import Element
from folium.plugins import FastMarkerCluster, HeatMap

from agentic_gis.spatial.geometry import to_wgs84
from agentic_gis.ui.layers import (
    SEQUENTIAL,
    MapLayerSpec,
    feature_role,
    resolve_role,
    uses_ordinal,
)

NYC_CENTER = (40.7128, -73.95)

BASEMAPS: dict[str, dict[str, Any]] = {
    "CARTO Dark Matter": {
        "url": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        "xyz": "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        "attr": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors '
        '&copy; <a href="https://carto.com/attributions">CARTO</a>',
        "theme": "dark",
        "max_zoom": 20,
        "subdomains": "abcd",
    },
    "CARTO Positron": {
        "url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        "xyz": "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        "attr": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors '
        '&copy; <a href="https://carto.com/attributions">CARTO</a>',
        "theme": "light",
        "max_zoom": 20,
        "subdomains": "abcd",
    },
    "OpenStreetMap": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "xyz": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attr": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        "theme": "light",
        "max_zoom": 19,
    },
    "NYC Basemap (NYC OTI)": {
        "url": "https://maps.nyc.gov/xyz/1.0.0/carto/basemap/{z}/{x}/{y}.jpg",
        "xyz": "https://maps.nyc.gov/xyz/1.0.0/carto/basemap/{z}/{x}/{y}.jpg",
        "attr": "Basemap &copy; NYC Office of Technology and Innovation",
        "theme": "light",
        "max_zoom": 21,
    },
    "No basemap (dark)": {"url": None, "attr": "", "theme": "dark", "max_zoom": 22},
    "No basemap (light)": {"url": None, "attr": "", "theme": "light", "max_zoom": 22},
}

SURFACE = {"dark": "#1a1a19", "light": "#fcfcfb"}
INK = {"dark": "#f0efec", "light": "#0b0b0b"}

# Vendored Leaflet assets (scripts/vendor_map_assets.py), served by Streamlit static serving.
STATIC_BASE = "/app/static/vendor"
LOCAL_ASSETS = {
    "leaflet.js": f"{STATIC_BASE}/leaflet/leaflet.js",
    "leaflet.css": f"{STATIC_BASE}/leaflet/leaflet.css",
    "jquery": f"{STATIC_BASE}/jquery/jquery.min.js",
    "leaflet.markercluster.js": f"{STATIC_BASE}/leaflet.markercluster/leaflet.markercluster.js",
    "MarkerCluster.css": f"{STATIC_BASE}/leaflet.markercluster/MarkerCluster.css",
    "MarkerCluster.Default.css": f"{STATIC_BASE}/leaflet.markercluster/MarkerCluster.Default.css",
    "leaflet-heat.js": f"{STATIC_BASE}/leaflet.heat/leaflet-heat.js",
    "leaflet_heat.min.js": f"{STATIC_BASE}/leaflet.heat/leaflet-heat.js",
}


def _local_url(url: str) -> str | None:
    name = url.rsplit("/", 1)[-1]
    if name.startswith("jquery") and name.endswith(".js"):
        return LOCAL_ASSETS["jquery"]
    return LOCAL_ASSETS.get(name)


def _walk(elem: Any) -> Any:
    yield elem
    for child in getattr(elem, "_children", {}).values():
        yield from _walk(child)


def localize_assets(m: folium.Map) -> folium.Map:
    """Point Folium's JS/CSS dependencies at vendored copies (no CDN access needed).

    Bootstrap / Font Awesome / awesome-markers are not used by these maps and are dropped.
    """
    for elem in _walk(m):
        for attr in ("default_js", "default_css"):
            items = getattr(elem, attr, None)
            if not items:
                continue
            local = []
            for key, url in items:
                lu = _local_url(url) or LOCAL_ASSETS.get(key)
                if lu:
                    local.append((key, lu))
            setattr(elem, attr, local)
    return m


def available_basemaps(custom_url: str | None = None, custom_attr: str | None = None) -> dict[str, dict]:
    maps = dict(BASEMAPS)
    if custom_url:
        maps = {
            "Custom (configured)": {
                "url": custom_url,
                "xyz": custom_url,
                "attr": custom_attr or "Custom basemap",
                "theme": "light",
                "max_zoom": 21,
            },
            **maps,
        }
    return maps


# ---------------------------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------------------------
def apply_filters(
    gdf: gpd.GeoDataFrame,
    boroughs: list[str] | None = None,
    area_geoms: gpd.GeoDataFrame | None = None,
    borough_geoms: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Client-side map filters (borough and/or neighborhood)."""
    out = gdf
    if boroughs:
        up = [b.upper() for b in boroughs]
        if "borough" in out.columns and out["borough"].notna().any():
            out = out[out["borough"].astype(str).str.upper().isin(up)]
        elif borough_geoms is not None and "borough" in borough_geoms:
            sel = borough_geoms[borough_geoms["borough"].isin(up)]
            if len(sel):
                out = out[out.geometry.intersects(sel.union_all())]
    if area_geoms is not None and len(area_geoms):
        out = out[out.geometry.intersects(area_geoms.union_all())]
    return out


def _jsonable_frame(gdf: gpd.GeoDataFrame, keep: list[str]) -> gpd.GeoDataFrame:
    cols = [c for c in dict.fromkeys(keep) if c in gdf.columns and c != "geometry"]
    out = gdf[cols + ["geometry"]].copy()
    for c in cols:
        s = out[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out[c] = s.dt.strftime("%Y-%m-%d %H:%M").fillna("")
        elif pd.api.types.is_float_dtype(s):
            out[c] = s.round(3).astype(object).where(s.notna(), None)
        elif not (pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s)):
            out[c] = s.astype("string").fillna("").astype(object)
    return out


def marker_radii(g: gpd.GeoDataFrame, source: gpd.GeoDataFrame, spec: MapLayerSpec) -> list[float]:
    """Marker radius per feature: constant, or sqrt-scaled by ``size_by`` within ``size_range``."""
    st = spec.style
    if st.color_role == "context":
        return [2.0] * len(g)
    col = st.size_by
    if not col or col not in source.columns:
        return [st.radius] * len(g)
    vals = g[col] if col in g.columns else source.loc[g.index, col]
    v = pd.to_numeric(vals, errors="coerce").fillna(0).clip(lower=0) ** 0.5
    lo, hi = float(v.min()), float(v.max())
    r0, r1 = st.size_range
    if hi <= lo:
        return [(r0 + r1) / 2] * len(g)
    return [round(r0 + (x - lo) / (hi - lo) * (r1 - r0), 1) for x in v]


def _sample(gdf: gpd.GeoDataFrame, n: int) -> gpd.GeoDataFrame:
    return gdf if len(gdf) <= n else gdf.sample(n, random_state=7)


def _pretty(col: str) -> str:
    return col.replace("_", " ").capitalize()


def layer_frame(spec: MapLayerSpec, gdf: gpd.GeoDataFrame, theme: str, max_features: int) -> gpd.GeoDataFrame:
    g = to_wgs84(gdf)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    if spec.render in ("points", "cluster", "heatmap") and len(g) > max_features:
        g = _sample(g, max_features)
    keep = list(spec.tooltip) + ([spec.style.color_by] if spec.style.color_by else [])
    out = _jsonable_frame(g, keep)
    if spec.style.color_by and spec.style.color_by in g.columns:
        vals = g[spec.style.color_by].tolist()
    else:
        vals = [None] * len(g)
    ordinal = uses_ordinal(spec)
    out["_fill"] = [resolve_role(feature_role(spec, v), theme, ordinal) for v in vals]  # type: ignore[arg-type]
    return out


# ---------------------------------------------------------------------------------------------
# Folium
# ---------------------------------------------------------------------------------------------
_LEGEND_CSS = """
<style>
.agis-legend {position:absolute; z-index:1000; bottom:24px; right:10px; max-width:270px; max-height:46%;
  overflow:auto; padding:8px 10px; border-radius:6px; font:12px/1.35 system-ui,-apple-system,"Segoe UI",sans-serif;
  box-shadow:0 1px 4px rgba(0,0,0,.35);}
.agis-legend.dark {background:rgba(26,26,25,.92); color:#f0efec; border:1px solid rgba(255,255,255,.10);}
.agis-legend.light {background:rgba(252,252,251,.94); color:#0b0b0b; border:1px solid rgba(11,11,11,.10);}
.agis-legend h4 {margin:6px 0 3px; font-size:12px; font-weight:600;}
.agis-legend .sub {opacity:.75; font-size:11px; margin-bottom:2px;}
.agis-legend .row {display:flex; align-items:center; gap:6px; margin:2px 0;}
.agis-legend .sw {width:14px; height:12px; border-radius:2px; flex:0 0 auto;}
.agis-legend .ln {width:16px; height:3px; border-radius:2px; flex:0 0 auto;}
.agis-legend .dot {width:10px; height:10px; border-radius:50%; flex:0 0 auto;}
.agis-badge {position:absolute; z-index:1000; bottom:28px; left:10px; padding:3px 8px; border-radius:4px;
  font:600 11px system-ui,sans-serif; letter-spacing:.04em; background:#fab219; color:#0b0b0b;}
.leaflet-container {font: 12px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;}
.leaflet-control-layers-expanded {max-width: 300px; font-size: 12px;}
.leaflet-tooltip {font-size: 12px; max-width: 320px; white-space: normal;}
</style>
"""
_DARK_CONTROLS_CSS = """
<style>
.leaflet-control-layers, .leaflet-bar a, .leaflet-control-scale-line, .leaflet-tooltip, .leaflet-popup-content-wrapper,
.leaflet-popup-tip {background:#1a1a19 !important; color:#f0efec !important; border-color:rgba(255,255,255,.18) !important;}
.leaflet-control-attribution {background:rgba(26,26,25,.75) !important; color:#c3c2b7 !important;}
.leaflet-control-attribution a {color:#86b6ef !important;}
.leaflet-control-layers-toggle {filter: invert(1) brightness(1.4);}
</style>
"""


def legend_html(specs: list[MapLayerSpec], theme: str, counts: dict[str, int]) -> str:
    blocks = []
    for s in specs:
        if not s.visible:
            continue
        shape = "dot" if s.geometry_kind == "point" else ("ln" if s.geometry_kind == "line" else "sw")
        items = []
        if s.render == "heatmap":
            ramp = SEQUENTIAL[theme]  # type: ignore[index]
            grad = ", ".join(ramp)
            items.append(
                f'<div class="row"><span class="sw" style="width:90px;background:linear-gradient(90deg,{grad})">'
                "</span><span>fewer → more reports</span></div>"
            )
        for it in s.legend:
            color = resolve_role(it.role, theme, uses_ordinal(s))  # type: ignore[arg-type]
            items.append(
                f'<div class="row"><span class="{shape}" style="background:{color}"></span>'
                f"<span>{html.escape(it.label)}</span></div>"
            )
        sub = s.legend_title or ""
        n = counts.get(s.id, s.feature_count)
        sub = f"{html.escape(_pretty(sub))} · {n:,} features" if sub else f"{n:,} features"
        blocks.append(f"<h4>{html.escape(s.title)}</h4><div class='sub'>{sub}</div>{''.join(items)}")
    if not blocks:
        return ""
    return f'<div class="agis-legend {theme}">{"".join(blocks)}</div>'


def _root(m: folium.Map) -> Any:
    return m.get_root()


def _draw_order(spec: MapLayerSpec) -> tuple[int, int]:
    """Heat maps and context layers first (bottom), analytical results on top."""
    rank = 0 if spec.render == "heatmap" else (1 if spec.style.color_role == "context" else 2)
    return rank, spec.z_index


def _fit_bounds(m: folium.Map, bounds: list[list[float]]) -> None:
    b = [x for x in bounds if x and all(np.isfinite(x))]
    if not b:
        m.fit_bounds([[40.49, -74.27], [40.92, -73.68]])
        return
    minx, miny = min(x[0] for x in b), min(x[1] for x in b)
    maxx, maxy = max(x[2] for x in b), max(x[3] for x in b)
    pad = 0.002
    m.fit_bounds([[miny - pad, minx - pad], [maxy + pad, maxx + pad]])


def render_folium(
    specs: list[MapLayerSpec],
    frames: dict[str, gpd.GeoDataFrame],
    basemap: str,
    basemaps: dict[str, dict] | None = None,
    max_features: int = 6000,
    synthetic: bool = False,
    local_assets: bool = True,
) -> folium.Map:
    """Build a Folium map. ``frames`` maps layer id -> (already filtered) GeoDataFrame."""
    bm = (basemaps or BASEMAPS).get(basemap) or BASEMAPS["CARTO Dark Matter"]
    theme = bm["theme"]
    m = folium.Map(
        location=NYC_CENTER,
        zoom_start=11,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
        max_zoom=bm["max_zoom"],
    )
    if bm.get("url"):
        kw = {"subdomains": bm["subdomains"]} if bm.get("subdomains") else {}
        folium.TileLayer(
            tiles=bm["url"], attr=bm["attr"], name=basemap, max_zoom=bm["max_zoom"], control=False, **kw
        ).add_to(m)
    else:
        _root(m).header.add_child(
            Element(f"<style>.leaflet-container{{background:{SURFACE[theme]} !important;}}</style>")
        )
    _root(m).header.add_child(Element(_LEGEND_CSS))
    if theme == "dark":
        _root(m).header.add_child(Element(_DARK_CONTROLS_CSS))
    surface = SURFACE[theme]
    bounds: list[list[float]] = []
    counts: dict[str, int] = {}
    # Context (reference) layers are drawn first so analytical results stay on top.
    for spec in sorted(specs, key=_draw_order):
        gdf = frames.get(spec.id)
        if gdf is None or len(gdf) == 0:
            counts[spec.id] = 0
            continue
        counts[spec.id] = len(gdf)
        g = layer_frame(spec, gdf, theme, max_features)
        if spec.visible:
            b = g.total_bounds
            bounds.append([float(x) for x in b])
        fg = folium.FeatureGroup(name=html.escape(spec.title), show=spec.visible)
        tip_fields = [c for c in spec.tooltip if c in g.columns]
        tooltip = (
            folium.GeoJsonTooltip(
                fields=tip_fields, aliases=[_pretty(c) + ":" for c in tip_fields], sticky=True
            )
            if tip_fields
            else None
        )
        popup = (
            folium.GeoJsonPopup(fields=tip_fields, aliases=[_pretty(c) for c in tip_fields], max_width=320)
            if tip_fields
            else None
        )
        if spec.render == "heatmap":
            pts = g.geometry.representative_point()
            heat = [[float(p.y), float(p.x)] for p in pts]
            ramp = SEQUENTIAL[theme]  # type: ignore[index]
            gradient = {0.25: ramp[1], 0.5: ramp[2], 0.75: ramp[3], 1.0: ramp[4]}
            HeatMap(heat, radius=14, blur=18, min_opacity=0.25, gradient=gradient).add_to(fg)
        elif spec.render == "cluster":
            pts = g.geometry.representative_point()
            labels = []
            for _, row in g.iterrows():
                parts = [
                    f"<b>{html.escape(_pretty(c))}:</b> {html.escape(str(row[c]))}" for c in tip_fields[:5]
                ]
                labels.append("<br>".join(parts))
            data: list[list[Any]] = [
                [float(p.y), float(p.x), lab, col]
                for p, lab, col in zip(pts, labels, g["_fill"], strict=False)
            ]
            callback = (
                "function (row) { var m = L.circleMarker(new L.LatLng(row[0], row[1]), {radius: 5, "
                f"color: '{surface}', weight: 1, fillColor: row[3], fillOpacity: 0.9}});"
                " if (row[2]) { m.bindTooltip(row[2], {sticky: true}); } return m; }"
            )
            FastMarkerCluster(
                data, callback=callback, options={"maxClusterRadius": 45, "disableClusteringAtZoom": 16}
            ).add_to(fg)
        elif spec.geometry_kind == "point" or spec.render == "points":
            is_ctx = spec.style.color_role == "context"
            if spec.geometry_kind != "point":
                g = g.set_geometry(g.geometry.representative_point())
            g["_r"] = marker_radii(g, gdf, spec)
            ring = surface
            opacity = 0.9 if not is_ctx else 0.3
            weight = 1 if not is_ctx else 0
            folium.GeoJson(
                json.loads(g.to_json(drop_id=True)),
                marker=folium.CircleMarker(radius=spec.style.radius, fill=True),
                style_function=lambda f, ring=ring, opacity=opacity, weight=weight: {
                    "fillColor": f["properties"]["_fill"],
                    "color": ring,
                    "weight": weight,
                    "fillOpacity": opacity,
                    "radius": f["properties"]["_r"],
                },
                tooltip=tooltip,
                popup=popup,
                name=spec.title,
            ).add_to(fg)
        else:
            is_line = spec.geometry_kind == "line"
            st = spec.style

            def style(f: dict, is_line: bool = is_line, st: Any = st) -> dict:
                fill = f["properties"]["_fill"]
                if is_line:
                    return {
                        "color": fill,
                        "weight": 4 if st.color_role != "context" else 2.5,
                        "opacity": 0.95,
                    }
                return {
                    "fillColor": fill,
                    "color": surface if st.color_role != "context" else fill,
                    "weight": 1 if st.color_role != "context" else 1.5,
                    "fillOpacity": st.fill_opacity,
                    "opacity": 0.9,
                }

            folium.GeoJson(
                json.loads(g.to_json(drop_id=True)),
                style_function=style,
                highlight_function=lambda f: {"weight": 3, "color": INK[theme], "fillOpacity": 0.8},
                tooltip=tooltip,
                popup=popup,
                name=spec.title,
                zoom_on_click=False,
            ).add_to(fg)
        fg.add_to(m)
    folium.LayerControl(collapsed=True, position="topright").add_to(m)
    leg = legend_html([s for s in specs if counts.get(s.id, 0)], theme, counts)
    if leg:
        _root(m).html.add_child(Element(leg))
    if synthetic:
        _root(m).html.add_child(Element('<div class="agis-badge">SYNTHETIC DEMO DATA</div>'))
    _fit_bounds(m, bounds)
    return localize_assets(m) if local_assets else m


# ---------------------------------------------------------------------------------------------
# PyDeck (deck.gl)
# ---------------------------------------------------------------------------------------------
def _rgba(hex_color: str, alpha: int = 220) -> list[int]:
    h = hex_color.lstrip("#")
    return [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha]


def _view_state(bounds: list[list[float]]) -> pdk.ViewState:
    b = [x for x in bounds if x and all(np.isfinite(x))]
    if not b:
        return pdk.ViewState(latitude=NYC_CENTER[0], longitude=NYC_CENTER[1], zoom=10)
    minx, miny = min(x[0] for x in b), min(x[1] for x in b)
    maxx, maxy = max(x[2] for x in b), max(x[3] for x in b)
    span = max(maxx - minx, (maxy - miny) * 1.3, 0.002)
    zoom = max(min(math.log2(360 / span) - 1.2, 17), 8)
    return pdk.ViewState(latitude=(miny + maxy) / 2, longitude=(minx + maxx) / 2, zoom=zoom)


def deck_basemap(bm: dict[str, Any], theme: str) -> tuple[str | None, str | None]:
    """(map_provider, map_style) for deck.gl.

    PyDeck accepts named CARTO styles, a MapLibre style URL (``style_url`` in a basemap entry), or no
    basemap. Raster XYZ basemaps (OSM, NYC OTI) are available in the Leaflet renderer.
    """
    if bm.get("style_url"):
        return "maplibre", bm["style_url"]
    if not bm.get("url"):
        return None, None
    return "carto", ("dark" if theme == "dark" else "light")


def render_pydeck(
    specs: list[MapLayerSpec],
    frames: dict[str, gpd.GeoDataFrame],
    basemap: str,
    basemaps: dict[str, dict] | None = None,
    max_features: int = 6000,
) -> pdk.Deck:
    bm = (basemaps or BASEMAPS).get(basemap) or BASEMAPS["CARTO Dark Matter"]
    theme = bm["theme"]
    layers = []
    bounds: list[list[float]] = []
    for spec in sorted(specs, key=_draw_order):
        gdf = frames.get(spec.id)
        if gdf is None or len(gdf) == 0 or not spec.visible:
            continue
        g = layer_frame(spec, gdf, theme, max_features)
        bounds.append([float(x) for x in g.total_bounds])
        tips = [c for c in spec.tooltip if c in g.columns][:6]
        g["_tip"] = [
            "<br>".join(f"<b>{html.escape(_pretty(c))}:</b> {html.escape(str(r[c]))}" for c in tips)
            for _, r in g.iterrows()
        ]
        g["_title"] = spec.title
        ctx = spec.style.color_role == "context"
        if spec.render == "heatmap":
            pts = g.geometry.representative_point()
            df = pd.DataFrame({"lon": pts.x, "lat": pts.y})
            ramp = SEQUENTIAL[theme]  # type: ignore[index]
            layers.append(
                pdk.Layer(
                    "HeatmapLayer",
                    df,
                    get_position=["lon", "lat"],
                    color_range=[_rgba(c)[:3] for c in ramp],
                    radius_pixels=40,
                    opacity=0.75,
                    id=spec.id,
                )
            )
        elif spec.geometry_kind == "point" or spec.render == "points":
            pts = g.geometry.representative_point()
            df = pd.DataFrame(g.drop(columns="geometry"))
            df["_r"] = marker_radii(g, gdf, spec)
            df["lon"], df["lat"] = pts.x.values, pts.y.values
            df["_rgba"] = [_rgba(c, 150 if ctx else 230) for c in df["_fill"]]
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    df,
                    get_position=["lon", "lat"],
                    get_fill_color="_rgba",
                    get_line_color=_rgba(SURFACE[theme]),
                    line_width_min_pixels=1,
                    stroked=True,
                    radius_units="pixels",
                    get_radius="_r",
                    pickable=True,
                    id=spec.id,
                )
            )
        else:
            g["_rgba"] = [
                _rgba(c, int(255 * (spec.style.fill_opacity if spec.geometry_kind != "line" else 0.95)))
                for c in g["_fill"]
            ]
            gj = json.loads(g.to_json(drop_id=True))
            layers.append(
                pdk.Layer(
                    "GeoJsonLayer",
                    gj,
                    get_fill_color="properties._rgba",
                    get_line_color="properties._rgba"
                    if spec.geometry_kind == "line"
                    else _rgba(SURFACE[theme]),
                    line_width_min_pixels=3 if spec.geometry_kind == "line" else 1,
                    stroked=True,
                    filled=spec.geometry_kind != "line",
                    pickable=True,
                    id=spec.id,
                )
            )
    tooltip = {
        "html": "<b>{_title}</b><br/>{_tip}",
        "style": {"backgroundColor": SURFACE[theme], "color": INK[theme], "fontSize": "12px"},
    }
    provider, style = deck_basemap(bm, theme)
    kwargs: dict[str, Any] = {"map_provider": provider}
    if style is not None:
        kwargs["map_style"] = style
    return pdk.Deck(
        layers=layers,
        initial_view_state=_view_state(bounds),
        tooltip=tooltip,  # type: ignore[arg-type]
        **kwargs,
    )
