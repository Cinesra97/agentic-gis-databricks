"""Agentic GIS — Streamlit application.

Run (PowerShell or bash):  uv run streamlit run app.py
"""

from __future__ import annotations

import hashlib
import logging

import geopandas as gpd
import streamlit as st
from streamlit_folium import st_folium

from agentic_gis.agent.graph import AgentSession, TurnResult, create_runtime
from agentic_gis.agent.workspace import Artifact
from agentic_gis.config.registry import DatasetRegistry, load_registry
from agentic_gis.config.settings import get_settings
from agentic_gis.connectors.base import QueryEngine
from agentic_gis.connectors.duckdb_engine import LocalDuckDBEngine
from agentic_gis.ui import chat, results
from agentic_gis.ui.layers import MapLayerSpec, build_layer_spec
from agentic_gis.ui.maps import apply_filters, available_basemaps, render_folium, render_pydeck
from agentic_gis.ui.sidebar import render_sidebar
from agentic_gis.ui.theme import inject_css

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(
    page_title="Agentic GIS · NYC", page_icon="🗺️", layout="wide", initial_sidebar_state="expanded"
)
inject_css()
settings = get_settings()


def _registry_fingerprint() -> str:
    h = hashlib.sha256()
    for p in (settings.datasets_config, settings.datasets_local_config):
        if p.exists():
            h.update(str(p.stat().st_mtime_ns).encode())
    for d in (settings.data_dir / "ingested",):
        if d.exists():
            for f in sorted(d.glob("*.parquet")):
                h.update(f"{f.name}{f.stat().st_mtime_ns}".encode())
    return h.hexdigest()[:16]


@st.cache_resource(show_spinner="Loading datasets into DuckDB…")
def get_local_engine(mode: str, fingerprint: str) -> tuple[DatasetRegistry, LocalDuckDBEngine]:
    registry = load_registry(settings)
    return registry, LocalDuckDBEngine(registry, mode, install_extensions=settings.duckdb_install_extensions)


def ensure_session(mode: str, provider: str, model: str, reset: bool) -> AgentSession:
    fp = _registry_fingerprint()
    key = (mode, provider, model, fp)
    if reset or st.session_state.get("session_key") != key or "agent" not in st.session_state:
        registry, local = get_local_engine(mode, fp)
        engines: dict[str, QueryEngine] = {"local": local}
        if mode == "databricks" and settings.databricks_configured:
            from agentic_gis.connectors.databricks_sql import DatabricksConnector

            engines["databricks"] = DatabricksConnector(settings)
        runtime = create_runtime(
            settings, mode=mode, llm_provider=provider, model=model, engines=engines, registry=registry
        )
        st.session_state["agent"] = AgentSession(runtime)
        st.session_state["session_key"] = key
        st.session_state.pop("view_turn", None)
    return st.session_state["agent"]


# ---------------------------------------------------------------------------------------------------
registry_for_sidebar = get_local_engine(st.session_state.get("mode", settings.mode), _registry_fingerprint())[
    0
]
sb = render_sidebar(settings, registry_for_sidebar)
session = ensure_session(sb.mode, sb.provider, sb.model, sb.new_conversation)
rt = session.runtime

mode_badge = {
    "demo": ("Demo · synthetic data", "orange"),
    "local": ("Public NYC data", "green"),
    "databricks": ("Databricks", "violet"),
}[sb.mode]
h1, h2 = st.columns([0.55, 0.45], vertical_alignment="center")
with h1:
    st.markdown(
        "<p class='agis-title'>Agentic GIS · New York City</p>"
        "<p class='agis-sub'>Ask a geographic question. The agent plans, queries authorized data, "
        "runs spatial analysis and maps the results.</p>",
        unsafe_allow_html=True,
    )
with h2:
    b1, b2, b3 = st.columns(3)
    b1.badge(mode_badge[0], color=mode_badge[1])  # type: ignore[arg-type]
    b2.badge(rt.llm.label if rt.llm else "Deterministic planner", color="blue", help=rt.llm_message)
    b3.badge(f"SQL approval: {settings.sql_approval}", color="gray")

left, right = st.columns([0.4, 0.6], gap="medium")

# ---- chat column -------------------------------------------------------------------------------
with left:
    box = st.container(height=640, border=False)
    with box:
        if not session.history and not session.pending_approval:
            chat.render_examples()
        chat.render_history(session)
        pending = session.pending_approval
        if pending:
            chat.render_approval(pending)
    decision = st.session_state.pop("approval_decision", None)
    if decision is not None and session.pending_approval:
        with box, st.status("Resuming after your decision…", expanded=True) as status:
            res = session.resume(bool(decision), on_event=chat.progress_callback(status))
            status.update(
                label="Done" if res.status != "awaiting_approval" else "Awaiting approval", state="complete"
            )
        st.session_state.pop("view_turn", None)
        st.rerun()
    question = st.chat_input(
        "Ask about NYC sanitation, infrastructure, places…", disabled=session.pending_approval is not None
    )
    question = question or st.session_state.pop("queued_question", None)
    if question:
        if session.pending_approval:
            st.warning("Approve or reject the pending SQL first.")
        else:
            with box:
                with st.chat_message("user"):
                    st.markdown(question)
                with st.status("Working…", expanded=True) as status:
                    try:
                        res = session.ask(question, on_event=chat.progress_callback(status))
                        label = {
                            "awaiting_approval": "SQL needs your approval",
                            "clarification": "Need a detail",
                            "complete": "Analysis complete",
                        }.get(res.status, res.status)
                        status.update(label=label, state="complete")
                    except Exception as exc:  # keep the UI alive; details are in the log
                        logging.exception("Agent failure")
                        status.update(label=f"Error: {exc}", state="error")
            st.session_state.pop("view_turn", None)
            st.rerun()


# ---- map + results column ------------------------------------------------------------------
def _context_boroughs() -> tuple[list[MapLayerSpec], dict[str, gpd.GeoDataFrame]]:
    ds = rt.registry.geography("borough", rt.mode)
    if ds is None:
        return [], {}
    g = rt.tool_context(0).load_reference(ds)
    art = Artifact(
        id="ref.boroughs",
        turn=0,
        step_id="ref",
        tool="reference",
        title="Borough boundaries",
        data=g,
        datasets=[ds.id],
    )
    spec = build_layer_spec(art, layer_id="ref-boroughs", context=True, style="polygons")
    return [spec], {spec.id: g}


with right:
    turns: list[TurnResult] = [t for t in session.history if t.layers or t.steps]
    view: TurnResult | None = None
    if turns:
        opts = list(range(len(turns)))
        default = len(turns) - 1
        labels = {
            i: f"Turn {turns[i].turn}: {(turns[i].plan or {}).get('interpretation', '')[:70]}" for i in opts
        }
        if len(turns) > 1:
            idx = st.selectbox(
                "Showing results for", opts, index=default, format_func=lambda i: labels[i], key="view_turn"
            )
        else:
            idx = default
        view = turns[idx]

    specs = [MapLayerSpec.model_validate(lyr) for lyr in (view.layers if view else [])]
    c1, c2, c3 = st.columns([0.36, 0.32, 0.32])
    visible_titles = c1.multiselect(
        "Layers",
        [s.title for s in specs],
        default=[s.title for s in specs],
        key=f"layers-{view.turn if view else 0}",
        placeholder="No layers",
    )
    boroughs = c2.multiselect(
        "Borough filter",
        ["Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"],
        key="f_boro",
        placeholder="All boroughs",
    )
    area_labels = sorted({lab for role, lab in rt.vocabulary.areas.values() if role == "neighborhood"})
    areas_sel = c3.multiselect(
        "Neighborhood filter", area_labels, key="f_area", placeholder="All neighborhoods"
    )

    for s in specs:
        s.visible = s.title in visible_titles
    frames: dict[str, gpd.GeoDataFrame] = {}
    area_geoms = None
    bgeoms = None
    if areas_sel:
        nds = rt.registry.geography("neighborhood", rt.mode)
        if nds is not None:
            ng = rt.tool_context(0).load_reference(nds)
            col = nds.name_column if nds.name_column in ng else ng.columns[0]
            area_geoms = ng[ng[col].astype(str).isin(areas_sel)]
    if boroughs:
        bds = rt.registry.geography("borough", rt.mode)
        bgeoms = rt.tool_context(0).load_reference(bds) if bds else None
    for s in specs:
        art = rt.store.get(s.artifact_id)
        if art is not None and art.is_spatial:
            frames[s.id] = apply_filters(art.data, boroughs, area_geoms, bgeoms)
    if not specs:
        specs, frames = _context_boroughs()
    synthetic = any(s.synthetic for s in specs)
    basemaps = available_basemaps(settings.custom_basemap_url, settings.custom_basemap_attribution)
    selected_feature = None
    if sb.renderer == "pydeck":
        deck = render_pydeck(specs, frames, sb.basemap, basemaps, settings.max_render_features)
        event = st.pydeck_chart(
            deck,
            height=560,
            on_select="rerun",
            selection_mode="single-object",
            key=f"deck-{view.turn if view else 0}",
        )
        try:
            objs = event.selection.get("objects", {}) if event else {}
            for items in objs.values():
                if items:
                    obj = items[0]
                    selected_feature = {"properties": obj.get("properties", obj)}
        except AttributeError:
            pass
    else:
        fmap = render_folium(
            specs,
            frames,
            sb.basemap,
            basemaps,
            settings.max_render_features,
            synthetic,
            local_assets=settings.map_assets == "local",
        )
        out = st_folium(
            fmap,
            height=560,
            use_container_width=True,
            returned_objects=["last_active_drawing"],
            key=f"map-{view.turn if view else 0}-{sb.basemap}-{len(boroughs)}-{len(areas_sel)}",
        )
        selected_feature = (out or {}).get("last_active_drawing")
    notes = [s.note for s in specs if s.note and s.visible]
    if notes:
        st.caption(" ".join(notes))

    tabs = st.tabs(
        ["Summary", "Results table", "Chart", "Reasoning & SQL", "Data & methods", "Export", "Feature"]
    )
    if view is None:
        with tabs[0]:
            st.info("Ask a question to see results here. Borough boundaries are shown for orientation.")
    else:
        with tabs[0]:
            results.render_answer(view)
        with tabs[1]:
            results.render_results_table(session, view)
        with tabs[2]:
            if not results.render_chart(session, view):
                st.caption("No time-series result in this turn. Ask e.g. “How has this changed over time?”")
        with tabs[3]:
            results.render_reasoning(view)
        with tabs[4]:
            results.render_provenance(view)
        with tabs[5]:
            results.render_exports(session, view)
    with tabs[6]:
        results.describe_feature(selected_feature)
