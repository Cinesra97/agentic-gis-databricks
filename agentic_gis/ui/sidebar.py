"""Sidebar: data mode, reasoning engine, Databricks connection, map settings, data catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from agentic_gis.agent.llm import OllamaClient
from agentic_gis.config.registry import DatasetRegistry, DatasetSpec, SourceSpec, SpatialSpec, TemporalSpec
from agentic_gis.config.settings import Settings
from agentic_gis.ui.maps import available_basemaps

MODE_LABELS = {
    "demo": "Demo — offline sample data (synthetic)",
    "local": "Public NYC data (ingested locally)",
    "databricks": "Databricks (Unity Catalog)",
}


@dataclass
class SidebarState:
    mode: str
    provider: str
    model: str
    basemap: str
    renderer: str
    new_conversation: bool


@st.cache_data(ttl=20, show_spinner=False)
def _ollama_models(base_url: str) -> list[str]:
    return OllamaClient(base_url, "probe").list_models()


def render_sidebar(settings: Settings, registry: DatasetRegistry) -> SidebarState:
    sb = st.sidebar
    sb.markdown("### 🗺️ Agentic GIS")
    sb.caption("Natural-language spatial analysis · local LLM · governed data access")

    modes = list(MODE_LABELS)
    mode = sb.radio(
        "Data source",
        modes,
        index=modes.index(settings.mode) if settings.mode in modes else 0,
        format_func=lambda m: MODE_LABELS[m],
        key="mode",
    )
    if mode == "databricks" and not settings.databricks_configured:
        sb.warning(
            "Databricks is not configured. Set DATABRICKS_SERVER_HOSTNAME, DATABRICKS_HTTP_PATH and "
            "credentials in .env, then restart. Only reference boundaries are available."
        )
    if mode == "local" and not any(d.role == "events" for d in registry.available("local")):
        sb.info(
            "No public datasets ingested yet. Run `uv run agentic-gis ingest nyc_311_sanitation` "
            "(needs internet access to data.cityofnewyork.us)."
        )

    sb.divider()
    providers = {"ollama": "Local LLM (Ollama)", "demo": "Deterministic planner (no LLM)"}
    if settings.openai_compat_base_url:
        providers["openai_compatible"] = "Approved OpenAI-compatible endpoint"
    pkeys = list(providers)
    provider = sb.radio(
        "Reasoning engine",
        pkeys,
        index=pkeys.index(settings.llm_provider) if settings.llm_provider in pkeys else 0,
        format_func=lambda p: providers[p],
        key="provider",
    )
    model = settings.ollama_model
    if provider == "ollama":
        models = _ollama_models(settings.ollama_base_url)
        if models:
            options = sorted(set(models) | {settings.ollama_model})
            model = sb.selectbox(
                "Ollama model", options, index=options.index(settings.ollama_model), key="model"
            )
        else:
            model = sb.text_input("Ollama model", settings.ollama_model, key="model_text")
            sb.caption(
                f"Ollama not reachable at {settings.ollama_base_url}. Questions will use the deterministic "
                "planner until it is running."
            )
        if sb.button("Test LLM connection", width="stretch"):
            status = OllamaClient(settings.ollama_base_url, model).test_connection()
            (sb.success if status.ok else sb.error)(status.message)
    elif provider == "openai_compatible":
        model = settings.openai_compat_model or ""
        sb.caption(f"Endpoint: {settings.openai_compat_base_url}")

    with sb.expander("Databricks connection", expanded=(mode == "databricks")):
        _databricks_panel(settings, registry)

    sb.divider()
    basemaps = available_basemaps(settings.custom_basemap_url, settings.custom_basemap_attribution)
    names = list(basemaps)
    default_bm = settings.default_basemap if settings.default_basemap in names else names[0]
    basemap = sb.selectbox("Basemap", names, index=names.index(default_bm), key="basemap")
    renderer = (
        sb.segmented_control(
            "Map engine",
            ["folium", "pydeck"],
            default=settings.map_renderer,
            format_func=lambda r: {"folium": "Leaflet", "pydeck": "deck.gl"}[r],
            key="renderer",
        )
        or "folium"
    )

    with sb.expander("Data catalog", expanded=False):
        _catalog(registry, mode)
    with sb.expander("Safety & governance", expanded=False):
        st.markdown(
            f"- SQL approval: **{settings.sql_approval}**\n"
            f"- Max rows per query: **{settings.max_result_rows:,}**\n"
            f"- Query timeout: **{settings.query_timeout_s}s**\n"
            f"- LLM may see data up to: **{settings.llm_max_sensitivity}**\n"
            f"- External LLM endpoints: **{'allowed' if settings.allow_external_llm else 'blocked'}**\n"
            f"- Audit log: `{settings.audit_log_path.name}`"
        )
        st.caption("Read-only, allowlisted, AST-validated SQL. No code execution by the agent.")
    if settings.admin_mode:
        with sb.expander("Register a dataset (admin)", expanded=False):
            _register_form(settings, registry, mode)

    sb.divider()
    new_conv = sb.button("New conversation", width="stretch", icon=":material/refresh:")
    return SidebarState(
        mode=mode,
        provider=provider,
        model=model,
        basemap=basemap,
        renderer=renderer,
        new_conversation=new_conv,
    )


def _catalog(registry: DatasetRegistry, mode: str) -> None:
    rows = []
    for ds in registry.all():
        if mode not in ds.modes:
            continue
        status = registry.status(ds)
        rows.append(
            {
                "id": ds.id,
                "name": ds.name,
                "role": ds.role,
                "status": status.state,
                "synthetic": "yes" if ds.synthetic else "",
                "sensitivity": ds.sensitivity,
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Only enabled datasets with status 'ready' are visible to the agent.")


def _databricks_panel(settings: Settings, registry: DatasetRegistry) -> None:
    if not settings.databricks_configured:
        st.caption("Not configured. See docs/DATABRICKS.md.")
        return
    st.caption(
        f"Host: `{settings.databricks_server_hostname}`  \nAuth: {settings.databricks_auth_type}  \n"
        f"Default: {settings.databricks_catalog or '—'}.{settings.databricks_schema or '—'}"
    )
    from agentic_gis.connectors.databricks_sql import DatabricksConnector

    if st.button("Test Databricks connection", width="stretch"):
        with st.spinner("Connecting…"):
            status = DatabricksConnector(settings).test_connection()
        (st.success if status.ok else st.error)(status.message)
        st.session_state["dbx_ok"] = status.ok
    if st.button("Discover tables in default schema", width="stretch"):
        try:
            with st.spinner("Listing tables…"):
                st.session_state["dbx_tables"] = DatabricksConnector(settings).list_tables()
        except Exception as exc:
            st.error(str(exc))
    tables: list[dict[str, Any]] = st.session_state.get("dbx_tables") or []
    if tables:
        registered = {
            d.source.table.lower() for d in registry.all() if d.backend == "databricks" and d.source.table
        }
        df = pd.DataFrame(tables)
        df["registered"] = df.apply(
            lambda r: f"{r['catalog']}.{r['schema']}.{r['table']}".lower() in registered, axis=1
        )
        st.dataframe(df[["table", "type", "registered"]], hide_index=True, width="stretch")
        st.caption("Discovered tables are not queryable until an administrator registers them.")


def _register_form(settings: Settings, registry: DatasetRegistry, mode: str) -> None:
    with st.form("register-ds"):
        st.caption(f"Writes to `{settings.datasets_local_config.name}` (git-ignored).")
        did = st.text_input("Dataset id (lowercase_with_underscores)")
        name = st.text_input("Display name")
        desc = st.text_area("Description")
        backend = st.selectbox("Backend", ["databricks", "local"])
        table = st.text_input("Databricks table (catalog.schema.table) or local file path")
        role = st.selectbox("Role", ["events", "infrastructure", "parcels", "streets", "boundary", "other"])
        fmt = st.selectbox("Geometry format", ["latlon", "wkt", "wkb", "geojson", "native", "none"])
        geom = st.text_input("Geometry column (or blank)")
        lat = st.text_input("Latitude column (for latlon)")
        lon = st.text_input("Longitude column (for latlon)")
        crs = st.text_input("CRS", "EPSG:4326")
        gtype = st.selectbox("Geometry type", ["point", "line", "polygon", "mixed"])
        tcol = st.text_input("Temporal column (optional)")
        ccol = st.text_input("Category column (optional)")
        sens = st.selectbox("Sensitivity", ["internal", "public", "confidential", "restricted"])
        submitted = st.form_submit_button("Register dataset")
    if submitted:
        try:
            source = (
                SourceSpec(type="databricks_table", table=table)
                if backend == "databricks"
                else SourceSpec(type="file", path=table)
            )
            spatial = None
            if fmt != "none":
                spatial = SpatialSpec(
                    geometry_column=geom or None,
                    geometry_type=gtype,
                    format=fmt,
                    crs=crs,
                    lat_column=lat or None,
                    lon_column=lon or None,
                )
            spec = DatasetSpec(
                id=did,
                name=name,
                description=desc or name,
                backend=backend,
                modes=[mode],
                role=role,
                sensitivity=sens,
                source=source,
                spatial=spatial,
                temporal=TemporalSpec(column=tcol) if tcol else None,
                category_column=ccol or None,
                schema_discovery=backend == "databricks",
            )
            registry.register(spec, settings.datasets_local_config)
            st.success(f"Registered {did}. Reload the app to use it.")
            st.cache_resource.clear()
        except Exception as exc:
            st.error(f"Could not register: {exc}")
