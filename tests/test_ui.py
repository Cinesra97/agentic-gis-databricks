"""Map-layer schema, renderers, and a Streamlit smoke test."""

import json

import pytest

from agentic_gis.agent.graph import AgentSession, create_runtime
from agentic_gis.ui.layers import (
    MapLayerSpec,
    build_layer_spec,
    feature_role,
    layer_json_schema,
    resolve_role,
)
from agentic_gis.ui.maps import BASEMAPS, apply_filters, render_folium, render_pydeck
from tests.conftest import make_settings


@pytest.fixture(scope="module")
def turn(tmp_path_factory, registry, demo_engine):
    rt = create_runtime(
        make_settings(tmp_path_factory.mktemp("ui")),
        mode="demo",
        llm_provider="demo",
        engines={"local": demo_engine},
        registry=registry,
    )
    s = AgentSession(rt)
    return rt, s.ask("Which neighborhoods have the largest increase in sanitation-related complaints?")


def test_layer_schema_is_valid_json_schema():
    schema = layer_json_schema()
    assert schema["title"] == "MapLayerSpec" and "render" in schema["properties"]
    json.dumps(schema)


def test_layers_round_trip_and_roles(turn):
    rt, r = turn
    specs = [MapLayerSpec.model_validate(lyr) for lyr in r.layers]
    choropleth = specs[0]
    assert choropleth.render == "choropleth" and choropleth.style.color_role == "diverging"
    assert choropleth.synthetic and "(synthetic)" in choropleth.title
    assert resolve_role(feature_role(choropleth, 0), "dark") == resolve_role("div:mid", "dark")
    assert resolve_role(feature_role(choropleth, 10**6), "light") != resolve_role(
        feature_role(choropleth, -(10**6)), "light"
    )


def test_folium_render_uses_local_assets_and_legend(turn):
    rt, r = turn
    specs = [MapLayerSpec.model_validate(lyr) for lyr in r.layers]
    frames = {s.id: rt.store.get(s.artifact_id).data for s in specs}
    m = render_folium(specs, frames, "CARTO Dark Matter", synthetic=True)
    html = m.get_root().render()
    assert "agis-legend" in html and "SYNTHETIC DEMO DATA" in html
    from agentic_gis.ui.maps import _walk

    urls = [u for e in _walk(m) for attr in ("default_js", "default_css") for _, u in getattr(e, attr, [])]
    assert urls and all(u.startswith("/app/static/vendor/") for u in urls)


def test_pydeck_render(turn):
    rt, r = turn
    specs = [MapLayerSpec.model_validate(lyr) for lyr in r.layers]
    frames = {s.id: rt.store.get(s.artifact_id).data for s in specs}
    deck = render_pydeck(specs, frames, "CARTO Dark Matter")
    spec = json.loads(deck.to_json())
    assert spec["layers"] and spec["mapProvider"] == "carto"
    blank = json.loads(render_pydeck(specs, frames, "No basemap (dark)").to_json())
    assert blank.get("mapProvider") in (None, "")


def test_filters(turn):
    rt, r = turn
    art = rt.store.get(r.primary_artifact)
    sub = apply_filters(art.data, ["Brooklyn"])
    assert len(sub) == 18 and set(sub["borough"]) == {"BROOKLYN"}


def test_build_layer_for_points(ctx, tools):
    ctx.step_id = "s1"
    out = tools.invoke("filter_dataset", {"dataset": "demo_catch_basins"}, ctx)
    spec = build_layer_spec(ctx.store.get(out.artifact_id), "x", style="auto")
    assert spec.render == "cluster" and spec.feature_count == out.rows
    assert all(b.get("attr") is not None for b in BASEMAPS.values())


def test_streamlit_app_smoke(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("AGIS_LLM_PROVIDER", "demo")
    monkeypatch.setenv("AGIS_AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    from agentic_gis.config.settings import reset_settings_cache

    reset_settings_cache()
    from pathlib import Path

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception
    at.chat_input[0].set_value("Show sanitation complaints near industrial properties.").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("industrial" in m.value.lower() for m in at.markdown)
    reset_settings_cache()
