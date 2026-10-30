"""Regressions from the live qwen2.5:7b run on the ingested 311 file (2026-10-30 follow-up).

* The model's filter_dataset call listed its own columns and dropped incident_address/street_name,
  so the largest cluster read "near QUEENS".
* LLM narration added two false limitations ("synthetic demonstration data", "does not cover the full
  period requested") on real data with no requested period.
* Re-ingestion guidance suggested --since alone, which cannot reach older records in a newest-first,
  row-limited download.

Offline; uses the 2026 fixture from ``test_reliability`` and the scripted fake Ollama server.
"""

from __future__ import annotations

import json
import re

import geopandas as gpd
import httpx
import pandas as pd
import pytest

from agentic_gis.agent.grounding import Evidence, check_narrative
from agentic_gis.agent.narrative import coverage_answer
from agentic_gis.agent.semantics import widen_sql_projection
from agentic_gis.agent.state import Narration
from agentic_gis.agent.tools import build_default_registry, describe_place
from agentic_gis.connectors.coverage import profile_frame
from agentic_gis.connectors.duckdb_engine import LocalDuckDBEngine
from agentic_gis.connectors.nyc_open_data import SocrataClient, ingest_dataset
from tests.fakes.fake_ollama import FakeOllama
from tests.test_reliability import (
    CLUSTERED,
    CLUSTERS,
    DS,
    QUESTION,
    ROWS,
    S_HOT,
    S_MAP,
    _write_fixture,
    narrator_calls,
    plan,
    s_filter,
    session,
    step,
)

# the column list from the live run (no incident_address / street_name)
NARROW = [
    "unique_key",
    "created_date",
    "borough",
    "community_board",
    "latitude",
    "longitude",
    "complaint_type",
    "geometry",
]
NARROW_PLAN = plan(
    s_filter(categories=["Illegal Dumping"], columns=NARROW), S_HOT, S_MAP, interpretation="Dumping"
)
LARGEST = "Largest cluster: 22 reports over 161 days near SYNTHETIC SITE 023, BROOKLYN."


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    reg = _write_fixture(tmp_path_factory.mktemp("synthetic"))
    return reg, LocalDuckDBEngine(reg, "local", install_extensions=False)


@pytest.fixture(scope="module")
def real_env(tmp_path_factory):
    reg = _write_fixture(tmp_path_factory.mktemp("real_metadata"), synthetic=False)
    return reg, LocalDuckDBEngine(reg, "local", install_extensions=False)


def narration(headline: str, summary: str, findings: list[str], limitations: list[str] | None = None) -> str:
    body = {
        "headline": headline,
        "summary": summary,
        "findings": [{"kind": "observation", "text": t} for t in findings],
    }
    if limitations is not None:
        body["limitations"] = limitations  # extra field a model may still emit; it must never be shown
    return json.dumps(body)


GOOD = narration(
    f"{CLUSTERS} places have repeated illegal dumping reports",
    f"{CLUSTERED:,} of the {ROWS:,} reports fall in {CLUSTERS} clusters of 5 or more reports within 75 ft.",
    ["The largest cluster has 22 reports near SYNTHETIC SITE 023, BROOKLYN."],
)


SYNTHETIC_LABEL = re.compile(r"synthetic (?:demo|demonstration) data|not findings about new york city", re.I)


def all_text(answer: dict) -> str:
    return " ".join(
        [
            answer["headline"],
            answer["summary"],
            *(f["text"] for f in answer["findings"]),
            *answer["limitations"],
        ]
    )


# ---------------------------------------------------------------------------------------------
# 1. Descriptive location attributes cannot be dropped by the planner
# ---------------------------------------------------------------------------------------------
def test_narrowed_filter_columns_keep_location_attributes(tmp_path, env):
    s, rt = session(tmp_path, env, FakeOllama([NARROW_PLAN]))
    r = s.ask(QUESTION)
    f = step(r, "filter_dataset")
    for col in (
        "incident_address",
        "street_name",
        "community_board",
        "borough",
        "latitude",
        "longitude",
        "geometry",
    ):
        assert f'"{col}"' in f["sql"], col
    assert any("Kept location columns" in n for n in f["notes"])
    art = rt.store.get(f["artifact_id"])
    assert {"incident_address", "street_name"} <= set(art.data.columns)
    assert f["rows"] == ROWS and step(r, "find_hotspots")["facts"]["clusters"] == CLUSTERS


def test_rule_plan_projection_is_unchanged(tmp_path, env):
    s, rt = session(tmp_path, env)
    f = step(s.ask(QUESTION), "filter_dataset")
    ds = rt.registry.get_any(DS)
    expected = [c.name for c in ds.columns if not c.sensitive]
    selected = re.findall(r'"(\w+)"', f["sql"].split(" FROM ")[0])
    assert selected == expected, "without a column list every registered non-sensitive column is read"


def test_llm_sql_projection_is_widened_before_approval(tmp_path, env):
    sql = f"SELECT unique_key, created_date, latitude, longitude, geometry FROM {DS} WHERE complaint_type = 'Illegal Dumping'"
    p = plan(
        {"id": "s1", "tool": "query_local_data", "purpose": "select", "args": {"sql": sql}},
        S_HOT,
        interpretation="Dumping",
    )
    s, _ = session(tmp_path, env, FakeOllama([p]))
    r = s.ask(QUESTION)
    shown = r.pending_approval["queries"][0]["sql"]
    assert '"incident_address"' in shown and '"street_name"' in shown
    assert any(v["kind"] == "added" and "location columns" in v["message"] for v in r.validation)
    r2 = s.resume(True)
    assert step(r2, "query_local_data")["rows"] == ROWS
    assert LARGEST in step(r2, "find_hotspots")["highlights"]


def test_aggregate_and_star_sql_are_not_rewritten(env):
    reg, _ = env
    for sql in (
        f"SELECT borough, COUNT(*) AS n FROM {DS} GROUP BY borough",
        f"SELECT * FROM {DS}",
        f"SELECT DISTINCT borough FROM {DS}",
    ):
        assert widen_sql_projection(sql, "duckdb", reg, "local", "local") == (sql, [])


# ---------------------------------------------------------------------------------------------
# 2-3. Repeat-report clusters name real locations; sensible fallbacks without an address
# ---------------------------------------------------------------------------------------------
def test_llm_and_rule_clusters_name_the_same_location(tmp_path, env):
    rules = step(session(tmp_path, env)[0].ask(QUESTION), "find_hotspots")
    llm = step(session(tmp_path, env, FakeOllama([NARROW_PLAN]))[0].ask(QUESTION), "find_hotspots")
    assert LARGEST in rules["highlights"] and LARGEST in llm["highlights"]
    assert llm["facts"]["largest_location"] == "near SYNTHETIC SITE 023, BROOKLYN"
    assert llm["facts"]["largest_location_level"] == "point"
    lat, lon = llm["facts"]["largest_center"]
    assert 40.4 < lat < 41.0 and -74.3 < lon < -73.6


def test_every_cluster_has_a_location_and_center(tmp_path, env):
    s, rt = session(tmp_path, env)
    r = s.ask(QUESTION)
    out = rt.store.get(step(r, "find_hotspots")["artifact_id"]).data
    assert {"location", "center_lat", "center_lon"} <= set(out.columns)
    assert out["location"].str.len().gt(5).all()
    assert not out["location"].str.fullmatch(r"near\s+[A-Z ]+").any(), "never just a borough"


def test_missing_address_falls_back_to_coordinates_and_area(env):
    reg, _ = env
    ds = reg.get_any(DS)
    labels = ds.descriptive_location_columns
    row = {"incident_address": None, "street_name": "", "community_board": "12 QUEENS", "borough": "QUEENS"}
    text, level = describe_place(row, labels, ds, 40.68123456, -73.77104)
    assert (text, level) == ("around 40.68123, -73.77104 (community board 12 QUEENS)", "coordinates")
    text, level = describe_place({"street_name": "149 AVENUE", "borough": "QUEENS"}, labels, ds, 40.7, -73.8)
    assert (text, level) == ("near 149 AVENUE, QUEENS", "point")
    text, level = describe_place({"borough": "QUEENS"}, labels, ds, 40.7, -73.8)
    assert text == "around 40.70000, -73.80000 (QUEENS)"
    assert describe_place({}, [], None, None, None) == ("at an unnamed location", "area")


def test_clusters_without_any_location_columns_use_reference_areas(ctx):
    tools = build_default_registry()
    ctx.step_id = "s1"
    out = tools.invoke(
        "filter_dataset", {"dataset": "demo_311_sanitation", "categories": ["Illegal Dumping"]}, ctx
    )
    art = ctx.store.get(out.artifact_id)
    keep = ["unique_key", "created_date", "complaint_type", "geometry"]
    ctx.step_id = "s2"
    stripped = ctx.save(
        "Dumping without location columns", gpd.GeoDataFrame(art.data[keep]), "filter_dataset", [art]
    )
    ctx.step_id = "s3"
    hs = tools.invoke("find_hotspots", {"input": stripped.id, "radius": 75, "min_points": 5}, ctx)
    assert hs.status == "ok" and hs.facts["clusters"] == CLUSTERS
    loc = hs.facts["largest_location"]
    assert re.fullmatch(r"around 40\.\d{5}, -7[34]\.\d{5} \((neighborhood|community district) .+\)", loc), loc
    assert any("no address or street column" in n for n in hs.notes)


# ---------------------------------------------------------------------------------------------
# 4-5. Provenance and scope claims come from metadata, not the model
# ---------------------------------------------------------------------------------------------
def test_real_dataset_is_never_labeled_synthetic(tmp_path, real_env):
    wrong = narration(
        f"{CLUSTERS} places have repeated illegal dumping reports",
        "These results come from synthetic demonstration data, not actual NYC findings.",
        [],
    )
    s, _ = session(tmp_path, real_env, FakeOllama([NARROW_PLAN], narrator=wrong), llm_narration=True)
    r = s.ask(QUESTION)
    a = r.answer
    assert a["narrator"] == "template" and any("synthetic" in p for p in a["narrative_rejected"])
    # (the sample's address values contain "SYNTHETIC SITE"; what must not appear is a synthetic label)
    assert not SYNTHETIC_LABEL.search(all_text(a)), "real data must not carry a synthetic label anywhere"
    assert any("row limit" in x for x in a["limitations"]), "the legitimate truncation note stays"


def test_deterministic_answer_on_real_data_has_no_synthetic_label(tmp_path, real_env):
    s, _ = session(tmp_path, real_env, FakeOllama([NARROW_PLAN]))
    a = s.ask(QUESTION).answer
    assert a["narrator"] == "template" and not SYNTHETIC_LABEL.search(all_text(a))


def test_model_limitations_are_never_shown(tmp_path, real_env):
    # exactly the two false limitations from the live run, attached to an otherwise correct answer
    fake = FakeOllama(
        [NARROW_PLAN],
        narrator=narration(
            f"{CLUSTERS} places have repeated illegal dumping reports",
            f"{CLUSTERED:,} of the {ROWS:,} reports fall in {CLUSTERS} clusters.",
            [],
            limitations=[
                "The analysis is based on synthetic demonstration data, not actual NYC findings.",
                "The analysis does not cover the full period requested.",
            ],
        ),
    )
    s, _ = session(tmp_path, real_env, fake, llm_narration=True)
    a = s.ask(QUESTION).answer
    assert a["narrator"].startswith("ollama:")
    text = " ".join(a["limitations"])
    assert not SYNTHETIC_LABEL.search(text) and "synthetic" not in text.lower() and "full period" not in text
    assert any("row limit" in x for x in a["limitations"])


@pytest.mark.parametrize(
    "summary,ok",
    [
        ("The analysis does not cover the full period requested.", False),
        ("Results are a partial period only.", False),
        (
            f"{CLUSTERED:,} of {ROWS:,} reports are clustered; the stored copy is a subset of the source.",
            True,
        ),
    ],
)
def test_coverage_claims_must_match_the_request(tmp_path, real_env, summary, ok):
    fake = FakeOllama([NARROW_PLAN], narrator=narration(f"{CLUSTERS} clusters", summary, []))
    s, _ = session(tmp_path, real_env, fake, llm_narration=True)
    a = s.ask(QUESTION).answer
    assert (a["narrator"] != "template") is ok, a["narrative_rejected"]


def test_partial_coverage_claim_allowed_when_period_was_requested(tmp_path, real_env):
    p = plan(s_filter(categories=["Illegal Dumping"]), S_HOT, interpretation="Dumping in 2026")
    fake = FakeOllama(
        [p], narrator=narration(f"{CLUSTERS} clusters", "The stored data only partially covers 2026.", [])
    )
    s, _ = session(tmp_path, real_env, fake, llm_narration=True)
    r = s.ask("Find repeated illegal dumping complaints in 2026.")
    assert r.answer["narrator"].startswith("ollama:"), r.answer["narrative_rejected"]
    assert any(x.startswith("Partial coverage") for x in r.answer["limitations"])


def test_synthetic_data_cannot_be_called_real():
    ev = Evidence(numbers={28.0}, synthetic=True, text=" ")
    bad = Narration(headline="28 clusters", summary="These are real NYC 311 complaints.", findings=[])
    assert any("as real data" in p for p in check_narrative(bad, ev))
    fine = Narration(headline="28 clusters", summary="This is not real NYC data.", findings=[])
    assert check_narrative(fine, ev) == []


# ---------------------------------------------------------------------------------------------
# 6. Deterministic narration by default; LLM narration only on request
# ---------------------------------------------------------------------------------------------
def test_narration_is_off_by_default_and_available_on_demand(tmp_path, real_env):
    fake = FakeOllama([NARROW_PLAN], narrator=GOOD)
    s, _ = session(tmp_path, real_env, fake)
    r = s.ask(QUESTION)
    a = r.answer
    assert narrator_calls(fake) == 0 and a["narrator"] == "template"
    assert a["headline"].startswith(f"{CLUSTERS} locations")
    assert any(LARGEST.removesuffix(".") in f["text"] for f in a["findings"])
    computed_limits = list(a["limitations"])
    r2 = s.narrate(r)
    assert narrator_calls(fake) == 1 and r2.answer["narrator"].startswith("ollama:")
    assert r2.answer["limitations"] == computed_limits, "limitations stay deterministic"
    assert r2.answer["narration_s"] is not None and s.history[-1].answer["narrator"].startswith("ollama:")
    s.narrate(r2)
    assert narrator_calls(fake) == 1, "an already narrated answer is not narrated again"
    prompt = next(
        q for q in fake.requests if not q["messages"][0]["content"].startswith("You are the planning")
    )
    assert prompt["options"]["num_predict"] == 450
    assert set(prompt["format"]["properties"]) == {"headline", "summary", "findings"}


def test_on_demand_narration_is_refused_for_empty_results(tmp_path, real_env):
    p = plan(
        s_filter(
            categories=["Illegal Dumping"],
            filters=[{"column": "incident_address", "op": "=", "value": "1 NOWHERE PLACE"}],
        ),
        S_HOT,
        interpretation="Dumping at 1 Nowhere Place",
    )
    fake = FakeOllama([p], narrator=GOOD)
    s, _ = session(tmp_path, real_env, fake)
    r = s.narrate(s.ask("Find repeated illegal dumping complaints at 1 Nowhere Place"))
    assert narrator_calls(fake) == 0 and r.answer["narrator"] == "template"
    assert any("No matching records" in x for x in r.answer["adjustments"])


# ---------------------------------------------------------------------------------------------
# Ingestion guidance and bounded windows
# ---------------------------------------------------------------------------------------------
def test_bounded_ingestion_counts_matches_and_uses_stable_paging(tmp_path, registry):
    rows = [
        {
            "unique_key": str(i),
            "created_date": "2022-02-01T10:00:00.000",
            "agency": "DSNY",
            "complaint_type": "Illegal Dumping",
            "borough": "QUEENS",
            "incident_address": "184-54 149 AVENUE",
            "latitude": "40.68",
            "longitude": "-73.77",
        }
        for i in range(30)
    ]
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        seen.append(params)
        if params.get("$select") == "count(*) AS n":
            return httpx.Response(200, json=[{"n": "40"}])
        off, lim = int(params.get("$offset", 0)), int(params.get("$limit", 1000))
        return httpx.Response(200, json=rows[off : off + lim])

    ds = registry.get_any(DS).model_copy(deep=True)
    ds.source.path = str(tmp_path / "window.parquet")
    client = SocrataClient(transport=httpx.MockTransport(handler))
    res = ingest_dataset(ds, client, since="2020-01-01", until="2023-12-31", limit=30)
    where = seen[0]["$where"]
    assert (
        "created_date >= '2020-01-01T00:00:00'" in where and "created_date < '2024-01-01T00:00:00'" in where
    )
    data_req = next(p for p in seen if p.get("$select") != "count(*) AS n")
    assert data_req["$order"].endswith(":id"), "offset paging needs a unique tie-breaker"
    assert res.matching_rows == 40 and res.limit_reached
    meta = json.loads((tmp_path / "window.provenance.json").read_text())
    assert meta["matching_rows"] == 40 and meta["limit_reached"] is True
    cov = profile_frame(ds, gpd.read_parquet(res.path))
    assert any("Only 30 of the 40 records" in n for n in cov.notes)
    with pytest.raises(ValueError):
        ingest_dataset(ds, client, since="2024-01-01", until="2023-01-01")


def test_coverage_answer_recommends_a_bounded_window():
    scope = {
        "datasets": [
            {
                "dataset": DS,
                "name": "311 sanitation complaints",
                "overlap": "none",
                "requested_from": "2020-01-01",
                "requested_to": "2023-12-31",
                "requested_text": "2020-01-01 to 2023-12-31",
                "coverage": {
                    "rows": 150_000,
                    "date_min": "2026-05-15",
                    "date_max": "2026-10-27",
                    "source": "NYC Open Data erm2-nwe9",
                    "notes": [],
                },
            }
        ]
    }
    tip = " ".join(coverage_answer(scope).limitations)
    assert "--since 2020-01-01 --until 2023-12-31" in tip
    assert "newest-first" in tip and "A start date alone is not enough" in tip


def test_describe_place_handles_pandas_missing_values(env):
    reg, _ = env
    ds = reg.get_any(DS)
    row = pd.Series(
        {"incident_address": pd.NA, "street_name": float("nan"), "community_board": "04 BROOKLYN"}
    )
    text, level = describe_place(row, ds.descriptive_location_columns, ds, 40.69, -73.94)
    assert level == "coordinates" and text.endswith("(community board 04 BROOKLYN)")
