"""Reliability regressions: unrequested filters, data coverage, empty results, grounded answers.

Offline. The fixture is built from the bundled SYNTHETIC 311 sample with its dates shifted into
2026-05-15 .. 2026-10-27 and registered as ``nyc_311_sanitation`` in local mode, with an
ingestion record saying the download stopped at its row limit. That reproduces the reported
situation (a 2026-only, truncated subset of a source that starts in 2010) without depending on
anyone's locally ingested file. Expected results on this data: 1,599 Illegal Dumping records and
28 repeat locations (>=5 reports within 75 ft) containing 350 reports.

The fake Ollama server is a test double: it scripts what a model might return (e.g. the invented
2020-2023 date filter from the bug report). It says nothing about the quality of a real model.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from agentic_gis.agent.graph import AgentSession, create_runtime
from agentic_gis.agent.grounding import Evidence, check_narrative
from agentic_gis.agent.semantics import QuestionScope, check_sql_scope
from agentic_gis.agent.state import Answer, Finding
from agentic_gis.agent.temporal import parse_time_window
from agentic_gis.agent.tools import RenderedSQL, ToolContext, build_default_registry
from agentic_gis.agent.workspace import ResultStore
from agentic_gis.config.registry import DatasetRegistry
from agentic_gis.config.settings import PROJECT_ROOT, Settings
from agentic_gis.connectors.duckdb_engine import LocalDuckDBEngine
from agentic_gis.security.audit import AuditLogger
from agentic_gis.security.sql_guard import SQLGuard
from tests.conftest import make_settings
from tests.fakes.fake_ollama import FakeOllama

DS = "nyc_311_sanitation"
START, END = "2026-05-15", "2026-10-27"
TODAY = date(2026, 10, 30)
QUESTION = "Find locations in NYC with repeated illegal dumping complaints."
ROWS, CLUSTERS, CLUSTERED = 1599, 28, 350
WHERE = "agency = 'DSNY' AND latitude IS NOT NULL AND created_date >= '2023-01-01T00:00:00'"


# ---------------------------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------------------------
def _shift(
    series: pd.Series, lo: pd.Timestamp, hi: pd.Timestamp, a: pd.Timestamp, b: pd.Timestamp
) -> pd.Series:
    frac = (series - lo) / (hi - lo)
    return a + frac * (b - a)


def _write_fixture(folder: Path, dumping_rows: int | None = None) -> DatasetRegistry:
    src = gpd.read_parquet(PROJECT_ROOT / "sample_data" / "synthetic_311_sanitation.parquet")
    lo, hi = src["created_date"].min(), src["created_date"].max()
    a, b = pd.Timestamp(f"{START} 00:00:00"), pd.Timestamp(f"{END} 12:00:00")
    df = src.copy()
    df["created_date"] = _shift(df["created_date"], lo, hi, a, b)
    df["closed_date"] = df["created_date"] + pd.Timedelta(days=2)
    if dumping_rows:  # scale Illegal Dumping up to an exact count (jittered copies, new ids)
        dump = df[df["complaint_type"] == "Illegal Dumping"]
        extra = dumping_rows - len(dump)
        rng = np.random.default_rng(7)
        add = dump.sample(extra, replace=True, random_state=7).copy()
        add["latitude"] = add["latitude"] + rng.normal(0, 0.002, extra)
        add["longitude"] = add["longitude"] + rng.normal(0, 0.002, extra)
        add["geometry"] = gpd.points_from_xy(add["longitude"], add["latitude"], crs="EPSG:4326")
        add["unique_key"] = np.arange(90_000_000, 90_000_000 + extra)
        df = gpd.GeoDataFrame(pd.concat([df, add], ignore_index=True), geometry="geometry", crs="EPSG:4326")
    path = folder / f"{DS}.parquet"
    df.to_parquet(path, index=False)
    sidecar = {
        "dataset_id": DS,
        "socrata_id": "erm2-nwe9",
        "where": WHERE,
        "order": "created_date DESC",
        "limit": len(df),
        "fetched_rows": len(df),
        "limit_reached": True,
        "rows": len(df),
        "retrieved_utc": "2026-10-28T09:00:00Z",
    }
    path.with_suffix(".provenance.json").write_text(json.dumps(sidecar), encoding="utf-8")
    overlay = folder / "datasets.local.yaml"
    overlay.write_text(
        "datasets:\n"
        f"  - id: {DS}\n"
        "    synthetic: true\n"  # the fixture is synthetic data and stays labeled as such
        "    source:\n"
        "      type: socrata\n"
        f"      path: {path.as_posix()}\n"
        "      format: geoparquet\n"
        "      socrata:\n"
        "        dataset_id: erm2-nwe9\n"
        f'        where: "{WHERE}"\n'
        "        order: created_date DESC\n"
        f"        limit: {len(df)}\n",
        encoding="utf-8",
    )
    return DatasetRegistry.load(Settings(_env_file=None).datasets_config, overlay)  # type: ignore[call-arg]


@pytest.fixture(scope="module")
def env(tmp_path_factory) -> tuple[DatasetRegistry, LocalDuckDBEngine]:
    reg = _write_fixture(tmp_path_factory.mktemp("fixture2026"))
    return reg, LocalDuckDBEngine(reg, "local", install_extensions=False)


@pytest.fixture(scope="module")
def scaled_env(tmp_path_factory) -> tuple[DatasetRegistry, LocalDuckDBEngine]:
    reg = _write_fixture(tmp_path_factory.mktemp("fixture_scaled"), dumping_rows=20_323)
    return reg, LocalDuckDBEngine(reg, "local", install_extensions=False)


def session(tmp_path, env, fake: FakeOllama | None = None, **kw):
    reg, eng = env
    st = make_settings(tmp_path, llm_provider="ollama" if fake else "demo", **kw)
    rt = create_runtime(
        st,
        mode="local",
        engines={"local": eng},
        registry=reg,
        llm_transport=fake.transport() if fake else None,
    )
    rt.today = TODAY
    return AgentSession(rt), rt


def step(r, tool: str, n: int = 0) -> dict:
    return [s for s in r.steps if s["tool"] == tool][n]


def plan(*steps: dict, interpretation: str = "Repeated illegal dumping, 2020 to 2023") -> str:
    return json.dumps({"interpretation": interpretation, "datasets": [DS], "steps": list(steps)})


def s_filter(**args) -> dict:
    return {"id": "s1", "tool": "filter_dataset", "purpose": "select", "args": {"dataset": DS, **args}}


S_HOT = {
    "id": "s2",
    "tool": "find_hotspots",
    "purpose": "repeat locations",
    "args": {"input": "s1", "method": "clusters", "radius": 75, "min_points": 5},
}
S_MAP = {"id": "s3", "tool": "generate_map_layer", "purpose": "map", "args": {"input": "s2"}}
INVENTED = plan(
    s_filter(categories=["Illegal Dumping"], date_from="2020-01-01", date_to="2023-12-31"), S_HOT, S_MAP
)


def planner_calls(fake: FakeOllama) -> int:
    return sum(
        1 for r in fake.requests if r["messages"][0]["content"].startswith("You are the planning module")
    )


def narrator_calls(fake: FakeOllama) -> int:
    return len(fake.requests) - planner_calls(fake)


def corrections(r, kind: str | None = None) -> list[str]:
    return [v["message"] for v in r.validation if kind is None or v["kind"] == kind]


# ---------------------------------------------------------------------------------------------
# Coverage profiling
# ---------------------------------------------------------------------------------------------
def test_coverage_is_measured_from_stored_data_not_configuration(env):
    _, eng = env
    cov = eng.coverage[DS]
    assert (cov.rows, cov.date_min, cov.date_max) == (8917, START, END)
    assert cov.limit_reached and cov.is_subset and cov.computed_by == "load"
    assert any("most recent records were kept" in n for n in cov.notes)
    assert any("starts at 2023-01-01, but the stored records start at 2026-05-15" in n for n in cov.notes)
    assert cov.overlap(None, None) == "all"
    assert cov.overlap("2020-01-01", "2023-12-31") == "none"
    assert cov.overlap("2026-01-01", "2026-12-31") == "partial"
    assert cov.overlap("2026-06-01", "2026-06-30") == "covered"


# ---------------------------------------------------------------------------------------------
# 1-3: temporal scope from the question (deterministic planner)
# ---------------------------------------------------------------------------------------------
def test_1_no_period_in_question_means_no_date_filter(tmp_path, env):
    s, _ = session(tmp_path, env)
    r = s.ask(QUESTION)
    assert r.status == "complete" and r.plan_source == "rules"
    f = step(r, "filter_dataset")
    assert "date_from" not in f["args"] and "date_to" not in f["args"]
    assert f["rows"] == ROWS
    entry = r.data_scope["datasets"][0]
    assert r.data_scope["period"] == "all available dates"
    assert entry["matching_rows"] == ROWS and entry["coverage"]["rows"] == 8917
    assert any("row limit" in w for w in r.data_scope["warnings"]), "truncated subset must be disclosed"


def test_2_request_for_2026_filters_2026_and_discloses_partial_coverage(tmp_path, env):
    s, _ = session(tmp_path, env)
    r = s.ask("Find locations with repeated illegal dumping complaints in 2026.")
    f = step(r, "filter_dataset")
    assert (f["args"]["date_from"], f["args"]["date_to"]) == ("2026-01-01", "2026-12-31")
    assert f["rows"] == ROWS and step(r, "find_hotspots")["rows"] == CLUSTERS
    assert any(
        w.startswith("Partial coverage") and "2026-05-15 to 2026-10-27" in w for w in r.data_scope["warnings"]
    )
    assert any("Partial coverage" in x for x in r.answer["limitations"])


@pytest.mark.parametrize("llm", [False, True])
def test_3_period_outside_coverage_reports_insufficient_coverage(tmp_path, env, llm):
    grounded = plan(
        s_filter(categories=["Illegal Dumping"], date_from="2020-01-01", date_to="2023-12-31"), S_HOT
    )
    fake = FakeOllama([grounded]) if llm else None
    s, _ = session(tmp_path, env, fake)
    r = s.ask("Find repeated illegal dumping complaints between 2020 and 2023.")
    assert r.status == "insufficient_coverage"
    assert r.steps == [], "nothing may run when the stored data cannot answer the question"
    a = r.answer
    assert "does not cover 2020-01-01 to 2023-12-31" in a["headline"]
    assert "2026-05-15 to 2026-10-27" in a["summary"] and "No analysis was run" in a["summary"]
    assert "not that there were no complaints" in a["summary"]
    if fake:
        assert narrator_calls(fake) == 0, "no narration call for an answer that needs no model"
    r2 = s.ask("Use all available dates")
    assert r2.status == "complete" and step(r2, "find_hotspots")["rows"] == CLUSTERS


# ---------------------------------------------------------------------------------------------
# 4, 7: invented filters in LLM plans
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "candidate",
    [
        INVENTED,
        # same dates hidden in 'filters', plus an invented borough, sub-type and sample size
        plan(
            s_filter(
                categories=["Illegal Dumping"],
                boroughs=["Queens"],
                limit=1000,
                filters=[
                    {"column": "created_date", "op": "between", "value": ["2020-01-01", "2023-12-31"]},
                    {"column": "descriptor", "op": "=", "value": "Removal Request"},
                ],
            ),
            S_HOT,
            S_MAP,
        ),
    ],
    ids=["date_args", "filters_list"],
)
def test_4_invented_filters_are_removed_and_results_match_rules(tmp_path, env, candidate):
    fake = FakeOllama([candidate])
    s, _ = session(tmp_path, env, fake)
    r = s.ask(QUESTION)
    assert r.status == "complete" and r.plan_source == "llm"
    f = step(r, "filter_dataset")
    assert "date_from" not in f["args"] and "boroughs" not in f["args"] and not f["args"]["filters"]
    assert f["rows"] == ROWS
    hs = step(r, "find_hotspots")
    assert (
        hs["status"] == "ok"
        and hs["facts"]["clusters"] == CLUSTERS
        and hs["facts"]["clustered_points"] == CLUSTERED
    )
    removed = " ".join(corrections(r, "removed"))
    assert "Removed the date filter 2020-01-01 to 2023-12-31" in removed
    assert "2020" not in r.plan["interpretation"], "interpretation must not repeat the removed dates"
    assert planner_calls(fake) == 1, "corrections are deterministic: no extra model call"
    assert any(t["title"].startswith("Plan checked against the question") for t in r.trace)
    text = (r.answer["headline"] + r.answer["summary"]).lower()
    assert "no complaints" not in text and "unknown input reference" not in json.dumps(r.steps).lower()


def test_7_llm_and_rule_plans_give_the_same_clusters(tmp_path, env):
    s_rules, _ = session(tmp_path, env)
    r1 = s_rules.ask(QUESTION)
    other_params = plan(
        s_filter(categories=["Illegal Dumping"]),
        {**S_HOT, "args": {"input": "s1", "method": "clusters", "radius": 150, "min_points": 3}},
    )
    fake = FakeOllama([other_params])
    s_llm, _ = session(tmp_path, env, fake)
    r2 = s_llm.ask(QUESTION)
    h1, h2 = step(r1, "find_hotspots"), step(r2, "find_hotspots")
    assert h1["facts"]["clusters"] == h2["facts"]["clusters"] == CLUSTERS
    assert h1["facts"]["largest_count"] == h2["facts"]["largest_count"]
    assert h2["args"]["radius"] == 75 and h2["args"]["min_points"] == 5
    assert any("standard radius 75 ft" in m for m in corrections(r2, "changed"))
    # a radius the user asks for is respected
    r3 = s_rules.ask("Find repeated illegal dumping complaints within 150 feet of each other.")
    assert step(r3, "find_hotspots")["args"]["radius"] == 150


# ---------------------------------------------------------------------------------------------
# 5: LLM SQL cannot bypass the request scope
# ---------------------------------------------------------------------------------------------
def _sql_plan(sql: str) -> str:
    return plan(
        {
            "id": "s1",
            "tool": "query_local_data",
            "purpose": "select",
            "args": {"sql": sql, "purpose": "dumping"},
        },
        S_HOT,
    )


def test_5_llm_sql_date_filter_is_removed_before_approval(tmp_path, env):
    sql = (
        f"SELECT * FROM {DS} WHERE complaint_type = 'Illegal Dumping' "
        "AND created_date BETWEEN '2020-01-01' AND '2023-12-31' LIMIT 500"
    )
    fake = FakeOllama([_sql_plan(sql)])
    s, _ = session(tmp_path, env, fake)
    r = s.ask(QUESTION)
    assert r.status == "awaiting_approval"
    shown = r.pending_approval["queries"][0]["sql"]
    assert "2020" not in shown and "Illegal Dumping" in shown and not re.search(r"LIMIT 500\b", shown)
    r2 = s.resume(True)
    assert r2.status == "complete" and step(r2, "query_local_data")["rows"] == ROWS
    assert step(r2, "find_hotspots")["facts"]["clusters"] == CLUSTERS
    assert any("Removed SQL condition" in m for m in corrections(r2, "removed"))


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT * FROM {DS} WHERE created_date >= '2020-01-01'",  # no category filter left
        f"SELECT * FROM {DS} WHERE NOT (complaint_type <> 'Illegal Dumping' OR YEAR(created_date) < 2024)",
    ],
    ids=["missing_category", "inside_not"],
)
def test_5_unrepairable_llm_sql_falls_back_to_rules(tmp_path, env, sql):
    fake = FakeOllama([_sql_plan(sql)])
    s, _ = session(tmp_path, env, fake)
    r = s.ask(QUESTION)
    assert r.status == "complete" and r.plan_source == "rules_fallback"
    assert step(r, "filter_dataset")["rows"] == ROWS
    assert corrections(r, "rejected") and planner_calls(fake) == 1


def test_5_scope_is_enforced_on_the_execution_path(tmp_path, env):
    reg, eng = env
    scope = QuestionScope.from_intent(
        {
            "question": QUESTION,
            "categories": ["Illegal Dumping"],
            "subject_dataset": DS,
            "categories_by_dataset": {DS: ["Illegal Dumping"]},
        },
        QUESTION,
    )
    ctx = ToolContext(
        settings=make_settings(tmp_path, sql_approval="never"),
        registry=reg,
        mode="local",
        store=ResultStore(),
        engines={"local": eng},
        guard=SQLGuard(),
        audit=AuditLogger(None),
        scope=scope.to_dict(),
    )
    bad = f"SELECT * FROM {DS} WHERE complaint_type = 'Illegal Dumping' AND YEAR(created_date) = 2021"
    review = ctx.validate_sql(RenderedSQL(sql=bad, params={}, backend="local", kind="llm_sql"))
    assert not review.ok and any(e.startswith("Request scope") for e in review.errors)
    out = build_default_registry().invoke("query_local_data", {"sql": bad}, ctx)
    assert out.status == "error" and "Request scope" in out.summary
    ok = f"SELECT * FROM {DS} WHERE complaint_type = 'Illegal Dumping'"
    assert ctx.validate_sql(RenderedSQL(sql=ok, params={}, backend="local", kind="llm_sql")).ok
    # template SQL from structured tools is not affected (its filters were validated in the plan)
    fixed = check_sql_scope(bad, "duckdb", scope, reg, "local", "local")
    assert "2021" not in fixed.sql and not fixed.violations


# ---------------------------------------------------------------------------------------------
# 6: all relevant records stay accessible
# ---------------------------------------------------------------------------------------------
def test_6_all_matching_records_are_used_at_scale(tmp_path, scaled_env):
    fake = FakeOllama([plan(s_filter(categories=["Illegal Dumping"], limit=1000), interpretation="Dumping")])
    s, _ = session(tmp_path, scaled_env, fake)
    r = s.ask("Show illegal dumping complaints")
    f = step(r, "filter_dataset")
    assert f["rows"] == 20_323 and not f["facts"]["truncated"]
    assert any("Removed the row limit 1000" in m for m in corrections(r))
    assert r.data_scope["datasets"][0]["matching_rows"] == 20_323
    s2, _ = session(tmp_path, scaled_env)
    assert step(s2.ask("Show illegal dumping complaints"), "filter_dataset")["rows"] == 20_323


# ---------------------------------------------------------------------------------------------
# 8-9: empty results and missing artifacts
# ---------------------------------------------------------------------------------------------
def test_8_empty_selection_skips_later_steps_without_errors_or_replans(tmp_path, env):
    q = "Find repeated illegal dumping complaints at 1 Nonexistent Place"
    p = plan(
        s_filter(
            categories=["Illegal Dumping"],
            filters=[{"column": "incident_address", "op": "=", "value": "1 NONEXISTENT PLACE"}],
        ),
        S_HOT,
        S_MAP,
        interpretation="Repeated dumping at 1 Nonexistent Place",
    )
    fake = FakeOllama([p])
    s, _ = session(tmp_path, env, fake)
    r = s.ask(q)
    assert r.status == "complete"
    f = step(r, "filter_dataset")
    assert f["status"] == "empty" and f["reason"] == "empty_filter" and f["rows"] == 0
    for later in r.steps[1:]:
        assert later["status"] == "skipped" and later["reason"] == "empty_input"
    assert "unknown input reference" not in json.dumps(r.steps).lower()
    assert planner_calls(fake) == 1 and narrator_calls(fake) == 0
    a = r.answer
    assert a["headline"].startswith("No matching records in the available data")
    assert "not evidence that no incidents occurred" in a["summary"]


def test_9_failed_step_skips_dependents_and_replans_at_most_once(tmp_path, env):
    bad = plan(
        s_filter(categories=["Illegal Dumping"]),
        {
            "id": "s2",
            "tool": "analyze_temporal_patterns",
            "purpose": "trend",
            "args": {"input": "s1", "date_column": "no_such_column"},
        },
        {"id": "s3", "tool": "summarize_results", "purpose": "describe", "args": {"input": "s2"}},
    )
    fake = FakeOllama([bad])
    s, _ = session(tmp_path, env, fake)
    r = s.ask("Show illegal dumping complaints over time")
    s2, s3 = step(r, "analyze_temporal_patterns"), step(r, "summarize_results")
    assert s2["status"] == "error" and s2["reason"] == "error"
    assert s3["status"] == "skipped" and s3["reason"] == "dependency_failed"
    assert "unknown input reference" not in json.dumps(r.steps).lower()
    assert planner_calls(fake) == 2, "one replan for a genuine error, then stop"
    assert r.status == "complete"
    assert any("s2 (analyze_temporal_patterns) did not complete" in x for x in r.answer["limitations"])
    assert any("s3 (summarize_results) was skipped" in x for x in r.answer["limitations"])


def test_9_empty_input_tool_output_is_a_skip_not_an_error(tmp_path, env):
    reg, eng = env
    ctx = ToolContext(
        settings=make_settings(tmp_path),
        registry=reg,
        mode="local",
        store=ResultStore(),
        engines={"local": eng},
        guard=SQLGuard(),
        audit=AuditLogger(None),
    )
    tools = build_default_registry()
    ctx.step_id = "s1"
    out = tools.invoke(
        "filter_dataset",
        {"dataset": DS, "filters": [{"column": "incident_address", "op": "=", "value": "nowhere"}]},
        ctx,
    )
    assert out.status == "empty" and out.reason == "empty_filter" and out.artifact_id
    ctx.step_id = "s2"
    hs = tools.invoke("find_hotspots", {"input": "s1"}, ctx)
    assert hs.status == "skipped" and hs.reason == "empty_input"
    ctx.step_id = "s3"
    summary = tools.invoke("summarize_results", {"input": "s1"}, ctx)
    assert summary.status == "ok" and summary.rows == 0


# ---------------------------------------------------------------------------------------------
# 10: narration must agree with the computed results
# ---------------------------------------------------------------------------------------------
def _narration(headline: str, summary: str, findings: list[str]) -> str:
    return json.dumps(
        {
            "headline": headline,
            "summary": summary,
            "findings": [{"kind": "observation", "text": t} for t in findings],
            "limitations": [],
            "suggested_follow_ups": [],
        }
    )


def test_10_contradictory_narrative_falls_back_to_template(tmp_path, env):
    wrong = _narration(
        "No illegal dumping complaints were found between 2020 and 2023",
        "There were no illegal dumping complaints in the data for this period.",
        ["Zero complaints were recorded."],
    )
    fake = FakeOllama([INVENTED], narrator=wrong)
    s, _ = session(tmp_path, env, fake)
    r = s.ask(QUESTION)
    a = r.answer
    assert a["narrator"] == "template" and a["narrative_rejected"]
    assert any("year 2020" in p for p in a["narrative_rejected"])
    assert any("no records were found" in p for p in a["narrative_rejected"])
    assert a["headline"].startswith(f"{CLUSTERS} locations")
    assert any(t["title"].startswith("LLM answer rejected") for t in r.trace)
    log = (tmp_path / "audit.jsonl").read_text()
    assert '"event": "narrative_rejected"' in log


def test_10_consistent_narrative_is_kept(tmp_path, env):
    good = _narration(
        f"{CLUSTERS} places have repeated illegal dumping reports",
        f"{CLUSTERED:,} of the {ROWS:,} reports fall in {CLUSTERS} clusters of 5 or more reports within 75 ft.",
        ["The largest cluster has 22 reports."],
    )
    fake = FakeOllama([INVENTED], narrator=good)
    s, _ = session(tmp_path, env, fake)
    r = s.ask(QUESTION)
    assert r.answer["narrator"].startswith("ollama:") and not r.answer["narrative_rejected"]
    prompt = next(
        x for x in fake.requests if not x["messages"][0]["content"].startswith("You are the planning")
    )
    body = prompt["messages"][1]["content"]
    assert "2026-05-15" in body and "2010" not in body, "narrator sees stored coverage, not publisher years"


def test_grounding_checks_numbers_years_places_and_absence():
    ev = Evidence(
        numbers={1599.0, 28.0, 350.0, 22.0}, years={2026}, text=" QUEENS BROOKLYN ", records=1599, clusters=28
    )
    ok = Answer(headline="28 clusters", summary="About 1,600 reports; 350 clustered in 2026.", findings=[])
    assert check_narrative(ok, ev) == []
    bad = Answer(
        headline="No complaints were found in 2021",
        summary="The Bronx had 4,000 reports.",
        findings=[Finding(kind="observation", text="There were no clusters.")],
    )
    problems = " | ".join(check_narrative(bad, ev, ["BUSHWICK"]))
    assert "2021" in problems and "4,000" in problems and "Bronx" in problems
    assert "no records were found" in problems and "no clusters" in problems


# ---------------------------------------------------------------------------------------------
# 11-12: follow-ups and replans keep the request scope
# ---------------------------------------------------------------------------------------------
def test_11_follow_up_date_changes(tmp_path, env):
    s, _ = session(tmp_path, env)
    r1 = s.ask("Find repeated illegal dumping complaints in 2026.")
    assert step(r1, "filter_dataset")["args"]["date_from"] == "2026-01-01"
    r2 = s.ask("Now only June")
    f2 = step(r2, "filter_dataset")
    assert (f2["args"]["date_from"], f2["args"]["date_to"]) == ("2026-06-01", "2026-06-30")
    assert f2["args"]["categories"] == ["Illegal Dumping"]
    assert 0 < f2["rows"] < ROWS
    r3 = s.ask("Use all available dates")
    f3 = step(r3, "filter_dataset")
    assert "date_from" not in f3["args"] and f3["rows"] == ROWS


def test_11_llm_follow_up_keeps_the_new_period(tmp_path, env):
    # the model keeps the old 2026 window for "only June"; the validator narrows it to June
    plans = [
        plan(s_filter(categories=["Illegal Dumping"], date_from="2026-01-01", date_to="2026-12-31"), S_HOT),
        plan(s_filter(categories=["Illegal Dumping"], date_from="2026-01-01", date_to="2026-12-31"), S_HOT),
    ]
    fake = FakeOllama(plans)
    s, _ = session(tmp_path, env, fake)
    s.ask("Find repeated illegal dumping complaints in 2026.")
    r = s.ask("Now only June")
    f = step(r, "filter_dataset")
    assert (f["args"]["date_from"], f["args"]["date_to"]) == ("2026-06-01", "2026-06-30")
    assert any("Changed the date filter" in m for m in corrections(r, "changed"))


def test_12_replanned_plan_cannot_reintroduce_invented_dates(tmp_path, env):
    failing_sql = _sql_plan(
        f"SELECT * FROM {DS} WHERE complaint_type = 'Illegal Dumping' AND no_such_column IS NOT NULL"
    )
    fake = FakeOllama([failing_sql, INVENTED])
    s, _ = session(tmp_path, env, fake, sql_approval="never")
    r = s.ask(QUESTION)
    assert planner_calls(fake) == 2, "the SQL error triggered exactly one replan"
    assert r.status == "complete" and r.plan_source == "llm"
    f = step(r, "filter_dataset")
    assert "date_from" not in f["args"] and f["rows"] == ROWS
    assert step(r, "find_hotspots")["facts"]["clusters"] == CLUSTERS
    assert any("Removed the date filter" in m for m in corrections(r, "removed"))


# ---------------------------------------------------------------------------------------------
# 13: approval and access controls are unchanged
# ---------------------------------------------------------------------------------------------
def test_13_approval_rejection_and_allowlist_still_apply(tmp_path, env):
    sql = f"SELECT * FROM {DS} WHERE complaint_type = 'Illegal Dumping'"
    fake = FakeOllama([_sql_plan(sql)])
    s, _ = session(tmp_path, env, fake)
    r = s.ask(QUESTION)
    assert r.status == "awaiting_approval" and not r.steps
    r2 = s.resume(False)
    assert r2.status == "rejected" and not r2.steps
    # tables outside the local-mode allowlist (here: the demo table) stay blocked
    reg, eng = env
    ctx = ToolContext(
        settings=make_settings(tmp_path),
        registry=reg,
        mode="local",
        store=ResultStore(),
        engines={"local": eng},
        guard=SQLGuard(),
        audit=AuditLogger(None),
        scope=QuestionScope(question=QUESTION, texts=[QUESTION]).to_dict(),
    )
    review = ctx.validate_sql(
        RenderedSQL(sql="SELECT * FROM demo_311_sanitation", params={}, backend="local", kind="llm_sql")
    )
    assert not review.ok
    # structured-tool SQL in an always-approve session still waits for approval
    s3, _ = session(tmp_path, env, sql_approval="always")
    r3 = s3.ask(QUESTION)
    assert r3.status == "awaiting_approval" and "$p0" in r3.pending_approval["queries"][0]["sql"]


# ---------------------------------------------------------------------------------------------
# Temporal parser
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("repeated illegal dumping complaints", (None, None, "none")),
        ("dumping in 2026", ("2026-01-01", "2026-12-31", "explicit")),
        ("between 2020 and 2023", ("2020-01-01", "2023-12-31", "explicit")),
        ("2020-2023 complaints", ("2020-01-01", "2023-12-31", "explicit")),
        ("since 2024", ("2024-01-01", None, "explicit")),
        ("before 2024", (None, "2023-12-31", "explicit")),
        ("in May 2026", ("2026-05-01", "2026-05-31", "explicit")),
        ("Q2 2026", ("2026-04-01", "2026-06-30", "explicit")),
        ("from 2026-06-01 to 2026-06-15", ("2026-06-01", "2026-06-15", "explicit")),
        ("in the last 6 months", ("2026-04-30", "2026-10-30", "relative")),
        ("last year", ("2025-01-01", "2025-12-31", "relative")),
        ("in June", ("2026-06-01", "2026-06-30", "month")),
        ("within 2000 feet of 2020 Broadway", (None, None, "none")),
        ("more than 2000 complaints", (None, None, "none")),
        ("complaints that may be repeated", (None, None, "none")),
        ("recent complaints", (None, None, "vague")),
        ("use all available dates", (None, None, "all")),
    ],
)
def test_time_window_parsing(text, expected):
    tw = parse_time_window(text, TODAY)
    assert (tw.date_from, tw.date_to, tw.kind) == expected
