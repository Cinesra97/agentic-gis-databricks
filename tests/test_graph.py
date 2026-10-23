"""End-to-end LangGraph workflow tests using the deterministic planner (no LLM)."""

import pytest

from agentic_gis.agent.graph import AgentSession, create_runtime
from tests.conftest import make_settings

EXAMPLES = [
    ("Find locations in NYC with repeated illegal dumping complaints.", "find_hotspots"),
    ("Identify areas with high concentrations of derelict vehicle complaints.", "find_hotspots"),
    ("Find catch basins within 300 feet of Atlantic Avenue and Flatbush Avenue.", "spatial_join"),
    ("Identify streets with repeated missed sweeping complaints.", "aggregate_by_geography"),
    ("Show sanitation complaints near industrial properties.", "spatial_join"),
    ("Compare dumping patterns across NYC boroughs over time.", "analyze_temporal_patterns"),
    ("Find areas where multiple types of sanitation complaints overlap.", "aggregate_by_geography"),
    ("Which neighborhoods have the largest increase in sanitation-related complaints?", "compare_periods"),
]


@pytest.fixture()
def session(tmp_path, registry, demo_engine):
    rt = create_runtime(
        make_settings(tmp_path),
        mode="demo",
        llm_provider="demo",
        engines={"local": demo_engine},
        registry=registry,
    )
    return AgentSession(rt)


@pytest.mark.parametrize("question,tool", EXAMPLES)
def test_example_questions_end_to_end(session, question, tool):
    r = session.ask(question)
    assert r.status == "complete", r.trace
    assert any(s["tool"] == tool and s["status"] == "ok" for s in r.steps), r.steps
    assert r.layers, "a map layer should be produced"
    assert r.citations and r.methodology
    ans = r.answer
    assert "synthetic" in ans["headline"].lower() and any("SYNTHETIC" in x for x in ans["limitations"])
    assert all(f["kind"] in ("observation", "inference") for f in ans["findings"])
    text = (ans["headline"] + ans["summary"] + " ".join(f["text"] for f in ans["findings"])).lower()
    assert "is an illegal dumping site" not in text
    assert any(s["node"] == "plan" for s in r.trace) and any(s["node"] == "review_sql" for s in r.trace)
    assert all(rv["ok"] for rv in r.sql_reviews)


def test_follow_up_uses_context(session):
    r1 = session.ask("Find locations in NYC with repeated illegal dumping complaints.")
    r2 = session.ask("Now show only Brooklyn")
    assert r2.status == "complete"
    f = next(s for s in r2.steps if s["tool"] == "filter_dataset")
    assert f["args"]["boroughs"] == ["Brooklyn"] and f["args"]["categories"] == ["Illegal Dumping"]
    r3 = session.ask("Which of those are near industrial properties?")
    join = next(s for s in r3.steps if s["tool"] == "spatial_join")
    assert join["args"]["left"] == "last" and join["status"] == "ok"
    assert r1.primary_artifact != r3.primary_artifact
    assert len(session.messages()) == 6


def test_clarification_round_trip(session):
    r = session.ask("Find catch basins within 300 feet of specified streets.")
    assert r.status == "clarification" and "streets" in r.answer["headline"].lower()
    r2 = session.ask("Grand Concourse")
    assert r2.status == "complete"
    f = next(
        s
        for s in r2.steps
        if s["tool"] == "filter_dataset" and s["args"]["dataset"] == "demo_street_corridors"
    )
    assert f["args"]["filters"][0]["value"] == ["GRAND CONCOURSE"]


def test_sql_approval_interrupt_and_resume(tmp_path, registry, demo_engine):
    rt = create_runtime(
        make_settings(tmp_path, sql_approval="always"),
        mode="demo",
        llm_provider="demo",
        engines={"local": demo_engine},
        registry=registry,
    )
    s = AgentSession(rt)
    r = s.ask("Find locations in NYC with repeated illegal dumping complaints.")
    assert r.status == "awaiting_approval" and r.pending_approval["queries"]
    assert not r.steps, "nothing may execute before approval"
    q = r.pending_approval["queries"][0]
    assert q["sql"].upper().startswith("SELECT") and "$p0" in q["sql"]
    r2 = s.resume(True)
    assert r2.status == "complete" and any(st["tool"] == "find_hotspots" for st in r2.steps)
    # The same SQL was approved earlier in this session, so a repeat question runs without asking again.
    r3 = s.ask("Find locations in NYC with repeated illegal dumping complaints.")
    assert r3.status == "complete"


def test_sql_rejection_runs_nothing(tmp_path, registry, demo_engine):
    rt = create_runtime(
        make_settings(tmp_path, sql_approval="always"),
        mode="demo",
        llm_provider="demo",
        engines={"local": demo_engine},
        registry=registry,
    )
    s = AgentSession(rt)
    s.ask("Show derelict vehicle complaints in Queens")
    r = s.resume(False)
    assert r.status == "rejected" and not r.steps and "No queries were run" in r.answer["headline"]
    assert s.pending_approval is None


def test_audit_log_written_without_secrets(tmp_path, registry, demo_engine):
    st = make_settings(tmp_path, databricks_token="dapi" + "a" * 32)
    rt = create_runtime(
        st, mode="demo", llm_provider="demo", engines={"local": demo_engine}, registry=registry
    )
    AgentSession(rt).ask("Show litter basket complaints in Manhattan")
    log = (tmp_path / "audit.jsonl").read_text()
    assert '"event": "question"' in log and '"event": "sql_execute"' in log and '"event": "answer"' in log
    assert "dapi" + "a" * 32 not in log
