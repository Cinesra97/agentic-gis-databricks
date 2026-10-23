"""Databricks connector + agent tests against an in-memory fake connection (offline)."""

import os

import geopandas as gpd
import pandas as pd
import pytest

from agentic_gis.agent.graph import AgentSession, create_runtime
from agentic_gis.config.registry import (
    ColumnSpec,
    DatasetRegistry,
    DatasetSpec,
    SourceSpec,
    SpatialSpec,
    TemporalSpec,
)
from agentic_gis.connectors.base import ConnectionNotConfiguredError, PermissionDeniedError
from agentic_gis.connectors.databricks_sql import DatabricksConnector, translate_error
from tests.conftest import make_settings
from tests.fakes.fake_databricks import FakeConnection, connect_factory

TOKEN = "dapi" + "f" * 32
FQ = "agency.sanitation.sr311"


def dbx_settings(tmp_path, **kw):
    return make_settings(
        tmp_path,
        databricks_server_hostname="adb-123.azuredatabricks.net",
        databricks_http_path="/sql/1.0/warehouses/abc",
        databricks_token=TOKEN,
        databricks_catalog="agency",
        databricks_schema="sanitation",
        **kw,
    )


@pytest.fixture(scope="module")
def sr311() -> pd.DataFrame:
    g = gpd.read_parquet("sample_data/synthetic_311_sanitation.parquet").head(3000)
    return pd.DataFrame(
        g.drop(columns="geometry")[
            [
                "unique_key",
                "created_date",
                "complaint_type",
                "borough",
                "incident_address",
                "latitude",
                "longitude",
            ]
        ]
    )


@pytest.fixture()
def dbx_registry(registry):
    spec = DatasetSpec(
        id="dbx_311",
        name="311 complaints (UC)",
        short_name="UC 311 complaints",
        description="test",
        backend="databricks",
        modes=["databricks"],
        role="events",
        sensitivity="internal",
        source=SourceSpec(type="databricks_table", table=FQ),
        category_column="complaint_type",
        spatial=SpatialSpec(
            format="latlon", lat_column="latitude", lon_column="longitude", geometry_type="point"
        ),
        temporal=TemporalSpec(column="created_date"),
        id_column="unique_key",
        columns=[
            ColumnSpec(name=c, type=t)
            for c, t in [
                ("unique_key", "integer"),
                ("created_date", "timestamp"),
                ("borough", "string"),
                ("incident_address", "string"),
                ("latitude", "double"),
                ("longitude", "double"),
            ]
        ]
        + [
            ColumnSpec(
                name="complaint_type",
                type="string",
                known_values=["Illegal Dumping", "Derelict Vehicles", "Missed Collection"],
            )
        ],
        join_keys={"borough": "borough"},
    )
    return DatasetRegistry([*registry.all(), spec], registry.geographies)


def test_connect_kwargs_and_auth_modes(tmp_path):
    kw = DatabricksConnector(dbx_settings(tmp_path)).connect_kwargs()
    assert kw["access_token"] == TOKEN and kw["catalog"] == "agency"
    assert kw["session_configuration"]["STATEMENT_TIMEOUT"] == "120"
    u2m = DatabricksConnector(dbx_settings(tmp_path, databricks_auth_type="oauth-u2m")).connect_kwargs()
    assert u2m["auth_type"] == "databricks-oauth" and "access_token" not in u2m
    with pytest.raises(ConnectionNotConfiguredError):
        DatabricksConnector(make_settings(tmp_path)).connect_kwargs()


def test_connection_test_and_discovery(tmp_path, sr311):
    conn = FakeConnection({FQ: sr311})
    c = DatabricksConnector(dbx_settings(tmp_path), connect_fn=connect_factory(conn))
    st = c.test_connection()
    assert st.ok and st.details["user"] == "analyst@example.gov"
    assert [t["table"] for t in c.list_tables()] == ["sr311"]
    cols = c.describe(FQ)
    assert "complaint_type" in [x.name for x in cols]


def test_execute_with_native_params(tmp_path, sr311):
    conn = FakeConnection({FQ: sr311})
    c = DatabricksConnector(dbx_settings(tmp_path), connect_fn=connect_factory(conn))
    res = c.execute(
        f"SELECT borough, COUNT(*) AS n FROM {FQ} WHERE complaint_type = :p0 GROUP BY 1",
        {"p0": "Illegal Dumping"},
        max_rows=100,
    )
    assert res.backend == "databricks" and res.rows >= 1
    assert conn.executed[-1][1] == {"p0": "Illegal Dumping"}


def test_permission_errors_are_friendly_and_redacted(tmp_path):
    err = translate_error(
        RuntimeError(f"[INSUFFICIENT_PERMISSIONS] User does not have SELECT. token={TOKEN}"), [TOKEN]
    )
    assert isinstance(err, PermissionDeniedError) and TOKEN not in str(err) and "grants" in str(err)
    conn = FakeConnection({}, fail_with="Invalid access token")
    st = DatabricksConnector(dbx_settings(tmp_path), connect_fn=connect_factory(conn)).test_connection()
    assert not st.ok and "authentication failed" in st.message.lower()


def test_agent_on_databricks_requires_approval(tmp_path, dbx_registry, sr311, demo_engine):
    from agentic_gis.connectors.duckdb_engine import LocalDuckDBEngine

    st = dbx_settings(tmp_path)
    conn = FakeConnection({FQ: sr311})
    engines = {
        "local": LocalDuckDBEngine(dbx_registry, "databricks"),
        "databricks": DatabricksConnector(st, connect_fn=connect_factory(conn)),
    }
    rt = create_runtime(st, mode="databricks", llm_provider="demo", engines=engines, registry=dbx_registry)
    session = AgentSession(rt)
    r = session.ask("Which boroughs have the most illegal dumping complaints?")
    assert r.status == "awaiting_approval"
    q = r.pending_approval["queries"][0]
    assert q["backend"] == "databricks" and FQ in q["sql"] and ":p0" in q["sql"]
    assert not conn.executed, "no warehouse query before approval"
    r2 = session.resume(True)
    assert r2.status == "complete"
    agg = next(s for s in r2.steps if s["tool"] == "aggregate_by_geography")
    assert agg["status"] == "ok" and agg["rows"] >= 1, [
        (s["tool"], s["status"], s["summary"]) for s in r2.steps
    ]
    assert any(FQ in sql for sql, _ in conn.executed)
    assert not any(c["synthetic"] for c in r2.citations if c["dataset_id"] == "dbx_311")


def test_schema_comment_injection_is_sanitized(tmp_path, dbx_registry, sr311):
    from agentic_gis.agent.tools import ToolContext, build_default_registry
    from agentic_gis.agent.workspace import ResultStore
    from agentic_gis.security.audit import AuditLogger
    from agentic_gis.security.sql_guard import SQLGuard

    st = dbx_settings(tmp_path)
    conn = FakeConnection({FQ: sr311})
    ctx = ToolContext(
        settings=st,
        registry=dbx_registry,
        mode="databricks",
        store=ResultStore(),
        engines={"databricks": DatabricksConnector(st, connect_fn=connect_factory(conn))},
        guard=SQLGuard(),
        audit=AuditLogger(None),
    )
    out = build_default_registry().invoke("get_schema", {"dataset": "dbx_311"}, ctx)
    assert out.status == "ok"
    assert all("ignore previous" not in (r["description"] or "").lower() for r in out.data)


@pytest.mark.databricks
def test_live_databricks_connection():
    """Real warehouse. Run with: DATABRICKS_LIVE_TESTS=1 uv run pytest -m databricks"""
    if os.environ.get("DATABRICKS_LIVE_TESTS") != "1":
        pytest.skip("set DATABRICKS_LIVE_TESTS=1 and DATABRICKS_* variables to enable")
    from agentic_gis.config.settings import Settings

    st = Settings()
    status = DatabricksConnector(st).test_connection()
    assert status.ok, status.message


def test_llm_prompt_includes_discovered_uc_columns(tmp_path, registry, sr311):
    import json

    from tests.fakes.fake_ollama import FakeOllama

    spec = DatasetSpec(
        id="dsny_discovered",
        name="Discovered UC table",
        description="schema discovered at runtime",
        backend="databricks",
        modes=["databricks"],
        role="events",
        sensitivity="internal",
        source=SourceSpec(type="databricks_table", table=FQ),
        schema_discovery=True,
    )
    reg = DatasetRegistry([*registry.all(), spec], registry.geographies)
    st = dbx_settings(tmp_path, llm_provider="ollama")
    conn = FakeConnection({FQ: sr311})
    from agentic_gis.connectors.duckdb_engine import LocalDuckDBEngine

    engines = {
        "local": LocalDuckDBEngine(reg, "databricks"),
        "databricks": DatabricksConnector(st, connect_fn=connect_factory(conn)),
    }
    fake = FakeOllama(
        [json.dumps({"interpretation": "x", "steps": [], "clarification": "Which columns matter?"})]
    )
    rt = create_runtime(st, mode="databricks", engines=engines, registry=reg, llm_transport=fake.transport())
    r = AgentSession(rt).ask("What is in the discovered table?")
    assert r.status == "clarification"
    prompt = fake.requests[0]["messages"][1]["content"]
    assert "complaint_type(" in prompt and "incident_address(" in prompt
    assert "ignore previous instructions" not in prompt.lower()
    assert not any("SELECT" in sql and "current_user" not in sql for sql, _ in conn.executed), "no row reads"
