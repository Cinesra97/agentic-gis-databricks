"""Shared fixtures. Tests run fully offline with bundled sample data."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentic_gis.agent.tools import ToolContext, build_default_registry
from agentic_gis.agent.workspace import ResultStore
from agentic_gis.config.registry import DatasetRegistry, load_registry
from agentic_gis.config.settings import Settings
from agentic_gis.connectors.duckdb_engine import LocalDuckDBEngine
from agentic_gis.security.audit import AuditLogger
from agentic_gis.security.sql_guard import SQLGuard

# Never pick up a developer's real credentials during offline tests. Live tests (opt-in via
# DATABRICKS_LIVE_TESTS=1 / OLLAMA_LIVE_TESTS=1) keep the environment so they can connect.
LIVE = os.environ.get("DATABRICKS_LIVE_TESTS") == "1" or os.environ.get("OLLAMA_LIVE_TESTS") == "1"
if not LIVE:
    for _var in list(os.environ):
        if _var.startswith(("AGIS_", "DATABRICKS_", "OLLAMA_")):
            os.environ.pop(_var)


def make_settings(tmp_path: Path | None = None, **kw) -> Settings:
    base = {"llm_provider": "demo", "audit_log_path": (tmp_path or Path("/tmp")) / "audit.jsonl"}
    base.update(kw)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture(scope="session")
def registry() -> DatasetRegistry:
    return load_registry(Settings(_env_file=None))  # type: ignore[call-arg]


@pytest.fixture(scope="session")
def demo_engine(registry: DatasetRegistry) -> LocalDuckDBEngine:
    return LocalDuckDBEngine(registry, "demo")


@pytest.fixture()
def ctx(settings: Settings, registry: DatasetRegistry, demo_engine: LocalDuckDBEngine) -> ToolContext:
    return ToolContext(
        settings=settings,
        registry=registry,
        mode="demo",
        store=ResultStore(),
        engines={"local": demo_engine},
        guard=SQLGuard(),
        audit=AuditLogger(None),
    )


@pytest.fixture(scope="session")
def tools():
    return build_default_registry()
