from pathlib import Path

import pytest
import yaml

from agentic_gis.config.registry import DatasetRegistry, RegistryError, UnknownDatasetError
from agentic_gis.config.settings import Settings
from tests.conftest import make_settings

ROOT = Path(__file__).resolve().parents[1]


def test_registry_modes(registry):
    demo = {d.id for d in registry.available("demo")}
    assert {"demo_311_sanitation", "nyc_boroughs", "demo_catch_basins"} <= demo
    assert not any(d.backend == "databricks" for d in registry.available("databricks"))  # examples disabled
    assert registry.geography("neighborhood", "demo").id == "nyc_community_districts"
    with pytest.raises(UnknownDatasetError):
        registry.get("dsny_catch_basins_exploded", "databricks")


def test_allowlist_and_search(registry):
    allowed = registry.allowed_tables("demo", "local")
    assert allowed["demo_311_sanitation"] == "demo_311_sanitation"
    assert registry.search("catch basins near streets", "demo")[0][0].id in {
        "demo_catch_basins",
        "demo_street_corridors",
    }


def test_env_substitution_and_overlay(tmp_path):
    base = yaml.safe_load((ROOT / "config/datasets.yaml").read_text())
    p = tmp_path / "d.yaml"
    p.write_text(yaml.safe_dump(base))
    overlay = tmp_path / "o.yaml"
    overlay.write_text(yaml.safe_dump({"datasets": [{"id": "dsny_catch_basins_exploded", "enabled": True}]}))
    reg = DatasetRegistry.load(
        p, overlay, env={"DATABRICKS_CATALOG": "prod_cat", "DATABRICKS_SCHEMA": "sanit"}
    )
    ds = reg.get_any("dsny_catch_basins_exploded")
    assert ds.enabled and ds.source.table == "prod_cat.sanit.catch_basins_exploded_final"
    with pytest.raises(RegistryError):
        DatasetRegistry.load(p, None, env={"DATABRICKS_CATALOG": "x; DROP"})


def test_settings_secrets_hidden(tmp_path):
    s = make_settings(
        tmp_path,
        databricks_server_hostname="https://adb-1.azuredatabricks.net/",
        databricks_http_path="/sql/1.0/warehouses/abc",
        databricks_token="dapi" + "0" * 32,
    )
    assert s.databricks_server_hostname == "adb-1.azuredatabricks.net"
    assert s.databricks_configured
    assert "dapi" not in repr(s) and "dapi" not in str(s.public_summary())
    assert s.secret_values() == ["dapi" + "0" * 32]


def test_settings_env_names(monkeypatch):
    monkeypatch.setenv("DATABRICKS_SERVER_HOSTNAME", "adb-2.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/x")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi" + "1" * 32)
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
    monkeypatch.setenv("AGIS_MAX_RESULT_ROWS", "123")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.databricks_configured and s.ollama_model == "llama3.1:8b" and s.max_result_rows == 123


def test_register_writes_overlay(tmp_path, registry):
    from agentic_gis.config.registry import DatasetSpec, SourceSpec

    reg = DatasetRegistry(registry.all(), registry.geographies)
    spec = DatasetSpec(
        id="uc_test",
        name="UC test",
        description="d",
        backend="databricks",
        modes=["databricks"],
        source=SourceSpec(type="databricks_table", table="a.b.c"),
        schema_discovery=True,
    )
    out = tmp_path / "local.yaml"
    reg.register(spec, out)
    assert "uc_test" in out.read_text()
    reloaded = DatasetRegistry.load(ROOT / "config/datasets.yaml", out, env={})
    assert reloaded.get_any("uc_test").source.table == "a.b.c"
