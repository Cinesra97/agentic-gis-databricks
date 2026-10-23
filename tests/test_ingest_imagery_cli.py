"""Public-data ingestion (mocked Socrata), imagery interfaces, and CLI."""

import json
from pathlib import Path

import geopandas as gpd
import httpx
import pytest

from agentic_gis.cli import main as cli_main
from agentic_gis.connectors.base import ConnectorError
from agentic_gis.connectors.nyc_open_data import SocrataClient, ingest_dataset
from agentic_gis.imagery.embeddings import (
    ClayEmbeddingModel,
    EmbeddingRecord,
    InMemoryEmbeddingStore,
    NotConfiguredError,
)
from agentic_gis.imagery.search import CANDIDATE_NOTE, ImagerySearchService
from agentic_gis.imagery.tiles import tile_grid


def socrata_transport(rows):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        if "erm2-nwe9" not in request.url.path:
            return httpx.Response(404)
        off = int(request.url.params.get("$offset", 0))
        lim = int(request.url.params.get("$limit", 1000))
        return httpx.Response(200, json=rows[off : off + lim])

    return httpx.MockTransport(handler), seen


def test_socrata_ingest(tmp_path, registry):
    rows = [
        {
            "unique_key": str(i),
            "created_date": "2024-02-01T10:00:00.000",
            "agency": "DSNY",
            "complaint_type": "Illegal Dumping",
            "borough": "BROOKLYN",
            "latitude": "40.69",
            "longitude": "-73.94",
        }
        for i in range(25)
    ] + [{"unique_key": "999", "created_date": "2024-02-01T10:00:00.000", "latitude": "0", "longitude": "0"}]
    transport, seen = socrata_transport(rows)
    ds = registry.get_any("nyc_311_sanitation").model_copy(deep=True)
    ds.source.path = str(tmp_path / "out.parquet")
    client = SocrataClient(transport=transport)
    res = ingest_dataset(ds, client, since="2024-01-01", limit=26)
    assert res.rows == 25  # out-of-NYC coordinates dropped
    g = gpd.read_parquet(res.path)
    assert g.crs.to_epsg() == 4326 and str(g["created_date"].dtype).startswith("datetime64")
    assert "created_date >= '2024-01-01T00:00:00'" in seen[0]["$where"]
    meta = json.loads(Path(res.path).with_suffix(".provenance.json").read_text())
    assert meta["socrata_id"] == "erm2-nwe9" and meta["rows"] == 25
    with pytest.raises(ValueError):
        ingest_dataset(ds, client, since="2024-13-45")


def test_socrata_errors():
    transport, _ = socrata_transport([])
    with pytest.raises(ConnectorError):
        SocrataClient(transport=transport).query("abcd-1234")
    with pytest.raises(ValueError):
        SocrataClient(transport=transport).query("../../etc")


def test_tile_grid_and_embeddings():
    tiles = tile_grid([-73.96, 40.70, -73.95, 40.71], 250, "m", source_id="ortho2024")
    assert len(tiles) >= 16 and tiles.crs.to_epsg() == 4326 and tiles["tile_id"].is_unique
    store = InMemoryEmbeddingStore()
    recs = [
        EmbeddingRecord(
            tile_id=t,
            source_id="ortho2024",
            footprint_wkt=g.wkt,
            model_name="test",
            model_version="0",
            vector=[float(i == k) for k in range(4)],
        )
        for i, (t, g) in enumerate(zip(tiles["tile_id"][:4], tiles.geometry[:4], strict=True))
    ]
    store.upsert(recs)
    svc = ImagerySearchService(store)
    assert svc.status().available
    hits = svc.similar_to_vector([1, 0, 0, 0], k=2)
    assert hits[0].tile_id == recs[0].tile_id and hits[0].status == "candidate"
    assert hits[0].verification == "unverified" and hits[0].note == CANDIDATE_NOTE
    with pytest.raises(NotConfiguredError):
        ClayEmbeddingModel().embed([], [])
    assert not ImagerySearchService(None).status().available


def test_cli_commands(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("AGIS_AUDIT_LOG_PATH", str(tmp_path / "a.jsonl"))
    from agentic_gis.config.settings import reset_settings_cache

    reset_settings_cache()
    assert cli_main(["validate-sql", "SELECT borough FROM demo_311_sanitation"]) == 0
    assert cli_main(["validate-sql", "DROP TABLE demo_311_sanitation"]) == 1
    assert cli_main(["datasets", "--mode", "demo"]) == 0
    rc = cli_main(
        [
            "ask",
            "--provider",
            "demo",
            "--mode",
            "demo",
            "--export",
            str(tmp_path / "out"),
            "Which boroughs have the most derelict vehicle complaints?",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0 and "synthetic" in out.lower()
    assert list((tmp_path / "out").glob("*.geojson"))
    assert cli_main(["layer-schema"]) == 0
    reset_settings_cache()
