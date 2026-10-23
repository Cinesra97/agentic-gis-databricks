# Handoff

_Last updated: 2026-10-23_

## Repository and branches

* GitHub: https://github.com/Cinesra97/agentic-gis-databricks
* Working branch: **`claude/funny-mendel-8l1m9p`** (all work so far).
* The repository was empty before this work, so GitHub made this branch the **default branch**
  automatically. There is no `main` yet. Recommended: create `main` from this branch and make it the
  default (GitHub → Settings → Branches), or open a pull request into a `main` you create.

Resume anywhere:

```powershell
git clone https://github.com/Cinesra97/agentic-gis-databricks.git
cd agentic-gis-databricks
git checkout claude/funny-mendel-8l1m9p
uv sync --python 3.12
uv run pytest
uv run streamlit run app.py
```

## Status by milestone

| Milestone | Status |
|---|---|
| 1 — Local app (Streamlit, LangGraph, Ollama client, DuckDB + Spatial, GeoPandas analysis, dynamic maps, exports) | Done; tested offline with the deterministic planner and a fake Ollama server |
| 2 — Databricks (connector, auth, schema discovery, read-only guarded queries, approval, connection test) | Done; tested against a fake connection; **not yet run against a real warehouse** |
| 3 — Enhanced agent (follow-ups, clarification, replanning, provenance, error handling, live UC schema in prompts) | Done |
| 4 — Imagery architecture (tiling, embedding records/stores, candidate-only search, roadmap) | Interfaces + placeholders done; no imagery or model integrated (by design) |

Checks at handoff: 97 tests passed, 2 opt-in live tests skipped; ruff and mypy clean.

## Not verified yet (needs your environment)

1. **Real Ollama model** end-to-end: `uv run agentic-gis ask --provider ollama -v "Which boroughs have the
   most illegal dumping complaints?"` — confirm `plan source: llm`. Then
   `$env:OLLAMA_LIVE_TESTS="1"; uv run pytest -m ollama`.
2. **Databricks**: `uv run agentic-gis check`, `uv run agentic-gis databricks-tables`, register the DSNY tables
   in `config/datasets.local.yaml` after inspecting their columns (docs/DATABRICKS.md), then ask a question in
   Databricks mode and approve the SQL.
3. **Windows** setup per docs/SETUP_WINDOWS.md (developed on Linux).
4. NYC Open Data ingestion against the live API; confirm the Socrata ids marked “verify”.
5. Basemap tiles in your network (tile servers were blocked during development).

## Known limitations / next steps

* Streamlit has no login → deploy behind SSO; Databricks identity is shared per deployment. Next: on-behalf-
  of-user auth in Databricks Apps.
* The deterministic planner covers common analysis patterns; unusual questions depend on the LLM planner.
  Evaluate LLM plan quality with a set of real questions and tune prompts/examples (`agent/prompts.py`).
* Street corridors in the sample data are approximate; real street analysis needs LION/CSCL centerlines
  (ingest or register them).
* Neighborhoods in demo mode are community districts; ingest `nyc_nta_2020` for NTAs.
* Audit log is a local JSONL file; ship it to central logging for production.
* Phase 2 imagery: see docs/IMAGERY_ROADMAP.md.

## Key decisions (see docs/ARCHITECTURE.md)

JSON plans validated against tool schemas (not free-form tool calling); deterministic planner for demo and
fallback; SQLGlot AST guard + approval for generated SQL; template SQL with bound parameters; GeoPandas in
EPSG:2263 for spatial work; locked in-memory DuckDB; vendored Leaflet assets for offline maps; one
`agentic_gis` package to avoid import collisions.
