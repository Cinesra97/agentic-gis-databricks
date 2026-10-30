# Handoff

_Last updated: 2026-10-30_

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
| Reliability fix (2026-10-30): invented LLM filters, coverage, empty results, grounded answers | Done; tested offline with a synthetic 2026 fixture and a scripted fake model; **not yet run with a real model or your ingested file** |

Checks at handoff: 137 tests passed, 2 opt-in live tests skipped; ruff (lint + format) and mypy clean.

## Reliability fix (2026-10-30)

Reported: with qwen2.5:7b, “Find locations in NYC with repeated illegal dumping complaints” on the local
`nyc_311_sanitation` file (150,000 rows, 2026-05-15 to 2026-10-27) returned “no complaints”.

Root causes and fixes:

1. **The model invented a 2020–2023 date filter** (the question has no period; the prompt's first example
   used a 2023 filter and the dataset publisher text says “2010 to Present”). → The planner prompt now has
   a “Request scope” section, stored coverage per dataset, and an example without dates (supplementary).
2. **Validation was structural only**, so the filter was accepted. → `agent/semantics.py` checks every
   plan (LLM, repair, replan, rules) against the scope parsed deterministically from the question
   (`agent/temporal.py` for periods): unrequested dates/boroughs/categories/filters/limits are removed,
   requested ones added, defaults made consistent with the rule planner; LLM SQL gets the same checks in
   planning and again before execution; unfixable plans fall back to the deterministic plan (no extra call).
3. **The empty result broke the dependency chain** (s2 returned no artifact; s3 failed with “Unknown input
   reference 's2'”). → Steps get reason codes; steps with empty or missing inputs are skipped.
4. **Reflect replanned on that symptom.** → Replans only for genuine execution errors.
5. **The LLM narrative was not checked** and claimed “no complaints”. → `agent/grounding.py` checks numbers,
   years, places and absence claims against the results; contradictions fall back to the deterministic
   summary. Narration is skipped for empty selections.
6. **Stored coverage was invisible** (the configured `since 2023` filter is not what was stored: the download
   stopped at its 150,000-row limit, newest first). → `connectors/coverage.py` measures rows and dates at
   load; out-of-range periods get an “insufficient coverage” answer; partial coverage and truncated
   downloads are disclosed in the answer, Summary and Data & methods tabs, sidebar catalog, CLI.

Verify on your machine (Windows PowerShell):

```powershell
git pull
uv sync --all-extras
uv run agentic-gis check                      # shows nyc_311_sanitation stored rows/dates, flags the row limit
uv run agentic-gis ask --provider ollama -v "Find locations in NYC with repeated illegal dumping complaints."
#   expect: [data scope] period: all available dates; 20,323 matching records; 448 clusters;
#   any [corrected] lines list filters the validator removed
uv run agentic-gis ask --provider ollama -v "Find repeated illegal dumping complaints between 2020 and 2023."
#   expect: [insufficient_coverage] with the stored dates
uv run streamlit run app.py                   # restart so the new code and coverage profiling load
```

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
* Request-scope validation relies on the deterministic intent parser. A category or place the parser does not
  recognize is only kept if its words appear in the question; LLM parameters the question does not set
  (cluster radius, grid size, proximity distance) are replaced by the documented defaults so LLM and rule
  plans agree. Relative periods (“last 6 months”) are resolved against today's date, and a month without a
  year means its most recent occurrence (or the previous question's year in a follow-up).
* Coverage is measured for local datasets only; Databricks tables report “coverage not measured” (measuring
  would need a warehouse query and approval).
* The narrative fact check is conservative (exact or rounded numbers, years, borough/vocabulary place names,
  absence claims); a correct LLM sentence with a derived number it cannot verify falls back to the template.
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
