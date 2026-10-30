# Testing

```powershell
uv run pytest                    # offline suite (no network, no LLM, no Databricks)
uv run pytest -m ollama          # + live local model   (set OLLAMA_LIVE_TESTS=1)
uv run pytest -m databricks      # + live warehouse     (set DATABRICKS_LIVE_TESTS=1 and DATABRICKS_*)
uv run ruff check . ; uv run ruff format --check . ; uv run mypy agentic_gis app.py
```

Result in the development environment (Linux, Python 3.13), 2026-10-30: **158 passed, 2 skipped** (the
two opt-in live tests); ruff (lint + format) and mypy clean.

## What the offline suite covers

| Area | Tests |
|---|---|
| SQL guard | allowed queries; DDL/DML, multi-statement, file reads, table functions, `COPY`/`ATTACH`/`INSTALL`/`PRAGMA`/`SET`, `SELECT INTO`, unknown tables, Databricks `secret()`/`ai_query`/`http_request`/UDFs/path tables/`IDENTIFIER()`, sensitive columns, LIMIT enforcement, comment stripping, name qualification |
| Registry & settings | modes, allowlist, env substitution (and injection refusal), overlay merge, admin registration, secret masking, env variable names, warehouse-id → HTTP path |
| Spatial | unit conversion, buffer area in feet, dwithin/nearest distances, name matching, DBSCAN on a planted cluster, Gi* on a planted hot spot, aggregation, period comparison, temporal stats, grid count-distinct, geometry parsing |
| Tools | parameterized template SQL, unknown-column errors, approval gate for raw SQL, unsafe SQL rejection, spatial pipeline consistency (buffer+within = dwithin), every analysis tool, no code-execution tool, imagery never fabricates, registry extensibility |
| LangGraph workflow | all 8 example questions end-to-end; follow-ups (borough filter, “those near …”); clarification round trip; SQL approval interrupt/resume, session-level approval reuse; rejection runs nothing; audit log contents without secrets |
| LLM (fake Ollama transport) | JSON extraction, schema inlining, connection test / missing model / unreachable, repair after invalid output, LLM SQL requires approval then executes, invalid plan → fallback, unreachable → fallback, unsafe LLM SQL → blocked → replan, narrative guardrails (verified→inference, claim rewriting, synthetic notice), prompt-injection sanitization |
| Databricks (fake connection) | connect kwargs for PAT/OAuth, connection test, table and column discovery, native parameters, permission/auth error translation with redaction, Databricks-mode agent run with approval before any warehouse query, schema-comment injection sanitized, live UC columns in LLM prompt without row reads |
| Locations and narration (`test_locations_narration.py`) | A narrowed LLM column list (the one from the live run) still reads `incident_address`/`street_name`; rule-plan projection unchanged; simple LLM `SELECT`s widened before approval, aggregate/`*`/`DISTINCT` SQL untouched; LLM and rule plans name the same largest cluster (“near SYNTHETIC SITE 023, BROOKLYN”); every cluster has a location and center; missing addresses fall back to coordinates + area (also from reference boundaries when the data has no location columns); real-dataset answers never carry a synthetic label; the two false limitations from the live run are never shown; coverage/truncation claims must match the request and ingestion record; narration off by default, on-demand rewrite keeps computed limitations, refuses empty results, sends a 3-field schema with a 450-token cap; bounded ingestion (`--until`, match count, `:id` paging tie-breaker) and bounded-window guidance |
| Reliability (`test_reliability.py`) | Fixture: the synthetic 311 sample shifted to 2026-05-15..2026-10-27, registered as `nyc_311_sanitation` (local mode) with a “row limit reached” ingestion record — independent of any locally ingested file. Covers: stored coverage measured from data; no period in the question → no date filter (1,599 records, 28 clusters / 350 reports); “in 2026” → 2026 filter + partial-coverage disclosure; “between 2020 and 2023” → insufficient coverage, nothing executed (rules and LLM); invented LLM dates/borough/sub-type/limit removed (also inside `filters`) with identical clusters to the rule plan; LLM SQL date conditions removed before approval, unrepairable SQL → rules fallback, execution path blocks out-of-scope SQL; 20,323-record scaled selection complete; empty selection → later steps skipped, no error, no replan, no narrator call; failed step → dependents skipped, one replan max; contradictory LLM narrative → template (+ audit event), consistent narrative kept; follow-up period changes (rules and LLM); replans cannot reintroduce dates; approval/rejection/allowlist unchanged; 17 time-phrase parsing cases; narrative fact checks |
| UI | layer schema, diverging/categorical roles, Folium render uses only vendored assets + legend + synthetic badge, PyDeck render, client-side filters, Streamlit `AppTest` smoke test (question round trip) |
| Ingestion / imagery / CLI | Socrata paging + type coercion + NYC bounds + provenance sidecar (mocked HTTP), tile grid, in-memory embedding search returns *candidates*, Clay placeholder refuses, CLI `validate-sql` / `datasets` / `ask --export` / `layer-schema` |

The fake Ollama server and fake Databricks connection in `tests/fakes/` are **test doubles**: they verify
the HTTP/connector wiring, validation, approval and fallback logic, not the quality of a real model or the
behavior of a real warehouse.

## Verified manually in the development environment

* Streamlit app served and exercised in headless Chromium (screenshots of chat, approval-free demo flow,
  choropleth, hot-spot grid, repeat-location markers, street segments, proximity results).
* `agentic-gis check` and `agentic-gis ask` with the deterministic planner.
* Reliability fix (2026-10-30): the reported failure was reproduced on commit `046d711` with the 2026
  fixture and a scripted model plan containing the invented 2020–2023 filter (0 rows → s2 skipped → s3
  “Unknown input reference 's2'” → replan → “no complaints” narrative). On the fixed code the same input
  gives 1,599 records and 28 clusters with one planner call. Local-mode UI checked in headless Chromium:
  insufficient-coverage answer, Summary data-scope line and coverage warnings, Data & methods data-scope table.

## Verified by the user (live, 2026-10-30)

With qwen2.5:7b on the ingested 311 file (150,000 rows, 2026-05-15 to 2026-10-27), `AGIS_LLM_NARRATION=false`:
plan source `llm`, 20,323 Illegal Dumping records, 448 clusters (75 ft, ≥5) containing 3,457 reports, no
synthetic warning, no invented date range, 60.8 s (178.7 s with narration). Remaining issues from that run
(largest cluster “near QUEENS”, false model-written limitations, `--since`-only re-ingest advice) were fixed
afterwards and are covered offline by `test_locations_narration.py`.

## Not verified in the development environment

The development sandbox could not reach these services, so the following were **not** tested:

| Component | Why | How to verify |
|---|---|---|
| A real Ollama model producing plans | Ollama downloads and model registries were blocked | `uv run agentic-gis ask --provider ollama -v "…"` and `$env:OLLAMA_LIVE_TESTS="1"; uv run pytest -m ollama` |
| Location naming with a real qwen2.5:7b on your ingested file (follow-up of 2026-10-30) | No model or portal access here; tested with the live run's column list in a scripted plan | `uv run agentic-gis ask --mode local --provider ollama -v "Find locations in NYC with repeated illegal dumping complaints"` — the filter SQL should list `incident_address` and `street_name`, and the largest cluster should read “58 reports over 156 days near 184-54 149 AVENUE, QUEENS” |
| LLM narration speed after the changes | Not measured with a real model; the request is ~36% smaller and output is capped at 450 tokens (measured on the fixture) | Add `--narrate` to the command above and compare the `narration …s` value with your earlier 118 s |
| `ingest --until` and the match count against the live Socrata API | Portal blocked; tested with a mocked API | `uv run agentic-gis ingest nyc_311_sanitation --since 2020-01-01 --until 2020-01-31 --limit 50000` — prints “Records matching the filter on the portal: …” |
| A real Databricks SQL warehouse | No credentials/workspace | `uv run agentic-gis check`, `databricks-tables`, then `$env:DATABRICKS_LIVE_TESTS="1"; uv run pytest -m databricks` |
| OAuth U2M / M2M flows | Same | `DATABRICKS_AUTH_TYPE=oauth-u2m` then `agentic-gis check` |
| Windows / PowerShell | Linux sandbox | Follow [SETUP_WINDOWS.md](SETUP_WINDOWS.md); run `uv run pytest` |
| NYC Open Data ingestion against the live API, and the Socrata ids marked “verify” | Portal blocked | `uv run agentic-gis ingest nyc_311_sanitation --limit 1000` |
| Basemap tiles (CARTO, OSM, NYC OTI) | Tile servers blocked; maps were checked with “No basemap” | Open the app and switch basemaps |
| `INSTALL spatial` from the DuckDB extension repository | Blocked; the offline wheel path was tested | `uv run agentic-gis check` on a networked machine |
| NYC GeoSearch geocoder option | Blocked | `AGIS_GEOCODER=nyc_geosearch` and ask about an address |
| Databricks Apps deployment (`app.yaml`) | No workspace | See DATABRICKS.md |
| Imagery reads (rasterio COG), Qdrant store | No imagery/index (Phase 2) | — |
