# Testing

```powershell
uv run pytest                    # offline suite (no network, no LLM, no Databricks)
uv run pytest -m ollama          # + live local model   (set OLLAMA_LIVE_TESTS=1)
uv run pytest -m databricks      # + live warehouse     (set DATABRICKS_LIVE_TESTS=1 and DATABRICKS_*)
uv run ruff check . ; uv run ruff format --check . ; uv run mypy agentic_gis app.py
```

Result in the development environment (Linux, Python 3.13): **97 passed, 2 skipped** (the two opt-in live
tests); ruff and mypy clean.

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
| UI | layer schema, diverging/categorical roles, Folium render uses only vendored assets + legend + synthetic badge, PyDeck render, client-side filters, Streamlit `AppTest` smoke test (question round trip) |
| Ingestion / imagery / CLI | Socrata paging + type coercion + NYC bounds + provenance sidecar (mocked HTTP), tile grid, in-memory embedding search returns *candidates*, Clay placeholder refuses, CLI `validate-sql` / `datasets` / `ask --export` / `layer-schema` |

The fake Ollama server and fake Databricks connection in `tests/fakes/` are **test doubles**: they verify
the HTTP/connector wiring, validation, approval and fallback logic, not the quality of a real model or the
behavior of a real warehouse.

## Verified manually in the development environment

* Streamlit app served and exercised in headless Chromium (screenshots of chat, approval-free demo flow,
  choropleth, hot-spot grid, repeat-location markers, street segments, proximity results).
* `agentic-gis check` and `agentic-gis ask` with the deterministic planner.

## Not verified in the development environment

The development sandbox could not reach these services, so the following were **not** tested:

| Component | Why | How to verify |
|---|---|---|
| A real Ollama model producing plans | Ollama downloads and model registries were blocked | `uv run agentic-gis ask --provider ollama -v "…"` and `$env:OLLAMA_LIVE_TESTS="1"; uv run pytest -m ollama` |
| A real Databricks SQL warehouse | No credentials/workspace | `uv run agentic-gis check`, `databricks-tables`, then `$env:DATABRICKS_LIVE_TESTS="1"; uv run pytest -m databricks` |
| OAuth U2M / M2M flows | Same | `DATABRICKS_AUTH_TYPE=oauth-u2m` then `agentic-gis check` |
| Windows / PowerShell | Linux sandbox | Follow [SETUP_WINDOWS.md](SETUP_WINDOWS.md); run `uv run pytest` |
| NYC Open Data ingestion against the live API, and the Socrata ids marked “verify” | Portal blocked | `uv run agentic-gis ingest nyc_311_sanitation --limit 1000` |
| Basemap tiles (CARTO, OSM, NYC OTI) | Tile servers blocked; maps were checked with “No basemap” | Open the app and switch basemaps |
| `INSTALL spatial` from the DuckDB extension repository | Blocked; the offline wheel path was tested | `uv run agentic-gis check` on a networked machine |
| NYC GeoSearch geocoder option | Blocked | `AGIS_GEOCODER=nyc_geosearch` and ask about an address |
| Databricks Apps deployment (`app.yaml`) | No workspace | See DATABRICKS.md |
| Imagery reads (rasterio COG), Qdrant store | No imagery/index (Phase 2) | — |
