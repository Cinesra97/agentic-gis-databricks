# Architecture

## Components

```mermaid
flowchart TB
  subgraph UI[Streamlit UI · app.py + agentic_gis/ui]
    chat[Chat + SQL approval] --- map[Map: Folium/Leaflet or PyDeck/deck.gl]
    map --- tabs[Results · Chart · Reasoning & SQL · Data & methods · Export]
  end
  subgraph Agent[LangGraph agent · agentic_gis/agent]
    graph[graph.py workflow] --> planner[planner.py<br/>LLM plan or rule plan]
    graph --> toolsreg[tools.py<br/>tool registry]
    graph --> narr[narrative.py<br/>answer + guardrails]
    planner --> llm[llm.py<br/>Ollama / OpenAI-compatible]
    toolsreg --> store[(workspace.py<br/>artifact store)]
  end
  subgraph Data[Data access · agentic_gis/connectors]
    guard[security/sql_guard.py] --> duck[(DuckDB in-memory<br/>+ Spatial)]
    guard --> dbx[(Databricks SQL warehouse<br/>Unity Catalog)]
    socrata[nyc_open_data.py<br/>operator ingestion] --> files[(data/ingested GeoParquet)]
  end
  spatial[spatial/<br/>GeoPandas, Shapely, PyProj] 
  registry[config/datasets.yaml<br/>allowlist + metadata]
  UI --> Agent
  toolsreg --> guard
  toolsreg --> spatial
  Agent --> registry
  files --> duck
```

## Request lifecycle

1. **start** — records the question, appends it to the conversation (checkpointed per session).
2. **discover** — ranks registered datasets for the question (keywords, tags, category values).
3. **plan**
   * LLM path: the prompt contains the authorized dataset digests (respecting the data-egress policy),
     the tool catalog rendered from each tool's JSON schema, two worked examples and the previous turn's
     results (wrapped as untrusted data). Ollama is called with the `Plan` JSON schema as `format`
     (structured output). The plan is validated (tools exist, arguments validate, references point to
     earlier steps/previous results/datasets). One repair round-trip, then fallback.
   * Rule path (demo mode and fallback): `RulePlanner` parses a generic intent (topics from registry
     category values and synonyms, subject/reference datasets from tags, boroughs/neighborhoods/ZIPs/streets
     from reference data, distances, dates, analysis type) and composes tool steps. Follow-ups merge with the
     previous intent; ambiguous requests (e.g. “specified streets”) produce a clarification question.
4. **review_sql** — renders every SQL statement the plan will run (template SQL from `filter_dataset`,
   or LLM SQL from `query_*`), validates it with the AST guard, and decides whether approval is required.
   Blocked SQL from an LLM plan triggers one replan with the errors.
5. **approve** — `langgraph.types.interrupt()` pauses the graph; the UI shows the SQL; `Command(resume=…)`
   continues. Approved fingerprints are remembered for the session (“newly generated SQL” only).
6. **execute** — runs steps in order through `ToolRegistry.invoke` (argument validation, error capture,
   audit logging). Steps that depend on a failed step are skipped.
7. **reflect** — if an LLM plan had failing steps, replan once with the observations.
8. **finalize** — chooses the main result, auto-creates a layer if none was requested, builds citations
   and the methodology list.
9. **respond** — template narrative (always computed) or LLM narrative from sanitized step digests, then
   `enforce_guardrails()` (relabels unsupported “verified” findings, rewrites violation claims, adds
   synthetic/report disclaimers).

## Key design decisions

| Decision | Why |
|---|---|
| **JSON plans instead of native tool calling** | Small local models are far more reliable with schema-constrained JSON (Ollama `format`) than with multi-turn tool calling. The plan is inspectable, validated as a whole, and shown to the user. |
| **Deterministic planner as demo mode + fallback** | The UI, tests and demos work without an LLM, and a flaky model never leaves the user without an answer. The fallback is always labeled. |
| **Plain HTTP to Ollama (httpx), no LangChain chat-model layer** | Fewer dependencies, precise control over structured output, timeouts and error messages; trivial to mock in tests; easy to swap providers. LangGraph is used for orchestration, checkpointing and human-in-the-loop. |
| **Artifacts outside graph state** | DataFrames stay in a per-session `ResultStore`; graph state is JSON (checkpointable). Tools exchange data by reference (`s1`, `t2.s3`, `last`), never raw rows from the LLM. |
| **Template SQL + GeoPandas for spatial work** | The agent pulls bounded, filtered rows (parameterized SQL, row limits) and does spatial analysis in GeoPandas/Shapely in EPSG:2263. This works identically for DuckDB and Databricks, whether or not spatial SQL functions are available. DuckDB Spatial is still loaded for approved LLM-written SQL. |
| **DuckDB in-memory, locked** | Local datasets are loaded into memory at startup, then `enable_external_access=false` and `lock_configuration=true` — SQL can't read files or load extensions even if the guard were bypassed. |
| **Standard map-layer schema** | Tools emit `MapLayerSpec` JSON; renderers are generic. Colors are stored as roles and resolved per basemap theme with CVD-validated palettes. |
| **Vendored Leaflet assets** | Folium normally loads JS from CDNs, which many government networks block. Assets are vendored (integrity-checked) and served by Streamlit static serving. |
| **Registry as allowlist** | One YAML file defines what exists, where it lives, its geometry/time semantics, sensitivity and provenance. The SQL guard, planners, prompts and citations all derive from it. |
| **Single `agentic_gis` package** | Avoids generic top-level names (`config`, `agent`, `ui`) colliding with installed packages; allows `pip install -e .`. |

## Spatial methods

| Method | Implementation | Parameters / caveats |
|---|---|---|
| Distances, buffers | Shapely in `AGIS_PROJECTED_CRS` (default EPSG:2263, US survey feet) | units: ft, m, km, mi, yd, block (≈264 ft) |
| Point-in-polygon, intersects, dwithin | `geopandas.sjoin` (+ `distance` for dwithin) | `summarize=true` → match counts per feature |
| Nearest | `geopandas.sjoin_nearest` with max distance | distances reported in feet |
| Repeat locations | DBSCAN semantics via STRtree neighbor pairs + label propagation | radius, min_points; clusters drawn as buffered hulls / sized markers |
| Hot spots | Getis-Ord Gi* on a square grid, queen contiguity incl. self; study area = convex hull + 1 cell | no multiple-testing correction (documented) |
| Aggregation | polygons (within), lines (nearest within max distance), grid; count / count_distinct / sum / mean | category breakdown columns |
| Temporal | counts per day/week/month/quarter/year, groups, least-squares trend, full-year % change | partial periods flagged |
| Period comparison | earliest vs latest year with data in Jan and Dec (or explicit periods) | low-baseline areas get no % change |

## Session and state

* One `AgentSession` per browser session: compiled graph + `InMemorySaver` checkpointer (thread id = session
  id) + `ResultStore`. Conversation messages accumulate in graph state; per-turn keys are reset each turn.
* The local DuckDB engine is shared across sessions (`st.cache_resource`); each query uses its own cursor.
* A Databricks connector is created per session (one connection per user session).

## Extending

See [EXTENDING.md](EXTENDING.md) for adding tools, datasets, LLM providers and basemaps.
