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
    graph --> sem[semantics.py + temporal.py<br/>request-scope validation]
    graph --> toolsreg[tools.py<br/>tool registry]
    graph --> narr[narrative.py + grounding.py<br/>answer, fact check, guardrails]
    planner --> llm[llm.py<br/>Ollama / OpenAI-compatible]
    toolsreg --> store[(workspace.py<br/>artifact store)]
  end
  subgraph Data[Data access · agentic_gis/connectors]
    guard[security/sql_guard.py] --> duck[(DuckDB in-memory<br/>+ Spatial)]
    guard --> dbx[(Databricks SQL warehouse<br/>Unity Catalog)]
    socrata[nyc_open_data.py<br/>operator ingestion] --> files[(data/ingested GeoParquet<br/>+ .provenance.json)]
    duck --> cov[coverage.py<br/>stored rows + date range]
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
   * The rule planner always parses the question first (cheap, deterministic). Its intent is the
     **request scope**: time period (`temporal.parse_time_window`: years, ranges, ISO/month dates,
     quarters, relative periods, bare months, “all dates”), boroughs, categories, areas/ZIPs/streets,
     distance, thresholds, plus earlier questions for follow-ups. Dataset titles, descriptions and
     publishers never count as a requested period.
   * LLM path: the prompt contains the request scope, the authorized dataset digests (with stored row counts
     and date ranges; respecting the data-egress policy), the tool catalog rendered from each tool's JSON
     schema, two worked examples and the previous turn's results (wrapped as untrusted data). Ollama is
     called with the `Plan` JSON schema as `format` (structured output). The plan is validated structurally
     (tools exist, arguments validate, references point to earlier steps/previous results/datasets; one
     repair round-trip) and then **semantically** (`semantics.SemanticValidator`): date filters must come
     from the question or the conversation (removed otherwise, also when hidden in `filters`); borough,
     category, attribute and area filters must be grounded in the question; unrequested row limits are
     removed; unspecified distances/thresholds use the documented defaults; filters the question asks for
     but the plan omits are added. LLM SQL predicates get the same checks (unrequested top-level `AND`
     conditions are removed; otherwise the plan is rejected). Every change is recorded (trace, UI, audit).
     A rejected plan is replaced by the deterministic plan without another model call.
   * Rule path (demo mode and fallback): `RulePlanner` parses a generic intent (topics from registry
     category values and synonyms, subject/reference datasets from tags, boroughs/neighborhoods/ZIPs/streets
     from reference data, distances, dates, analysis type) and composes tool steps. Follow-ups merge with the
     previous intent (dates are kept, replaced, or cleared by “all dates”); ambiguous requests (e.g.
     “specified streets”) produce a clarification question. Rule plans pass through the same semantic check.
4. **check_coverage** — compares the requested period with the dates actually stored
   (`connectors/coverage.py`: measured once at load from the in-memory frame, merged with the ingestion
   sidecar). Entirely outside → status `insufficient_coverage`, nothing runs, the answer says which dates
   are stored. Partial overlap or a truncated download (row limit reached) → disclosed as warnings.
5. **review_sql** — renders every SQL statement the plan will run (template SQL from `filter_dataset`,
   or LLM SQL from `query_*`), validates it with the AST guard (and, for LLM SQL, the request scope), and
   decides whether approval is required. Blocked SQL from an LLM plan triggers one replan with the errors.
6. **approve** — `langgraph.types.interrupt()` pauses the graph; the UI shows the SQL; `Command(resume=…)`
   continues. Approved fingerprints are remembered for the session (“newly generated SQL” only).
7. **execute** — runs steps in order through `ToolRegistry.invoke` (argument validation, error capture,
   audit logging). Each step records a `reason` (`ok`, `empty_filter`, `empty_result`, `zero_clusters`,
   `empty_input`, `error`, `dependency_failed`, `missing_input`, `blocked_by_policy`, `needs_approval`,
   `unavailable`). A step whose input has no artifact or zero rows is skipped (tools may opt in with
   `accepts_empty_input`), so an empty selection never surfaces as “Unknown input reference”.
8. **reflect** — replans an LLM plan once, only for genuine execution errors (not for empty results or
   skipped dependents: a new plan cannot create data that is not stored).
9. **finalize** — chooses the main result, auto-creates a layer if none was requested, builds citations,
   the methodology list, and the data scope (matching vs stored records).
10. **respond** — template narrative (always computed; explains empty selections, zero clusters and
    failures by their root cause). The LLM narrates only when there are results; its answer is checked
    against the computed facts (`grounding.check_narrative`: numbers, years, places, “no records”/“no
    clusters” claims) and replaced by the template if it contradicts them. Computed caveats are always kept.
    Then `enforce_guardrails()` (relabels unsupported “verified” findings, rewrites violation claims, adds
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
| **Request scope enforced in code, not only in prompts** | Small local models add plausible-looking filters (e.g. a date range copied from an example or a dataset title). The deterministic intent parser already knows what the question asked; checking every plan against it is cheap, transparent and needs no extra model call. |
| **Coverage measured, not configured** | The configured ingestion window says what was requested from the portal, not what is stored (downloads stop at row limits). An empty result for an unstored period must not read as “no complaints”. |
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
