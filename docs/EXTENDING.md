# Extending the agent

## Add a tool

Tools are plain functions with a Pydantic input model. Register them on a `ToolRegistry`; the planner
prompt, argument validation, audit logging and error handling come for free.

```python
from pydantic import BaseModel, Field
from agentic_gis.agent.tools import REF, ToolContext, ToolOutput, build_default_registry


class CountWithinArgs(BaseModel):
    input: str = Field(description="Features to count", json_schema_extra=REF)  # REF marks references
    area: str = Field(description="Polygons to count within", json_schema_extra=REF)


registry = build_default_registry()


@registry.tool("count_within", "Count input features inside each polygon.", CountWithinArgs)
def count_within(ctx: ToolContext, a: CountWithinArgs) -> ToolOutput:
    pts_art, pts = ctx.resolve_spatial(a.input)
    area_art, areas = ctx.resolve_spatial(a.area)
    # ... compute a GeoDataFrame `out` ...
    art = ctx.save("Counts", out, "count_within", [pts_art, area_art], notes=["Point-in-polygon counts."])
    return ToolOutput(summary=f"Counted {len(pts):,} features.", artifact_id=art.id, rows=len(out))
```

Pass the registry to `create_runtime(..., tools=registry)`. Rules:

* Accept data only by reference (`ctx.resolve(...)`), never raw records from the LLM.
* Mark reference arguments with `json_schema_extra=REF` so plan validation can resolve them (dataset ids
  become explicit load steps).
* If the tool runs SQL, provide `sql_renderer=` so the SQL is validated and shown for approval *before*
  execution, and execute it through `ctx.run_sql(...)`.
* Put method parameters and caveats in `notes`; they appear in the methodology section.
* Add tests in `tests/` (see `tests/test_tools.py`).

## Add a dataset

See [DATASETS.md](DATASETS.md). Tags, synonyms and `known_values` drive discovery and the deterministic
planner; descriptions and columns drive the LLM planner.

## Add an LLM provider

Subclass `LLMClient` in `agentic_gis/agent/llm.py` (implement `chat`, `test_connection`, `list_models`) and
add a branch in `build_llm()`. Respect `check_llm_endpoint()` so non-local endpoints require explicit
approval. `generate_json()` (schema-constrained output + one repair round) is inherited.

## Add a basemap

Set `AGIS_CUSTOM_BASEMAP_URL` / `AGIS_CUSTOM_BASEMAP_ATTRIBUTION` for an organization tile server, or add an
entry to `BASEMAPS` in `agentic_gis/ui/maps.py` (`url`, `xyz`, `attr`, `theme`, `max_zoom`). Follow the
provider's usage policy and attribution requirements.

## Add a geography

Register a boundary dataset with `role: boundary` and a `geography_role` (e.g. `police_precinct`), plus a
`name_column`. It becomes available to `aggregate_by_geography`, `compare_periods` and `filter_by_area`
automatically, and its names become planner vocabulary.
