from agentic_gis.agent.tools import FilterDatasetArgs, render_filter_sql


def run(tools, ctx, step, name, args):
    ctx.step_id = step
    return tools.invoke(name, args, ctx)


def test_filter_dataset_is_parameterized(tools, ctx):
    r = render_filter_sql(
        ctx,
        FilterDatasetArgs(
            dataset="demo_311_sanitation",
            categories=["dumping"],
            boroughs=["brooklyn"],
            date_from="2023-01-01",
        ),
    )
    assert "$p0" in r.sql and "Illegal Dumping" in r.params.values() and "BROOKLYN" in r.params.values()
    assert "dumping" not in r.sql.lower().replace("illegal", "")  # values never inlined
    out = run(
        tools,
        ctx,
        "s1",
        "filter_dataset",
        {
            "dataset": "demo_311_sanitation",
            "categories": ["dumping"],
            "boroughs": ["Brooklyn"],
            "date_from": "2023-01-01",
        },
    )
    art = ctx.store.get(out.artifact_id)
    assert out.status == "ok" and art.is_spatial and set(art.data["borough"]) == {"BROOKLYN"}
    assert (art.data["created_date"] >= "2023-01-01").all() and art.synthetic


def test_filter_rejects_unknown_column(tools, ctx):
    out = run(
        tools,
        ctx,
        "s1",
        "filter_dataset",
        {"dataset": "demo_311_sanitation", "filters": [{"column": "nope", "op": "=", "value": 1}]},
    )
    assert out.status == "error" and "does not exist" in out.summary


def test_query_tool_requires_approval_then_runs(tools, ctx):
    args = {"sql": "SELECT borough, COUNT(*) AS n FROM demo_311_sanitation GROUP BY 1", "purpose": "count"}
    out = run(tools, ctx, "s1", "query_local_data", args)
    assert out.status == "needs_approval"
    ctx.approved.add(ctx.guard.validate(args["sql"], ctx.policy_for("local")).fingerprint)
    out = run(tools, ctx, "s1", "query_local_data", args)
    assert out.status == "ok" and out.rows == 5


def test_query_tool_blocks_unsafe_sql(tools, ctx):
    out = run(tools, ctx, "s1", "query_local_data", {"sql": "DROP TABLE demo_311_sanitation"})
    assert out.status == "error" and "rejected" in out.summary.lower()


def test_spatial_pipeline(tools, ctx):
    run(
        tools,
        ctx,
        "s1",
        "filter_dataset",
        {
            "dataset": "demo_street_corridors",
            "filters": [{"column": "street_name", "op": "=", "value": "ATLANTIC AVENUE"}],
        },
    )
    run(tools, ctx, "s2", "filter_dataset", {"dataset": "demo_catch_basins"})
    out = run(
        tools,
        ctx,
        "s3",
        "spatial_join",
        {"left": "s2", "right": "s1", "predicate": "dwithin", "distance": 300},
    )
    assert out.status == "ok" and out.rows > 50
    buf = run(tools, ctx, "s4", "buffer_features", {"input": "s1", "distance": 300})
    inside = run(tools, ctx, "s5", "spatial_join", {"left": "s2", "right": "s4", "predicate": "within"})
    assert inside.rows == out.rows and buf.status == "ok"
    layer = run(tools, ctx, "s6", "generate_map_layer", {"input": "s3"})
    assert layer.layer["render"] in ("points", "cluster") and layer.layer["synthetic"]


def test_analysis_tools(tools, ctx):
    run(tools, ctx, "s1", "filter_dataset", {"dataset": "demo_311_sanitation"})
    agg = run(tools, ctx, "s2", "aggregate_by_geography", {"input": "s1", "geography": "borough"})
    assert agg.rows == 5 and agg.headline
    cmp_ = run(tools, ctx, "s3", "compare_periods", {"input": "s1", "geography": "neighborhood"})
    assert cmp_.status == "ok" and "Largest increase" in (cmp_.headline or "")
    tmp = run(
        tools, ctx, "s4", "analyze_temporal_patterns", {"input": "s1", "freq": "year", "group_by": "borough"}
    )
    assert tmp.status == "ok" and tmp.data
    hs = run(tools, ctx, "s5", "find_hotspots", {"input": "s1", "method": "gi_star", "cell_size": 2000})
    assert hs.status == "ok"
    area = run(
        tools,
        ctx,
        "s6",
        "filter_by_area",
        {"input": "s1", "area_type": "neighborhood", "names": ["Bushwick"]},
    )
    assert area.status == "ok" and 0 < area.rows < 8917
    geo = run(tools, ctx, "s7", "geocode_place", {"query": "Atlantic Avenue"})
    assert geo.status == "ok"


def test_invalid_args_and_unknown_tool(tools, ctx):
    assert run(tools, ctx, "s1", "buffer_features", {"input": "s1", "distance": -5}).status == "error"
    assert run(tools, ctx, "s1", "run_python", {"code": "import os"}).status == "error"
    assert "run_python" not in tools.names()


def test_imagery_tool_never_fabricates(tools, ctx):
    out = run(tools, ctx, "s1", "search_imagery", {"query": "dumped mattresses"})
    assert out.status == "unavailable" and out.artifact_id is None


def test_registry_extensible(tools):
    from pydantic import BaseModel

    from agentic_gis.agent.tools import ToolOutput, ToolRegistry, ToolSpec

    class Args(BaseModel):
        n: int

    reg = ToolRegistry()
    reg.register(ToolSpec("double_it", "Doubles", Args, lambda c, a: ToolOutput(summary=str(a.n * 2))))
    assert "double_it" in reg.names() and "double_it" in reg.catalog_for_prompt()
