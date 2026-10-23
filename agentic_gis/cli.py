"""Command-line interface.

agentic-gis check                      # configuration, DuckDB Spatial, Ollama, Databricks
agentic-gis ask "question" [--approve] # run the agent headless (same graph as the UI)
agentic-gis datasets [--mode demo]     # list registered datasets and their status
agentic-gis ingest <dataset_id>        # download a public NYC Open Data dataset
agentic-gis validate-sql "SELECT ..."  # check SQL against the safety policy
agentic-gis databricks-tables          # discover tables in the default catalog.schema
agentic-gis layer-schema               # print the map-layer JSON schema
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agentic_gis.config.registry import load_registry
from agentic_gis.config.settings import get_settings


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_check(args: argparse.Namespace) -> int:
    from agentic_gis.agent.llm import OllamaClient
    from agentic_gis.connectors.duckdb_engine import LocalDuckDBEngine

    s = get_settings()
    ok = True
    print("== Settings ==")
    _print(s.public_summary())
    reg = load_registry(s)
    print("\n== Datasets ==")
    for mode in ("demo", "local", "databricks"):
        ready = [d.id for d in reg.available(mode)]
        print(f"{mode:<11} ready: {', '.join(ready) or '(none)'}")
    print("\n== DuckDB ==")
    eng = LocalDuckDBEngine(reg, args.mode or s.mode, install_extensions=s.duckdb_install_extensions)
    st = eng.test_connection()
    print(("OK   " if st.ok else "FAIL ") + st.message)
    if eng.load_errors:
        ok = False
        _print(eng.load_errors)
    print("\n== Ollama ==")
    o = OllamaClient(s.ollama_base_url, s.ollama_model).test_connection()
    print(("OK   " if o.ok else "WARN ") + o.message)
    if o.details.get("models"):
        print("Installed models:", ", ".join(o.details["models"]))
    print("\n== Databricks ==")
    if s.databricks_configured:
        from agentic_gis.connectors.databricks_sql import DatabricksConnector

        d = DatabricksConnector(s).test_connection()
        print(("OK   " if d.ok else "FAIL ") + d.message)
        ok = ok and d.ok
    else:
        print("SKIP not configured (optional)")
    return 0 if ok else 1


def cmd_ask(args: argparse.Namespace) -> int:
    from agentic_gis.agent.graph import AgentSession, create_runtime
    from agentic_gis.spatial.geometry import for_export, to_geojson_text

    s = get_settings()
    rt = create_runtime(s, mode=args.mode, llm_provider=args.provider, model=args.model)
    print(f"[planner] {rt.llm_message}", file=sys.stderr)
    session = AgentSession(rt)

    def progress(node: str, data: dict) -> None:
        if args.verbose:
            print(f"  … {node}", file=sys.stderr)

    rc = 0
    for q in args.question:
        print(f"\n>>> {q}")
        res = session.ask(q, on_event=progress)
        if res.status == "awaiting_approval":
            for item in (res.pending_approval or {}).get("queries", []):
                print(f"\n[SQL needs approval] step {item['step_id']} ({item['backend']}):\n{item['sql']}")
                if item.get("params"):
                    print(f"params: {json.dumps(item['params'], default=str)}")
            if not args.approve:
                print("\nNot approved. Re-run with --approve to execute.", file=sys.stderr)
                return 2
            res = session.resume(True, on_event=progress)
        ans = res.answer or {}
        print(f"\n[{res.status}] plan source: {res.plan_source} · {res.elapsed_s}s")
        print(f"\n{ans.get('headline', '')}\n\n{ans.get('summary', '')}")
        for f in ans.get("findings", []):
            print(f"  - ({f['kind']}) {f['text']}")
        for lim in ans.get("limitations", []):
            print(f"  ! {lim}")
        if args.verbose:
            print("\nSteps:")
            for st in res.steps:
                print(f"  {st['step_id']:<4} {st['tool']:<26} {st['status']:<8} {st.get('summary', '')}")
            for r in res.sql_reviews:
                print(f"\nSQL {r['step_id']} ({r.get('backend')}):\n{r.get('sql')}")
        if args.export:
            out = Path(args.export)
            out.mkdir(parents=True, exist_ok=True)
            for st in res.steps:
                art = rt.store.get(st.get("artifact_id") or "")
                if art is None or art.turn != res.turn:
                    continue
                base = out / f"t{art.turn}_{art.step_id}"
                for_export(art.data).to_csv(base.with_suffix(".csv"), index=False)
                if art.is_spatial:
                    base.with_suffix(".geojson").write_text(to_geojson_text(art.data), encoding="utf-8")
            print(f"\nExported results to {out}")
        if res.status not in ("complete", "clarification"):
            rc = 1
    return rc


def cmd_datasets(args: argparse.Namespace) -> int:
    s = get_settings()
    reg = load_registry(s)
    for ds in reg.all():
        if args.mode and args.mode not in ds.modes:
            continue
        st = reg.status(ds)
        flag = " SYNTHETIC" if ds.synthetic else ""
        print(f"{ds.id:<30} {st.state:<14} {ds.backend:<10} {ds.sensitivity:<12} {ds.name}{flag}")
        if st.message:
            print(f"{'':<30} {st.message}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from agentic_gis.connectors.nyc_open_data import SocrataClient, ingest_dataset

    s = get_settings()
    reg = load_registry(s)
    socrata = [d for d in reg.all() if d.source.type == "socrata"]
    if args.list or not args.dataset:
        for d in socrata:
            sc = d.source.socrata
            assert sc is not None
            verified = "" if sc.verified_id else "  (verify dataset id on the portal)"
            print(f"{d.id:<26} {sc.dataset_id}  -> {d.source.path}{verified}")
        return 0
    ds = reg.get_any(args.dataset)
    token = s.socrata_app_token.get_secret_value() if s.socrata_app_token else None
    client = SocrataClient(s.socrata_domain, token)
    res = ingest_dataset(ds, client, since=args.since, limit=args.limit)
    print(f"Ingested {res.rows:,} rows into {res.path} (where: {res.where})")
    print("Restart the app (or click 'New conversation') to load the data in 'local' mode.")
    return 0


def cmd_validate_sql(args: argparse.Namespace) -> int:
    from agentic_gis.security.sql_guard import SQLGuard, SQLPolicy

    s = get_settings()
    reg = load_registry(s)
    sens = {d.id: d.sensitive_columns for d in reg.available(args.mode, backend=args.backend)}
    policy = SQLPolicy(
        dialect="duckdb" if args.backend == "local" else "databricks",
        allowed_tables=reg.allowed_tables(args.mode, args.backend),
        max_rows=s.max_result_rows,
        sensitive_columns=sens,
        default_catalog=s.databricks_catalog,
        default_schema=s.databricks_schema,
    )
    res = SQLGuard().validate(args.sql, policy)
    _print(res.model_dump())
    return 0 if res.ok else 1


def cmd_databricks_tables(args: argparse.Namespace) -> int:
    from agentic_gis.connectors.databricks_sql import DatabricksConnector

    s = get_settings()
    if not s.databricks_configured:
        print("Databricks is not configured (see .env.example).", file=sys.stderr)
        return 1
    for t in DatabricksConnector(s).list_tables(args.catalog, args.schema):
        print(f"{t['catalog']}.{t['schema']}.{t['table']}  [{t['type']}]  {t.get('comment') or ''}")
    return 0


def cmd_layer_schema(args: argparse.Namespace) -> int:
    from agentic_gis.ui.layers import schema_json

    print(schema_json())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentic-gis", description="Agentic GIS command-line tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="Check configuration and connections")
    c.add_argument("--mode", choices=["demo", "local", "databricks"])
    c.set_defaults(func=cmd_check)
    a = sub.add_parser("ask", help="Ask the agent one or more questions (follow-ups share context)")
    a.add_argument("question", nargs="+")
    a.add_argument("--mode", choices=["demo", "local", "databricks"])
    a.add_argument("--provider", choices=["ollama", "demo", "openai_compatible"])
    a.add_argument("--model")
    a.add_argument("--approve", action="store_true", help="Approve generated SQL without prompting")
    a.add_argument("--export", help="Directory for CSV/GeoJSON exports")
    a.add_argument("-v", "--verbose", action="store_true")
    a.set_defaults(func=cmd_ask)
    d = sub.add_parser("datasets", help="List registered datasets")
    d.add_argument("--mode", choices=["demo", "local", "databricks"])
    d.set_defaults(func=cmd_datasets)
    i = sub.add_parser("ingest", help="Download a public dataset from NYC Open Data")
    i.add_argument("dataset", nargs="?")
    i.add_argument("--since", help="Start date YYYY-MM-DD (temporal datasets)")
    i.add_argument("--limit", type=int)
    i.add_argument("--list", action="store_true")
    i.set_defaults(func=cmd_ingest)
    v = sub.add_parser("validate-sql", help="Validate SQL against the guard")
    v.add_argument("sql")
    v.add_argument("--mode", default="demo", choices=["demo", "local", "databricks"])
    v.add_argument("--backend", default="local", choices=["local", "databricks"])
    v.set_defaults(func=cmd_validate_sql)
    t = sub.add_parser("databricks-tables", help="List tables visible in Unity Catalog")
    t.add_argument("--catalog")
    t.add_argument("--schema")
    t.set_defaults(func=cmd_databricks_tables)
    sub.add_parser("layer-schema", help="Print the map-layer JSON schema").set_defaults(func=cmd_layer_schema)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
