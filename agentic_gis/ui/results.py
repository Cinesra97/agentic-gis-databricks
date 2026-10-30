"""Results, reasoning, provenance and export panels."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from agentic_gis.agent.graph import AgentSession, TurnResult
from agentic_gis.agent.workspace import Artifact
from agentic_gis.spatial.geometry import for_export, to_geojson_text
from agentic_gis.ui.theme import KIND_BADGE, OTHER_GRAY, SERIES_DARK


def turn_artifacts(session: AgentSession, turn: TurnResult) -> list[Artifact]:
    arts = []
    for s in turn.steps:
        a = session.runtime.store.get(s.get("artifact_id") or "")
        if a is not None and a.turn == turn.turn and a not in arts:
            arts.append(a)
    return arts


def render_answer(turn: TurnResult) -> None:
    ans = turn.answer or {}
    if not ans:
        st.info("No answer yet.")
        return
    st.markdown(f"<div class='agis-headline'>{ans.get('headline', '')}</div>", unsafe_allow_html=True)
    st.markdown(ans.get("summary", ""))
    findings = ans.get("findings", [])
    if findings:
        for f in findings:
            label, color = KIND_BADGE.get(f.get("kind", ""), ("Note", "gray"))
            c1, c2 = st.columns([0.2, 0.8], vertical_alignment="top")
            c1.badge(label, color=color)  # type: ignore[arg-type]
            c2.markdown(f.get("text", ""))
    lims = ans.get("limitations", [])
    if lims:
        with st.expander(f"Limitations and caveats ({len(lims)})", expanded=False):
            for item in lims:
                st.markdown(f"- {item}")
    render_data_scope(turn, compact=True)
    meta = []
    meta.append(f"narrative: {ans.get('narrator', 'template')}")
    meta.append(f"{turn.elapsed_s:.1f}s")
    st.caption(" · ".join(meta))
    if ans.get("adjustments"):
        st.caption("Guardrails applied: " + "; ".join(ans["adjustments"]))


PLAN_SOURCES = {
    "llm": "LLM plan (checked against the question)",
    "rules": "deterministic plan",
    "rules_fallback": "deterministic fallback plan",
}


def data_scope_rows(turn: TurnResult) -> list[dict[str, Any]]:
    rows = []
    for d in (turn.data_scope or {}).get("datasets", []):
        cov = d.get("coverage") or {}
        where = ", ".join(d.get("boroughs") or []) or ", ".join(d.get("areas") or []) or "citywide"
        if d.get("boroughs") and d.get("areas"):
            where += " · " + ", ".join(d["areas"])
        stored = f"{cov['date_min']} to {cov['date_max']}" if cov.get("date_min") else "not measured"
        complete = "unknown" if not cov.get("computed_by") or cov.get("computed_by") == "unknown" else "yes"
        if cov.get("limit_reached"):
            complete = "no — download stopped at the row limit"
        elif cov.get("is_subset"):
            complete = "subset (ingestion filter)"
        matching = d.get("matching_rows")
        if d.get("rows_in_areas") is not None:
            matching = f"{d['rows_in_areas']:,} in area ({(matching or 0):,} before area filter)"
        rows.append(
            {
                "Dataset": d.get("name"),
                "Stored dates": stored,
                "Stored records": f"{cov['rows']:,}" if cov.get("rows") is not None else "—",
                "Date range applied": d.get("requested_text"),
                "Categories": ", ".join(map(str, d.get("categories") or [])) or "all",
                "Geography": where,
                "Other filters": "; ".join(
                    f"{f['column']} {f['op']} {f.get('value')}" for f in d.get("filters") or []
                )
                or "—",
                "Matching records": f"{matching:,}" if isinstance(matching, int) else (matching or "—"),
                "Complete copy": complete,
            }
        )
    return rows


def _step_prefix(entry: dict[str, Any]) -> str:
    return f"{entry['step']}: " if entry.get("step") else ""


def render_data_scope(turn: TurnResult, compact: bool = True) -> None:
    """What data the answer is based on: dataset, stored coverage, filters, counts, corrections."""
    ds = turn.data_scope or {}
    corrections = [v for v in turn.validation if v.get("kind") in ("removed", "changed", "added", "rejected")]
    assumptions = [v for v in turn.validation if v.get("kind") == "assumption"]
    if not ds and not corrections:
        return
    rows = data_scope_rows(turn)
    if compact:
        bits = [f"Planner: {PLAN_SOURCES.get(turn.plan_source, turn.plan_source)}"]
        if ds.get("period"):
            bits.append(f"period: {ds['period']}")
        for r in rows[:2]:
            bits.append(
                f"{r['Dataset']}: {r['Matching records']} matching · stored {r['Stored records']} "
                f"({r['Stored dates']})"
            )
        st.caption(" · ".join(bits))
    else:
        st.markdown("**Data scope**")
        st.caption(
            f"Planner: {PLAN_SOURCES.get(turn.plan_source, turn.plan_source)} · requested period: "
            f"{ds.get('period', 'all available dates')}"
        )
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        shown = set(ds.get("warnings", []))
        for d in ds.get("datasets", []):
            cov = d.get("coverage") or {}
            for n in cov.get("notes") or []:
                if f"{d.get('name')}: {n}" not in shown:
                    st.caption(f"• {d.get('name')}: {n}")
            if cov.get("ingest_where"):
                st.caption(f"• {d.get('name')} ingestion filter: `{cov['ingest_where']}`")
    for w in ds.get("warnings", []):
        st.warning(w, icon="⚠️")
    if corrections:
        with st.expander(f"Plan corrections ({len(corrections)})", expanded=not compact):
            st.caption(
                "Filters or parameters the question did not ask for were corrected before anything ran."
            )
            for c in corrections:
                st.markdown(f"- {_step_prefix(c)}{c['message']}")
    if assumptions and not compact:
        st.markdown("**Assumptions**")
        for a in assumptions:
            st.markdown(f"- {_step_prefix(a)}{a['message']}")


def render_results_table(session: AgentSession, turn: TurnResult) -> None:
    arts = turn_artifacts(session, turn)
    if not arts:
        st.info("This turn produced no tabular results.")
        return
    default = next((i for i, a in enumerate(arts) if a.id == turn.primary_artifact), len(arts) - 1)
    idx = st.selectbox(
        "Result set",
        range(len(arts)),
        index=default,
        format_func=lambda i: f"{arts[i].step_id} · {arts[i].title} ({arts[i].rows:,} rows)",
        key=f"res-select-{turn.turn}",
    )
    art = arts[idx]
    df = for_export(art.data)
    if art.synthetic:
        st.caption("⚠ Synthetic demonstration data — not real NYC records.")
    st.dataframe(df.head(5000), width="stretch", height=360, hide_index=True)
    if len(df) > 5000:
        st.caption(f"Showing the first 5,000 of {len(df):,} rows. Exports contain all rows.")
    if art.notes:
        st.caption(" ".join(art.notes))


def render_chart(session: AgentSession, turn: TurnResult) -> bool:
    arts = [a for a in turn_artifacts(session, turn) if a.extras.get("chart")]
    if not arts:
        return False
    for art in arts:
        df = art.data
        wide = df.pivot_table(index="period", columns="group", values="count", aggfunc="sum").fillna(0)
        groups = list(wide.columns)
        colors = [
            OTHER_GRAY if g == "Other" else SERIES_DARK[i % len(SERIES_DARK)] for i, g in enumerate(groups)
        ]
        st.markdown(f"**{art.title}**")
        st.line_chart(
            wide,
            color=colors if len(groups) > 1 else None,  # type: ignore[arg-type]
            height=320,
            x_label="Period",
            y_label="Reports",
        )
        summary = art.extras.get("summary")
        if summary:
            st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")
    return True


def render_reasoning(turn: TurnResult) -> None:
    plan = turn.plan or {}
    if plan:
        st.markdown(f"**Interpretation:** {plan.get('interpretation', '')}")
        if plan.get("assumptions"):
            st.caption("Assumptions: " + "; ".join(plan["assumptions"]))
    for i, entry in enumerate(turn.trace):
        title = entry.get("title", "")
        icon = {"warning": "⚠️", "error": "⛔", "approval": "✅", "plan": "🧭", "step": "▸"}.get(
            entry.get("kind", ""), "•"
        )
        detail = entry.get("detail")
        if not detail:
            st.markdown(f"{icon} {title}")
            continue
        with st.expander(f"{icon} {title}", expanded=False):
            if isinstance(detail, dict) and detail.get("sql"):
                st.code(detail["sql"], language="sql")
                rest = {k: v for k, v in detail.items() if k != "sql"}
                st.json(rest, expanded=1)
            elif isinstance(detail, list | dict):
                st.json(detail, expanded=2)
            else:
                st.write(detail)
        _ = i
    if turn.sql_reviews:
        st.markdown("**SQL executed or proposed**")
        for r in turn.sql_reviews:
            status = "allowed" if r.get("ok") else "blocked"
            appr = (
                " · approval required"
                if r.get("requires_approval")
                else (" · previously approved" if r.get("previously_approved") else "")
            )
            st.caption(f"{r['step_id']} · {r.get('backend')} · {r.get('kind')} · {status}{appr}")
            if r.get("sql"):
                st.code(r["sql"], language="sql")
            if r.get("params"):
                st.caption("Parameters: " + json.dumps(r["params"], default=str))
            for e in r.get("errors") or []:
                st.error(e)


def render_provenance(turn: TurnResult) -> None:
    render_data_scope(turn, compact=False)
    if turn.citations:
        st.markdown("**Datasets used**")
        rows = []
        for c in turn.citations:
            rows.append(
                {
                    "Dataset": c["name"],
                    "Publisher": c.get("publisher"),
                    "Synthetic": "yes" if c.get("synthetic") else "no",
                    "Sensitivity": c.get("sensitivity"),
                    "License / terms": c.get("license"),
                    "Source": c.get("source_url"),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        for c in turn.citations:
            st.caption(f"• {c.get('citation')}" + (f" — {c['notes']}" if c.get("notes") else ""))
    if turn.methodology:
        st.markdown("**Methodology**")
        for m in turn.methodology:
            st.markdown(f"- {m}")


def build_report(session: AgentSession, turn: TurnResult) -> str:
    ans = turn.answer or {}
    lines = [
        "# Agentic GIS analysis report",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M}; mode: {session.runtime.mode}_",
        "",
        f"**Question:** {next((m['content'] for m in session.messages() if m['turn'] == turn.turn and m['role'] == 'user'), '')}",
        "",
        f"## {ans.get('headline', '')}",
        "",
        ans.get("summary", ""),
        "",
        "## Findings",
    ]
    for f in ans.get("findings", []):
        lines.append(f"- **{f['kind']}** — {f['text']}")
    lines += ["", "## Limitations"] + [f"- {x}" for x in ans.get("limitations", [])]
    rows = data_scope_rows(turn)
    if rows:
        lines += ["", "## Data scope"]
        for r in rows:
            lines.append("- " + "; ".join(f"{k}: {v}" for k, v in r.items()))
        lines += [f"- ⚠ {w}" for w in (turn.data_scope or {}).get("warnings", [])]
    fixes = [v for v in turn.validation if v.get("kind") in ("removed", "changed", "added", "rejected")]
    if fixes:
        lines += ["", "## Plan corrections"] + [f"- {v['message']}" for v in fixes]
    lines += ["", "## Plan", f"Interpretation: {(turn.plan or {}).get('interpretation', '')}", ""]
    for s in turn.steps:
        lines.append(f"- `{s['step_id']}` **{s['tool']}** ({s['status']}): {s.get('summary', '')}")
    if turn.sql_reviews:
        lines += ["", "## SQL"]
        for r in turn.sql_reviews:
            lines += [f"Step {r['step_id']} ({r.get('backend')}):", "```sql", r.get("sql") or "", "```"]
    lines += ["", "## Data sources"] + [f"- {c.get('citation')}" for c in turn.citations]
    lines += ["", "## Methodology"] + [f"- {m}" for m in turn.methodology]
    return "\n".join(lines) + "\n"


def render_exports(session: AgentSession, turn: TurnResult) -> None:
    arts = turn_artifacts(session, turn)
    if not arts:
        st.info("Nothing to export for this turn.")
        return
    for art in arts:
        c1, c2, c3 = st.columns([0.52, 0.24, 0.24], vertical_alignment="center")
        c1.markdown(
            f"**{art.step_id}** · {art.title}  \n<span class='agis-muted'>{art.rows:,} rows</span>",
            unsafe_allow_html=True,
        )
        base = f"agis_t{art.turn}_{art.step_id}" + ("_synthetic" if art.synthetic else "")
        c2.download_button(
            "CSV",
            for_export(art.data).to_csv(index=False).encode("utf-8"),
            file_name=f"{base}.csv",
            mime="text/csv",
            key=f"csv-{art.id}",
            on_click="ignore",
            width="stretch",
        )
        if art.is_spatial:
            c3.download_button(
                "GeoJSON",
                to_geojson_text(art.data).encode("utf-8"),
                file_name=f"{base}.geojson",
                mime="application/geo+json",
                key=f"gj-{art.id}",
                on_click="ignore",
                width="stretch",
            )
    st.divider()
    c1, c2 = st.columns(2)
    c1.download_button(
        "Report (Markdown)",
        build_report(session, turn).encode("utf-8"),
        file_name=f"agis_report_turn{turn.turn}.md",
        mime="text/markdown",
        on_click="ignore",
        width="stretch",
    )
    c2.download_button(
        "Map layers (JSON)",
        json.dumps(turn.layers, indent=2, default=str).encode("utf-8"),
        file_name=f"agis_layers_turn{turn.turn}.json",
        mime="application/json",
        on_click="ignore",
        width="stretch",
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for art in arts:
            base = f"t{art.turn}_{art.step_id}"
            zf.writestr(f"{base}.csv", for_export(art.data).to_csv(index=False))
            if art.is_spatial:
                zf.writestr(f"{base}.geojson", to_geojson_text(art.data))
        zf.writestr("report.md", build_report(session, turn))
        zf.writestr("layers.json", json.dumps(turn.layers, indent=2, default=str))
    st.download_button(
        "Everything (ZIP)",
        buf.getvalue(),
        file_name=f"agis_turn{turn.turn}.zip",
        mime="application/zip",
        on_click="ignore",
        width="stretch",
    )


def describe_feature(feature: dict[str, Any] | None) -> None:
    if not feature:
        st.caption("Click a feature on the map to inspect its attributes.")
        return
    props = {k: v for k, v in (feature.get("properties") or {}).items() if not str(k).startswith("_")}
    if props:
        st.dataframe(
            pd.DataFrame({"Attribute": list(props), "Value": [str(v) for v in props.values()]}),
            hide_index=True,
            width="stretch",
        )
