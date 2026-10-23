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
    meta = []
    src = {"llm": "LLM plan", "rules": "deterministic plan", "rules_fallback": "deterministic fallback plan"}
    meta.append(src.get(turn.plan_source, turn.plan_source))
    meta.append(f"narrative: {ans.get('narrator', 'template')}")
    meta.append(f"{turn.elapsed_s:.1f}s")
    st.caption(" · ".join(meta))
    if ans.get("adjustments"):
        st.caption("Guardrails applied: " + "; ".join(ans["adjustments"]))


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
