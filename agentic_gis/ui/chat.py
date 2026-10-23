"""Chat panel: conversation history, SQL approval card, example prompts."""

from __future__ import annotations

import json

import streamlit as st

from agentic_gis.agent.graph import AgentSession, TurnResult
from agentic_gis.ui.theme import KIND_BADGE, NODE_LABELS

EXAMPLES = [
    "Find locations in NYC with repeated illegal dumping complaints.",
    "Identify areas with high concentrations of derelict vehicle complaints.",
    "Find catch basins within 300 feet of Atlantic Avenue.",
    "Identify streets with repeated missed sweeping complaints.",
    "Show sanitation complaints near industrial properties.",
    "Compare dumping patterns across NYC boroughs over time.",
    "Find areas where multiple types of sanitation complaints overlap.",
    "Which neighborhoods have the largest increase in sanitation-related complaints?",
]


def queue_question(q: str) -> None:
    st.session_state["queued_question"] = q


def _assistant_block(turn: TurnResult, show_followups: bool) -> None:
    ans = turn.answer or {}
    st.markdown(f"**{ans.get('headline', '')}**")
    if ans.get("summary"):
        st.markdown(ans["summary"])
    for f in ans.get("findings", [])[:4]:
        label, color = KIND_BADGE.get(f.get("kind", ""), ("Note", "gray"))
        st.markdown(f":{color}-badge[{label}] {f.get('text', '')}")
    if turn.status == "clarification":
        st.caption("Reply below to continue.")
    if show_followups and ans.get("suggested_follow_ups"):
        cols = st.columns(len(ans["suggested_follow_ups"][:3]))
        for i, q in enumerate(ans["suggested_follow_ups"][:3]):
            cols[i].button(
                q,
                key=f"fu-{turn.turn}-{i}",
                on_click=queue_question,
                args=(q,),
                width="stretch",
                type="tertiary",
            )


def render_history(session: AgentSession) -> None:
    msgs = session.messages()
    turns = {t.turn: t for t in session.history}
    last_turn = session.history[-1].turn if session.history else None
    for m in msgs:
        if m["role"] == "user":
            with st.chat_message("user"):
                st.markdown(m["content"])
        else:
            t = turns.get(m["turn"])
            with st.chat_message("assistant", avatar="🗺️"):
                if t is None:
                    st.markdown(m["content"])
                else:
                    _assistant_block(t, show_followups=(t.turn == last_turn))


def render_examples() -> None:
    st.markdown("**Try a question** — or type your own below.")
    for i, q in enumerate(EXAMPLES):
        st.button(q, key=f"ex-{i}", on_click=queue_question, args=(q,), width="stretch")


def render_approval(pending: dict) -> None:
    with st.container(border=True):
        st.markdown("**🔐 Review SQL before it runs**")
        st.caption(
            "The agent generated the following read-only SQL. It passed the AST safety check "
            "(allowlisted tables, no side effects, row limit). Nothing runs until you approve."
        )
        if pending.get("interpretation"):
            st.markdown(f"_Plan:_ {pending['interpretation']}")
        for q in pending.get("queries", []):
            st.markdown(
                f"**{q['step_id']}** · {q.get('purpose', '')}  \n"
                f"<span class='agis-sql-note'>Backend: {q.get('backend')} · "
                f"{'LLM-written SQL' if q.get('kind') == 'llm_sql' else 'template query'} · datasets: "
                f"{', '.join(q.get('datasets') or [])}</span>",
                unsafe_allow_html=True,
            )
            st.code(q.get("sql", ""), language="sql")
            if q.get("params"):
                st.caption("Parameters: " + json.dumps(q["params"], default=str))
            for w in q.get("warnings") or []:
                st.warning(w)
        c1, c2 = st.columns(2)
        c1.button(
            "Approve and run",
            type="primary",
            width="stretch",
            key="approve-yes",
            on_click=lambda: st.session_state.update(approval_decision=True),
        )
        c2.button(
            "Reject",
            width="stretch",
            key="approve-no",
            on_click=lambda: st.session_state.update(approval_decision=False),
        )


def progress_callback(status_box):  # type: ignore[no-untyped-def]
    def cb(node: str, data: dict) -> None:
        label = NODE_LABELS.get(node, node)
        extra = ""
        if node == "plan" and data.get("plan"):
            steps = data["plan"].get("steps") or []
            extra = f" — {len(steps)} steps ({data.get('plan_source', '')})"
        if node == "execute" and data.get("steps"):
            ok = sum(1 for s in data["steps"] if s["status"] == "ok")
            extra = f" — {ok}/{len(data['steps'])} steps ok"
        status_box.write(f"✓ {label}{extra}")

    return cb
