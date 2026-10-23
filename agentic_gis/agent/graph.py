"""LangGraph workflow for the GIS agent.

    start -> discover -> plan -> review_sql -> [approve] -> execute -> reflect -> finalize -> respond
                          ^                                              |
                          +------------------ replan (LLM only) ---------+

* ``plan`` asks the LLM for a JSON plan (validated against tool schemas and the dataset
  allowlist; one repair attempt) and falls back to the deterministic planner.
* ``review_sql`` renders and validates every SQL statement the plan will run.
* ``approve`` pauses the graph with :func:`langgraph.types.interrupt` until the user approves
  or rejects newly generated SQL (configurable via ``AGIS_SQL_APPROVAL``).
* ``respond`` writes the answer (LLM or template) and applies claim guardrails.

Graph state is JSON-serializable and checkpointed per session (thread) with an in-memory
saver, which keeps conversation context for follow-up questions.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agentic_gis.agent.llm import LLMClient, LLMError, LLMOutputError, LLMUnavailableError, build_llm
from agentic_gis.agent.narrative import (
    citations_for,
    datasets_for_llm,
    enforce_guardrails,
    step_digest_for_llm,
    template_answer,
)
from agentic_gis.agent.planner import RulePlanner, Vocabulary, build_planner_prompt, validate_plan
from agentic_gis.agent.prompts import NARRATOR_SYSTEM, NARRATOR_USER, PLANNER_SYSTEM
from agentic_gis.agent.state import AgentState, Answer, Plan
from agentic_gis.agent.tools import ToolContext, ToolRegistry, build_default_registry
from agentic_gis.agent.workspace import Artifact, ResultStore
from agentic_gis.config.registry import DatasetRegistry, DatasetSpec, UnknownDatasetError, load_registry
from agentic_gis.config.settings import Settings, get_settings
from agentic_gis.connectors.base import QueryEngine
from agentic_gis.security.audit import AuditLogger
from agentic_gis.security.policy import DataEgressPolicy
from agentic_gis.security.sanitize import wrap_untrusted
from agentic_gis.security.sql_guard import SQLGuard
from agentic_gis.spatial.operations import normalize_name
from agentic_gis.ui.layers import build_layer_spec

log = logging.getLogger(__name__)

ANALYTIC_TOOLS = {
    "find_hotspots",
    "aggregate_by_geography",
    "compare_periods",
    "spatial_join",
    "nearest_features",
    "filter_by_area",
    "buffer_features",
    "query_local_data",
    "query_databricks",
    "geocode_place",
}


# =========================================================================================
# Runtime
# =========================================================================================
@dataclass
class AgentRuntime:
    settings: Settings
    registry: DatasetRegistry
    mode: str
    engines: dict[str, QueryEngine]
    tools: ToolRegistry
    llm: LLMClient | None
    guard: SQLGuard
    audit: AuditLogger
    store: ResultStore
    session_id: str
    approved: set[str] = field(default_factory=set)
    llm_message: str = ""
    _vocabulary: Vocabulary | None = None

    def tool_context(self, turn: int, step_id: str = "s0") -> ToolContext:
        return ToolContext(
            settings=self.settings,
            registry=self.registry,
            mode=self.mode,
            store=self.store,
            engines=self.engines,
            guard=self.guard,
            audit=self.audit,
            session_id=self.session_id,
            turn=turn,
            step_id=step_id,
            approved=self.approved,
            reference_cache=self._reference_cache,
        )

    def __post_init__(self) -> None:
        self._reference_cache: dict[str, Any] = {}

    @property
    def policy(self) -> DataEgressPolicy:
        return DataEgressPolicy(self.settings, self.llm.is_local if self.llm else True)

    @property
    def vocabulary(self) -> Vocabulary:
        if self._vocabulary is None:
            self._vocabulary = build_vocabulary(self.tool_context(0), self.registry, self.mode)
        return self._vocabulary


def build_vocabulary(ctx: ToolContext, registry: DatasetRegistry, mode: str) -> Vocabulary:
    v = Vocabulary()
    for role in ["neighborhood", "community_district", *registry.geography_roles(mode)]:
        if role in ("borough", "zip"):
            continue
        ds = registry.geography(role, mode)
        if ds is None:
            continue
        try:
            g = ctx.load_reference(ds)
        except Exception as exc:  # pragma: no cover - reference data problems are reported elsewhere
            log.warning("Vocabulary: could not load %s: %s", ds.id, exc)
            continue
        label_col = ds.name_column if ds.name_column in g else None
        for col in [c for c in (ds.name_column, "neighborhood", "ntaname") if c and c in g]:
            for label, val in zip(g[label_col or col], g[col], strict=False):
                for part in re.split(r"\s+and\s+|\s*[-,/()]\s*", str(val)):
                    n = normalize_name(part)
                    if len(n) >= 4 and not re.fullmatch(r"[A-Z]{2}\d{2}", n):
                        v.areas.setdefault(n, (role, str(label)))
    for ds in registry.available(mode):
        if ds.role != "streets" or not ds.name_column:
            continue
        name_col = ds.column(ds.name_column)
        values = list(name_col.known_values) if name_col is not None else []
        if not values and ds.backend == "local":
            try:
                g = ctx.load_reference(ds)
                values = sorted(g[ds.name_column].dropna().astype(str).unique())[:5000]
            except Exception:  # pragma: no cover
                values = []
        for val in values:
            v.streets[normalize_name(val)] = val
    return v


def create_engines(
    settings: Settings, registry: DatasetRegistry, mode: str
) -> tuple[dict[str, QueryEngine], list[str]]:
    """Build query engines for a mode. Returns (engines, messages)."""
    from agentic_gis.connectors.duckdb_engine import LocalDuckDBEngine

    msgs = []
    engines: dict[str, QueryEngine] = {
        "local": LocalDuckDBEngine(registry, mode, install_extensions=settings.duckdb_install_extensions)
    }
    if mode == "databricks":
        if settings.databricks_configured:
            from agentic_gis.connectors.databricks_sql import DatabricksConnector

            engines["databricks"] = DatabricksConnector(settings)
        else:
            msgs.append("Databricks is not configured; only local reference data is available.")
    return engines, msgs


def create_runtime(
    settings: Settings | None = None,
    mode: str | None = None,
    llm_provider: str | None = None,
    model: str | None = None,
    engines: dict[str, QueryEngine] | None = None,
    registry: DatasetRegistry | None = None,
    session_id: str | None = None,
    llm_transport: Any = None,
    tools: ToolRegistry | None = None,
) -> AgentRuntime:
    settings = settings or get_settings()
    mode = mode or settings.mode
    registry = registry or load_registry(settings)
    if engines is None:
        engines, _ = create_engines(settings, registry, mode)
    sid = session_id or uuid.uuid4().hex[:12]
    try:
        sel = build_llm(settings, model=model, provider=llm_provider, transport=llm_transport)
        llm, msg = sel.client, sel.message
    except LLMError as exc:
        llm, msg = None, f"LLM disabled: {exc}"
    except Exception as exc:  # ExternalLLMNotAllowedError and similar
        llm, msg = None, f"LLM disabled: {exc}"
    audit = AuditLogger(settings.audit_log_path, settings.secret_values(), settings.audit_log_sql)
    return AgentRuntime(
        settings=settings,
        registry=registry,
        mode=mode,
        engines=engines,
        tools=tools or build_default_registry(),
        llm=llm,
        guard=SQLGuard(),
        audit=audit,
        store=ResultStore(),
        session_id=sid,
        llm_message=msg,
    )


# =========================================================================================
# Graph
# =========================================================================================
def _trace(
    state: AgentState, node: str, title: str, detail: Any = None, kind: str = "info"
) -> list[dict[str, Any]]:
    entry = {"node": node, "title": title, "kind": kind, "ts": time.time()}
    if detail is not None:
        entry["detail"] = detail
    return [*state.get("trace", []), entry]


def _datasets_used(rt: AgentRuntime, ids: list[str]) -> list[DatasetSpec]:
    out = []
    for did in dict.fromkeys(ids):
        try:
            out.append(rt.registry.get_any(did))
        except UnknownDatasetError:
            continue
    return out


def build_graph(rt: AgentRuntime, checkpointer: Any = None) -> Any:
    settings = rt.settings

    # ---- start -----------------------------------------------------------------------------
    def start(state: AgentState) -> dict[str, Any]:
        q = state["question"].strip()
        ctx = state.get("context") or {}
        effective = q  # clarification answers are merged with the previous intent by the planners
        rt.audit.log("question", rt.session_id, turn=state["turn"], mode=rt.mode, question=q)
        return {
            "effective_question": effective,
            "context": ctx,
            "messages": [{"role": "user", "content": q, "turn": state["turn"]}],
            "trace": _trace(
                state,
                "start",
                "Question received",
                {"question": q, "mode": rt.mode, "planner": rt.llm.label if rt.llm else "deterministic"},
            ),
        }

    # ---- discover ----------------------------------------------------------------------------
    def discover(state: AgentState) -> dict[str, Any]:
        ranked = rt.registry.search(state["effective_question"], rt.mode, limit=6)
        cands = [
            {"id": d.id, "name": d.name, "role": d.role, "score": round(s, 2), "synthetic": d.synthetic}
            for d, s in ranked
        ]
        detail = [f"{c['id']} ({c['role']}, score {c['score']})" for c in cands]
        return {"candidates": cands, "trace": _trace(state, "discover", "Dataset discovery", detail)}

    # ---- plan ----------------------------------------------------------------------------------
    def plan(state: AgentState) -> dict[str, Any]:
        turn = state["turn"]
        ctx = rt.tool_context(turn)
        context = state.get("context") or {}
        prev_intent = context.get("last_intent")
        rules = RulePlanner(rt.registry, rt.mode, rt.vocabulary)
        feedback = state.get("replan_feedback")
        notes: list[str] = []
        trace = state.get("trace", [])
        intent = None
        source = "rules"
        chosen: Plan | None = None
        if rt.llm is not None:
            try:
                prompt = build_planner_prompt(
                    state["effective_question"],
                    rt.registry,
                    rt.mode,
                    rt.tools,
                    ctx,
                    rt.policy,
                    context,
                    rt.store,
                    turn,
                )
                if context.get("pending_clarification"):
                    prompt += f"\n\nThe user was asked: {context['pending_clarification']}"
                if feedback:
                    prompt += (
                        "\n\nYour previous plan could not be completed:\n"
                        + wrap_untrusted("errors", feedback)
                        + "\nWrite a corrected plan."
                    )
                candidate = rt.llm.generate_json(PLANNER_SYSTEM, prompt, Plan, purpose="plan")
                check = validate_plan(
                    candidate, rt.tools, rt.registry, rt.mode, rt.store, turn, settings.max_plan_steps, ctx
                )
                if check.errors:
                    repair = (
                        prompt
                        + "\n\nThis plan was rejected by the validator:\n"
                        + wrap_untrusted("validator", check.errors)
                        + "\nReturn a corrected plan."
                    )
                    candidate = rt.llm.generate_json(PLANNER_SYSTEM, repair, Plan, purpose="plan_repair")
                    check = validate_plan(
                        candidate,
                        rt.tools,
                        rt.registry,
                        rt.mode,
                        rt.store,
                        turn,
                        settings.max_plan_steps,
                        ctx,
                    )
                if check.errors:
                    notes.append("LLM plan failed validation: " + "; ".join(check.errors[:3]))
                    raise LLMOutputError("invalid plan")
                chosen, source = check.plan, "llm"
            except LLMUnavailableError as exc:
                notes.append(str(exc))
                if not settings.llm_fallback_to_rules:
                    return {
                        "status": "error",
                        "error": str(exc),
                        "trace": _trace(state, "plan", "LLM unavailable", str(exc), "error"),
                    }
            except LLMError as exc:
                notes.append(f"LLM planning failed: {str(exc)[:200]}")
            for call in rt.llm.calls[-3:]:
                rt.audit.log(
                    "llm_call",
                    rt.session_id,
                    turn=turn,
                    purpose=call.purpose,
                    model=call.model,
                    ok=call.ok,
                    elapsed_ms=call.elapsed_ms,
                    attempts=call.attempts,
                    error=call.error,
                )
        try:
            p_rules, intent = rules.plan(state["effective_question"], prev_intent)
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("Rule planner failed")
            p_rules, intent = Plan(interpretation="", steps=[], clarification=f"Planner error: {exc}"), None
        if chosen is None:
            check = validate_plan(
                p_rules, rt.tools, rt.registry, rt.mode, rt.store, turn, settings.max_plan_steps, ctx
            )
            chosen = check.plan
            source = "rules" if rt.llm is None else "rules_fallback"
            if check.errors and not chosen.clarification:
                notes += check.errors
        detail = {
            "source": source,
            "interpretation": chosen.interpretation,
            "steps": [f"{s.id} {s.tool}: {s.purpose}" for s in chosen.steps],
            "assumptions": chosen.assumptions,
            "notes": notes,
        }
        title = {
            "llm": f"Plan written by {rt.llm.label if rt.llm else 'LLM'}",
            "rules": "Plan from deterministic planner",
            "rules_fallback": "LLM unavailable or invalid; used deterministic planner",
        }[source]
        status = "clarification" if chosen.clarification else "running"
        rt.audit.log(
            "plan",
            rt.session_id,
            turn=turn,
            source=source,
            interpretation=chosen.interpretation,
            steps=[{"id": s.id, "tool": s.tool} for s in chosen.steps],
        )
        return {
            "plan": chosen.model_dump(),
            "plan_source": source,
            "intent": intent.to_dict() if intent else None,
            "status": status,
            "replan_feedback": None,
            "llm_used": source == "llm",
            "trace": [
                *trace,
                {"node": "plan", "title": title, "kind": "plan", "detail": detail, "ts": time.time()},
            ],
        }

    # ---- review SQL --------------------------------------------------------------------------
    def review_sql(state: AgentState) -> dict[str, Any]:
        turn = state["turn"]
        ctx = rt.tool_context(turn)
        p = Plan.model_validate(state["plan"])
        reviews = []
        for step in p.steps:
            spec = rt.tools.get(step.tool)
            if spec.sql_renderer is None:
                continue
            ctx.step_id = step.id
            entry: dict[str, Any] = {"step_id": step.id, "tool": step.tool, "purpose": step.purpose}
            try:
                args = spec.args_model.model_validate(step.args)
                rendered = spec.sql_renderer(ctx, args)
                review = ctx.validate_sql(rendered)
                needs = ctx.requires_approval(rendered.kind, rendered.backend)
                entry.update(
                    {
                        "backend": rendered.backend,
                        "kind": rendered.kind,
                        "ok": review.ok,
                        "sql": review.sql or rendered.sql,
                        "params": rendered.params,
                        "errors": review.errors,
                        "warnings": review.warnings,
                        "datasets": review.datasets,
                        "fingerprint": review.fingerprint,
                        "requires_approval": bool(
                            review.ok and needs and review.fingerprint not in rt.approved
                        ),
                        "previously_approved": bool(needs and review.fingerprint in rt.approved),
                    }
                )
            except Exception as exc:
                entry.update(
                    {
                        "ok": False,
                        "errors": [str(exc).strip("'\"")[:300]],
                        "sql": "",
                        "requires_approval": False,
                        "backend": "?",
                        "kind": "?",
                    }
                )
            reviews.append(entry)
            rt.audit.log(
                "sql_validated",
                rt.session_id,
                turn=turn,
                step=step.id,
                ok=entry["ok"],
                errors=entry.get("errors"),
                sql=entry.get("sql"),
                fingerprint=entry.get("fingerprint"),
            )
        bad = [r for r in reviews if not r["ok"]]
        update: dict[str, Any] = {"sql_reviews": reviews}
        detail = [
            {
                "step": r["step_id"],
                "backend": r.get("backend"),
                "ok": r["ok"],
                "errors": r.get("errors"),
                "needs_approval": r.get("requires_approval"),
            }
            for r in reviews
        ]
        if bad and state.get("plan_source") == "llm" and state.get("replans", 0) < settings.max_replans:
            update["replan_feedback"] = [f"{r['step_id']}: {'; '.join(r['errors'])}" for r in bad]
            update["replans"] = state.get("replans", 0) + 1
            update["trace"] = _trace(
                state, "review_sql", "SQL rejected by policy; asking the planner to revise", detail, "warning"
            )
        else:
            n_ok = len(reviews) - len(bad)
            title = f"Validated {n_ok} SQL statement(s)" + (f"; {len(bad)} blocked" if bad else "")
            update["trace"] = _trace(state, "review_sql", title, detail, "warning" if bad else "info")
        return update

    def route_after_review(state: AgentState) -> str:
        if state.get("replan_feedback"):
            return "plan"
        if any(r.get("requires_approval") for r in state.get("sql_reviews", [])):
            return "approve"
        return "execute"

    # ---- approve (human in the loop) -------------------------------------------------------
    def approve(state: AgentState) -> dict[str, Any]:
        pending = [r for r in state.get("sql_reviews", []) if r.get("requires_approval")]
        decision = interrupt(
            {
                "type": "sql_approval",
                "question": state["question"],
                "interpretation": (state.get("plan") or {}).get("interpretation"),
                "queries": [
                    {
                        k: r.get(k)
                        for k in (
                            "step_id",
                            "purpose",
                            "backend",
                            "kind",
                            "sql",
                            "params",
                            "warnings",
                            "datasets",
                            "fingerprint",
                        )
                    }
                    for r in pending
                ],
            }
        )
        approved = decision.get("approved") if isinstance(decision, dict) else bool(decision)
        fps = (decision.get("fingerprints") if isinstance(decision, dict) else None) or [
            r["fingerprint"] for r in pending
        ]
        if approved:
            rt.approved.update(fps)
            rt.audit.log("sql_approved", rt.session_id, turn=state["turn"], fingerprints=fps)
            return {
                "approval": {"approved": True, "fingerprints": fps},
                "trace": _trace(state, "approve", f"User approved {len(fps)} query(ies)", None, "approval"),
            }
        rt.audit.log("sql_rejected", rt.session_id, turn=state["turn"], fingerprints=fps)
        return {
            "approval": {"approved": False},
            "status": "rejected",
            "trace": _trace(
                state, "approve", "User rejected the SQL; nothing was executed", None, "approval"
            ),
        }

    def route_after_approve(state: AgentState) -> str:
        return "respond" if state.get("status") == "rejected" else "execute"

    # ---- execute ---------------------------------------------------------------------------------
    def execute(state: AgentState) -> dict[str, Any]:
        turn = state["turn"]
        ctx = rt.tool_context(turn)
        p = Plan.model_validate(state["plan"])
        blocked = {r["step_id"]: r for r in state.get("sql_reviews", []) if not r["ok"]}
        step_ids = {s.id for s in p.steps}
        failed: set[str] = set()
        records = []
        trace = list(state.get("trace", []))
        for step in p.steps:
            spec = rt.tools.get(step.tool)
            deps = [str(step.args[f]) for f in spec.ref_fields if step.args.get(f) in step_ids]
            rec: dict[str, Any] = {
                "step_id": step.id,
                "tool": step.tool,
                "purpose": step.purpose,
                "args": step.args,
            }
            if step.id in blocked:
                rec.update(
                    status="error", summary="Blocked by SQL policy: " + "; ".join(blocked[step.id]["errors"])
                )
            elif any(d in failed for d in deps):
                rec.update(
                    status="skipped", summary=f"Skipped because a required step failed ({', '.join(deps)})."
                )
            else:
                ctx.step_id = step.id
                out = rt.tools.invoke(step.tool, step.args, ctx)
                rec.update(out.model_dump(exclude={"data", "layer"}))
                if out.data is not None and step.tool in (
                    "summarize_results",
                    "analyze_temporal_patterns",
                    "list_datasets",
                    "describe_dataset",
                    "get_schema",
                ):
                    rec["data"] = out.data
            if rec["status"] in ("error", "skipped", "needs_approval", "unavailable"):
                failed.add(step.id)
            records.append(rec)
            trace.append(
                {
                    "node": "execute",
                    "title": f"{step.id} · {step.tool}",
                    "kind": "step",
                    "detail": {
                        k: rec.get(k)
                        for k in (
                            "status",
                            "summary",
                            "highlights",
                            "rows",
                            "sql",
                            "warnings",
                            "notes",
                            "elapsed_ms",
                            "args",
                        )
                        if rec.get(k) not in (None, [], "")
                    },
                    "ts": time.time(),
                }
            )
        return {"steps": records, "layers": ctx.layers, "trace": trace}

    # ---- reflect -------------------------------------------------------------------------------
    def reflect(state: AgentState) -> dict[str, Any]:
        errors = [s for s in state.get("steps", []) if s["status"] == "error"]
        if errors and state.get("plan_source") == "llm" and state.get("replans", 0) < settings.max_replans:
            fb = [f"{s['step_id']} {s['tool']}: {s['summary']}" for s in errors]
            return {
                "replan_feedback": fb,
                "replans": state.get("replans", 0) + 1,
                "trace": _trace(
                    state, "reflect", "Some steps failed; asking the planner to revise", fb, "warning"
                ),
            }
        empties = [s for s in state.get("steps", []) if s["status"] == "empty"]
        detail = None
        if empties:
            detail = [f"{s['step_id']} returned no rows" for s in empties]
        title = "Results checked" + (f" ({len(errors)} step errors)" if errors else "")
        return {"trace": _trace(state, "reflect", title, detail, "warning" if errors or empties else "info")}

    def route_after_reflect(state: AgentState) -> str:
        return "plan" if state.get("replan_feedback") else "finalize"

    # ---- finalize --------------------------------------------------------------------------------
    def finalize(state: AgentState) -> dict[str, Any]:
        turn = state["turn"]
        steps = state.get("steps", [])
        arts_raw = [
            rt.store.get(s["artifact_id"]) for s in steps if s.get("artifact_id") and s["status"] == "ok"
        ]
        arts: list[Artifact] = [a for a in arts_raw if a is not None and a.turn == turn]
        primary = None
        for s in reversed(steps):
            a = rt.store.get(s.get("artifact_id") or "")
            if a is not None and a.turn == turn and s["tool"] in ANALYTIC_TOOLS and a.is_spatial:
                primary = a
                break
        if primary is None:
            primary = next((a for a in reversed(arts) if a.is_spatial), arts[-1] if arts else None)
        layers = list(state.get("layers", []))
        if not layers and primary is not None and primary.is_spatial and primary.rows:
            spec = build_layer_spec(
                primary, layer_id=f"layer-{primary.id}-auto", max_render_features=settings.max_render_features
            )
            layers.append(spec.model_dump())
        if primary is not None:
            rt.store.set_primary(turn, primary.id)
        ds_ids = [d for a in arts for d in a.datasets] + list((state.get("plan") or {}).get("datasets", []))
        datasets = _datasets_used(rt, ds_ids)
        methodology: list[str] = []
        for s in steps:
            for n in s.get("notes", []):
                if n not in methodology:
                    methodology.append(n)
        methodology.append(
            f"Distances, buffers and grids are computed in {settings.projected_crs}; results are "
            "displayed in WGS84 (EPSG:4326)."
        )
        methodology.append(
            f"SQL is validated by an AST-based read-only guard (allowlisted tables, row limit "
            f"{settings.max_result_rows:,}, timeout {settings.query_timeout_s}s)."
        )
        return {
            "layers": layers,
            "primary_artifact": primary.id if primary else None,
            "citations": citations_for(datasets),
            "methodology": methodology,
            "trace": _trace(
                state, "finalize", f"Prepared {len(layers)} map layer(s) and {len(datasets)} citation(s)"
            ),
        }

    # ---- respond ------------------------------------------------------------------------------------
    def respond(state: AgentState) -> dict[str, Any]:
        turn = state["turn"]
        p = state.get("plan") or {}
        status = state.get("status", "running")
        context = dict(state.get("context") or {})
        datasets = _datasets_used(
            rt, [c["dataset_id"] for c in state.get("citations", [])] + p.get("datasets", [])
        )
        intent = state.get("intent") or {}
        adjustments: list[str] = []
        narrator = "template"
        if status == "clarification":
            answer = Answer(
                headline=p.get("clarification") or "Could you clarify?",
                summary=p.get("clarification") or "",
                findings=[],
                limitations=[],
                suggested_follow_ups=[],
            )
            context["pending_clarification"] = p.get("clarification")
            context["last_question"] = state["effective_question"]
            context["last_intent"] = intent or context.get("last_intent")
        elif status == "rejected":
            answer = Answer(
                headline="No queries were run.",
                summary="The generated SQL was not approved, so nothing was executed. You can rephrase "
                "the question or approve the query next time.",
                findings=[],
                limitations=[],
                suggested_follow_ups=[],
            )
            context.pop("pending_clarification", None)
        elif status == "error":
            answer = Answer(
                headline="The request could not be completed.",
                summary=state.get("error") or "",
                findings=[],
                limitations=[],
                suggested_follow_ups=[],
            )
        else:
            steps = state.get("steps", [])
            answer = template_answer(
                state["question"],
                p.get("interpretation", ""),
                steps,
                intent.get("analysis"),
                datasets,
                state.get("primary_artifact"),
            )
            if rt.llm is not None and rt.policy.may_include_results(datasets) and steps:
                try:
                    user = NARRATOR_USER.format(
                        question=state["question"],
                        interpretation=p.get("interpretation", ""),
                        synthetic=any(d.synthetic for d in datasets),
                        verification=[d.id for d in datasets if d.verification_source] or "none",
                        results=wrap_untrusted("step_results", [step_digest_for_llm(s) for s in steps]),
                        datasets=wrap_untrusted("datasets", datasets_for_llm(datasets)),
                    )
                    answer = rt.llm.generate_json(NARRATOR_SYSTEM, user, Answer, purpose="narrate")
                    narrator = rt.llm.label
                except LLMError as exc:
                    adjustments.append(
                        f"LLM narration unavailable ({str(exc)[:120]}); used template summary."
                    )
            context.pop("pending_clarification", None)
            context["last_question"] = state["question"]
            context["last_interpretation"] = p.get("interpretation")
            context["last_intent"] = intent or None
            context["last_primary"] = state.get("primary_artifact")
        answer, guard_adj = enforce_guardrails(answer, datasets)
        adjustments += guard_adj
        rt.audit.log(
            "answer",
            rt.session_id,
            turn=turn,
            status=status,
            narrator=narrator,
            headline=answer.headline[:200],
            adjustments=adjustments,
        )
        final_status = status if status in ("clarification", "rejected", "error") else "complete"
        content = f"**{answer.headline}**\n\n{answer.summary}"
        return {
            "answer": {**answer.model_dump(), "narrator": narrator, "adjustments": adjustments},
            "status": final_status,
            "context": context,
            "messages": [{"role": "assistant", "content": content, "turn": turn}],
            "trace": _trace(state, "respond", f"Answer written ({narrator})", adjustments or None),
        }

    def route_after_plan(state: AgentState) -> str:
        return "respond" if state.get("status") in ("clarification", "error") else "review_sql"

    g = StateGraph(AgentState)
    for name, fn in [
        ("start", start),
        ("discover", discover),
        ("plan", plan),
        ("review_sql", review_sql),
        ("approve", approve),
        ("execute", execute),
        ("reflect", reflect),
        ("finalize", finalize),
        ("respond", respond),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "start")
    g.add_edge("start", "discover")
    g.add_edge("discover", "plan")
    g.add_conditional_edges("plan", route_after_plan, {"respond": "respond", "review_sql": "review_sql"})
    g.add_conditional_edges(
        "review_sql", route_after_review, {"plan": "plan", "approve": "approve", "execute": "execute"}
    )
    g.add_conditional_edges("approve", route_after_approve, {"respond": "respond", "execute": "execute"})
    g.add_edge("execute", "reflect")
    g.add_conditional_edges("reflect", route_after_reflect, {"plan": "plan", "finalize": "finalize"})
    g.add_edge("finalize", "respond")
    g.add_edge("respond", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


# =========================================================================================
# Session facade (used by the Streamlit UI, CLI and tests)
# =========================================================================================
@dataclass
class TurnResult:
    status: str
    turn: int
    answer: dict[str, Any] | None
    plan: dict[str, Any] | None
    plan_source: str
    steps: list[dict[str, Any]]
    layers: list[dict[str, Any]]
    sql_reviews: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    methodology: list[str]
    trace: list[dict[str, Any]]
    primary_artifact: str | None
    pending_approval: dict[str, Any] | None = None
    elapsed_s: float = 0.0


PER_TURN_RESET: dict[str, Any] = {
    "plan": None,
    "plan_source": "",
    "intent": None,
    "sql_reviews": [],
    "approval": None,
    "steps": [],
    "layers": [],
    "answer": None,
    "citations": [],
    "methodology": [],
    "trace": [],
    "replans": 0,
    "status": "running",
    "error": None,
    "primary_artifact": None,
    "replan_feedback": None,
    "candidates": [],
    "llm_used": False,
}


class AgentSession:
    def __init__(self, runtime: AgentRuntime):
        self.runtime = runtime
        self.graph = build_graph(runtime)
        self.config = {"configurable": {"thread_id": runtime.session_id}, "recursion_limit": 60}
        self.turn = 0
        self.history: list[TurnResult] = []

    def _run(self, payload: Any, on_event: Callable[[str, dict[str, Any]], None] | None) -> TurnResult:
        t0 = time.perf_counter()
        for update in self.graph.stream(payload, self.config, stream_mode="updates"):
            if on_event is None:
                continue
            for node, data in update.items():
                if node != "__interrupt__":
                    on_event(node, data if isinstance(data, dict) else {})
        snap = self.graph.get_state(self.config)
        vals = snap.values
        pending = None
        for task in snap.tasks:
            for intr in getattr(task, "interrupts", ()) or ():
                pending = intr.value
        status = "awaiting_approval" if pending else vals.get("status", "complete")
        result = TurnResult(
            status=status,
            turn=vals.get("turn", self.turn),
            answer=vals.get("answer"),
            plan=vals.get("plan"),
            plan_source=vals.get("plan_source", ""),
            steps=vals.get("steps", []),
            layers=vals.get("layers", []),
            sql_reviews=vals.get("sql_reviews", []),
            citations=vals.get("citations", []),
            methodology=vals.get("methodology", []),
            trace=vals.get("trace", []),
            primary_artifact=vals.get("primary_artifact"),
            pending_approval=pending,
            elapsed_s=round(time.perf_counter() - t0, 2),
        )
        if not pending:
            self.history.append(result)
        return result

    def ask(self, question: str, on_event: Callable[[str, dict[str, Any]], None] | None = None) -> TurnResult:
        if self.pending_approval is not None:
            raise RuntimeError("A query is awaiting approval; call resume() first")
        self.turn += 1
        payload = {
            **PER_TURN_RESET,
            "question": question,
            "turn": self.turn,
            "mode": self.runtime.mode,
            "session_id": self.runtime.session_id,
        }
        return self._run(payload, on_event)

    def resume(
        self,
        approved: bool,
        fingerprints: list[str] | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> TurnResult:
        return self._run(Command(resume={"approved": approved, "fingerprints": fingerprints}), on_event)

    @property
    def pending_approval(self) -> dict[str, Any] | None:
        snap = self.graph.get_state(self.config)
        for task in snap.tasks:
            for intr in getattr(task, "interrupts", ()) or ():
                return intr.value
        return None

    def messages(self) -> list[dict[str, Any]]:
        return list(self.graph.get_state(self.config).values.get("messages", []))
