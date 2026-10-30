"""Semantic validation: does a plan do what the question asked, and nothing more?

Structural validation (``planner.validate_plan``) checks tools, argument types and references.
It cannot tell that ``date_from: 2020-01-01`` was never requested. This module compares every
plan (LLM, LLM repair, LLM replan and rule plans) with the request scope parsed
deterministically from the question and the conversation:

* date filters must come from the question (or the follow-up context); without a period in
  the question all dates are used. Dataset titles/descriptions never count;
* borough, category, area and attribute filters must be grounded in the question;
* unrequested row limits are removed; distances and thresholds default to documented values;
* LLM-written SQL gets the same checks (predicates on date, borough, category and other
  columns); unrequested predicates are removed deterministically or the plan is rejected.

Every change is recorded as a :class:`Correction` so the trace and UI can show it. Rejected
LLM plans fall back to the deterministic plan, without another model call.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any

import sqlglot
from sqlglot import exp

from agentic_gis.agent.state import Plan, PlanStep
from agentic_gis.agent.temporal import describe_window, parse_time_window, window_contains
from agentic_gis.agent.tools import normalize_borough, resolve_categories
from agentic_gis.config.registry import (
    DatasetRegistry,
    DatasetSpec,
    UnknownDatasetError,
    location_columns_for,
)
from agentic_gis.spatial.geometry import UNIT_TO_METERS
from agentic_gis.spatial.operations import normalize_name

BOROUGH_DISPLAY = {
    "MANHATTAN": "Manhattan",
    "BRONX": "Bronx",
    "BROOKLYN": "Brooklyn",
    "QUEENS": "Queens",
    "STATEN ISLAND": "Staten Island",
}
# Defaults used when the question does not specify a parameter (same as the rule planner).
DEFAULT_CLUSTER_RADIUS_FT = 75.0
DEFAULT_CLUSTER_MIN_POINTS = 5
DEFAULT_CELL_FT = 1000.0
DEFAULT_PROXIMITY_FT = 500.0
_REF_TOOLS_INPUT = ("input", "left", "right", "target", "geography", "area")


@dataclass
class Correction:
    kind: str  # removed | changed | added | rejected | assumption
    message: str
    step: str | None = None
    field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def _ft(value: float, units: str) -> float:
    return value * UNIT_TO_METERS.get(units, 0.3048) / 0.3048


@dataclass
class QuestionScope:
    """What the user asked for, parsed deterministically (no LLM)."""

    question: str
    texts: list[str] = field(default_factory=list)  # question + earlier questions it builds on
    date_from: str | None = None
    date_to: str | None = None
    time_kind: str = "none"
    time_phrase: str | None = None
    years: list[int] = field(default_factory=list)
    boroughs: list[str] = field(default_factory=list)  # canonical upper-case names
    categories_by_dataset: dict[str, list[str]] = field(default_factory=dict)
    subject_categories: list[str] = field(default_factory=list)
    subject_dataset: str | None = None
    reference_dataset: str | None = None
    areas: list[dict[str, str]] = field(default_factory=list)
    zips: list[str] = field(default_factory=list)
    streets: list[str] = field(default_factory=list)
    distance: float | None = None
    units: str = "ft"
    min_points: int | None = None
    geography: str | None = None
    analysis: str = "list"
    use_previous: bool = False

    @classmethod
    def from_intent(cls, intent: dict[str, Any] | None, question: str) -> QuestionScope:
        if not intent:
            tw = parse_time_window(question)
            return cls(
                question=question,
                texts=[question],
                date_from=tw.date_from,
                date_to=tw.date_to,
                time_kind=tw.kind,
                time_phrase=tw.phrase,
                years=tw.years,
            )
        texts = [*intent.get("context_questions", []), intent.get("question") or question]
        years: set[int] = set()
        for t in texts:
            years |= set(parse_time_window(t).years)
        return cls(
            question=question,
            texts=[t for t in texts if t],
            date_from=intent.get("date_from"),
            date_to=intent.get("date_to"),
            time_kind=intent.get("time_kind") or "none",
            time_phrase=intent.get("time_phrase"),
            years=sorted(years),
            boroughs=[b for b in (normalize_borough(x) for x in intent.get("boroughs") or []) if b],
            categories_by_dataset=dict(intent.get("categories_by_dataset") or {}),
            subject_categories=list(intent.get("categories") or []),
            subject_dataset=intent.get("subject_dataset"),
            reference_dataset=intent.get("reference_dataset"),
            areas=list(intent.get("areas") or []),
            zips=list(intent.get("zips") or []),
            streets=list(intent.get("street_names") or []),
            distance=intent.get("distance"),
            units=intent.get("units") or "ft",
            min_points=intent.get("min_points"),
            geography=intent.get("geography"),
            analysis=intent.get("analysis") or "list",
            use_previous=bool(intent.get("use_previous")),
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QuestionScope:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ---- grounding helpers --------------------------------------------------------------
    @property
    def _blob(self) -> str:
        return " " + " ".join(_norm(t) for t in self.texts) + " "

    @property
    def _blob_names(self) -> str:
        return " " + " ".join(normalize_name(t) for t in self.texts) + " "

    def mentions(self, value: Any) -> bool:
        v = _norm(str(value))
        if not v:
            return False
        if f" {v} " in self._blob:
            return True
        n = normalize_name(str(value))
        return bool(n) and f" {n} " in self._blob_names

    @property
    def numbers(self) -> set[float]:
        return {float(x.replace(",", "")) for t in self.texts for x in re.findall(r"\d[\d,]*(?:\.\d+)?", t)}

    @property
    def has_dates(self) -> bool:
        return bool(self.date_from or self.date_to)

    @property
    def where(self) -> str:
        if self.boroughs:
            return ", ".join(BOROUGH_DISPLAY.get(b, b) for b in self.boroughs)
        if self.areas:
            return ", ".join(a.get("name", "") for a in self.areas)
        if self.zips:
            return "ZIP " + ", ".join(self.zips)
        return "citywide"

    def period_text(self) -> str:
        if not self.has_dates:
            return "all available dates"
        return describe_window(self.date_from, self.date_to) + (
            f" (from '{self.time_phrase}')" if self.time_phrase else ""
        )

    def prompt_text(self) -> str:
        """Request scope for the planner prompt (parsed from the question; enforced in code)."""
        period = (
            f"{describe_window(self.date_from, self.date_to)} (from '{self.time_phrase}')"
            if self.has_dates
            else "none named; use all dates and do not set date_from/date_to"
        )
        cats = ", ".join(self.subject_categories) or "none recognized"
        lines = [
            "Request scope (parsed from the question; plans outside it are corrected):",
            f"- time period: {period}",
            f"- boroughs: {', '.join(BOROUGH_DISPLAY.get(b, b) for b in self.boroughs) or 'none named (citywide)'}",
            f"- categories: {cats}",
        ]
        if self.areas or self.zips:
            lines.append(f"- areas: {', '.join([a.get('name', '') for a in self.areas] + self.zips)}")
        if self.streets:
            lines.append(f"- streets: {', '.join(self.streets)}")
        lines.append(
            f"- distance: {self.distance:g} {self.units}"
            if self.distance is not None
            else "- distance: none named"
        )
        return "\n".join(lines)

    def distance_for(self, kind: str) -> tuple[float, str] | None:
        """The question's distance if it is meant for ``kind`` ('proximity' or 'cluster'/'grid')."""
        if self.distance is None:
            return None
        is_prox = self.analysis == "proximity"
        if (kind == "proximity") != is_prox:
            return None
        return float(self.distance), self.units

    def area_grounded(self, name: str) -> bool:
        n = normalize_name(name)
        for a in self.areas:
            for key in ("name", "label"):
                v = normalize_name(str(a.get(key, "")))
                if v and (v == n or v in n or n in v):
                    return True
        if name.strip() in self.zips or (normalize_borough(name) or "") in self.boroughs:
            return True
        return self.mentions(name)


@dataclass
class SemanticResult:
    plan: Plan
    corrections: list[Correction] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)  # reasons the plan cannot be used as is

    @property
    def changed(self) -> bool:
        return any(c.kind in ("removed", "changed", "added") for c in self.corrections)


# =========================================================================================
# Category grounding (shared by structured filters and SQL predicates)
# =========================================================================================
def _generic_words(ds: DatasetSpec) -> set[str]:
    own = set(re.findall(r"[a-z0-9]+", f"{ds.name} {ds.short_name or ''}".lower()))
    return own | {
        "complaint",
        "complaints",
        "request",
        "requests",
        "report",
        "reports",
        "service",
        "311",
        "all",
    }


def _category_grounded(ds: DatasetSpec, term: str, scope: QuestionScope) -> bool:
    scope_cats = scope.categories_by_dataset.get(ds.id) or (
        scope.subject_categories if ds.id == scope.subject_dataset else []
    )
    exact, _like, _ = resolve_categories(ds, [term])
    if scope_cats and exact and set(exact) <= set(scope_cats):
        return True
    words = set(_norm(term).split())
    if not words or words <= _generic_words(ds):
        return False  # "sanitation complaints" names the dataset, not a category
    if scope_cats:
        # the question's categories were recognized; a different category is not requested
        return bool(exact) is False and scope.mentions(term)
    return scope.mentions(term)


def _scope_categories(ds: DatasetSpec, scope: QuestionScope) -> list[str]:
    return scope.categories_by_dataset.get(ds.id) or (
        scope.subject_categories if ds.id == scope.subject_dataset else []
    )


# =========================================================================================
# Plan validation
# =========================================================================================
class SemanticValidator:
    def __init__(self, registry: DatasetRegistry, mode: str, max_rows: int = 50_000):
        self.registry = registry
        self.mode = mode
        self.max_rows = max_rows

    def _ds(self, dataset_id: Any) -> DatasetSpec | None:
        try:
            return self.registry.get_any(str(dataset_id))
        except UnknownDatasetError:
            return None

    def check(self, plan: Plan, scope: QuestionScope, source: str = "llm") -> SemanticResult:
        res = SemanticResult(plan=plan)
        if plan.clarification and not plan.steps:
            return res
        steps: list[PlanStep] = []
        rewire: dict[str, str] = {}
        for step in plan.steps:
            args = {
                k: (rewire.get(v, v) if k in _REF_TOOLS_INPUT and isinstance(v, str) else v)
                for k, v in step.args.items()
            }
            st = PlanStep(id=step.id, tool=step.tool, args=args, purpose=step.purpose)
            handler = getattr(self, f"_check_{step.tool}", None)
            keep = handler(st, scope, res) if handler else True
            if keep is False:
                src = st.args.get("input")
                if isinstance(src, str):
                    rewire[st.id] = src
                continue
            steps.append(st)
        self._check_coverage_of_scope(steps, scope, res)
        dates_touched = any(c.field in ("date_from", "date_to", "dates") for c in res.corrections)
        geo_touched = any(c.field in ("boroughs", "names") for c in res.corrections)
        interpretation = plan.interpretation
        assumptions = list(plan.assumptions)
        if dates_touched:
            interpretation = _scrub_dates(interpretation) + f" ({scope.period_text()})"
            assumptions = [a for a in assumptions if not _mentions_ungrounded_year(a, scope)]
        res.plan = Plan(
            interpretation=interpretation.strip(),
            geography=scope.where if geo_touched or not plan.geography else plan.geography,
            datasets=plan.datasets,
            steps=steps,
            clarification=plan.clarification,
            assumptions=assumptions,
        )
        return res

    # ---- per-tool checks (return False to drop the step) ---------------------------------
    def _check_filter_dataset(self, st: PlanStep, scope: QuestionScope, res: SemanticResult) -> bool:
        a = st.args
        ds = self._ds(a.get("dataset"))
        if ds is None:
            return True
        tcol = ds.temporal.column.lower() if ds.temporal else None
        date_cols = {c.name.lower() for c in ds.columns if c.type in ("timestamp", "date")}
        if tcol:
            date_cols.add(tcol)
        llm_from, llm_to = a.get("date_from"), a.get("date_to")
        moved = False
        kept_filters: list[dict[str, Any]] = []
        for f in list(a.get("filters") or []):
            col = str(f.get("column", "")).lower()
            if col in date_cols:
                lo, hi = _filter_bounds(f)
                if col == tcol:
                    llm_from = max(filter(None, [llm_from, lo]), default=None)
                    llm_to = min(filter(None, [llm_to, hi]), default=None)
                    moved = True
                elif not (scope.has_dates and scope.mentions(col.replace("_date", "")) and lo):
                    res.corrections.append(
                        Correction(
                            "removed",
                            f"Removed a date condition on {f.get('column')} ({f.get('op')} {f.get('value')}): "
                            "the question does not ask for it.",
                            st.id,
                            "dates",
                        )
                    )
                else:
                    kept_filters.append(f)
                continue
            kept_filters.append(f)
        # ---- dates -------------------------------------------------------------------
        if ds.temporal:
            new_from, new_to = llm_from, llm_to
            had = bool(llm_from or llm_to)
            where = " (given in 'filters')" if moved else ""
            if scope.has_dates:
                if had and not window_contains(scope.date_from, scope.date_to, llm_from, llm_to):
                    new_from, new_to = scope.date_from, scope.date_to
                    res.corrections.append(
                        Correction(
                            "changed",
                            f"Changed the date filter{where} from {describe_window(llm_from, llm_to)} to "
                            f"{scope.period_text()} to match the question.",
                            st.id,
                            "date_from",
                        )
                    )
                elif not had and (ds.role == "events" or ds.id == scope.subject_dataset):
                    new_from, new_to = scope.date_from, scope.date_to
                    res.corrections.append(
                        Correction(
                            "added", f"Added the requested period: {scope.period_text()}.", st.id, "date_from"
                        )
                    )
            elif had:
                new_from = new_to = None
                why = (
                    f"'{scope.time_phrase}' does not name a period"
                    if scope.time_kind == "vague"
                    else "you asked for all dates"
                    if scope.time_kind == "all"
                    else "the question does not mention a time period"
                )
                res.corrections.append(
                    Correction(
                        "removed",
                        f"Removed the date filter {describe_window(llm_from, llm_to)}{where}: {why}, so all "
                        "available records are used.",
                        st.id,
                        "date_from",
                    )
                )
            for k, v in (("date_from", new_from), ("date_to", new_to)):
                if v:
                    a[k] = v
                else:
                    a.pop(k, None)
        # ---- boroughs ------------------------------------------------------------------
        llm_boros = list(a.get("boroughs") or [])
        keep_b: list[str] = []
        implied = bool(scope.areas or scope.zips or scope.streets)
        for raw in llm_boros:
            b = normalize_borough(str(raw))
            if b and (b in scope.boroughs or scope.mentions(raw)):
                keep_b.append(BOROUGH_DISPLAY.get(b, b))
            elif b and implied and not scope.boroughs:
                keep_b.append(BOROUGH_DISPLAY.get(b, b))
                res.corrections.append(
                    Correction(
                        "assumption",
                        f"Kept the borough filter {BOROUGH_DISPLAY.get(b, b)} (implied by the place named in the question).",
                        st.id,
                        "boroughs",
                    )
                )
            else:
                res.corrections.append(
                    Correction(
                        "removed",
                        f"Removed the borough filter '{raw}': the question does not mention it.",
                        st.id,
                        "boroughs",
                    )
                )
        if scope.boroughs and (ds.role == "events" or ds.id == scope.subject_dataset):
            have = {normalize_borough(b) for b in keep_b}
            missing = [BOROUGH_DISPLAY.get(b, b) for b in scope.boroughs if b not in have]
            if missing:
                keep_b += missing
                res.corrections.append(
                    Correction(
                        "added",
                        f"Added the requested borough filter: {', '.join(missing)}.",
                        st.id,
                        "boroughs",
                    )
                )
        if keep_b:
            a["boroughs"] = list(dict.fromkeys(keep_b))
        else:
            a.pop("boroughs", None)
        # ---- categories ------------------------------------------------------------------
        if ds.category_column:
            keep_c: list[str] = []
            for term in list(a.get("categories") or []):
                if _category_grounded(ds, str(term), scope):
                    keep_c.append(str(term))
                else:
                    res.corrections.append(
                        Correction(
                            "removed",
                            f"Removed the category filter '{term}': the question does not ask for it.",
                            st.id,
                            "categories",
                        )
                    )
            scope_cats = _scope_categories(ds, scope)
            if scope_cats and not scope.use_previous:
                have = set(resolve_categories(ds, keep_c)[0])
                missing = [c for c in scope_cats if c not in have]
                if missing and (ds.role == "events" or ds.id == scope.subject_dataset):
                    keep_c += missing
                    res.corrections.append(
                        Correction(
                            "added",
                            f"Added the requested category filter: {', '.join(missing)}.",
                            st.id,
                            "categories",
                        )
                    )
            if keep_c:
                a["categories"] = list(dict.fromkeys(keep_c))
            else:
                a.pop("categories", None)
        # ---- other attribute filters -----------------------------------------------------------
        final_filters = []
        for f in kept_filters:
            ok, why = self._attr_filter_grounded(ds, f, scope)
            if ok:
                final_filters.append(f)
            else:
                res.corrections.append(
                    Correction(
                        "removed",
                        f"Removed the filter {f.get('column')} {f.get('op')} {f.get('value')!r}: {why}.",
                        st.id,
                        "filters",
                    )
                )
        a["filters"] = final_filters
        # ---- row limit -------------------------------------------------------------------------
        lim = a.get("limit")
        if lim is not None and float(lim) not in scope.numbers:
            a.pop("limit", None)
            res.corrections.append(
                Correction(
                    "removed",
                    f"Removed the row limit {lim}: the question does not ask for a sample, so all matching "
                    "records are used.",
                    st.id,
                    "limit",
                )
            )
        return True

    def _attr_filter_grounded(
        self, ds: DatasetSpec, f: dict[str, Any], scope: QuestionScope
    ) -> tuple[bool, str]:
        col = str(f.get("column", ""))
        cl = col.lower()
        op = f.get("op", "=")
        val = f.get("value")
        vals = val if isinstance(val, list) else ([] if val is None else [val])
        bcol = (ds.join_keys.get("borough") or "borough").lower()
        if op in ("is_null", "not_null"):
            words = [w for w in re.split(r"[_\s]+", cl) if len(w) > 2 and w not in ("date", "the")]
            if any(scope.mentions(w) for w in words):
                return True, ""
            return False, "the question does not ask to exclude or select empty values"
        if cl == bcol:
            boros = [normalize_borough(str(v)) for v in vals]
            if boros and all(b and (b in scope.boroughs) for b in boros):
                return True, ""
            return False, "the borough is not in the question"
        if ds.category_column and cl == ds.category_column.lower():
            if vals and all(_category_grounded(ds, str(v), scope) for v in vals):
                return True, ""
            return False, "the category is not in the question"
        if ds.name_column and cl == ds.name_column.lower():
            streets = {normalize_name(s) for s in scope.streets}
            if vals and all(normalize_name(str(v)) in streets or scope.mentions(v) for v in vals):
                return True, ""
            return False, "the name is not in the question"
        if "zip" in cl:
            if vals and all(str(v) in scope.zips or scope.mentions(v) for v in vals):
                return True, ""
            return False, "the ZIP code is not in the question"
        if not vals:
            return False, "the condition is not in the question"
        for v in vals:
            if isinstance(v, int | float) and not isinstance(v, bool):
                if float(v) not in scope.numbers:
                    return False, f"{v} is not in the question"
            elif not scope.mentions(v):
                return False, f"'{v}' is not in the question"
        return True, ""

    def _check_filter_by_area(self, st: PlanStep, scope: QuestionScope, res: SemanticResult) -> bool:
        names = [str(n) for n in st.args.get("names") or []]
        keep = [n for n in names if scope.area_grounded(n)]
        if not keep:
            res.corrections.append(
                Correction(
                    "removed",
                    f"Removed the area filter {', '.join(names)}: the question does not name these places.",
                    st.id,
                    "names",
                )
            )
            return False
        if len(keep) < len(names):
            dropped = [n for n in names if n not in keep]
            st.args["names"] = keep
            res.corrections.append(
                Correction(
                    "removed",
                    f"Removed area names not in the question: {', '.join(dropped)}.",
                    st.id,
                    "names",
                )
            )
        return True

    def _check_compare_periods(self, st: PlanStep, scope: QuestionScope, res: SemanticResult) -> bool:
        for key in ("period_a", "period_b"):
            per = st.args.get(key)
            if not per:
                continue
            ok = isinstance(per, list) and len(per) == 2
            if ok:
                ys = (
                    {int(str(per[0])[:4]), int(str(per[1])[:4])}
                    if all(str(p)[:4].isdigit() for p in per)
                    else set()
                )
                ok = bool(ys) and (
                    ys <= set(scope.years)
                    or (
                        scope.has_dates
                        and window_contains(scope.date_from, scope.date_to, str(per[0]), str(per[1]))
                    )
                )
            if not ok:
                st.args.pop(key, None)
                res.corrections.append(
                    Correction(
                        "removed",
                        f"Removed {key} {per}: the question does not name this period; the earliest and latest "
                        "full years in the data are compared.",
                        st.id,
                        "dates",
                    )
                )
        return True

    def _check_find_hotspots(self, st: PlanStep, scope: QuestionScope, res: SemanticResult) -> bool:
        a = st.args
        units = a.get("units", "ft")
        if a.get("method", "clusters") == "clusters":
            given = scope.distance_for("cluster")
            target_r, target_u = given or (DEFAULT_CLUSTER_RADIUS_FT, "ft")
            if abs(_ft(float(a.get("radius", 300)), units) - _ft(target_r, target_u)) > 0.5:
                why = "the radius in the question" if given else "the standard radius"
                res.corrections.append(
                    Correction(
                        "changed",
                        f"Used {why} {target_r:g} {target_u} instead of {a.get('radius')} {units}.",
                        st.id,
                        "radius",
                    )
                )
                a["radius"], a["units"] = target_r, target_u
            target_n = scope.min_points or DEFAULT_CLUSTER_MIN_POINTS
            if int(a.get("min_points", 5)) != target_n:
                why = "the threshold in the question" if scope.min_points else "the standard threshold"
                res.corrections.append(
                    Correction(
                        "changed",
                        f"Used {why} of {target_n} reports per location instead of {a.get('min_points')}.",
                        st.id,
                        "min_points",
                    )
                )
                a["min_points"] = target_n
        else:
            target_c, target_u = scope.distance_for("grid") or (DEFAULT_CELL_FT, "ft")
            if abs(_ft(float(a.get("cell_size", 1000)), units) - _ft(target_c, target_u)) > 0.5:
                res.corrections.append(
                    Correction(
                        "changed",
                        f"Used a {target_c:g} {target_u} grid instead of {a.get('cell_size')} {units}.",
                        st.id,
                        "cell_size",
                    )
                )
                a["cell_size"], a["units"] = target_c, target_u
        return True

    def _check_spatial_join(self, st: PlanStep, scope: QuestionScope, res: SemanticResult) -> bool:
        a = st.args
        if a.get("predicate") != "dwithin":
            return True
        units = a.get("units", "ft")
        given = scope.distance_for("proximity")
        target, tu = given or (DEFAULT_PROXIMITY_FT, "ft")
        cur = a.get("distance")
        if cur is None or abs(_ft(float(cur), units) - _ft(target, tu)) > 0.5:
            why = "the distance in the question" if given else "the standard distance"
            res.corrections.append(
                Correction(
                    "changed", f"Used {why} {target:g} {tu} instead of {cur} {units}.", st.id, "distance"
                )
            )
            a["distance"], a["units"] = target, tu
        return True

    def _check_buffer_features(self, st: PlanStep, scope: QuestionScope, res: SemanticResult) -> bool:
        a = st.args
        given = scope.distance_for("proximity")
        if given and abs(_ft(float(a.get("distance", 0)), a.get("units", "ft")) - _ft(*given)) > 0.5:
            res.corrections.append(
                Correction(
                    "changed",
                    f"Used the distance in the question ({given[0]:g} {given[1]}).",
                    st.id,
                    "distance",
                )
            )
            a["distance"], a["units"] = given
        elif not given:
            res.corrections.append(
                Correction(
                    "assumption",
                    f"Buffer distance {a.get('distance')} {a.get('units', 'ft')} was chosen by the planner.",
                    st.id,
                    "distance",
                )
            )
        return True

    def _check_nearest_features(self, st: PlanStep, scope: QuestionScope, res: SemanticResult) -> bool:
        a = st.args
        given = scope.distance_for("proximity")
        if given:
            a["max_distance"], a["units"] = given
        elif a.get("max_distance") is not None:
            res.corrections.append(
                Correction(
                    "removed",
                    f"Removed the search limit {a['max_distance']} {a.get('units', 'ft')}: not in the question.",
                    st.id,
                    "max_distance",
                )
            )
            a.pop("max_distance", None)
        return True

    def _check_aggregate_by_geography(self, st: PlanStep, scope: QuestionScope, res: SemanticResult) -> bool:
        a = st.args
        roles = set(self.registry.geography_roles(self.mode))
        geo = a.get("geography")
        if (
            scope.geography in roles
            and isinstance(geo, str)
            and geo in roles
            and geo != scope.geography
            and scope.analysis in ("by_geography", "temporal", "change")
        ):
            res.corrections.append(
                Correction(
                    "changed", f"Aggregated by {scope.geography} as asked (not {geo}).", st.id, "geography"
                )
            )
            a["geography"] = scope.geography
        grid = scope.distance_for("grid")
        if a.get("geography") == "grid" and grid:
            a["cell_size"], a["units"] = grid
        top = a.get("top_n")
        if top is not None and float(top) not in scope.numbers:
            a.pop("top_n", None)
            res.corrections.append(
                Correction(
                    "removed",
                    f"Removed top_n={top}: the question does not ask for a top list.",
                    st.id,
                    "top_n",
                )
            )
        mv = a.get("min_value")
        if scope.min_points is not None and mv is not None and float(mv) != float(scope.min_points):
            a["min_value"] = scope.min_points
            res.corrections.append(
                Correction(
                    "changed", f"Used the threshold in the question ({scope.min_points}).", st.id, "min_value"
                )
            )
        elif mv is not None and float(mv) not in scope.numbers:
            res.corrections.append(
                Correction(
                    "assumption",
                    f"Areas with a value below {mv:g} are not shown (planner default).",
                    st.id,
                    "min_value",
                )
            )
        return True

    def _check_query_local_data(self, st: PlanStep, scope: QuestionScope, res: SemanticResult) -> bool:
        return self._check_sql(st, scope, res, "local")

    def _check_query_databricks(self, st: PlanStep, scope: QuestionScope, res: SemanticResult) -> bool:
        return self._check_sql(st, scope, res, "databricks")

    def _check_sql(self, st: PlanStep, scope: QuestionScope, res: SemanticResult, backend: str) -> bool:
        dialect = "duckdb" if backend == "local" else "databricks"
        out = check_sql_scope(
            str(st.args.get("sql", "")), dialect, scope, self.registry, self.mode, backend, self.max_rows
        )
        for c in out.corrections:
            c.step = st.id
        res.corrections += out.corrections
        if out.violations:
            res.rejected += [f"{st.id}: {v}" for v in out.violations]
            return True
        sql, added = widen_sql_projection(out.sql, dialect, self.registry, self.mode, backend)
        if added:
            res.corrections.append(
                Correction(
                    "added",
                    f"Added location columns {', '.join(added)} to the SQL so the places in the result can be named.",
                    st.id,
                    "columns",
                )
            )
        if sql != st.args.get("sql"):
            st.args["sql"] = sql
        return True

    # ---- whole-plan checks ---------------------------------------------------------------
    def _check_coverage_of_scope(
        self, steps: list[PlanStep], scope: QuestionScope, res: SemanticResult
    ) -> None:
        """Geographic restrictions in the question must appear somewhere in the plan."""
        if scope.use_previous:
            return
        uses_sql = any(s.tool in ("query_local_data", "query_databricks") for s in steps)
        if (scope.areas or scope.zips) and not uses_sql:
            areas = [a.get("name", "") for a in scope.areas] + scope.zips
            applied = any(
                s.tool == "filter_by_area"
                and any(scope.area_grounded(str(n)) for n in s.args.get("names") or [])
                for s in steps
            )
            if not applied:
                res.rejected.append(
                    f"The question names {', '.join(areas)}, but the plan does not restrict to it."
                )


# =========================================================================================
# Helpers
# =========================================================================================
def _filter_bounds(f: dict[str, Any]) -> tuple[str | None, str | None]:
    """Date bounds of an AttributeFilter on a date column (inclusive)."""
    op, val = f.get("op", "="), f.get("value")
    vals = val if isinstance(val, list) else [val]
    ds: list[str] = [d for d in (_as_date(v) for v in vals) if d]
    if not ds:
        return None, None
    if op == "between" and len(ds) == 2:
        return min(ds), max(ds)
    if op in (">=",):
        return ds[0], None
    if op == ">":
        return (date.fromisoformat(ds[0]) + timedelta(days=1)).isoformat(), None
    if op == "<=":
        return None, ds[0]
    if op == "<":
        return None, (date.fromisoformat(ds[0]) - timedelta(days=1)).isoformat()
    return min(ds), max(ds)


def _as_date(v: Any) -> str | None:
    s = str(v)
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
        except ValueError:
            return None
    if re.fullmatch(r"(19|20)\d\d", s):
        return f"{s}-01-01"
    return None


_DATE_PHRASE = re.compile(
    r"\s*,?\s*\b(?:from|between|in|during|for|since|after|before|until|through)?\s*"
    r"(?:\d{4}-\d{2}-\d{2}|(?:19|20)\d\d)(?:\s*(?:to|and|-|–|through|until)\s*(?:\d{4}-\d{2}-\d{2}|(?:19|20)\d\d))?",
    re.I,
)


def _scrub_dates(text: str) -> str:
    return re.sub(r"\s{2,}", " ", _DATE_PHRASE.sub("", text)).strip(" ,;")


def _mentions_ungrounded_year(text: str, scope: QuestionScope) -> bool:
    years = {int(y) for y in re.findall(r"\b((?:19|20)\d\d)\b", text)}
    return bool(years - set(scope.years))


# =========================================================================================
# LLM SQL
# =========================================================================================
@dataclass
class SQLScopeResult:
    sql: str
    corrections: list[Correction] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)


def _conjuncts(e: Any) -> list[exp.Expression]:
    if isinstance(e, exp.And):
        return _conjuncts(e.left) + _conjuncts(e.right)
    if isinstance(e, exp.Paren) and isinstance(e.this, exp.And):
        return _conjuncts(e.this)
    return [e]


def _literal_values(e: exp.Expression) -> list[Any]:
    out: list[Any] = []
    for lit in e.find_all(exp.Literal):
        if lit.is_string:
            out.append(lit.this)
        else:
            try:
                out.append(float(lit.this))
            except ValueError:
                out.append(lit.this)
    return out


def _date_literals(values: list[Any]) -> tuple[list[str], list[int]]:
    dates, years = [], []
    for v in values:
        if isinstance(v, float):
            if v.is_integer() and 1900 <= v <= 2100:
                years.append(int(v))
            continue
        d = _as_date(v)
        if d and len(str(v)) >= 10:
            dates.append(d)
        elif d:
            years.append(int(d[:4]))
    return dates, years


def _tree_datasets(tree: Any, registry: DatasetRegistry, mode: str, backend: str) -> list[DatasetSpec]:
    tables = registry.allowed_tables(mode, backend)
    datasets: list[DatasetSpec] = []
    for t in tree.find_all(exp.Table):
        names = [t.name.lower(), ".".join(p for p in (t.catalog, t.db, t.name) if p).lower()]
        for n in names:
            for key, did in tables.items():
                if key == n or key.split(".")[-1] == n:
                    ds = registry.get_any(did)
                    if ds not in datasets:
                        datasets.append(ds)
    return datasets


def widen_sql_projection(
    sql: str, dialect: str, registry: DatasetRegistry, mode: str, backend: str
) -> tuple[str, list[str]]:
    """Add missing location/spatial columns to a simple record-level SELECT over one dataset.

    Only plain ``SELECT cols FROM table [WHERE ...]`` queries are widened (no joins, grouping,
    aggregates, DISTINCT, window functions, CTEs or subqueries in FROM), so the row set is unchanged.
    Returns the (possibly) rewritten SQL and the columns added.
    """
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return sql, []
    if not isinstance(tree, exp.Select):
        return sql, []
    frm = tree.args.get("from_") or tree.args.get("from")  # key name differs across sqlglot versions
    if (
        frm is None
        or not isinstance(frm.this, exp.Table)
        or tree.args.get("joins")
        or tree.args.get("group")
        or tree.args.get("distinct")
        or tree.args.get("with_")
        or tree.args.get("with")
        or tree.find(exp.AggFunc)
        or tree.find(exp.Window)
    ):
        return sql, []
    if any(
        isinstance(e, exp.Star) or isinstance(getattr(e, "this", None), exp.Star) for e in tree.expressions
    ):
        return sql, []
    found = _tree_datasets(tree, registry, mode, backend)
    if len(found) != 1 or found[0].spatial is None or not found[0].columns:
        return sql, []
    ds = found[0]
    sp = ds.spatial
    if sp is None:
        return sql, []
    present = {e.alias_or_name.lower() for e in tree.expressions}
    spatial_cols = [
        c for c in (sp.geometry_column, sp.lat_column, sp.lon_column) if c and c in ds.column_names
    ]
    want = location_columns_for(ds, ds.column_names) + spatial_cols
    add = [c for c in want if c.lower() not in present and c.lower() not in ds.sensitive_columns]
    if not add:
        return sql, []
    qualifier = frm.this.alias_or_name if frm.this.alias else None
    for c in add:
        tree = tree.select(exp.column(c, table=qualifier, quoted=True), append=True, copy=False)
    return tree.sql(dialect=dialect), add


def sql_datasets(sql: str, dialect: str, registry: DatasetRegistry, mode: str, backend: str) -> list[str]:
    """Registered datasets referenced by a SQL statement (empty if it cannot be parsed)."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return []
    return [d.id for d in _tree_datasets(tree, registry, mode, backend)] if tree is not None else []


def check_sql_scope(
    sql: str,
    dialect: str,
    scope: QuestionScope,
    registry: DatasetRegistry,
    mode: str,
    backend: str,
    max_rows: int = 50_000,
) -> SQLScopeResult:
    """Check the WHERE predicates of LLM SQL against the request scope.

    Unrequested top-level AND conditions are removed; conditions that cannot be removed safely
    (inside OR/NOT) and requested filters that are missing make the SQL unusable (violations).
    Parsing problems are left to the SQL guard, which rejects unparseable SQL.
    """
    out = SQLScopeResult(sql=sql)
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return out
    if tree is None:
        return out
    datasets = _tree_datasets(tree, registry, mode, backend)
    if not datasets:
        return out
    tcols: set[str] = set()
    bcols: set[str] = set()
    ccols: dict[str, DatasetSpec] = {}
    for ds in datasets:
        if ds.temporal:
            tcols.add(ds.temporal.column.lower())
        tcols |= {c.name.lower() for c in ds.columns if c.type in ("timestamp", "date")}
        bcols.add((ds.join_keys.get("borough") or "borough").lower())
        if ds.category_column:
            ccols[ds.category_column.lower()] = ds
    found_t = found_b = False
    found_c: set[str] = set()
    changed = False
    for sel in list(tree.find_all(exp.Select)):
        where = sel.args.get("where")
        if where is None:
            continue
        keep: list[exp.Expression] = []
        for conj in _conjuncts(where.this):
            cols = {c.name.lower() for c in conj.find_all(exp.Column)}
            vals = _literal_values(conj)
            problem = None
            kind = None
            if cols & tcols:
                kind = "date"
                dates, years = _date_literals(vals)
                relative = any(conj.find_all(exp.CurrentDate, exp.CurrentTimestamp)) or bool(
                    re.search(r"\b(now|current_date|current_timestamp|today)\b", conj.sql().lower())
                )
                if not dates and not years and not relative:
                    if isinstance(conj, exp.Not | exp.Is) or "null" in conj.sql().lower():
                        kind = None  # null checks only
                    else:
                        problem = "a date condition without dates"
                elif not scope.has_dates:
                    problem = "a date condition, but the question does not mention a time period"
                elif relative and scope.time_kind != "relative":
                    problem = "a relative date condition the question does not ask for"
                else:
                    lo = scope.date_from or "0001-01-01"
                    hi = (
                        (date.fromisoformat(scope.date_to) + timedelta(days=1)).isoformat()
                        if scope.date_to
                        else "9999-12-31"
                    )
                    bad = [d for d in dates if not (lo <= d <= hi)]
                    ylo, yhi = int(lo[:4]), int(hi[:4]) if scope.date_to else 9999
                    bad_y = [y for y in years if not (ylo <= y <= yhi)]
                    if bad or bad_y:
                        problem = (
                            f"dates {', '.join([*bad, *map(str, bad_y)])} outside the requested period "
                            f"{scope.period_text()}"
                        )
                    else:
                        found_t = True
            elif cols & bcols:
                kind = "borough"
                boros = [normalize_borough(str(v)) for v in vals if isinstance(v, str)]
                if boros and all(b and b in scope.boroughs for b in boros):
                    found_b = True
                else:
                    problem = f"a borough condition ({', '.join(str(v) for v in vals)}) not in the question"
            elif cols & set(ccols):
                kind = "category"
                cname = next(iter(cols & set(ccols)))
                cds = ccols[cname]
                strs = [str(v).strip("%") for v in vals if isinstance(v, str)]
                if strs and all(_category_grounded(cds, s, scope) for s in strs):
                    found_c.add(cds.id)
                else:
                    problem = f"a category condition ({', '.join(strs)}) not in the question"
            else:
                strs = [str(v).strip("%") for v in vals if isinstance(v, str) and str(v).strip("%")]
                ungrounded = [s for s in strs if not scope.mentions(s)]
                if ungrounded:
                    kind = "attribute"
                    problem = f"a condition on {', '.join(sorted(cols)) or 'a value'} ({', '.join(ungrounded)}) not in the question"
            if problem is None:
                keep.append(conj)
                continue
            changed = True  # a top-level AND condition can be dropped without changing the rest
            out.corrections.append(
                Correction(
                    "removed", f"Removed SQL condition `{conj.sql(dialect=dialect)}`: {problem}.", None, kind
                )
            )
        if len(keep) != len(_conjuncts(where.this)):
            if keep:
                cond: Any = keep[0]
                for k in keep[1:]:
                    cond = exp.and_(cond, k)
                sel.set("where", exp.Where(this=cond))
            else:
                sel.set("where", None)
    # violations that live inside OR/NOT cannot be removed: detect them after repair
    for sel in tree.find_all(exp.Select):
        where = sel.args.get("where")
        if where is None:
            continue
        for node in where.find_all(exp.Or, exp.Not):
            cols = {c.name.lower() for c in node.find_all(exp.Column)}
            if cols & tcols and not scope.has_dates:
                out.violations.append(
                    "SQL contains a date condition inside OR/NOT, but the question has no period."
                )
    for ds in datasets:
        if ds.role != "events":
            continue
        if scope.has_dates and ds.temporal and not found_t:
            out.violations.append(f"SQL does not apply the requested period ({scope.period_text()}).")
        if _scope_categories(ds, scope) and ds.id not in found_c and not scope.use_previous:
            out.violations.append(
                f"SQL does not filter to the requested categories ({', '.join(_scope_categories(ds, scope))})."
            )
        if scope.boroughs and not found_b:
            out.violations.append(
                f"SQL does not filter to the requested borough(s) ({', '.join(BOROUGH_DISPLAY.get(b, b) for b in scope.boroughs)})."
            )
    lim = tree.args.get("limit")
    if lim is not None:
        lit = lim.expression if isinstance(lim, exp.Limit) else None
        n = None
        if isinstance(lit, exp.Literal) and not lit.is_string:
            try:
                n = int(float(lit.this))
            except ValueError:
                n = None
        if n is not None and n < max_rows and float(n) not in scope.numbers and not tree.args.get("order"):
            tree.set("limit", None)
            changed = True
            out.corrections.append(
                Correction(
                    "removed",
                    f"Removed `LIMIT {n}` from the SQL: the question does not ask for a sample.",
                    None,
                    "limit",
                )
            )
    if changed:
        out.sql = tree.sql(dialect=dialect)
    out.violations = list(dict.fromkeys(out.violations))
    return out
