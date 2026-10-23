"""Planning: deterministic intent-based planner + LLM planner prompt construction + plan validation.

The rule planner is not a lookup table of example questions. It parses a question into a
generic :class:`Intent` (topic categories, subject/reference datasets, geography, distance,
time window, analysis type) using vocabulary taken from the dataset registry and reference
data, then composes tool steps. It powers demo mode and is the fallback when the LLM is
unavailable or returns an invalid plan.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from agentic_gis.agent.state import Plan, PlanStep
from agentic_gis.agent.tools import BOROUGH_ALIASES, ToolRegistry
from agentic_gis.agent.workspace import ResultStore
from agentic_gis.config.registry import DatasetRegistry, DatasetSpec
from agentic_gis.security.policy import DataEgressPolicy
from agentic_gis.security.sanitize import clean_text, wrap_untrusted
from agentic_gis.spatial.operations import normalize_name

log = logging.getLogger(__name__)

BOROUGH_DISPLAY = {
    "MANHATTAN": "Manhattan",
    "BRONX": "Bronx",
    "BROOKLYN": "Brooklyn",
    "QUEENS": "Queens",
    "STATEN ISLAND": "Staten Island",
}

_RX = {
    "change": r"\b(increase[sd]?|increasing|growth|grew|grow(ing)?|ris(e|ing)|rose|jump|spike|surge|decrease[sd]?|"
    r"decline[sd]?|declining|drop(ped)?|fell|change[sd]?|changing|worsen\w*|improv\w*)\b",
    "temporal": r"\b(over time|trends?|trending|time series|monthly|weekly|yearly|annual(ly)?|by (month|week|year|"
    r"quarter)|per (month|week|year)|season\w*|year[- ]over[- ]year|across (the )?years|history)\b",
    "overlap": r"\b(overlap\w*|co-?occur\w*|coincid\w*|multiple (types|kinds|categories)|several (types|kinds)|"
    r"different (types|kinds)|more than one type|mix(ed)? of)\b",
    "proximity": r"\b(near|nearby|close to|closest|adjacent|within|around|next to|proximity|vicinity|surrounding|"
    r"along|next door|neighbo(u)?ring)\b",
    "repeat": r"\b(repeat(ed|ing)?|recurr\w*|repeatedly|chronic|persistent|same (location|place|spot|address)|"
    r"multiple (complaints|reports|times)|again and again|problem (locations|spots|sites))\b",
    "hotspot": r"\b(hot ?spots?|concentrat\w*|clusters?|clustering|dense|density|heat ?map|high (number|volume|"
    r"levels?|rates?) of|hot zones?)\b",
    "by_geo": r"\b(by|per|each|across|which|what|top|most|rank\w*|compare|comparison|breakdown)\b",
    "followup_ref": r"\b(those|these|them|the same|previous|above|earlier|that (result|data|analysis|map|area)s?|"
    r"(this|that) result)\b",
    "followup_start": r"^(and|now|what about|how about|only|just|also|instead|then|ok(ay)?|same|but|"
    r"break (it|that|them) down|zoom (in|to)|limit (it|that|them|to)|restrict|filter (it|that|"
    r"them|those|these|to))\b",
}
GEO_WORDS = {
    "borough": r"\bboroughs?\b",
    "neighborhood": r"\bneighbou?rhoods?\b|\bntas?\b",
    "community_district": r"\bcommunity (districts?|boards?)\b|\bcds?\b",
    "zip": r"\bzip( ?codes?)?s?\b|\bpostal\b",
}
UNITS = {
    "feet": "ft",
    "foot": "ft",
    "ft": "ft",
    "'": "ft",
    "meters": "m",
    "meter": "m",
    "metres": "m",
    "m": "m",
    "miles": "mi",
    "mile": "mi",
    "mi": "mi",
    "km": "km",
    "kilometers": "km",
    "yards": "yd",
    "yd": "yd",
    "blocks": "block",
    "block": "block",
}


@dataclass
class Vocabulary:
    """Names the rule planner can recognize, built from the registry and reference data."""

    streets: dict[str, str] = field(default_factory=dict)  # normalized -> display
    areas: dict[str, tuple[str, str]] = field(default_factory=dict)  # normalized part -> (role, label)


@dataclass
class Intent:
    question: str
    analysis: str = "list"
    explicit_analysis: bool = False
    event_dataset: str | None = None
    categories: list[str] = field(default_factory=list)
    reference_categories: list[str] = field(default_factory=list)
    subject_dataset: str | None = None
    reference_dataset: str | None = None
    street_names: list[str] = field(default_factory=list)
    boroughs: list[str] = field(default_factory=list)
    areas: list[dict[str, str]] = field(default_factory=list)
    zips: list[str] = field(default_factory=list)
    distance: float | None = None
    units: str = "ft"
    date_from: str | None = None
    date_to: str | None = None
    freq: str | None = None
    group_by: str | None = None
    geography: str | None = None
    min_points: int | None = None
    use_previous: bool = False
    clarification: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Intent:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _find(rx: str, text: str) -> re.Match[str] | None:
    return re.search(rx, text, re.I)


def _first_pos(text: str, terms: list[str]) -> int:
    best = 10**6
    for t in terms:
        m = re.search(rf"\b{re.escape(t.lower())}", text)
        if m:
            best = min(best, m.start())
    return best


class RulePlanner:
    def __init__(self, registry: DatasetRegistry, mode: str, vocabulary: Vocabulary | None = None):
        self.registry = registry
        self.mode = mode
        self.vocab = vocabulary or Vocabulary()

    # ---------------------------------------------------------------------------------------
    # Intent parsing
    # ---------------------------------------------------------------------------------------
    def parse(self, question: str, previous: dict[str, Any] | None = None) -> Intent:
        q = question.strip()
        ql = q.lower()
        it = Intent(question=q)
        available = self.registry.available(self.mode)
        events = [d for d in available if d.role == "events"]
        others = [d for d in available if d.role in ("infrastructure", "parcels", "streets")]

        # --- topics (category values) and datasets mentioned ---------------------------------
        mention_pos: dict[str, int] = {}
        cats_by_ds: dict[str, list[str]] = {}
        for ds in events:
            cats, pos = self._match_categories(ds, ql)
            generic = _find(r"\b(311|complaints?|requests?|reports?|sanitation|dsny|service requests?)\b", ql)
            if cats or generic:
                it.event_dataset = it.event_dataset or ds.id
                cats_by_ds[ds.id] = cats
                p = pos if cats else (generic.start() if generic else 10**6)
                mention_pos[ds.id] = min(p, generic.start() if generic else 10**6)
        for ds in others:
            terms = [t for t in ds.tags if len(t) > 3] + list(ds.synonyms)
            p = _first_pos(ql, terms)
            if p < 10**6:
                mention_pos[ds.id] = p
                if ds.category_column:
                    cats_by_ds[ds.id] = self._match_categories(ds, ql)[0]

        # --- geography -------------------------------------------------------------------------
        for alias, boro in sorted(BOROUGH_ALIASES.items(), key=lambda kv: -len(kv[0])):
            if len(alias) <= 2:
                continue
            if re.search(rf"\b{re.escape(alias)}\b", ql) and boro not in it.boroughs:
                it.boroughs.append(boro)
        qn = normalize_name(q)
        # --- streets (before areas, so "Jamaica Avenue" is not read as the Jamaica neighborhood) ---
        for norm, display in sorted(self.vocab.streets.items(), key=lambda kv: -len(kv[0])):
            if re.search(rf"\b{re.escape(norm)}\b", qn) and display not in it.street_names:
                if not any(norm in normalize_name(s) for s in it.street_names):
                    it.street_names.append(display)
        street_norms = [normalize_name(s) for s in it.street_names]
        for norm, (role, label) in sorted(self.vocab.areas.items(), key=lambda kv: -len(kv[0])):
            if any(norm in sn for sn in street_norms):
                continue
            if len(norm) >= 4 and re.search(rf"\b{re.escape(norm)}\b", qn):
                if norm in {b for b in BOROUGH_ALIASES.values()}:
                    continue
                if not any(a["label"] == label for a in it.areas):
                    it.areas.append({"role": role, "name": norm.title(), "label": label})
        it.zips = sorted(set(re.findall(r"\b(1[0-1]\d{3})\b", q)))
        for role, rx in GEO_WORDS.items():
            if _find(rx, ql):
                it.geography = role
                break

        # --- distance / thresholds / time ----------------------------------------------------------
        m = re.search(
            r"(\d+(?:\.\d+)?)\s*(feet|foot|ft|'|meters?|metres|m|miles?|mi|km|kilometers|yards|yd|"
            r"blocks?)\b",
            ql,
        )
        if m:
            it.distance, it.units = float(m.group(1)), UNITS[m.group(2)]
        m = re.search(
            r"\b(?:at least|more than|over|minimum of|>=?)\s*(\d+)\s*(?:complaints|reports|times|records)", ql
        )
        if m:
            it.min_points = int(m.group(1)) + (1 if re.search(r"more than|over", m.group(0)) else 0)
        years = sorted({int(y) for y in re.findall(r"\b(20[0-3]\d)\b", q)})
        if re.search(r"\b(since|after|from)\s+20[0-3]\d\b", ql) and years:
            it.date_from = f"{years[0]}-01-01"
            if len(years) > 1:
                it.date_to = f"{years[-1]}-12-31"
        elif re.search(r"\bbetween\s+20[0-3]\d\s+and\s+20[0-3]\d\b", ql) and len(years) >= 2:
            it.date_from, it.date_to = f"{years[0]}-01-01", f"{years[-1]}-12-31"
        elif re.search(r"\b(before|until|through)\s+20[0-3]\d\b", ql) and years:
            it.date_to = f"{years[-1]}-12-31"
        elif len(years) == 1:
            it.date_from, it.date_to = f"{years[0]}-01-01", f"{years[0]}-12-31"
        for f in ("week", "month", "quarter", "year"):
            if re.search(rf"\b(by|per|each|every) {f}\b|\b{f}ly\b|\b{f}\b", ql):
                it.freq = f
                break
        if it.freq is None and re.search(r"\bannual(ly)?\b|\bover the years\b", ql):
            it.freq = "year"

        # --- analysis type -----------------------------------------------------------------------
        streets_ds = next((d for d in others if d.role == "streets"), None)
        asks_streets = bool(
            re.search(r"\bstreets?\b|\bavenues?\b|\bblocks? faces?\b|\bcorridors?\b|\broads?\b", ql)
        )
        has_ref_target = any(d.id in mention_pos for d in others)
        if _find(_RX["overlap"], ql):
            it.analysis = "overlap"
        elif _find(_RX["change"], ql) and not _find(r"\bwithin\b", ql):
            it.analysis = "change"
        elif (
            asks_streets
            and streets_ds
            and _find(_RX["repeat"] + "|" + _RX["hotspot"] + r"|\bmost\b", ql)
            and not it.street_names
            and it.event_dataset
        ):
            it.analysis = "by_street"
            it.reference_dataset = streets_ds.id
        elif (it.distance is not None or _find(_RX["proximity"], ql)) and (has_ref_target or it.street_names):
            it.analysis = "proximity"
        elif _find(_RX["repeat"], ql):
            it.analysis = "repeat"
        elif _find(_RX["hotspot"], ql):
            it.analysis = "hotspot"
        elif _find(_RX["temporal"], ql) or (it.freq and _find(r"\bcompare|trend|pattern", ql)):
            it.analysis = "temporal"
        elif it.geography and _find(_RX["by_geo"], ql):
            it.analysis = "by_geography"
        else:
            it.analysis = "list"
        it.explicit_analysis = it.analysis != "list"
        if it.analysis == "temporal":
            if it.geography == "borough" or _find(r"\bboroughs\b", ql):
                it.group_by = "borough"
            elif len(it.categories) > 1 or _find(r"\b(by|per|each) (type|category|complaint type)\b", ql):
                it.group_by = "category"
            it.freq = it.freq or ("year" if _find(r"\byears?\b", ql) else "month")

        # --- subject / reference for proximity ---------------------------------------------------
        if it.analysis == "proximity":
            ranked = sorted(mention_pos.items(), key=lambda kv: kv[1])
            ids = [k for k, _ in ranked]
            if it.street_names and streets_ds and streets_ds.id not in ids:
                ids.append(streets_ds.id)
            if len(ids) >= 2:
                it.subject_dataset, it.reference_dataset = ids[0], ids[1]
            elif len(ids) == 1:
                it.subject_dataset = ids[0]
            if it.street_names and streets_ds:
                if it.subject_dataset == streets_ds.id and it.reference_dataset:
                    it.subject_dataset, it.reference_dataset = it.reference_dataset, streets_ds.id
                elif it.reference_dataset is None:
                    it.reference_dataset = streets_ds.id
        else:
            it.subject_dataset = it.event_dataset or next(
                iter(sorted(mention_pos, key=lambda k: mention_pos[k])), None
            )
        it.categories = cats_by_ds.get(it.subject_dataset or "", [])
        it.reference_categories = cats_by_ds.get(it.reference_dataset or "", [])

        # --- clarification: streets referenced but not named ---------------------------------------
        needs_names = (
            it.analysis == "proximity"
            and streets_ds is not None
            and it.reference_dataset == streets_ds.id
            and not it.street_names
        )
        if needs_names and _find(
            r"\b(specified|specific|certain|particular|given|selected|named|these|those|some)\s+"
            r"(streets?|roads?|avenues?|corridors?)\b|\bstreets?\b",
            ql,
        ):
            examples = ", ".join(sorted(self.vocab.streets.values())[:8])
            it.clarification = "Which streets should I use? Please name one or more streets" + (
                f" (for example: {examples})." if examples else "."
            )

        # --- follow-up merging -------------------------------------------------------------------------
        if previous:
            it = self._merge_followup(it, Intent.from_dict(previous), ql)
        if not it.subject_dataset and not it.clarification:
            if events:
                it.subject_dataset = it.event_dataset = events[0].id
                it.notes.append(f"No dataset named; assumed {events[0].label}.")
        return it

    def _match_categories(self, ds: DatasetSpec, ql: str) -> tuple[list[str], int]:
        found: list[str] = []
        pos = 10**6
        known = ds.known_categories()
        if not known:  # without declared values we cannot map words to categories reliably
            return found, pos
        phrases: list[tuple[str, str]] = [(k.lower(), v) for k, v in ds.synonyms.items() if v in known]
        phrases += [(k.lower(), k) for k in known]
        own = set(re.findall(r"[a-z]+", f"{ds.name} {ds.short_name or ''}".lower()))
        generic = {w for t in ds.tags for w in re.findall(r"[a-z]+", t.lower()) if w in own} | {
            "complaint",
            "complaints",
            "condition",
            "request",
            "collection",
            "street",
            "vehicles",
            "service",
            "needs",
        }
        for k in known:  # distinctive words of each category ("dumping", "derelict", "sweeping")
            for w in re.findall(r"[a-z]+", k.lower()):
                if len(w) > 4 and w not in generic:
                    phrases.append((w, k))
        spans: list[tuple[int, int]] = []
        for phrase, value in sorted(phrases, key=lambda kv: -len(kv[0])):
            m = re.search(rf"\b{re.escape(phrase)}", ql)
            if not m or any(m.start() < e and s < m.end() for s, e in spans):
                continue
            spans.append((m.start(), m.end()))
            if value not in found:
                found.append(value)
                pos = min(pos, m.start())
        return found, pos

    def _merge_followup(self, it: Intent, prev: Intent, ql: str) -> Intent:
        refers = bool(_find(_RX["followup_ref"], ql)) or bool(_find(_RX["followup_start"], ql.strip()))
        has_topic = bool(it.categories) or it.event_dataset is not None or it.subject_dataset is not None
        words = len(re.findall(r"[a-z0-9]+", ql))
        if prev.clarification and (it.street_names or not has_topic):
            refers = True
        if not refers and (has_topic or words > 6):
            return it  # a fresh, self-contained question
        merged = Intent.from_dict(prev.to_dict())
        merged.question = it.question
        merged.clarification = it.clarification
        merged.notes = [f"Follow-up to: “{prev.question}”"]
        if prev.clarification and it.street_names:
            merged.street_names = it.street_names
            merged.clarification = None
            return merged
        if it.categories:
            merged.categories = it.categories
        if it.boroughs:
            merged.boroughs = it.boroughs
        if it.areas:
            merged.areas = it.areas
        if it.zips:
            merged.zips = it.zips
        if it.street_names:
            merged.street_names = it.street_names
        if it.distance is not None:
            merged.distance, merged.units = it.distance, it.units
        if it.date_from or it.date_to:
            merged.date_from, merged.date_to = it.date_from, it.date_to
        if it.min_points:
            merged.min_points = it.min_points
        if it.explicit_analysis:
            merged.analysis = it.analysis
            merged.freq = it.freq or merged.freq
            merged.group_by = it.group_by
            merged.geography = it.geography or merged.geography
            if it.analysis == "proximity":
                ref_ds = it.reference_dataset or it.subject_dataset
                if ref_ds and ref_ds != prev.subject_dataset:
                    merged.reference_dataset = ref_ds
                    merged.reference_categories = it.reference_categories or (
                        it.categories if ref_ds == it.subject_dataset else []
                    )
                if _find(_RX["followup_ref"], ql):
                    merged.use_previous = True
                    merged.notes.append("Using the previous result as input.")
        elif it.geography:
            merged.geography = it.geography
            merged.analysis = "by_geography" if prev.analysis in ("list", "by_geography") else prev.analysis
        if it.freq and not it.explicit_analysis:
            merged.freq = it.freq
        return merged

    # ---------------------------------------------------------------------------------------
    # Plan composition
    # ---------------------------------------------------------------------------------------
    def plan(self, question: str, previous_intent: dict[str, Any] | None = None) -> tuple[Plan, Intent]:
        it = self.parse(question, previous_intent)
        if it.clarification:
            return Plan(
                interpretation="The request needs one more detail.", steps=[], clarification=it.clarification
            ), it
        steps: list[PlanStep] = []
        n = 0

        def add(tool: str, purpose: str, **args: Any) -> str:
            nonlocal n
            n += 1
            sid = f"s{n}"
            steps.append(
                PlanStep(
                    id=sid, tool=tool, args={k: v for k, v in args.items() if v is not None}, purpose=purpose
                )
            )
            return sid

        def load(dataset_id: str, purpose: str, with_filters: bool = True, **extra: Any) -> str:
            ds = self.registry.get(dataset_id, self.mode)
            args: dict[str, Any] = {"dataset": dataset_id, **extra}
            if with_filters:
                if it.categories and dataset_id == it.subject_dataset:
                    args["categories"] = it.categories
                if it.boroughs:
                    args["boroughs"] = [BOROUGH_DISPLAY.get(b, b) for b in it.boroughs]
                if ds.temporal and (it.date_from or it.date_to) and ds.role == "events":
                    args["date_from"], args["date_to"] = it.date_from, it.date_to
            sid = add("filter_dataset", purpose, **args)
            area_names = [a["name"] for a in it.areas]
            if with_filters and area_names:
                role = it.areas[0]["role"]
                sid = add(
                    "filter_by_area",
                    f"Keep records inside {', '.join(area_names)}",
                    input=sid,
                    area_type=role,
                    names=area_names,
                )
            if with_filters and it.zips:
                sid = add(
                    "filter_by_area",
                    f"Keep records in ZIP {', '.join(it.zips)}",
                    input=sid,
                    area_type="zip",
                    names=it.zips,
                )
            return sid

        subject = it.subject_dataset
        if subject is None:
            return Plan(
                interpretation="No suitable dataset is available.",
                steps=[],
                clarification="I could not find a dataset for this question. Which data should I use? "
                "Try the Data catalog in the sidebar.",
            ), it
        sds = self.registry.get(subject, self.mode)
        topic = ", ".join(it.categories) if it.categories else sds.label
        where = (
            ", ".join(BOROUGH_DISPLAY.get(b, b) for b in it.boroughs)
            or ", ".join(a["name"] for a in it.areas)
            or "citywide"
        )
        src = "last" if it.use_previous else None
        if src:
            topic = "the previous results"

        def subject_input(purpose: str) -> str:
            return src or load(subject, purpose)

        geo = (
            it.geography if it.geography in ("borough", "neighborhood", "community_district", "zip") else None
        )
        a = it.analysis
        if a == "repeat":
            s1 = subject_input(f"Select {topic} records ({where})")
            radius = it.distance if it.distance is not None else 75.0
            s2 = add(
                "find_hotspots",
                f"Group reports within {radius:g} {it.units} of each other into repeat locations",
                input=s1,
                method="clusters",
                radius=radius,
                units=it.units,
                min_points=it.min_points or 5,
            )
            add("generate_map_layer", "Map repeat-report locations", input=s2, color_by="count")
            add(
                "generate_map_layer",
                "Show the individual reports for context",
                input=s1,
                style="points",
                context=True,
            )
            interp = f"Find locations with repeated {topic} reports ({where})"
        elif a == "hotspot":
            s1 = subject_input(f"Select {topic} records ({where})")
            cell = it.distance if it.distance is not None else 1000.0
            s2 = add(
                "find_hotspots",
                "Test for statistically significant concentrations (Getis-Ord Gi*)",
                input=s1,
                method="gi_star",
                cell_size=cell,
                units=it.units,
            )
            add("generate_map_layer", "Map hot and cold spots", input=s2, color_by="hotspot_class")
            add("generate_map_layer", "Heat map of individual reports", input=s1, style="heatmap")
            interp = f"Identify areas with high concentrations of {topic} ({where})"
        elif a == "proximity":
            ref = it.reference_dataset
            if ref is None:
                return Plan(
                    interpretation="Proximity analysis needs a reference layer.",
                    steps=[],
                    clarification="Near what? Name a layer such as catch basins, streets, or industrial "
                    "properties.",
                ), it
            rds = self.registry.get(ref, self.mode)
            s1 = subject_input(f"Select {topic} ({where})")
            if rds.role == "streets" and it.street_names:
                s2 = add(
                    "filter_dataset",
                    f"Select segments of {', '.join(it.street_names)}",
                    dataset=ref,
                    filters=[
                        {"column": rds.name_column or "street_name", "op": "in", "value": it.street_names}
                    ],
                    boroughs=[BOROUGH_DISPLAY.get(b, b) for b in it.boroughs] or None,
                )
            else:
                s2 = add(
                    "filter_dataset",
                    f"Select {rds.label} ({where})",
                    dataset=ref,
                    categories=it.reference_categories or None,
                    boroughs=[BOROUGH_DISPLAY.get(b, b) for b in it.boroughs] or None,
                )
            dist = it.distance if it.distance is not None else 500.0
            s3 = add(
                "spatial_join",
                f"Keep {sds.label if not src else 'previous results'} within {dist:g} "
                f"{it.units} of {rds.label}",
                left=s1,
                right=s2,
                predicate="dwithin",
                distance=dist,
                units=it.units,
                summarize=True,
            )
            add("generate_map_layer", f"Map {sds.label} near {rds.label}", input=s3)
            if rds.role == "streets" or len(steps) < 10:
                add("generate_map_layer", f"Show {rds.label} for reference", input=s2, context=True)
            add("summarize_results", "Summarize the matched features", input=s3)
            ref_desc = ", ".join(it.street_names) if it.street_names else rds.label
            interp = f"Find {topic} within {dist:g} {it.units} of {ref_desc} ({where})"
            if it.distance is None:
                it.notes.append("No distance given; used 500 ft.")
        elif a == "overlap":
            s1 = subject_input(f"Select sanitation complaints ({where})")
            cat = sds.category_column
            cell = it.distance if it.distance is not None else 1000.0
            s2 = add(
                "aggregate_by_geography",
                "Count distinct complaint types per grid cell",
                input=s1,
                geography=geo or "grid",
                cell_size=cell,
                units=it.units,
                measure="count_distinct",
                value_column=cat,
                category_column=cat,
                min_value=it.min_points or 3,
            )
            add(
                "generate_map_layer",
                "Map areas where several complaint types overlap",
                input=s2,
                color_by="value",
            )
            interp = f"Find areas where multiple types of {topic} overlap ({where})"
        elif a == "change":
            s1 = subject_input(f"Select {topic} ({where})")
            role = geo or "neighborhood"
            s2 = add(
                "compare_periods",
                f"Compare counts per {role} between the earliest and latest full year",
                input=s1,
                geography=role,
            )
            add("generate_map_layer", "Map the change in reports", input=s2, color_by="change")
            add("analyze_temporal_patterns", "Yearly totals for context", input=s1, freq="year")
            interp = f"Rank {role.replace('_', ' ')}s by the change in {topic} ({where})"
        elif a == "temporal":
            s1 = subject_input(f"Select {topic} ({where})")
            group = sds.category_column if it.group_by == "category" else it.group_by
            add(
                "analyze_temporal_patterns",
                f"Count reports per {it.freq}" + (f" by {group}" if group else ""),
                input=s1,
                freq=it.freq or "month",
                group_by=group,
            )
            s3 = add(
                "aggregate_by_geography",
                "Totals by borough for the map",
                input=s1,
                geography=geo or "borough",
            )
            add("generate_map_layer", "Map totals", input=s3, color_by="value")
            interp = f"Describe how {topic} changed over time ({where})" + (f", by {group}" if group else "")
        elif a == "by_street" and it.reference_dataset:
            ref = it.reference_dataset
            s1 = subject_input(f"Select {topic} ({where})")
            s2 = load(
                ref,
                "Load street segments",
                with_filters=False,
                boroughs=[BOROUGH_DISPLAY.get(b, b) for b in it.boroughs] or None,
            )
            min_v = it.min_points or 3
            s3 = add(
                "aggregate_by_geography",
                f"Assign each report to the nearest street segment and keep segments "
                f"with at least {min_v} reports",
                input=s1,
                geography=s2,
                max_distance=100,
                min_value=min_v,
            )
            add("generate_map_layer", "Map street segments by number of reports", input=s3, color_by="value")
            add("generate_map_layer", "Show all reports for context", input=s1, style="points", context=True)
            interp = f"Find street segments with repeated {topic} ({where})"
        elif a == "by_geography":
            s1 = subject_input(f"Select {topic} ({where})")
            role = geo or "neighborhood"
            s2 = add("aggregate_by_geography", f"Count reports per {role}", input=s1, geography=role)
            add("generate_map_layer", f"Map counts by {role}", input=s2, color_by="value")
            interp = f"Compare {topic} across {role.replace('_', ' ')}s ({where})"
        else:
            s1 = subject_input(f"Select {topic} ({where})")
            add("summarize_results", "Summarize the selected records", input=s1)
            add("generate_map_layer", "Map the records", input=s1)
            if sds.temporal:
                add("analyze_temporal_patterns", "Monthly counts", input=s1, freq=it.freq or "month")
            interp = f"Show {topic} ({where})"
        if it.date_from or it.date_to:
            interp += f", {it.date_from or '…'} to {it.date_to or '…'}"
        datasets = [d for d in (subject, it.reference_dataset) if d]
        return Plan(
            interpretation=interp, geography=where, datasets=datasets, steps=steps, assumptions=it.notes
        ), it


# =========================================================================================
# LLM planner prompt
# =========================================================================================
def dataset_digest(ds: DatasetSpec, level: str, live_columns: list[Any] | None = None) -> str:
    if level == "name_only":
        return f"- {ds.id}: {ds.label} [{ds.role}] (details withheld by data policy)"
    geom = ds.spatial.geometry_type if ds.spatial else "none"
    parts = [
        f"- {ds.id}: {clean_text(ds.name, 120)[0]} [role={ds.role}, geometry={geom}"
        + (f", date={ds.temporal.column}" if ds.temporal else "")
        + (f", category={ds.category_column}" if ds.category_column else "")
        + (", SYNTHETIC" if ds.synthetic else "")
        + "]"
    ]
    if level == "full":
        cats = ds.known_categories()
        if cats:
            parts.append(f"    categories: {', '.join(cats[:20])}")
        cols = [f"{c.name}({c.type})" for c in ds.columns if not c.sensitive and c.type != "geometry"][:25]
        if not cols and live_columns:
            sensitive = ds.sensitive_columns
            cols = [
                f"{clean_text(c.name, 64)[0]}({clean_text(c.type, 32)[0]})"
                for c in live_columns
                if c.name.lower() not in sensitive
            ][:40]
        if cols:
            parts.append(f"    columns: {', '.join(cols)}")
        if ds.name_column:
            parts.append(f"    name column: {ds.name_column}")
    return "\n".join(parts)


def build_planner_prompt(
    question: str,
    registry: DatasetRegistry,
    mode: str,
    tools: ToolRegistry,
    ctx: Any,
    policy: DataEgressPolicy,
    context: dict[str, Any],
    store: ResultStore,
    turn: int,
) -> str:
    from agentic_gis.agent.prompts import PLANNER_USER

    digests = []
    for d in registry.available(mode):
        level = policy.dataset_level(d)
        live = None
        if level == "full" and d.schema_discovery and not d.columns and ctx is not None:
            try:  # Unity Catalog metadata lookup (cached per session); never reads table rows
                live = ctx.columns_for(d)
            except Exception as exc:  # permission or connectivity problems: plan without columns
                log.info("Schema discovery for %s failed: %s", d.id, exc)
        digests.append(dataset_digest(d, level, live))
    datasets = "\n".join(digests)
    ctx_lines = []
    if context.get("last_question"):
        prev = {
            "previous_question": context.get("last_question"),
            "previous_interpretation": context.get("last_interpretation"),
            "previous_results": [a.digest() for a in store.all() if a.turn == turn - 1][:8],
            "main_previous_result": context.get("last_primary"),
        }
        ctx_lines.append('Conversation context (refer to the previous main result as "last"):')
        ctx_lines.append(wrap_untrusted("conversation", prev))
    return PLANNER_USER.format(
        question=clean_text(question, 1000)[0],
        mode=mode,
        geographies=", ".join(registry.geography_roles(mode)),
        context="\n".join(ctx_lines) + ("\n" if ctx_lines else ""),
        datasets=datasets,
        tools=tools.catalog_for_prompt(ctx),
    )


# =========================================================================================
# Plan validation / normalization (shared by both planners)
# =========================================================================================
@dataclass
class PlanCheck:
    plan: Plan
    errors: list[str]
    warnings: list[str]


def validate_plan(
    plan: Plan,
    tools: ToolRegistry,
    registry: DatasetRegistry,
    mode: str,
    store: ResultStore,
    turn: int,
    max_steps: int,
    ctx: Any = None,
    geography_roles: Callable[[], list[str]] | None = None,
) -> PlanCheck:
    """Validate tools/arguments/references; insert explicit load steps for dataset references."""
    errors: list[str] = []
    warnings: list[str] = []
    roles = set(geography_roles() if geography_roles else registry.geography_roles(mode)) | {"grid"}
    available_tools = set(tools.names(ctx))
    ready = {d.id for d in registry.available(mode)}
    out_steps: list[PlanStep] = []
    seen_ids: set[str] = set()
    load_n = 0
    dataset_loads: dict[str, str] = {}
    if len(plan.steps) > max_steps:
        errors.append(f"Plan has {len(plan.steps)} steps; the maximum is {max_steps}.")
    for step in plan.steps[:max_steps]:
        if step.tool not in available_tools:
            errors.append(f"{step.id}: unknown or unavailable tool '{step.tool}'.")
            continue
        if step.id in seen_ids:
            errors.append(f"{step.id}: duplicate step id.")
            continue
        spec = tools.get(step.tool)
        args = dict(step.args)
        for fname in spec.ref_fields:
            ref = args.get(fname)
            if ref is None:
                continue
            if not isinstance(ref, str):
                errors.append(f"{step.id}.{fname}: reference must be a string.")
                continue
            if fname == "geography" and ref in roles:
                continue
            if ref in seen_ids or (ref.startswith("t") and store.get(ref) is not None):
                continue
            if ref in ("last", "previous", "prev"):
                if store.last_primary(turn) is None:
                    errors.append(f"{step.id}.{fname}: there is no previous result to use.")
                continue
            if ref in ready:
                if ref not in dataset_loads:
                    load_n += 1
                    lid = f"d{load_n}"
                    out_steps.append(
                        PlanStep(
                            id=lid,
                            tool="filter_dataset",
                            args={"dataset": ref},
                            purpose=f"Load {registry.get_any(ref).label}",
                        )
                    )
                    seen_ids.add(lid)
                    dataset_loads[ref] = lid
                args[fname] = dataset_loads[ref]
                continue
            errors.append(
                f"{step.id}.{fname}: '{ref}' is not an earlier step, a previous result, or an "
                "available dataset."
            )
        if "dataset" in args and step.tool in ("filter_dataset", "describe_dataset", "get_schema"):
            if args["dataset"] not in ready:
                errors.append(f"{step.id}: dataset '{args['dataset']}' is not available in {mode} mode.")
                continue
        try:
            model = tools.validate_args(step.tool, args)
            args = model.model_dump(exclude_none=True, exclude_defaults=False)
        except Exception as exc:
            msg = str(exc).splitlines()
            errors.append(f"{step.id} ({step.tool}): invalid arguments: {' '.join(msg[:3])[:300]}")
            continue
        out_steps.append(PlanStep(id=step.id, tool=step.tool, args=args, purpose=step.purpose))
        seen_ids.add(step.id)
    if not plan.clarification and not out_steps:
        errors.append("The plan has no executable steps.")
    normalized = Plan(
        interpretation=plan.interpretation,
        geography=plan.geography,
        datasets=plan.datasets,
        steps=out_steps,
        clarification=plan.clarification,
        assumptions=plan.assumptions,
    )
    return PlanCheck(normalized, errors, warnings)


def plan_to_text(plan: Plan) -> str:
    return json.dumps(plan.model_dump(), indent=1)
