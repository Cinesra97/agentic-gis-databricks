"""Check an LLM-written answer against the computed results before showing it.

The narrator may only restate facts from the step results, the data scope and the question.
:func:`check_narrative` looks for numbers, years, place names, "nothing was found" claims, and
claims about the data itself (real vs synthetic, date coverage, truncated downloads) that the
evidence, the dataset metadata or the question do not support. If it finds any, the graph discards
the LLM answer and uses the deterministic template answer (and records why in the trace and audit
log). Limitations and provenance notes are always produced deterministically; the model only
rewrites the explanation.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from agentic_gis.agent.state import Answer, Narration
from agentic_gis.spatial.operations import normalize_name

_BOROUGHS = {
    "manhattan": "MANHATTAN",
    "bronx": "BRONX",
    "brooklyn": "BROOKLYN",
    "queens": "QUEENS",
    "staten island": "STATEN ISLAND",
}
_NUM = re.compile(r"(?<![\w.,/-])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(\s?%)?(?![\w])")
_YEAR = re.compile(r"(?<![\w.,-])((?:19|20)\d\d)(?![\w,])")
_NO_RECORDS = re.compile(
    r"\b(?:no|zero|0)\s+(?!(?:\w+\s+){0,3}(?:clusters?|hot ?spots?|locations?|areas?|cells?|places?|sites?)\b)"
    r"(?:[\w-]+\s+){0,3}(?:complaints?|reports?|records?|requests?|incidents?|cases?)\b"
    r"|\b(?:did not|didn't|could not|couldn't|were unable to) find any\b"
    r"|\bthere (?:were|are|was|is) no (?:[\w-]+\s+){0,3}(?:complaints?|reports?|records?|requests?|incidents?)\b",
    re.I,
)
# Claims about the data itself. Synthetic status comes from dataset metadata, period coverage from the
# request scope + stored coverage, truncation from the ingestion record -- never from the model.
_SYNTHETIC_CLAIM = re.compile(
    r"\b(?:synthetic|simulated|fictional|fictitious|made[- ]up|dummy data|fake data|demonstration data|"
    r"demo data|sample data|test data)\b|\bnot (?:real|actual)(?: nyc| new york city)? (?:data|findings|records)\b",
    re.I,
)
_REAL_CLAIM = re.compile(
    r"(?<!not )(?<!n't )\b(?:real|actual|official|authoritative)\s+(?:nyc\s+|new york city\s+)?"
    r"(?:311\s+)?(?:data|records|complaints|findings|incidents)\b",
    re.I,
)
_PERIOD_CLAIM = re.compile(
    r"\b(?:does not|doesn't|did not|didn't|do not|don't|may not|might not|cannot|can't|fails? to|not)\s+"
    r"(?:fully\s+|completely\s+|entirely\s+)?(?:cover|span|include|reflect)\w*\s+(?:the\s+|all\s+(?:of\s+)?)?"
    r"(?:full|entire|whole|complete|requested|specified|selected|asked)?\s*(?:time\s+)?"
    r"(?:period|date range|time range|timeframe|time frame|dates|years?|months?)\b"
    r"|\bpartial(?:ly)?\s+(?:period|coverage|covered|year|time range|date range)\b"
    r"|\b(?:only|just)\s+(?:partially|partly)\s+cover\w*"
    r"|\b(?:requested|specified|selected) (?:time\s+)?(?:period|date range|time range|timeframe)\b",
    re.I,
)
_SUBSET_CLAIM = re.compile(
    r"\b(?:row limit|record limit|download limit|truncated|incomplete (?:download|copy|dataset|data ?set|data|records)|"
    r"(?:a |only a )?subset of (?:the )?(?:data|records|dataset|complaints)|partial (?:download|copy|dataset))\b",
    re.I,
)
_NO_CLUSTERS = re.compile(
    r"\bno\s+(?:[\w-]+\s+){0,3}(?:clusters?|hot ?spots?|repeat(?:ed)?(?:[- ]report)? locations?)\b"
    r"|\b(?:zero|0)\s+(?:clusters?|hot ?spots?|locations?)\b",
    re.I,
)


@dataclass
class Evidence:
    numbers: set[float] = field(default_factory=set)
    years: set[int] = field(default_factory=set)
    text: str = ""  # normalized evidence text for place-name checks
    records: int | None = None  # rows of the main record selection
    clusters: int | None = None  # clusters / hot spots found (None when not computed)
    synthetic: bool | None = None  # from dataset metadata (None = unknown, not checked)
    period_requested: bool = False  # the question asked for a time period
    period_partial: bool = False  # ... and the stored data covers it only partly (or not at all)
    subset: bool = False  # the stored copy is a truncated or filtered subset of the source


def _flatten(obj: Any) -> Iterable[str]:
    if obj is None:
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten(v)
    elif isinstance(obj, list | tuple | set):
        for v in obj:
            yield from _flatten(v)
    else:
        yield str(obj)


def _numbers_in(text: str) -> set[float]:
    out = set()
    for m in _NUM.finditer(text):
        try:
            out.add(float(m.group(1).replace(",", "") + (m.group(2) or "")))
        except ValueError:
            continue
    return out


def build_evidence(
    question: str,
    interpretation: str,
    steps: list[dict[str, Any]],
    data_scope: dict[str, Any] | None,
    template: Answer | None = None,
    extra_numbers: Iterable[float] = (),
    synthetic: bool | None = None,
) -> Evidence:
    """Facts the narrator may use. Years come only from results, the scope and the question."""
    ev = Evidence(synthetic=synthetic)
    for d in (data_scope or {}).get("datasets", []):
        cov = d.get("coverage") or {}
        if d.get("requested_from") or d.get("requested_to"):
            ev.period_requested = True
            if d.get("overlap") in ("partial", "none"):
                ev.period_partial = True
        if cov.get("limit_reached") or cov.get("is_subset"):
            ev.subset = True
    chunks: list[str] = [question, interpretation]
    for s in steps:
        for k in ("summary", "headline", "highlights", "notes", "warnings", "facts", "rows", "purpose"):
            chunks.extend(_flatten(s.get(k)))
        if isinstance(s.get("data"), list):  # temporal summaries etc. (already small)
            chunks.extend(_flatten(s["data"][:50]))
    if data_scope:
        for d in data_scope.get("datasets", []):
            cov = d.get("coverage") or {}
            chunks.extend(
                _flatten(
                    [
                        d.get("requested_text"),
                        d.get("matching_rows"),
                        d.get("categories"),
                        d.get("boroughs"),
                        d.get("areas"),
                        cov.get("rows"),
                        cov.get("date_min"),
                        cov.get("date_max"),
                        cov.get("ingest_limit"),
                    ]
                )
            )
        chunks.extend(_flatten(data_scope.get("warnings")))
    if template is not None:
        chunks += [
            template.headline,
            template.summary,
            *(f.text for f in template.findings),
            *template.limitations,
        ]
    blob = "\n".join(c for c in chunks if c)
    ev.numbers = _numbers_in(blob) | set(extra_numbers)
    ev.years = {int(y) for y in _YEAR.findall(blob)}
    ev.text = " " + normalize_name(blob) + " "
    main = next(
        (s for s in steps if s.get("tool") == "filter_dataset" and s.get("status") in ("ok", "empty")), None
    )
    if main is None:
        main = next(
            (s for s in steps if s.get("tool", "").startswith("query_") and s.get("rows") is not None), None
        )
    if main is not None:
        ev.records = int(main.get("rows") or 0)
    hs = next(
        (s for s in steps if s.get("tool") == "find_hotspots" and s.get("status") in ("ok", "empty")), None
    )
    if hs is not None:
        facts = hs.get("facts") or {}
        ev.clusters = int(facts.get("clusters", hs.get("rows") or 0))
    return ev


def _supported(value: float, evidence: set[float], pct: bool) -> bool:
    if value <= 12 and float(value).is_integer() and not pct:
        return True  # small counts ("top 3", "5 reports", months) are not checked
    for e in evidence:
        if abs(value - e) <= 0.51:
            return True
        if pct and abs(value - e) <= 1.0:
            return True
        if e and value >= 100 and value % 10 == 0 and abs(value - e) / abs(e) <= 0.05:
            return True  # rounded ("about 3,500" for 3,457)
    return False


def check_narrative(
    answer: Answer | Narration,
    ev: Evidence,
    place_names: Iterable[str] = (),
) -> list[str]:
    """Problems with an LLM answer (empty list = consistent with the evidence).

    Every model-written sentence is checked, including limitations when the model supplied any.
    """
    problems: list[str] = []
    texts = [
        answer.headline,
        answer.summary,
        *(f.text for f in answer.findings),
        *getattr(answer, "limitations", []),
    ]
    body = "\n".join(texts)
    if ev.synthetic is False and _SYNTHETIC_CLAIM.search(body):
        problems.append("describes the data as synthetic or demonstration data, but the datasets are real")
    if ev.synthetic is True and _REAL_CLAIM.search(body):
        problems.append("describes synthetic demonstration data as real data")
    if _PERIOD_CLAIM.search(body) and not ev.period_partial:
        problems.append(
            "claims a requested period is not (fully) covered, but "
            + (
                "the stored data covers the requested period"
                if ev.period_requested
                else "the question did not ask for a period"
            )
        )
    if _SUBSET_CLAIM.search(body) and not ev.subset:
        problems.append("says the data is a truncated subset, but the ingestion record does not show that")
    for m in _NUM.finditer(body):
        raw = m.group(1)
        value = float(raw.replace(",", "") + (m.group(2) or ""))
        is_year = "," not in raw and not m.group(2) and not m.group(3) and 1900 <= value <= 2100
        if is_year:
            if int(value) not in ev.years:
                problems.append(
                    f"mentions the year {int(value)}, which is not in the results or the question"
                )
        elif not _supported(value, ev.numbers, bool(m.group(3))):
            problems.append(f"states the number {m.group(0).strip()}, which is not in the results")
    if ev.records and _NO_RECORDS.search(body):
        problems.append(f"says no records were found, but {ev.records:,} matching records were analyzed")
    if ev.clusters and _NO_CLUSTERS.search(body):
        problems.append(f"says no clusters were found, but {ev.clusters:,} were found")
    norm_body = " " + normalize_name(body) + " "
    for name, canon in _BOROUGHS.items():
        if re.search(rf"\b{name}\b", body, re.I) and f" {canon} " not in ev.text:
            problems.append(f"mentions {name.title()}, which is not in the results")
    for place in place_names:
        n = normalize_name(place)
        if len(n) >= 5 and f" {n} " in norm_body and f" {n} " not in ev.text:
            problems.append(f"mentions {place.title()}, which is not in the results")
    return list(dict.fromkeys(problems))[:6]
