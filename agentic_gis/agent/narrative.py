"""Answer construction: deterministic templates, LLM narration, and claim guardrails."""

from __future__ import annotations

import re
from typing import Any

from agentic_gis.agent.state import Answer, Finding
from agentic_gis.config.registry import DatasetRegistry, DatasetSpec

SYNTHETIC_NOTICE = (
    "Results are computed from SYNTHETIC demonstration data generated for this app; they are not "
    "findings about New York City."
)
REPORTS_NOTICE = (
    "Complaint (311) records are reports from the public. They indicate where conditions were "
    "reported, not where violations were verified."
)

_QUALIFIERS = r"(complaint|report|reported|candidate|possible|potential|suspected|alleged|review)"
_SITE_CLAIMS = [
    (
        re.compile(r"\b(an?|the)\s+illegal[- ]dumping\s+(site|location|spot|ground|area)\b", re.I),
        "a location with repeated illegal dumping reports",
    ),
    (
        re.compile(r"\billegal[- ]dumping\s+(sites|locations|spots|grounds)\b", re.I),
        "locations with repeated illegal dumping reports",
    ),
    (
        re.compile(
            r"\b(confirmed|verified|proven)\s+(illegal[- ]dumping|dumping|violations?|derelict vehicles?)\b",
            re.I,
        ),
        r"reported \2",
    ),
    (re.compile(r"\bdumping\s+hot\s?spots?\b", re.I), "dumping-report hot spots"),
]


def _sanitize_claims(text: str) -> tuple[str, bool]:
    changed = False
    for pat, repl in _SITE_CLAIMS:

        def sub(m: re.Match[str], repl: str = repl, src: str = text) -> str:
            window = src[max(0, m.start() - 40) : m.end() + 40].lower()
            if (
                re.search(_QUALIFIERS, window)
                and "confirmed" not in m.group(0).lower()
                and "verified" not in m.group(0).lower()
            ):
                return m.group(0)
            return m.expand(repl)

        new = pat.sub(sub, text)
        if new != text:
            changed = True
            text = new
    return text, changed


def enforce_guardrails(answer: Answer, datasets: list[DatasetSpec]) -> tuple[Answer, list[str]]:
    """Apply claim policy and mandatory disclosures. Returns (answer, list of adjustments made)."""
    adjustments: list[str] = []
    verification = any(d.verification_source for d in datasets)
    synthetic = any(d.synthetic for d in datasets)
    events = any(d.role == "events" for d in datasets)

    findings = []
    for f in answer.findings:
        kind = f.kind
        if kind == "verified" and not verification:
            kind = "inference"
            adjustments.append(
                "A finding labeled 'verified' was relabeled 'inference' (no verification dataset)."
            )
        text, changed = _sanitize_claims(f.text)
        if changed:
            adjustments.append("Rephrased a finding that implied a confirmed violation.")
        findings.append(Finding(kind=kind, text=text))
    summary, c1 = _sanitize_claims(answer.summary)
    headline, c2 = _sanitize_claims(answer.headline)
    if c1 or c2:
        adjustments.append("Rephrased wording that implied a confirmed violation.")
    limitations = list(dict.fromkeys(answer.limitations))
    if synthetic:
        if "synthetic" not in summary.lower():
            summary = "**Synthetic demo data.** " + summary
        if "synthetic" not in headline.lower():
            headline = headline.rstrip(".") + " (synthetic demo data)"
        if SYNTHETIC_NOTICE not in limitations:
            limitations.insert(0, SYNTHETIC_NOTICE)
    if events and not any("report" in x.lower() and "verif" in x.lower() for x in limitations):
        limitations.append(REPORTS_NOTICE)
    return Answer(
        headline=headline,
        summary=summary,
        findings=findings,
        limitations=limitations,
        suggested_follow_ups=answer.suggested_follow_ups[:4],
    ), adjustments


PREFERRED_KEY_TOOL = {"temporal": "analyze_temporal_patterns", "change": "compare_periods"}


def template_answer(
    question: str,
    interpretation: str,
    steps: list[dict[str, Any]],
    analysis: str | None,
    datasets: list[DatasetSpec],
    primary_artifact: str | None = None,
) -> Answer:
    """Deterministic narrative from tool outputs (used in demo mode and as LLM fallback)."""
    ok = [s for s in steps if s.get("status") in ("ok", "empty")]
    failed = [s for s in steps if s.get("status") in ("error", "needs_approval", "skipped", "unavailable")]
    analytic = [
        s
        for s in ok
        if s["tool"] not in ("generate_map_layer", "describe_dataset", "list_datasets", "get_schema")
    ]
    key = None
    pref = PREFERRED_KEY_TOOL.get(analysis or "")
    if pref:
        key = next((s for s in analytic if s["tool"] == pref), None)
    if key is None and primary_artifact:
        key = next(
            (
                s
                for s in analytic
                if s.get("artifact_id") == primary_artifact and s["tool"] != "generate_map_layer"
            ),
            None,
        )
    if key is None:
        key = next(
            (s for s in reversed(analytic) if s["tool"] not in ("summarize_results", "filter_dataset")),
            analytic[-1] if analytic else None,
        )
    if key is None:
        headline = "The analysis could not be completed."
        summary = "No step produced results. " + " ".join(s.get("summary", "") for s in failed[:2])
    else:
        headline = key.get("headline") or key.get("summary", interpretation)
        lead = (
            f"I interpreted the question as: {interpretation[0].lower() + interpretation[1:]}."
            if interpretation
            else ""
        )
        bits = [lead]
        first = next((s for s in ok if s["tool"] == "filter_dataset"), None)
        if first:
            bits.append(first.get("summary", ""))
        bits.append(key.get("summary", ""))
        if key.get("highlights"):
            bits.append(key["highlights"][0])
        summary = " ".join(b for b in bits if b)
    findings: list[Finding] = []
    ordered = [key] + [s for s in analytic if s is not key] if key else analytic
    for s in ordered:
        for h in s.get("highlights", [])[:3]:
            findings.append(Finding(kind="observation", text=h))
    if analysis in ("repeat", "hotspot") and findings:
        findings.append(
            Finding(
                kind="inference",
                text=(
                    "Places with many reports are candidates for field review; repeated reports can also reflect a few "
                    "active reporters, high foot traffic, or duplicate submissions."
                ),
            )
        )
    if analysis == "change" and findings:
        findings.append(
            Finding(
                kind="inference",
                text=(
                    "Increases in reports may reflect changes in conditions, reporting behavior, outreach, or service "
                    "levels; check operational data before drawing conclusions."
                ),
            )
        )
    if analysis == "proximity" and findings:
        findings.append(
            Finding(
                kind="inference",
                text=(
                    "Proximity shows spatial co-location only; it does not establish that one feature causes the other."
                ),
            )
        )
    limitations: list[str] = []
    for s in steps:
        for w in s.get("warnings", []):
            limitations.append(w)
    for s in failed:
        limitations.append(f"Step {s['step_id']} ({s['tool']}) did not complete: {s.get('summary', '')}")
    follow = _follow_ups(analysis)
    return Answer(
        headline=headline,
        summary=summary,
        findings=findings[:8],
        limitations=limitations[:6],
        suggested_follow_ups=follow,
    )


def _follow_ups(analysis: str | None) -> list[str]:
    return {
        "repeat": [
            "How have these repeat locations changed over time?",
            "Show only Brooklyn",
            "Which of these are near industrial properties?",
        ],
        "hotspot": [
            "Compare this across boroughs over time",
            "Show only the Bronx",
            "Which neighborhoods have the most of these complaints?",
        ],
        "proximity": ["Increase the distance to 500 feet", "Show only Queens", "Summarize these by borough"],
        "change": [
            "Show the monthly trend for the top neighborhood",
            "Repeat this by ZIP code",
            "Where are the repeat locations?",
        ],
        "temporal": [
            "Which neighborhoods had the largest increase?",
            "Show this by month",
            "Where are the hot spots?",
        ],
        "overlap": [
            "Which complaint types overlap most often?",
            "Show only Brooklyn",
            "Where are the hot spots of illegal dumping?",
        ],
    }.get(
        analysis or "",
        ["Show this by borough", "How has this changed over time?", "Where are the hot spots?"],
    )


def step_digest_for_llm(step: dict[str, Any]) -> dict[str, Any]:
    return {
        k: step.get(k)
        for k in (
            "step_id",
            "tool",
            "purpose",
            "status",
            "summary",
            "highlights",
            "rows",
            "notes",
            "warnings",
        )
        if step.get(k) not in (None, [], "")
    }


def datasets_for_llm(datasets: list[DatasetSpec]) -> list[dict[str, Any]]:
    return [
        {
            "id": d.id,
            "name": d.name,
            "synthetic": d.synthetic,
            "publisher": d.provenance.publisher,
            "verification_source": d.verification_source,
        }
        for d in datasets
    ]


def citations_for(
    datasets: list[DatasetSpec], registry: DatasetRegistry | None = None
) -> list[dict[str, Any]]:
    out = []
    for d in datasets:
        out.append(
            {
                "dataset_id": d.id,
                "name": d.name,
                "publisher": d.provenance.publisher,
                "source_url": d.provenance.source_url,
                "license": d.provenance.license,
                "citation": d.provenance.citation or d.name,
                "synthetic": d.synthetic,
                "sensitivity": d.sensitivity,
                "notes": d.provenance.notes,
            }
        )
    return out
