"""Defenses against prompt injection via retrieved records, schemas, and tool outputs.

Data that originates outside the application (record text, column comments, error messages)
is treated as untrusted. Before it is placed in an LLM prompt it is:

* truncated and stripped of control characters;
* scanned for instruction-like phrases, which are replaced by a marker;
* wrapped in ``<untrusted_data>`` tags that the system prompt declares to be data only.

Even if an injection survives, the LLM can only emit a JSON plan that is validated against
registered tool schemas, the dataset allowlist, and the SQL guard, and SQL needs approval.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime
from typing import Any

import pandas as pd

FILTER_MARKER = "[removed: possible embedded instruction]"

_INJECTION_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"\bignore\b.{0,40}\b(instruction|prompt|rule|direction)s?\b",
        r"\bdisregard\b.{0,40}\b(instruction|prompt|rule|above|previous|system)",
        r"\bforget\b.{0,30}\b(instruction|prompt|rule)s?\b",
        r"\byou are (now|no longer)\b",
        r"\b(system|developer) (prompt|message|instruction)s?\b",
        r"</?\s*(system|assistant|user|untrusted_data|instructions?|tool)\s*>",
        r"\bnew instructions?\b",
        r"\boverride\b.{0,30}\b(safety|rules?|policy|instructions?)\b",
        r"\b(run|execute)\b.{0,20}\b(sql|query|command|code|python|shell)\b",
        r"\b(drop|truncate|delete\s+from|insert\s+into|alter)\s+table\b",
        r"\bact as\b",
        r"\bjailbreak\b",
        r"\bprompt injection\b",
    )
]
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def clean_text(value: Any, max_len: int = 160) -> tuple[str, bool]:
    """Return (sanitized string, flagged)."""
    text = "" if value is None else str(value)
    text = _CONTROL.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    flagged = looks_like_injection(text)
    if flagged:
        text = FILTER_MARKER
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text, flagged


def _jsonable(v: Any, max_len: int) -> tuple[Any, bool]:
    if v is None:
        return None, False
    if isinstance(v, bool | int):
        return v, False
    if isinstance(v, float):
        return (None if math.isnan(v) or math.isinf(v) else round(v, 4)), False
    if isinstance(v, datetime | date | pd.Timestamp):
        return str(v)[:19], False
    try:
        if pd.isna(v):
            return None, False
    except (TypeError, ValueError):
        pass
    return clean_text(v, max_len)


def sanitize_records(
    df: pd.DataFrame, max_rows: int = 5, max_len: int = 120, exclude: set[str] | None = None
) -> tuple[list[dict[str, Any]], int]:
    """Convert a few rows into prompt-safe dicts. Returns (records, flagged_count)."""
    exclude = {c.lower() for c in (exclude or set())} | {"geometry"}
    cols = [c for c in df.columns if str(c).lower() not in exclude]
    out: list[dict[str, Any]] = []
    flagged = 0
    for _, row in df.head(max_rows).iterrows():
        rec = {}
        for c in cols:
            val, f = _jsonable(row[c], max_len)
            flagged += int(f)
            rec[str(c)] = val
        out.append(rec)
    return out, flagged


def wrap_untrusted(label: str, payload: Any) -> str:
    """Serialize payload as JSON inside untrusted-data tags (tag-breaking sequences escaped)."""
    text = json.dumps(payload, default=str, ensure_ascii=False, indent=1)
    text = re.sub(r"<\s*/?\s*untrusted_data", "<\\\\untrusted_data", text, flags=re.I)
    safe_label = re.sub(r"[^A-Za-z0-9_.\-]", "_", label)[:60]
    return f'<untrusted_data source="{safe_label}">\n{text}\n</untrusted_data>'
