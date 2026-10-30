"""Deterministic parsing of the time period a question asks about.

Shared by the rule planner (to build plans) and the semantic validator (to decide whether a
date filter in an LLM plan is supported by the question). Only words in the question count:
dataset titles, descriptions and publisher names are never a source of date filters.

    >>> parse_time_window("illegal dumping in 2026").date_from
    '2026-01-01'
    >>> parse_time_window("repeated illegal dumping complaints").kind
    'none'
"""

from __future__ import annotations

import calendar
import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any

MONTHS: dict[str, int] = {}
for _i in range(1, 13):
    MONTHS[calendar.month_name[_i].lower()] = _i
    MONTHS[calendar.month_abbr[_i].lower()] = _i
MONTHS["sept"] = 9
_MONTH_RX = "|".join(sorted(MONTHS, key=len, reverse=True))
_YEAR_RX = r"(?:19[5-9]\d|20[0-4]\d)"
# A 4-digit number followed by one of these is a quantity or an address, not a year.
_NOT_YEAR_AFTER = re.compile(
    r"\s*(?:feet|foot|ft\b|'|meters?|metres?|m\b|miles?|mi\b|km\b|kilometers?|yards?|yd\b|blocks?|%|"
    r"st\b|street|ave\b|avenue|blvd|boulevard|broadway|road|rd\b|place\b|pl\b|lane\b|parkway|pkwy)",
    re.I,
)
_NOT_YEAR_BEFORE = re.compile(
    r"(?:more than|over|at least|fewer than|less than|under|top|than|minimum of|maximum of|limit|[<>]=?)\s*$",
    re.I,
)
_NUM_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "eighteen": 18,
    "twenty-four": 24,
    "a": 1,
    "a couple of": 2,
    "a few": 3,
    "few": 3,
}
ALL_DATES_RX = re.compile(
    r"\b(?:all (?:dates|time|years|records|data|periods?|available (?:data|dates|records|years))|"
    r"(?:entire|whole|full) (?:period|dataset|data ?set|history|time ?range|date range|record)|"
    r"any (?:date|time|year)|no date (?:filter|limit|restriction)s?|regardless of (?:date|time|year)|"
    r"across all (?:dates|years|time)|without (?:a )?date (?:filter|limit|restriction)s?|"
    r"remove the date (?:filter|limit|restriction))\b",
    re.I,
)
VAGUE_RX = re.compile(r"\b(?:recent(?:ly)?|lately|these days|nowadays|currently|latest)\b", re.I)


@dataclass
class TimeWindow:
    """The time period requested in a piece of text.

    ``kind`` is one of: ``none`` (no time words), ``explicit`` (dates, months, quarters or
    years), ``relative`` (e.g. "last 6 months", resolved against ``today``), ``month`` (a
    month without a year; the most recent such month is assumed), ``all`` (the user asked
    for every date, which clears inherited filters) and ``vague`` ("recently"; no filter).
    """

    date_from: str | None = None
    date_to: str | None = None
    kind: str = "none"
    phrase: str | None = None
    years: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def has_dates(self) -> bool:
        return bool(self.date_from or self.date_to)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Period:
    start: date
    end: date
    pos: int
    stop: int


def _month_end(y: int, m: int) -> date:
    return date(y, m, calendar.monthrange(y, m)[1])


def _add_months(d: date, months: int) -> date:
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def _is_year_at(text: str, start: int, end: int) -> bool:
    if start > 0 and text[start - 1] in "#$":
        return False
    if _NOT_YEAR_BEFORE.search(text[:start]):
        return False
    return not _NOT_YEAR_AFTER.match(text, end)


def _periods(text: str) -> list[_Period]:
    """Explicit calendar periods in reading order: ISO dates, month ranges, months, quarters, years."""
    out: list[_Period] = []
    taken: list[tuple[int, int]] = []

    def free(a: int, b: int) -> bool:
        return not any(a < e and s < b for s, e in taken)

    def add(p: _Period) -> None:
        out.append(p)
        taken.append((p.pos, p.stop))

    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
        add(_Period(d, d, m.start(), m.end()))
    for m in re.finditer(rf"\b({_MONTH_RX})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+({_YEAR_RX})\b", text, re.I):
        try:
            d = date(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2)))
        except ValueError:
            continue
        if free(m.start(), m.end()):
            add(_Period(d, d, m.start(), m.end()))
    # "May to July 2026", "between May and July 2026", "Jan-Mar 2026"
    rx_mrange = re.compile(
        rf"\b({_MONTH_RX})\.?\s*(?:-|–|to|through|thru|until|and)\s*({_MONTH_RX})\.?,?\s+({_YEAR_RX})\b", re.I
    )
    for m in rx_mrange.finditer(text):
        if not free(m.start(), m.end()):
            continue
        y = int(m.group(3))
        m1, m2 = MONTHS[m.group(1).lower()], MONTHS[m.group(2).lower()]
        y1 = y - 1 if m1 > m2 else y
        add(_Period(date(y1, m1, 1), _month_end(y, m2), m.start(), m.end()))
    for m in re.finditer(rf"\b({_MONTH_RX})\.?,?\s+(?:of\s+)?({_YEAR_RX})\b", text, re.I):
        if not free(m.start(), m.end()):
            continue
        y, mo = int(m.group(2)), MONTHS[m.group(1).lower()]
        add(_Period(date(y, mo, 1), _month_end(y, mo), m.start(), m.end()))
    rx_q = re.compile(
        rf"\b(?:q([1-4])|(first|second|third|fourth|1st|2nd|3rd|4th) quarter)\s*(?:of\s+|,\s*)?({_YEAR_RX})\b",
        re.I,
    )
    words = {"first": 1, "second": 2, "third": 3, "fourth": 4, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
    for m in rx_q.finditer(text):
        if not free(m.start(), m.end()):
            continue
        q = int(m.group(1)) if m.group(1) else words[m.group(2).lower()]
        y = int(m.group(3))
        add(_Period(date(y, 3 * q - 2, 1), _month_end(y, 3 * q), m.start(), m.end()))
    for m in re.finditer(rf"(?<!\d)({_YEAR_RX})(?!\d)", text):
        if not free(m.start(), m.end()) or not _is_year_at(text, m.start(), m.end()):
            continue
        y = int(m.group(1))
        add(_Period(date(y, 1, 1), date(y, 12, 31), m.start(), m.end()))
    return sorted(out, key=lambda p: p.pos)


def _before(text: str, pos: int, rx: str) -> bool:
    return (
        re.search(rf"\b(?:{rx})\s+(?:the\s+)?(?:(?:start|end|beginning) of\s+)?$", text[:pos], re.I)
        is not None
    )


def _resolve_relative(text: str, today: date) -> tuple[date, date, str] | None:
    tl = text.lower()
    num = r"(\d{1,3}|" + "|".join(sorted(map(re.escape, _NUM_WORDS), key=len, reverse=True)) + ")"
    m = re.search(rf"\b(?:last|past|previous|prior|recent)\s+{num}\s+(day|week|month|year)s?\b", tl)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else _NUM_WORDS[m.group(1)]
        unit = m.group(2)
        if unit == "day":
            start = today - timedelta(days=n)
        elif unit == "week":
            start = today - timedelta(weeks=n)
        elif unit == "month":
            start = _add_months(today, -n)
        else:
            start = _add_months(today, -12 * n)
        return start, today, m.group(0)
    m = re.search(
        r"\b(?:in the |over the |during the )?(past|last|previous|this|current)\s+(day|week|month|year)\b", tl
    )
    if m:
        which, unit = m.group(1), m.group(2)
        if which in ("this", "current"):
            if unit == "year":
                return date(today.year, 1, 1), today, m.group(0)
            if unit == "month":
                return date(today.year, today.month, 1), today, m.group(0)
            if unit == "week":
                return today - timedelta(days=today.weekday()), today, m.group(0)
            return today, today, m.group(0)
        if which == "past":  # "past year" = the last 12 months
            delta = {"day": timedelta(days=1), "week": timedelta(weeks=1)}.get(unit)
            start = today - delta if delta else _add_months(today, -1 if unit == "month" else -12)
            return start, today, m.group(0)
        # "last year" / "previous month" = the previous calendar period
        if unit == "year":
            return date(today.year - 1, 1, 1), date(today.year - 1, 12, 31), m.group(0)
        if unit == "month":
            first = date(today.year, today.month, 1)
            prev_end = first - timedelta(days=1)
            return date(prev_end.year, prev_end.month, 1), prev_end, m.group(0)
        if unit == "week":
            start = today - timedelta(days=today.weekday() + 7)
            return start, start + timedelta(days=6), m.group(0)
        return today - timedelta(days=1), today - timedelta(days=1), m.group(0)
    m = re.search(r"\b(year[- ]to[- ]date|ytd)\b", tl)
    if m:
        return date(today.year, 1, 1), today, m.group(0)
    m = re.search(r"\b(today|yesterday)\b", tl)
    if m:
        d = today if m.group(1) == "today" else today - timedelta(days=1)
        return d, d, m.group(0)
    return None


def _month_only(text: str, today: date) -> tuple[date, date, str, list[str]] | None:
    """Months named without a year ("in May", "between May and July")."""
    prep = r"in|during|since|from|between|until|before|after|through|thru|of|for|only|just|about"
    rx_range = re.compile(
        rf"\b(?:{prep})\s+({_MONTH_RX})\s*(?:-|–|to|through|thru|until|and)\s*({_MONTH_RX})\b(?!\.?,?\s+{_YEAR_RX})",
        re.I,
    )
    rx_one = re.compile(rf"\b(?:{prep})\s+({_MONTH_RX})\b(?!\.?,?\s+(?:of\s+)?{_YEAR_RX})", re.I)

    def latest_year(month: int) -> int:
        return today.year if month <= today.month else today.year - 1

    bare = re.search(rf"^\s*({_MONTH_RX})(?:\s+only)?\s*[?.!]*\s*$|\b({_MONTH_RX})\s+only\b", text, re.I)
    if bare:
        mo = MONTHS[(bare.group(1) or bare.group(2)).lower()]
        y = latest_year(mo)
        return (
            date(y, mo, 1),
            _month_end(y, mo),
            bare.group(0).strip(),
            [f"No year given for '{bare.group(0).strip()}'; assumed {date(y, mo, 1):%B %Y}."],
        )
    m = rx_range.search(text)
    if m and not _verb_may(text, m.end(2)):
        m1, m2 = MONTHS[m.group(1).lower()], MONTHS[m.group(2).lower()]
        y2 = latest_year(m2)
        y1 = y2 - 1 if m1 > m2 else y2
        start, end = date(y1, m1, 1), _month_end(y2, m2)
        return (
            start,
            end,
            m.group(0),
            [f"No year given for '{m.group(0).strip()}'; assumed {start:%B %Y}–{end:%B %Y}."],
        )
    for m in rx_one.finditer(text):
        if m.group(1).lower() == "may" and _verb_may(text, m.end(1)):
            continue
        if m.group(1).lower() == "march" and re.match(r"\s+(to|on|in)\b", text[m.end(1) :], re.I):
            continue
        mo = MONTHS[m.group(1).lower()]
        y = latest_year(mo)
        start, end = date(y, mo, 1), _month_end(y, mo)
        kw = m.group(0).split()[0].lower()
        if kw in ("since", "after", "from"):
            end = today
        elif kw in ("before", "until", "through", "thru"):
            start = date(1900, 1, 1)
        return (
            start,
            end,
            m.group(0),
            [f"No year given for '{m.group(0).strip()}'; assumed {date(y, mo, 1):%B %Y}."],
        )
    return None


def _verb_may(text: str, end: int) -> bool:
    return (
        re.match(r"\s+(be|have|not|also|still|need|want|help|show|include)\b", text[end:], re.I) is not None
    )


def parse_time_window(text: str, today: date | None = None) -> TimeWindow:
    """Parse the time period in ``text``. Returns ``kind='none'`` when the text names no period."""
    today = today or date.today()
    t = text.strip()
    tw = TimeWindow()
    all_m = ALL_DATES_RX.search(t)
    periods = _periods(t)
    if all_m and not periods:
        tw.kind, tw.phrase = "all", all_m.group(0)
        tw.notes.append("All available dates requested; no date filter.")
        return tw
    if periods:
        tw.kind = "explicit"
        tw.years = sorted({p.start.year for p in periods} | {p.end.year for p in periods})
        first, last = periods[0], periods[-1]
        start: date | None
        end: date | None
        if len(periods) == 1 and _before(
            t, first.pos, "since|from|starting(?: in| from)?|beginning(?: in)?|as of"
        ):
            start, end = first.start, None
        elif len(periods) == 1 and _before(t, first.pos, "after"):
            start, end = first.end + timedelta(days=1), None
        elif len(periods) == 1 and _before(t, first.pos, "before|prior to"):
            start, end = None, first.start - timedelta(days=1)
        elif len(periods) == 1 and _before(
            t, first.pos, "until|through|thru|up to|up until|by|no later than"
        ):
            start, end = None, first.end
        else:  # one period, a range ("between 2020 and 2023"), or several periods ("2023 vs 2024")
            start, end = min(p.start for p in periods), max(p.end for p in periods)
        tw.date_from = start.isoformat() if start else None
        tw.date_to = end.isoformat() if end else None
        lead = re.search(
            r"\b(?:in|during|for|since|from|after|before|until|through|between|of)\s+$", t[: first.pos], re.I
        )
        tw.phrase = t[(lead.start() if lead else first.pos) : last.stop].strip()
        if all_m:
            tw.notes.append(f"Both '{all_m.group(0)}' and a specific period were mentioned; used the period.")
        return tw
    rel = _resolve_relative(t, today)
    if rel:
        start, end, phrase = rel
        tw.kind, tw.phrase = "relative", phrase
        tw.date_from, tw.date_to = start.isoformat(), end.isoformat()
        tw.years = sorted({start.year, end.year})
        tw.notes.append(f"'{phrase}' resolved relative to today ({today.isoformat()}): {start} to {end}.")
        return tw
    mo = _month_only(t, today)
    if mo:
        start, end, phrase, notes = mo
        tw.kind, tw.phrase, tw.notes = "month", phrase.strip(), notes
        tw.date_from = start.isoformat() if start.year > 1900 else None
        tw.date_to = end.isoformat()
        tw.years = sorted({start.year, end.year} - {1900})
        return tw
    if all_m:
        tw.kind, tw.phrase = "all", all_m.group(0)
        tw.notes.append("All available dates requested; no date filter.")
        return tw
    vague = VAGUE_RX.search(t)
    if vague:
        tw.kind, tw.phrase = "vague", vague.group(0)
        tw.notes.append(f"'{vague.group(0)}' does not name a period; no date filter was applied.")
    return tw


def window_contains(
    outer_from: str | None, outer_to: str | None, inner_from: str | None, inner_to: str | None
) -> bool:
    """True if [inner_from, inner_to] lies inside [outer_from, outer_to] (None = open end)."""
    if outer_from and (inner_from is None or inner_from < outer_from):
        return False
    return not (outer_to and (inner_to is None or inner_to > outer_to))


def describe_window(date_from: str | None, date_to: str | None) -> str:
    if date_from and date_to:
        return f"{date_from} to {date_to}"
    if date_from:
        return f"from {date_from}"
    if date_to:
        return f"through {date_to}"
    return "all dates"
