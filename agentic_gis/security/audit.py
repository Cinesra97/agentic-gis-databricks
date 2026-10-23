"""Append-only JSONL audit log with secret redaction.

The audit log records *what happened* (questions, plans, SQL validated/approved/executed,
tool calls, errors, row counts) but never result rows or credentials.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("agentic_gis.audit")

_PATTERNS = [
    (re.compile(r"dapi[0-9a-f]{32}(-\d+)?", re.I), "dapi***"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-~+/]+=*"), "Bearer ***"),
    (re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]+"), "***jwt***"),
    (
        re.compile(
            r"(?i)\b(token|password|passwd|secret|api[_-]?key|client[_-]?secret)\b\s*[=:]\s*['\"]?[^\s'\",;]+"
        ),
        r"\1=***",
    ),
]


def redact_text(text: str, secrets: list[str] | None = None) -> str:
    for s in secrets or []:
        if s and len(s) >= 4:
            text = text.replace(s, "***")
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text


def redact(obj: Any, secrets: list[str] | None = None) -> Any:
    if isinstance(obj, str):
        return redact_text(obj, secrets)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and re.search(r"(?i)(token|secret|password|api_key)", k):
                out[k] = "***"
            else:
                out[k] = redact(v, secrets)
        return out
    if isinstance(obj, list | tuple):
        return [redact(v, secrets) for v in obj]
    return obj


class AuditLogger:
    def __init__(self, path: Path | None, secrets: list[str] | None = None, log_sql: bool = True):
        self.path = path
        self.secrets = secrets or []
        self.log_sql = log_sql
        self._lock = threading.Lock()

    def log(self, event: str, session_id: str | None = None, **details: Any) -> None:
        if not self.log_sql:
            for key in ("sql", "original_sql"):
                if key in details:
                    details[key] = "[omitted by configuration]"
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "event": event,
            "session": session_id,
            **redact(details, self.secrets),
        }
        line = json.dumps(record, default=str, ensure_ascii=False)
        log.debug(line)
        if self.path is None:
            return
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError as exc:  # audit failures must not crash the app, but are surfaced in logs
            log.error("Could not write audit log %s: %s", self.path, exc)
