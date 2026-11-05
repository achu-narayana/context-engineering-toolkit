"""Input kind detection. Pure heuristics, no model.

Kinds: json | log | chat | email | code | prose
"""
from __future__ import annotations

import json
import re

CHAT_LINE_RE = re.compile(
    r"^\s*(?:\[?\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AP]M)?\]?\s*)?"   # optional [12:34] / 12:34 PM
    r"(?:\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?\]?\s*)?"  # optional [2024-01-01 12:34]
    r"(?:<)?@?([A-Z][\w .'-]{0,30}?|[a-z][\w.-]{1,20})(?:>)?\s*[:>]\s+\S",
    re.MULTILINE,
)
EMAIL_HEADER_RE = re.compile(r"^(From|To|Subject|Sent|Date|Cc):\s*\S", re.MULTILINE | re.IGNORECASE)
EMAIL_QUOTE_RE = re.compile(r"^On .{5,120} wrote:\s*$|^-{3,}\s*Original Message\s*-{3,}$", re.MULTILINE)
LOG_LINE_RE = re.compile(
    r"^\s*(?:\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?|\d{2}:\d{2}:\d{2}(?:[.,]\d+)?|\[\d{4}-\d{2}-\d{2}[^\]]*\])"
    r"[^\n]*?\b(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|ERR|FATAL|CRITICAL|Exception|Traceback)\b",
    re.MULTILINE,
)
CODE_HINT_RE = re.compile(
    r"^\s*(def |class |import |from \w+ import |#include|public static|function |const |let |var |fn |package |SELECT |=> |\{|\}|\s*//|\s*#!)",
    re.MULTILINE,
)


def looks_like_json(text: str) -> bool:
    s = text.strip()
    if not s or s[0] not in "{[":
        return False
    try:
        json.loads(s)
        return True
    except Exception:
        # Tolerate trailing garbage like "...,}" by checking the first 2KB structure
        return bool(re.match(r'^[\[{]\s*(?:"[^"]+"\s*:|[\[{"\d])', s)) and s.count('"') > 10


def detect_kind(text: str) -> str:
    """Return one of json | log | chat | email | code | prose."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    n = max(1, len(lines))
    if looks_like_json(text):
        return "json"
    if len(LOG_LINE_RE.findall(text)) >= max(3, int(0.4 * n)):
        return "log"
    if EMAIL_HEADER_RE.search(text) and (len(EMAIL_HEADER_RE.findall(text)) >= 2 or EMAIL_QUOTE_RE.search(text)):
        return "email"
    chat_hits = len(CHAT_LINE_RE.findall(text))
    if chat_hits >= max(3, int(0.35 * n)):
        return "chat"
    code_hits = len(CODE_HINT_RE.findall(text))
    if code_hits >= max(4, int(0.3 * n)):
        return "code"
    return "prose"
