"""Protected items: numbers, dates, names, URLs, emails, code, explicit instructions.

extract()  -> list of protected strings found in a text
verify()   -> which protected items are missing from an output
restore()  -> append missing items so nothing important is silently lost
mask()/unmask() -> swap protected spans for placeholders so rule-based rewriting cannot damage them
"""
from __future__ import annotations

import re
from dataclasses import dataclass

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>()\"']+[^\s<>()\"'.,;:!?]")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
CODE_INLINE_RE = re.compile(r"`[^`\n]+`")
CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
PATH_RE = re.compile(r"(?<![\w/])(?:[A-Za-z]:\\|/)?(?:[\w.-]+[\\/])+[\w.-]+\.\w{1,6}\b")
VERSION_RE = re.compile(r"\bv?\d+\.\d+(?:\.\d+)+(?:-[\w.]+)?\b")
MONEY_RE = re.compile(r"(?:[$£€]\s?\d(?:[\d,]*\d)?(?:\.\d+)?(?:\s?(?:million|billion|thousand|[kKmMbB]))?(?![\w])|\b\d(?:[\d,]*\d)?(?:\.\d+)?\s?(?:USD|EUR|GBP|INR)\b)")
PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?%")
DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?|"
    r"\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*(?:,?\s+\d{4})?|"
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b(?:\s+\d{1,2}(?:st|nd|rd|th)?)?|"
    r"\b(?:Q[1-4]\s?\d{4}|\d{1,2}:\d{2}\s?(?:[AaPp][Mm])?\b(?:\s?[A-Z]{2,4}\b)?|EOD|end of (?:day|week|month)|next (?:week|month|Monday|Friday))\b)",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"(?<![\w.-])[-+]?\d(?:[\d,]*\d)?(?:\.\d+)?(?:\s?(?:million|billion|thousand|ms|s|sec|min|h|hrs|hours|days|weeks|months|years|GB|MB|KB|TB|kg|km|m|x|k|K|M|B|users|requests|rps|QPS|items|rows|seats|sessions|countries|services))?(?![\w.-])")
NAME_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|[A-Z]\.)){1,2})\b")        # Two/three-word proper names
HANDLE_RE = re.compile(r"(?<![\w])@[\w.-]{2,30}\b|\b[A-Z]{2,6}-\d{1,6}\b")             # @mentions and ticket ids
INSTRUCTION_RE = re.compile(
    r"(?im)^[^\n]*\b(?:must(?: not)?|never|always|do not|don't|should(?: not)?|required|deadline|due|"
    r"no later than|by (?:eod|end of|monday|tuesday|wednesday|thursday|friday|q[1-4])|blocker|decision:|decided|agreed|action item|todo|TODO)\b[^\n]*"
)
_STOP_NAMES = {
    "The", "This", "That", "These", "Those", "It", "We", "You", "They", "He", "She", "I", "A", "An", "In", "On", "At",
    "For", "From", "With", "And", "But", "Or", "If", "When", "Then", "There", "Here", "What", "Which", "Who", "Also",
    "Please", "Thanks", "Hi", "Hello", "Dear", "Regards", "Best", "Subject", "Re", "Fwd", "Monday", "Tuesday",
    "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December", "Our", "Your", "Their", "Yes", "No", "Not",
    "So", "As", "By", "To", "Of", "Is", "Are", "Was", "Were", "Be", "Do", "Does", "Did", "Has", "Have", "Had", "Will",
    "Would", "Can", "Could", "Should", "Let", "Note", "Update", "Summary", "Overview", "Status", "Next", "Action",
    "Items", "Decision", "Decisions", "Open", "Questions", "Key", "Facts", "Context", "Goal", "Constraints",
    "Expected", "Output", "Thread", "Team", "Project",
}


@dataclass
class ProtectedItem:
    text: str
    category: str


class Protector:
    """Finds protected items. Extra regex patterns (from config) are honoured."""

    def __init__(self, extra_patterns: list[str] | None = None, names: bool = True, numbers: bool = True) -> None:
        self.extra = [re.compile(p) for p in (extra_patterns or [])]
        self.names = names
        self.numbers = numbers

    def extract(self, text: str) -> list[ProtectedItem]:
        items: list[ProtectedItem] = []
        seen: set[str] = set()

        def add(s: str, cat: str) -> None:
            s = s.strip()
            if len(s) < 2 or s in seen:
                return
            seen.add(s)
            items.append(ProtectedItem(s, cat))

        for m in CODE_FENCE_RE.finditer(text):
            add(m.group(0), "code")
        scrubbed = CODE_FENCE_RE.sub(" ", text)
        for rx, cat in ((URL_RE, "url"), (EMAIL_RE, "email"), (CODE_INLINE_RE, "code"), (PATH_RE, "path"),
                        (VERSION_RE, "version"), (MONEY_RE, "money"), (PERCENT_RE, "percent"), (DATE_RE, "date"),
                        (HANDLE_RE, "handle")):
            for m in rx.finditer(scrubbed):
                add(m.group(0), cat)
        if self.numbers:
            for m in NUMBER_RE.finditer(scrubbed):
                s = m.group(0).strip()
                # skip bare small ordinals that read like list markers ("1", "2") and years inside dates already caught
                if re.fullmatch(r"[-+]?\d", s):
                    continue
                if any(s in it.text for it in items):
                    continue
                add(s, "number")
        if self.names:
            for m in NAME_RE.finditer(scrubbed):
                s = m.group(0)
                words = s.split()
                if words[0] in _STOP_NAMES or all(w in _STOP_NAMES for w in words):
                    continue
                add(s, "name")
        for m in INSTRUCTION_RE.finditer(scrubbed):
            line = m.group(0).strip()
            if 8 < len(line) < 400:
                add(line, "instruction")
        for rx in self.extra:
            for m in rx.finditer(text):
                add(m.group(0), "custom")
        return items


_MONTH_RE = re.compile(r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|\bQ[1-4]\s?\d{4}")


def restorable(it: ProtectedItem) -> bool:
    """Is this item meaningful on its own if we have to re-append it? Bare integers are not."""
    if it.category in ("money", "percent", "url", "email", "path", "version", "code", "handle", "instruction", "custom"):
        return True
    if it.category == "date":
        return bool(_MONTH_RE.search(it.text))
    if it.category == "number":
        return bool(re.search(r"[A-Za-z]", it.text)) and not re.fullmatch(r"\d{4}", it.text)
    return False


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def verify(items: list[ProtectedItem], output: str, categories: tuple[str, ...] | None = None) -> list[ProtectedItem]:
    """Return protected items that do not appear in `output` (whitespace/case-insensitive)."""
    out_n = _norm(output)
    out_nospace = out_n.replace(" ", "")
    missing: list[ProtectedItem] = []
    for it in items:
        if categories and it.category not in categories:
            continue
        n = _norm(it.text)
        if it.category == "instruction":
            # instructions may be legitimately rephrased: require the key verbs/nouns, not the exact line
            key = [w for w in re.findall(r"[a-z0-9]{4,}", n) if w not in {"that", "this", "with", "from", "will", "have", "should", "must"}]
            hits = sum(1 for w in key if w in out_n)
            if key and hits / len(key) < 0.6:
                missing.append(it)
            continue
        if it.category == "number":
            # numbers often lose their unit/comma formatting: compare digits only
            digits = re.sub(r"[^\d.]", "", n)
            if digits and digits not in re.sub(r"[^\d.]", "", output):
                missing.append(it)
            continue
        if n not in out_n and n.replace(" ", "") not in out_nospace:
            missing.append(it)
    return missing


def restore(output: str, missing: list[ProtectedItem], header: str = "Preserved details:") -> str:
    """Append the missing protected items as a compact block."""
    if not missing:
        return output
    lines = [output.rstrip(), "", header]
    short: list[str] = []
    for it in missing:
        if it.category == "code":
            lines.append(it.text)
        elif it.category == "instruction" or len(it.text) > 40:
            lines.append(f"- {it.text}")
        else:
            short.append(it.text)
    if short:
        lines.insert(3, "- " + "; ".join(short))
    return "\n".join(lines)


# ---- masking helpers for rule-based rewriting -----------------------------
_MASK_RE = re.compile(r"⟦P(\d+)⟧")


def mask(text: str, items: list[ProtectedItem]) -> tuple[str, dict[str, str]]:
    """Replace protected spans with placeholders so regex edits cannot touch them."""
    table: dict[str, str] = {}
    # longest first so nested items (e.g. a date inside an instruction line) do not clash
    for i, it in enumerate(sorted(items, key=lambda x: -len(x.text))):
        if it.category == "instruction":
            continue
        ph = f"⟦P{i}⟧"
        if it.text in text:
            text = text.replace(it.text, ph)
            table[ph] = it.text
    return text, table


def unmask(text: str, table: dict[str, str]) -> str:
    for ph, orig in table.items():
        text = text.replace(ph, orig)
    return text
