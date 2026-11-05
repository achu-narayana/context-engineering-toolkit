"""Segmentation of input into sections and sentences.

A Section is the unit the selector ranks and the budget manager allocates tokens to.
Code fences are kept as single, protected sections.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

FENCE_RE = re.compile(r"```[\s\S]*?```")
HEADING_RE = re.compile(r"^(#{1,6}\s+\S.*|[A-Z][A-Za-z0-9 ,:&/-]{2,60}:?\s*)$")
BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
# Sentence splitter that does not split on decimals, abbreviations or inside URLs.
_SENT_RE = re.compile(
    r"(?<!\b[A-Z])(?<!\b[A-Z][a-z])(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bvs)(?<!\be\.g)(?<!\bi\.e)(?<!\d)"
    r"(?<=[.!?])\s+(?=[A-Z\"'(\[\d])"
)


@dataclass
class Section:
    id: int
    text: str
    kind: str = "paragraph"          # paragraph | heading | list | code | turn | message | record
    position: float = 0.0            # 0 = start of document, 1 = end
    tokens: int = 0
    protected: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0               # relevance/centrality score from the selector
    keep: bool = True
    reason: str = ""
    budget: int = 0                  # tokens allocated by the budget manager
    output: str = ""                 # compressed text
    output_tokens: int = 0

    @property
    def is_code(self) -> bool:
        return self.kind == "code"


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if "\n" in text and BULLET_RE.search(text):
        # bullet lists: each line is a unit
        return [ln.strip() for ln in text.splitlines() if ln.strip()]
    parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def segment_prose(text: str) -> list[Section]:
    """Split prose/markdown into paragraph-level sections; code fences stay whole."""
    sections: list[Section] = []
    pos = 0
    pieces: list[tuple[str, str]] = []
    for m in FENCE_RE.finditer(text):
        before = text[pos:m.start()]
        if before.strip():
            pieces.append(("text", before))
        pieces.append(("code", m.group(0)))
        pos = m.end()
    tail = text[pos:]
    if tail.strip():
        pieces.append(("text", tail))

    for kind, chunk in pieces:
        if kind == "code":
            sections.append(Section(id=len(sections), text=chunk.strip(), kind="code"))
            continue
        paras = re.split(r"\n\s*\n", chunk)
        buf: list[str] = []
        for para in paras:
            para = para.strip("\n")
            if not para.strip():
                continue
            stripped = para.strip()
            if HEADING_RE.match(stripped) and len(stripped) < 80 and "\n" not in stripped:
                sections.append(Section(id=len(sections), text=stripped, kind="heading"))
            elif BULLET_RE.match(stripped):
                sections.append(Section(id=len(sections), text=stripped, kind="list"))
            else:
                # merge very short consecutive lines (e.g. hard-wrapped text)
                sections.append(Section(id=len(sections), text=re.sub(r"[ \t]*\n[ \t]*", " ", stripped), kind="paragraph"))
        del buf
    # If a single huge paragraph, split it further by sentences into ~6-sentence sections.
    out: list[Section] = []
    for s in sections:
        if s.kind == "paragraph" and len(s.text) > 1800:
            sents = split_sentences(s.text)
            for i in range(0, len(sents), 6):
                out.append(Section(id=len(out), text=" ".join(sents[i:i + 6]), kind="paragraph"))
        else:
            s.id = len(out)
            out.append(s)
    n = max(1, len(out) - 1)
    for s in out:
        s.position = s.id / n
    return out


def segment_lines(text: str, kind: str) -> list[Section]:
    """One section per non-empty line (used for logs)."""
    out = [Section(id=i, text=ln.rstrip(), kind=kind) for i, ln in enumerate(l for l in text.splitlines() if l.strip())]
    n = max(1, len(out) - 1)
    for s in out:
        s.position = s.id / n
    return out
