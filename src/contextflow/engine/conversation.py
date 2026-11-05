"""Conversation compressor for chat and email threads.

Older turns are summarised; decisions, tasks, open questions and key facts are kept explicitly;
the latest turn(s) are kept close to verbatim. Works with or without the LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .segment import Section, split_sentences

CHAT_TURN_RE = re.compile(
    r"^\s*(?:\[?(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?|\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AP]M)?)\]?\s*)?"
    r"<?@?(?P<speaker>[A-Z][\w .'-]{0,30}?|[a-z][\w.-]{1,20})>?\s*[:>]\s+(?P<text>\S.*)$"
)
EMAIL_SPLIT_RE = re.compile(r"(?=^From:\s)", re.MULTILINE)
EMAIL_HDR_RE = re.compile(r"^(From|To|Cc|Subject|Date|Sent):\s*(.*)$", re.MULTILINE | re.IGNORECASE)
QUOTED_RE = re.compile(r"^On .{5,120} wrote:\s*$[\s\S]*|^-{2,}\s*Original Message\s*-{2,}[\s\S]*|^>.*$", re.MULTILINE)
SIGNOFF_RE = re.compile(r"^\s*(?:--|__|thanks|thank you|best|regards|kind regards|cheers|sincerely|sent from my)\b[\s\S]*", re.IGNORECASE | re.MULTILINE)

DECISION_RE = re.compile(r"\b(?:decid(?:e|ed|es|ing)|decisions?|agreed?|let'?s go with|we'?ll go with|going with|approved|signed? off|confirmed|final(?:ised|ized)?|settled on|consensus|we will use|we are using|moving forward with|green ?light)\b", re.IGNORECASE)
TASK_RE = re.compile(r"\b(?:todo|action item|I'?ll (?:do|send|write|update|fix|handle|take|draft|set up|ship|open|create)|will (?:do|send|write|update|fix|handle|take|draft|set up|ship|open|create)|can you|could you|please (?:send|update|review|check|fix|share|add|confirm)|need(?:s)? to|owner|assigned|by (?:eod|eow|tomorrow|monday|tuesday|wednesday|thursday|friday|end of)|deadline|due\b)", re.IGNORECASE)
QUESTION_RE = re.compile(r"\?\s*$")
FACT_RE = re.compile(r"\d|\bhttps?://|\b(?:because|since|blocked|blocker|risk|issue|bug|cost|budget|latency|version)\b", re.IGNORECASE)


@dataclass
class Turn:
    speaker: str
    text: str
    ts: str = ""
    subject: str = ""
    index: int = 0


@dataclass
class Facts:
    decisions: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    participants: list[str] = field(default_factory=list)


def parse_chat(text: str) -> list[Turn]:
    turns: list[Turn] = []
    for ln in text.splitlines():
        if not ln.strip():
            continue
        m = CHAT_TURN_RE.match(ln)
        if m:
            turns.append(Turn(speaker=m.group("speaker").strip(), text=m.group("text").strip(), ts=(m.group("ts") or "").strip(), index=len(turns)))
        elif turns:
            turns[-1].text += "\n" + ln.strip()   # continuation line
        else:
            turns.append(Turn(speaker="", text=ln.strip(), index=0))
    return turns


def parse_email(text: str) -> list[Turn]:
    parts = [p for p in EMAIL_SPLIT_RE.split(text) if p.strip()]
    turns: list[Turn] = []
    for p in parts:
        hdrs = {k.lower(): v.strip() for k, v in EMAIL_HDR_RE.findall(p)}
        body = EMAIL_HDR_RE.sub("", p)
        body = QUOTED_RE.sub("", body)
        body = SIGNOFF_RE.sub("", body).strip()
        body_lines = [ln for ln in body.splitlines()]
        # salutation ("Hi all," / "Nora,") and a bare name as the last line
        while body_lines and re.fullmatch(r"\s*(?:(?:hi|hello|hey|dear|good (?:morning|afternoon|evening))\b[^\n]{0,40}|[A-Z][\w.' -]{0,30}),?\s*", body_lines[0]) and len(body_lines[0].split()) <= 4:
            body_lines.pop(0)
        while body_lines and (not body_lines[-1].strip() or (len(body_lines[-1].split()) <= 3 and not re.search(r"[.!?:;,\d]", body_lines[-1]))):
            body_lines.pop()
        body = "\n".join(body_lines).strip()
        if not body and not hdrs:
            continue
        sender = hdrs.get("from", "")
        sender = re.sub(r"<.*?>", "", sender).strip() or sender
        turns.append(Turn(speaker=sender or "unknown", text=body, ts=hdrs.get("date", hdrs.get("sent", "")), subject=hdrs.get("subject", ""), index=len(turns)))
    # chronological order: many clients paste newest-first
    if len(turns) > 1 and _looks_newest_first(turns):
        turns.reverse()
        for i, t in enumerate(turns):
            t.index = i
    return turns


def _looks_newest_first(turns: list[Turn]) -> bool:
    subj = [t.subject for t in turns if t.subject]
    if subj and subj[0].lower().startswith(("re:", "fwd:")) and not subj[-1].lower().startswith(("re:", "fwd:")):
        return True
    return False


def turns_to_sections(turns: list[Turn], kind: str) -> list[Section]:
    out: list[Section] = []
    n = max(1, len(turns) - 1)
    for t in turns:
        label = f"{t.speaker}" + (f" ({t.ts})" if t.ts else "")
        out.append(Section(id=t.index, text=t.text, kind=kind, position=t.index / n, meta={"speaker": t.speaker, "ts": t.ts, "label": label, "subject": t.subject}))
    return out


DEFERRAL_RE = re.compile(r"\b(?:don'?t know|not sure|no idea|leave (?:that|it) for|next week|later|tbd|to be decided|no owner|unassigned|parking|park (?:that|it)|follow up|will find out|good question)\b", re.IGNORECASE)
ACK_RE = re.compile(r"^\W*(?:ok(?:ay)?|yep|yes|yeah|no|nope|sure|fine(?: by me)?|great|cool|thanks(?: \w+)?|thank you|will do|done|agreed|same|\+1|morning(?: all)?|hi(?: all)?|hello|got it|sounds good|makes sense|perfect|nice|lol|haha|\U0001F44D)\W*$", re.IGNORECASE)


def is_open_question(turns: list[Turn], i: int) -> bool:
    """A question is open if nobody else answers within two turns, the reply defers it, or it is at the end."""
    n = len(turns)
    if i >= n - 3:
        return True
    asker = turns[i].speaker
    replies = [t for t in turns[i + 1:i + 3] if t.speaker and t.speaker != asker]
    if not replies:
        return True
    return any(DEFERRAL_RE.search(r.text) for r in replies)


def extract_facts(turns: list[Turn]) -> Facts:
    f = Facts()
    f.participants = list(dict.fromkeys(t.speaker for t in turns if t.speaker))
    for t in turns:
        for s in split_sentences(t.text):
            s_clean = s.strip()
            if len(s_clean) < 6 or len(s_clean.split()) < 4:
                continue
            tag = f"{t.speaker}: {s_clean}" if t.speaker else s_clean
            if DECISION_RE.search(s_clean) and len(s_clean) < 300:
                f.decisions.append(tag)
            elif TASK_RE.search(s_clean) and len(s_clean) < 300 and not QUESTION_RE.search(s_clean):
                f.tasks.append(tag)
            elif QUESTION_RE.search(s_clean):
                if is_open_question(turns, t.index):
                    f.questions.append(tag)
            elif FACT_RE.search(s_clean) and len(s_clean) < 240:
                f.facts.append(tag)
    # de-dup while keeping order
    for attr in ("decisions", "tasks", "questions", "facts"):
        seen: set[str] = set()
        uniq = []
        for x in getattr(f, attr):
            k = re.sub(r"\W+", " ", x.lower()).strip()
            if k not in seen:
                seen.add(k)
                uniq.append(x)
        setattr(f, attr, uniq)
    return f


def _key(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def render_thread(
    sections: list[Section],
    facts: Facts,
    mode: str,
    kind: str,
    latest_verbatim: int,
    summarize_turn,
    count,
    target: int,
) -> str:
    """Assemble the compressed thread within `target` tokens.

    Priority order: header, decisions, tasks, open questions, latest turn(s), then the best older turns
    until the budget runs out. Acknowledgement-only turns and sentences already listed as facts are skipped.
    """
    if not sections:
        return ""
    n = len(sections)
    header = f"{'Email' if kind == 'email' else 'Chat'} thread: {n} messages"
    if facts.participants:
        header += ", participants: " + ", ".join(facts.participants[:8])
    subj = next((s.meta.get("subject") for s in sections if s.meta.get("subject")), "")
    if subj:
        header += f"\nSubject: {subj}"
    lines = [header]
    limit = 5 if mode == "aggressive" else 10
    listed: set[str] = set()
    for title, items in (("Decisions:", facts.decisions), ("Tasks / action items:", facts.tasks), ("Open questions:", facts.questions)):
        if items:
            lines += ["", title]
            for it in items[:limit]:
                lines.append(f"- {it}")
                listed.add(_key(it.split(": ", 1)[-1]))

    def label(s: Section) -> str:
        return s.meta.get("label") or s.meta.get("speaker") or ""

    latest = [s for s in sections if s.id >= n - latest_verbatim]
    latest_lines = ["", "Latest:"]
    for s in latest:
        body = (s.output or summarize_turn(s)).strip()
        if body:
            latest_lines.append(f"{label(s)}: {body}".strip(": "))

    used = count("\n".join(lines + latest_lines))
    remaining = max(0, target - used)

    older = [s for s in sections if s.keep and s.id < n - latest_verbatim]
    picked: list[tuple[Section, str]] = []
    for s in sorted(older, key=lambda x: -x.score):
        if ACK_RE.match(s.text.strip()):
            continue
        summ = (s.output or summarize_turn(s)).strip()
        if not summ:
            continue
        sents = [x for x in split_sentences(summ) if _key(x.lstrip("-* ")) not in listed]
        bullets = any(x.lstrip().startswith(("-", "*", "•")) for x in sents)
        summ = ("\n  ".join(sents) if bullets else " ".join(sents)).strip()
        if not summ or _key(summ) in listed:
            continue
        line = f"- {label(s)}: {summ}" if label(s) else f"- {summ}"
        t = count(line)
        if t > remaining and picked:
            continue
        picked.append((s, line))
        remaining -= t
        if remaining <= 0:
            break
    if picked:
        lines += ["", "Earlier discussion:"] + [ln for _, ln in sorted(picked, key=lambda p: p[0].id)]
    lines += latest_lines
    return "\n".join(lines)
