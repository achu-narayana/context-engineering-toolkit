"""Extractive compression and rule-based cleanup (the no-LLM core).

cleanup()      Light mode: remove filler, hedges, repeated sentences, greetings/sign-offs.
extract()      Pick the highest-value sentences in a section to fit a token budget (TextRank over
               sentence embeddings; sentences with protected items get priority; original order kept).
telegraphic()  Aggressive style: strip articles and connective filler, bullet each unit.
"""
from __future__ import annotations

import re
from typing import Callable

import numpy as np

from .embeddings import embed, cosine_matrix
from .protect import ProtectedItem, mask, unmask
from .segment import BULLET_RE, split_sentences
from .select import textrank

PRONOUN_START_RE = re.compile(r"^(?:She|He|They|It|This|That|These|Those|Her|His|Their|Its)\b")

FILLER_PHRASES = [
    r"\b(?:basically|actually|literally|honestly|frankly|obviously|clearly|of course|needless to say)\b,?\s*",
    r"\b(?:as you (?:may |might )?know|as (?:I|we) mentioned(?: before| earlier)?|like I said|at the end of the day|"
    r"in order to|for all intents and purposes|it goes without saying that|the fact of the matter is that|"
    r"it is worth noting that|it should be noted that|please note that|just a quick note that|I just wanted to say that|"
    r"I (?:just )?wanted to (?:let you know|reach out|follow up|check in|circle back)(?: that| and)?|"
    r"hope (?:you(?:'re| are) (?:doing )?well|this (?:email |message )?finds you well)[.,!]?)\s*",
    r"\b(?:I think(?: that)?|I believe(?: that)?|I feel like|I guess|in my opinion|personally,?|to be honest,?|"
    r"you know,?|I mean,?|kind of|sort of|pretty much|more or less|very|really|quite|rather|somewhat|extremely)\s+",
    r"\b(?:um+|uh+|hmm+|erm+|ah+|oh+|well,|so,|okay,|ok,|right,|anyway,|also,)\s+",
]
_FILLER_RES = [re.compile(p, re.IGNORECASE) for p in FILLER_PHRASES]
GREETING_RE = re.compile(
    r"^\s*(?:hi|hello|hey|dear|good (?:morning|afternoon|evening)|greetings)\b[^\n]{0,40}[,!.:-]?\s*$|"
    r"^\s*(?:thanks(?: again| a lot| so much)?|thank you(?: again| so much)?|cheers|best(?: regards| wishes)?|kind regards|regards|"
    r"sincerely|warm(?:ly| regards)|talk soon|let me know if you have any questions[.!]?|have a (?:great|nice|good) (?:day|weekend|one)[.!]?)[,!.]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_ARTICLE_RE = re.compile(r"\b(?:a|an|the|that|which|who is|which is|that is|there (?:is|are)|it is|this is)\b\s*", re.IGNORECASE)
_CONNECT_RE = re.compile(r"\b(?:in order to|so that|as well as|in addition(?: to)?|furthermore|moreover|however|therefore|thus|hence|"
                         r"additionally|consequently|nevertheless|on the other hand|for example|for instance|such as|please|kindly|"
                         r"we (?:would|will) (?:like|need) to|I (?:would|will) (?:like|need) to|going to|be able to)\b,?\s*", re.IGNORECASE)
_SPACE_RE = re.compile(r"[ \t]{2,}")


def cleanup(text: str, items: list[ProtectedItem], drop_greetings: bool = True) -> str:
    """Remove filler without changing meaning. Protected spans are masked during editing."""
    masked, table = mask(text, items)
    if drop_greetings:
        masked = GREETING_RE.sub("", masked)
    for rx in _FILLER_RES:
        masked = rx.sub("", masked)
    # A removed sentence-initial filler can leave a dangling comma: ", the system..." -> "The system..."
    masked = re.sub(r"(^|[.!?]\s+|\n)\s*[,;:]\s*", r"\1", masked)
    # Fix capitalisation after a removed sentence-initial filler
    masked = re.sub(r"(^|[.!?]\s+|\n)([a-z])", lambda m: m.group(1) + m.group(2).upper(), masked)
    masked = re.sub(r"\s+([,.;:!?])", r"\1", masked)
    masked = re.sub(r"\(\s*\)", "", masked)
    masked = _SPACE_RE.sub(" ", masked)
    masked = re.sub(r"\n{3,}", "\n\n", masked)
    # repeated sentences
    out: list[str] = []
    seen: set[str] = set()
    for para in masked.split("\n"):
        kept = []
        for sent in split_sentences(para) if para.strip() else []:
            key = re.sub(r"\W+", " ", sent).strip().lower()
            if key and key in seen:
                continue
            seen.add(key)
            kept.append(sent)
        out.append(" ".join(kept) if kept else para)
    return unmask("\n".join(out), table).strip()


def extract(
    text: str,
    budget_tokens: int,
    count: Callable[[str], int],
    items: list[ProtectedItem],
    embedding_model: str,
    allow_download: bool = True,
    goal: str | None = None,
) -> str:
    """Keep the best sentences that fit in `budget_tokens`, in original order."""
    sents = split_sentences(text)
    if len(sents) <= 1 or count(text) <= budget_tokens:
        return text
    is_list = "\n" in text and BULLET_RE.search(text) is not None
    joiner = "\n" if is_list else " "
    vecs = embed(sents + ([goal] if goal else []), embedding_model, allow_download)
    sv = vecs[: len(sents)]
    sim = cosine_matrix(sv)
    scores = textrank(sim)
    if goal:
        g = cosine_matrix(sv, vecs[len(sents):])[:, 0]
        g = (g - g.min()) / (g.max() - g.min() + 1e-9)
        scores = 0.55 * scores + 0.45 * g
    # protected-item bonus and lead bonus
    prot_texts = [it.text for it in items]
    for i, s in enumerate(sents):
        bonus = 0.0
        if any(p in s for p in prot_texts):
            bonus += 0.35
        if i == 0:
            bonus += 0.15
        scores[i] += bonus
    order = np.argsort(-scores)
    chosen: list[int] = []
    used = 0
    for i in order:
        t = count(sents[i])
        if used + t <= budget_tokens or not chosen:
            chosen.append(int(i))
            used += t
            # redundancy: skip sentences too similar to any chosen one
        if used >= budget_tokens:
            break
    # a chosen sentence that opens with a pronoun needs its antecedent: pull in the previous sentence if it fits
    for i in list(chosen):
        if i > 0 and (i - 1) not in chosen and PRONOUN_START_RE.match(sents[i]):
            t = count(sents[i - 1])
            if used + t <= budget_tokens * 1.25:
                chosen.append(i - 1)
                used += t
    # drop near-duplicates among chosen
    final: list[int] = []
    for i in sorted(chosen):
        if any(sim[i, j] > 0.9 for j in final):
            continue
        final.append(i)
    return joiner.join(sents[i] for i in final)


def telegraphic(text: str, items: list[ProtectedItem]) -> str:
    """Bullet, article-free, terse form. Protected spans untouched."""
    masked, table = mask(text, items)
    masked = GREETING_RE.sub("", masked)
    for rx in _FILLER_RES:
        masked = rx.sub("", masked)
    units: list[str] = []
    for para in masked.split("\n"):
        if not para.strip():
            continue
        for sent in split_sentences(para):
            s = _CONNECT_RE.sub("", sent)
            s = _ARTICLE_RE.sub("", s)
            s = re.sub(r"\b(?:is|are|was|were|be|been|being)\b\s+", "", s)
            s = re.sub(r"\s+([,.;:!?])", r"\1", s)
            s = _SPACE_RE.sub(" ", s).strip(" .;,")
            if s:
                s = s[0].upper() + s[1:]
                units.append(s)
    seen: set[str] = set()
    out: list[str] = []
    for u in units:
        k = u.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(u if u.startswith(("-", "*", "•")) else f"- {u}")
    return unmask("\n".join(out), table)
