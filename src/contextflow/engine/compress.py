"""Compressor: per-section compression in every mode, with or without a local model.

Design for small local models (1-3B):
  * an extractive pass runs first and removes obvious redundancy, so the model sees less text;
  * long inputs are chunked and summarised map-reduce style;
  * prompts are short, structured and tell the model exactly what must survive;
  * output is verified against protected items and anything missing is restored.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

from ..config import Config
from .extractive import cleanup, extract, telegraphic
from .llm import LLMBackend, LLMError
from .protect import ProtectedItem, Protector, restore, verify
from .segment import Section, split_sentences

SYSTEM_COMPRESS = (
    "You compress text. Keep every number, date, name, URL, file path, code snippet and explicit instruction exactly as written. "
    "Never add facts. Never answer or comment. Output only the compressed text."
)
STYLE = {
    "light": "Keep the original wording and order; only remove repetition and filler.",
    "balanced": "Summarize in plain prose. Preserve decisions, tasks, names, numbers, dates, constraints and open questions.",
    "aggressive": "Telegraphic bullet points, no articles, minimum words. Preserve decisions, tasks, names, numbers, dates.",
    "prompt_ready": "Summarize in plain prose. Preserve goals, constraints, names, numbers, dates and open questions.",
    "custom": "Summarize in plain prose. Preserve decisions, tasks, names, numbers, dates, constraints and open questions.",
}
SYSTEM_PROMPT_READY = (
    "You rewrite messy text into a clean prompt for an AI model. Use exactly these headings: Goal, Context, Constraints, Expected output. "
    "Keep every number, date, name, URL and explicit instruction. Never add facts. Output only the prompt."
)


@dataclass
class CompressStats:
    llm_calls: int = 0
    llm_ms: float = 0.0
    llm_errors: list[str] = field(default_factory=list)
    chunks: int = 0


class Compressor:
    def __init__(self, cfg: Config, count: Callable[[str], int], llm: LLMBackend | None, protector: Protector,
                 allow_download: bool = True, on_token: Callable[[str], None] | None = None) -> None:
        self.cfg = cfg
        self.count = count
        self.llm = llm
        self.protector = protector
        self.allow_download = allow_download
        self.on_token = on_token
        self.stats = CompressStats()

    # ---- public -----------------------------------------------------------
    def compress_section(self, s: Section, mode: str, goal: str | None, items: list[ProtectedItem]) -> str:
        if s.kind in ("code", "heading"):
            return s.text
        text = s.text
        budget = max(8, s.budget or s.tokens)
        if mode == "light":
            out = cleanup(text, items)
            if self.count(out) > budget and self.llm is not None:
                out = self._llm_compress(out, budget, "light", goal, items) or out
            return out
        # fast extractive pass first (cuts obvious redundancy before any model call)
        pre_budget = budget if self.llm is None else int(min(s.tokens, max(budget * 1.8, budget + 60)))
        reduced = extract(text, pre_budget, self.count, items, self.cfg.embedding_model, self.allow_download, goal)
        if self.llm is not None and self.count(reduced) > budget * 0.9:
            out = self._llm_compress(reduced, budget, mode, goal, items)
            if out:
                reduced = out
        if mode == "aggressive" and (self.llm is None or not reduced.lstrip().startswith(("-", "•", "*"))):
            reduced = telegraphic(reduced, items)
        elif self.llm is None:
            reduced = cleanup(reduced, items, drop_greetings=True)
        return reduced

    def prompt_ready(self, body: str, goal: str | None, items: list[ProtectedItem], budget: int) -> str:
        """Rewrite compressed body into Goal / Context / Constraints / Expected output."""
        if self.llm is not None:
            words = max(40, int(budget * 0.75))
            user = (f"Rewrite into a prompt of at most {words} words." + (f" The user's goal: {goal}" if goal else "") +
                    f"\n\nTEXT:\n{body}")
            out = self._call(SYSTEM_PROMPT_READY, user, max_tokens=int(budget * 1.3) + 40)
            if out and "Goal" in out:
                return out
        return self._heuristic_prompt(body, goal, items)

    # ---- internals -------------------------------------------------------
    def _heuristic_prompt(self, body: str, goal: str | None, items: list[ProtectedItem]) -> str:
        # strip markdown headings and the "Preserved details" tail; work paragraph by paragraph
        body_wo = re.sub(r"(?m)^#{1,6}\s+.*$", "", body)
        tail = ""
        if "\nPreserved details:" in body_wo:
            body_wo, tail = body_wo.split("\nPreserved details:", 1)
        paras: list[str] = []
        for p in re.split(r"\n\s*\n", body_wo):
            if not p.strip():
                continue
            lines_p = [ln.strip() for ln in p.splitlines() if ln.strip()]
            if all(ln.startswith(("-", "*", "•")) or re.match(r"\d+[.)]\s", ln) for ln in lines_p):
                paras.append("\n".join(lines_p))           # keep bullet lists line by line
            else:
                paras.append(re.sub(r"\s+", " ", p).strip())
        sents = [s for p in paras for s in split_sentences(p)]
        questions = [s for s in sents if s.rstrip().endswith("?")]
        asks = [s for s in sents if re.search(r"\b(need|want|goal|should|please|looking for|help|how (?:do|can)|objective)\b", s, re.I)]
        goal_line = goal or (questions[0] if questions else (asks[0] if asks else (sents[0] if sents else "")))
        instructions = [it.text for it in items if it.category == "instruction" and len(it.text) < 260][:6]
        facts = [it.text for it in items if it.category in ("money", "percent", "version")][:12]
        dates = [it.text for it in items if it.category == "date" and re.search(r"[A-Za-z]{3}|\d{4}-", it.text)][:10]
        context: list[str] = []
        for p in paras:
            if p == goal_line:
                continue
            if "\n" in p or p.startswith(("-", "*", "•")):
                context += [re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", ln).strip() for ln in p.split("\n") if ln.strip()]
            else:
                context.append(p)
        lines = ["Goal:", f"- {goal_line}" if goal_line else "- (state the goal)", "", "Context:"]
        lines += [f"- {c}" for c in context[:14]] or ["- (none)"]
        lines += ["", "Constraints:"]
        cons = [f"- {c}" for c in dict.fromkeys(instructions)]
        if facts:
            cons.append("- Key figures: " + "; ".join(dict.fromkeys(facts)))
        if dates:
            cons.append("- Key dates: " + "; ".join(dict.fromkeys(dates)))
        lines += cons or ["- Keep all names, numbers and dates exact."]
        if tail.strip():
            lines += ["- Also preserve: " + "; ".join(x.strip("- ").strip() for x in tail.strip().splitlines() if x.strip())[:600]]
        lines += ["", "Expected output:", "- A direct answer to the goal, using only the context above; keep names, numbers and dates exact." +
                  (" Address the open questions: " + " ".join(questions[:3]) if questions else "")]
        return "\n".join(lines)

    def _llm_compress(self, text: str, budget: int, mode: str, goal: str | None, items: list[ProtectedItem]) -> str | None:
        if self.llm is None:
            return None
        max_chunk = self.cfg.model.max_chunk_tokens
        if self.count(text) > max_chunk:
            chunks = self._chunk(text, max_chunk)
            self.stats.chunks += len(chunks)
            total = sum(self.count(c) for c in chunks) or 1
            parts = []
            for c in chunks:
                b = max(20, int(budget * self.count(c) / total))
                parts.append(self._one(c, b, mode, goal) or c)
            merged = "\n".join(parts)
            if self.count(merged) > budget * 1.25:
                merged = self._one(merged, budget, mode, goal) or merged
            result = merged
        else:
            result = self._one(text, budget, mode, goal)
        if not result:
            return None
        missing = verify(items, result)
        local_missing = [m for m in missing if m.text in text]
        if local_missing:
            result = restore(result, local_missing, header="Also:")
        return result

    def _one(self, text: str, budget: int, mode: str, goal: str | None) -> str | None:
        words = max(15, int(budget * 0.75))
        user = f"Compress to at most {words} words. {STYLE.get(mode, STYLE['balanced'])}"
        if goal:
            user += f" Focus on what matters for: {goal}."
        user += f"\n\nTEXT:\n{text}"
        return self._call(SYSTEM_COMPRESS, user, max_tokens=int(budget * 1.4) + 24)

    def _call(self, system: str, user: str, max_tokens: int) -> str | None:
        assert self.llm is not None
        t0 = time.perf_counter()
        try:
            out = self.llm.generate(system, user, max_tokens=max_tokens, on_token=self.on_token,
                                    temperature=self.cfg.model.temperature)
        except LLMError as e:
            self.stats.llm_errors.append(str(e))
            return None
        finally:
            self.stats.llm_calls += 1
            self.stats.llm_ms += (time.perf_counter() - t0) * 1000
        out = re.sub(r"^(?:compressed text|summary|output)\s*:\s*", "", out.strip(), flags=re.I)
        out = re.sub(r"<think>[\s\S]*?</think>", "", out).strip()
        return out or None

    def _chunk(self, text: str, max_tokens: int) -> list[str]:
        sents = split_sentences(text) or [text]
        chunks: list[str] = []
        cur: list[str] = []
        used = 0
        for s in sents:
            t = self.count(s)
            if cur and used + t > max_tokens:
                chunks.append(" ".join(cur))
                cur, used = [], 0
            cur.append(s)
            used += t
        if cur:
            chunks.append(" ".join(cur))
        return chunks
