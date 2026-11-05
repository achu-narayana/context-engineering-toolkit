"""The pipeline: one function, every entry point.

compress(text, mode, goal, target_tokens, use_llm, redact, config, on_progress, on_token) -> CompressResult
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable

from ..config import Config, MODES
from ..tokens import get_counter, would_have_cost
from . import conversation as conv
from .budget import allocate, target_for
from .compress import Compressor
from .detect import detect_kind
from .llm import get_backend
from .protect import Protector, restorable, restore, verify
from .redact import redact as _redact
from .report import Report, SectionTrace
from .segment import Section, segment_prose
from .select import select_sections
from .tooloutput import summarize_json, summarize_logs

ProgressCallback = Callable[[str, str], None]
MAX_INPUT_CHARS = 400_000


@dataclass
class CompressResult:
    text: str
    report: Report
    original: str

    @property
    def ok(self) -> bool:
        return not any(w.startswith("error:") for w in self.report.warnings)


def _preview(s: str, n: int = 90) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def compress(
    text: str,
    mode: str = "balanced",
    goal: str | None = None,
    target_tokens: int | None = None,
    use_llm: bool | None = None,
    redact: bool | None = None,
    config: Config | None = None,
    on_progress: ProgressCallback | None = None,
    on_token: Callable[[str], None] | None = None,
    store_memory: bool | None = None,
    allow_download: bool = True,
) -> CompressResult:
    cfg = config or Config.load()
    mode = (mode or cfg.default_mode).lower().replace("-", "_")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if target_tokens is None and mode == "custom":
        target_tokens = cfg.target_tokens or None
    if mode == "custom" and not target_tokens:
        raise ValueError("custom mode needs target_tokens")

    def progress(stage: str, detail: str = "") -> None:
        if on_progress:
            on_progress(stage, detail)

    t_start = time.perf_counter()
    stages: dict[str, float] = {}
    warnings: list[str] = []
    counter = get_counter(cfg.cost_tokenizer)
    count = counter.count
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        proc.cpu_percent(None)
    except Exception:  # noqa: BLE001
        proc = None

    original = text or ""
    if not original.strip():
        rep = _empty_report(cfg, mode, counter, "error: empty selection, nothing to compress")
        return CompressResult(text=original, report=rep, original=original)
    if len(original) > MAX_INPUT_CHARS:
        warnings.append(f"input truncated to {MAX_INPUT_CHARS} characters for processing")
        original_proc = original[:MAX_INPUT_CHARS]
    else:
        original_proc = original

    # 0) redaction
    t = time.perf_counter()
    redactions: dict[str, int] = {}
    work = original_proc
    if redact if redact is not None else cfg.redaction.enabled:
        rr = _redact(work, cfg.redaction)
        work, redactions = rr.text, rr.counts
    stages["redact"] = (time.perf_counter() - t) * 1000

    # 1) detect
    t = time.perf_counter()
    kind = detect_kind(work)
    progress("detect", kind)
    tokens_in = count(work)
    ratio = cfg.ratio_for(mode)
    target = target_for(mode, tokens_in, ratio, target_tokens)
    stages["detect"] = (time.perf_counter() - t) * 1000

    # 2) LLM availability (never fatal)
    llm = None
    want_llm = cfg.model.backend != "none" if use_llm is None else use_llm
    if want_llm:
        try:
            llm = get_backend(cfg)
            if llm is None:
                warnings.append("no local model available; ran in no-LLM mode")
        except Exception as e:  # noqa: BLE001
            warnings.append(f"model backend failed to load ({e}); ran in no-LLM mode")
    backend = llm.name if llm else "no-llm"
    protector = Protector(cfg.protected_patterns)
    comp = Compressor(cfg, count, llm, protector, allow_download, on_token)

    # 3) tool output shortcut (rule-based)
    if kind in ("json", "log"):
        t = time.perf_counter()
        progress("compress", "structured summary")
        try:
            out = summarize_json(work, mode) if kind == "json" else summarize_logs(work, mode)
        except json.JSONDecodeError as e:
            warnings.append(f"JSON parse failed ({e}); treating as prose")
            kind = "prose"
            out = ""
        if kind in ("json", "log"):
            if llm is not None and count(out) > target * 1.3:
                progress("compress", "model pass over structured summary")
                items = protector.extract(out)
                better = comp._llm_compress(out, target, mode, goal, items)
                if better:
                    out = better
            stages["compress"] = (time.perf_counter() - t) * 1000
            rep = _finish(cfg, mode, kind, backend, counter, work, out, target, [], [], [], redactions, warnings, stages, t_start, proc, comp)
            return CompressResult(text=out, report=rep, original=original)

    # 4) segment
    t = time.perf_counter()
    facts = None
    if kind in ("chat", "email"):
        turns = conv.parse_chat(work) if kind == "chat" else conv.parse_email(work)
        if len(turns) < 2:
            kind = "prose"
            sections = segment_prose(work)
        else:
            sections = conv.turns_to_sections(turns, "turn" if kind == "chat" else "message")
            facts = conv.extract_facts(turns)
    else:
        sections = segment_prose(work)
    if not sections:
        sections = [Section(id=0, text=work.strip(), kind="paragraph")]
    progress("segment", f"{len(sections)} sections")
    stages["segment"] = (time.perf_counter() - t) * 1000

    # 5) protect
    t = time.perf_counter()
    all_items = protector.extract(work)
    for s in sections:
        s.tokens = count(s.text)
        s.protected = [it.text for it in all_items if it.text in s.text]
    progress("protect", f"{len(all_items)} protected items")
    stages["protect"] = (time.perf_counter() - t) * 1000

    # 6) select
    t = time.perf_counter()
    select_sections(sections, goal, target, mode, embedding_model=cfg.embedding_model, allow_download=allow_download)
    kept_n = sum(1 for s in sections if s.keep)
    progress("select", f"{kept_n}/{len(sections)} sections kept")
    stages["select"] = (time.perf_counter() - t) * 1000

    # 7) budget
    t = time.perf_counter()
    allocate(sections, target)
    progress("budget", f"{target} tokens target")
    stages["budget"] = (time.perf_counter() - t) * 1000

    # 8) compress
    t = time.perf_counter()
    progress("compress", backend)
    section_items = {s.id: [it for it in all_items if it.text in s.text] for s in sections}
    for s in sections:
        if s.keep:
            s.output = comp.compress_section(s, mode, goal, section_items[s.id])
            s.output_tokens = count(s.output)
    if facts is not None:
        latest = 1 if mode == "aggressive" else 2
        body = conv.render_thread(sections, facts, mode, kind, latest,
                                  lambda sec: comp.compress_section(sec, mode, goal, section_items[sec.id]) if not sec.output else sec.output,
                                  count, target)
    else:
        body = _assemble(sections)
    stages["compress"] = (time.perf_counter() - t) * 1000

    # 9) verify protected items from KEPT sections (dropped sections were dropped on purpose)
    t = time.perf_counter()
    kept_items = [it for it in all_items if any(it.text in s.text for s in sections if s.keep) and it.category != "name"]
    missing = [it for it in verify(kept_items, body) if restorable(it)]
    # critical items from dropped sections are never silently lost either (money, dates, URLs, instructions...)
    dropped_items = [it for it in all_items if restorable(it) and it.category not in ("number", "handle", "path")
                     and not any(it.text in s.text for s in sections if s.keep)]
    seen = {m.text for m in missing}
    missing += [it for it in verify(dropped_items, body) if it.text not in seen and (mode != "aggressive" or it.category in ("money", "instruction", "url"))]
    if missing:
        body = restore(body, missing)
    if mode == "prompt_ready":
        # build the structured prompt from the verified body, then make sure the rewrite lost nothing
        progress("compress", "building prompt")
        t2 = time.perf_counter()
        prompt_body = comp.prompt_ready(body, goal, all_items, target)
        again = [it for it in verify(kept_items + dropped_items, prompt_body) if restorable(it) and it.category != "number"]
        body = restore(prompt_body, again, header="Also preserve:") if again else prompt_body
        missing = list({m.text: m for m in missing + again}.values())
        stages["prompt"] = (time.perf_counter() - t2) * 1000
    progress("verify", f"{len(missing)} items restored")
    stages["verify"] = (time.perf_counter() - t) * 1000

    # 10) memory
    stored = 0
    if (cfg.memory_enabled if store_memory is None else store_memory):
        try:
            from .memory import MemoryStore, decide
            rows = [(s.text, _preview(s.text, 200), s.kind, kind) for s in sections if decide(s, ratio) == "store"]
            if rows:
                stored = MemoryStore(embedding_model=cfg.embedding_model).add_many(rows)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"memory store skipped: {e}")

    rep = _finish(cfg, mode, kind, backend, counter, work, body, target, sections, all_items, missing, redactions, warnings, stages, t_start, proc, comp)
    rep.memory_stored = stored
    return CompressResult(text=body, report=rep, original=original)


def _assemble(sections: list[Section]) -> str:
    """Join kept sections; a heading whose whole body was dropped is dropped too."""
    parts: list[str] = []
    for i, s in enumerate(sections):
        if not s.keep or not s.output.strip():
            continue
        if s.kind == "heading":
            level = len(s.text) - len(s.text.lstrip("#")) if s.text.startswith("#") else 2
            body_kept = False
            for nxt in sections[i + 1:]:
                if nxt.kind == "heading":
                    nxt_level = len(nxt.text) - len(nxt.text.lstrip("#")) if nxt.text.startswith("#") else 2
                    if nxt_level <= level:
                        break
                    continue          # a sub-heading: keep looking inside this heading's scope
                if nxt.keep and nxt.output.strip():
                    body_kept = True
                    break
            if not body_kept:
                s.keep = False
                s.reason = "heading with no remaining body"
                continue
        parts.append(s.output.strip())
    return "\n\n".join(parts)


def _finish(cfg, mode, kind, backend, counter, work, out, target, sections, items, missing, redactions, warnings, stages, t_start, proc, comp) -> Report:
    count = counter.count
    tokens_in = count(work)
    tokens_out = count(out)
    cpu = rss = 0.0
    if proc is not None:
        try:
            cpu = proc.cpu_percent(None)
            rss = proc.memory_info().rss / 1e6
        except Exception:  # noqa: BLE001
            pass
    if comp.stats.llm_errors:
        warnings.append("model errors: " + "; ".join(comp.stats.llm_errors[:3]))
    if comp.stats.llm_calls:
        stages["llm_total"] = comp.stats.llm_ms
    if tokens_out >= tokens_in:
        warnings.append("output is not shorter than input (nothing safe to remove)")
    cost_in = would_have_cost(tokens_in, cfg.price_per_million)
    cost_out = would_have_cost(tokens_out, cfg.price_per_million)
    return Report(
        mode=mode, input_kind=kind, backend=backend, tokens_in=tokens_in, tokens_out=tokens_out, target_tokens=target,
        latency_ms=(time.perf_counter() - t_start) * 1000, cpu_percent=cpu, rss_mb=rss,
        would_have_cost_usd=cost_in, cost_after_usd=cost_out, saved_usd=round(cost_in - cost_out, 6),
        price_per_million=cfg.price_per_million, tokenizer=counter.name, tokenizer_approximate=counter.approximate,
        sections=[SectionTrace(id=s.id, kind=s.kind, kept=s.keep, reason=s.reason, score=round(s.score, 3), tokens_in=s.tokens,
                               tokens_out=s.output_tokens if s.keep else 0, preview=_preview(s.text), protected=s.protected[:12]) for s in sections],
        protected_total=len(items), protected_restored=[m.text for m in missing], redactions=redactions, warnings=warnings,
        stages_ms={k: round(v, 1) for k, v in stages.items()},
    )


def _empty_report(cfg: Config, mode: str, counter, warning: str) -> Report:
    return Report(mode=mode, input_kind="empty", backend="none", tokens_in=0, tokens_out=0, target_tokens=0, latency_ms=0.0,
                  cpu_percent=0.0, rss_mb=0.0, would_have_cost_usd=0.0, cost_after_usd=0.0, saved_usd=0.0,
                  price_per_million=cfg.price_per_million, tokenizer=counter.name, tokenizer_approximate=counter.approximate,
                  warnings=[warning])
