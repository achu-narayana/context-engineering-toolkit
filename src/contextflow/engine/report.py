"""Context debugger data: what was kept, dropped, why, and what it cost."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class SectionTrace:
    id: int
    kind: str
    kept: bool
    reason: str
    score: float
    tokens_in: int
    tokens_out: int
    preview: str
    protected: list[str] = field(default_factory=list)


@dataclass
class Report:
    mode: str
    input_kind: str
    backend: str                      # "no-llm" | "llama_cpp:<model>" | "openai_compat:<model>"
    tokens_in: int
    tokens_out: int
    target_tokens: int
    latency_ms: float
    cpu_percent: float
    rss_mb: float
    would_have_cost_usd: float        # cost of the ORIGINAL at the configured price
    cost_after_usd: float
    saved_usd: float
    price_per_million: float
    tokenizer: str
    tokenizer_approximate: bool
    sections: list[SectionTrace] = field(default_factory=list)
    protected_total: int = 0
    protected_restored: list[str] = field(default_factory=list)
    redactions: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    stages_ms: dict[str, float] = field(default_factory=dict)
    memory_stored: int = 0

    # ---- derived ------------------------------------------------------
    @property
    def ratio(self) -> float:
        return round(self.tokens_out / self.tokens_in, 3) if self.tokens_in else 1.0

    @property
    def saved_pct(self) -> float:
        return round(100 * (1 - self.ratio), 1)

    @property
    def dropped(self) -> list[SectionTrace]:
        return [s for s in self.sections if not s.kept]

    @property
    def kept(self) -> list[SectionTrace]:
        return [s for s in self.sections if s.kept]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(ratio=self.ratio, saved_pct=self.saved_pct, tokens_saved=self.tokens_in - self.tokens_out)
        return d

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_text(self) -> str:
        lines = [
            f"mode={self.mode} kind={self.input_kind} backend={self.backend}",
            f"tokens: {self.tokens_in} -> {self.tokens_out} (target {self.target_tokens}, ratio {self.ratio}, saved {self.saved_pct}%)",
            f"latency: {self.latency_ms:.0f} ms | cpu {self.cpu_percent:.0f}% | rss {self.rss_mb:.0f} MB",
            f"would-have-cost: ${self.would_have_cost_usd:.5f} -> ${self.cost_after_usd:.5f} (saved ${self.saved_usd:.5f} at ${self.price_per_million}/M, {self.tokenizer}{' approx' if self.tokenizer_approximate else ''})",
            f"protected items: {self.protected_total} tracked, {len(self.protected_restored)} restored after compression",
        ]
        if self.redactions:
            lines.append("redactions: " + ", ".join(f"{k}={v}" for k, v in self.redactions.items() if v))
        if self.stages_ms:
            lines.append("stages: " + ", ".join(f"{k} {v:.0f}ms" for k, v in self.stages_ms.items()))
        lines.append(f"sections: {len(self.kept)} kept, {len(self.dropped)} dropped")
        for s in self.sections:
            flag = "KEEP" if s.kept else "DROP"
            lines.append(f"  [{flag}] #{s.id} {s.kind:9s} {s.tokens_in:5d}->{s.tokens_out:4d} score={s.score:.2f}  {s.reason}")
            lines.append(f"         {s.preview}")
        if self.warnings:
            lines.append("warnings: " + "; ".join(self.warnings))
        return "\n".join(lines)
