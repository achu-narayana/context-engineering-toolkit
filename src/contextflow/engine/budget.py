"""Token budget manager: turn a target into per-section allocations.

Rules:
  * The total budget is split across kept sections proportionally to their score.
  * Every kept section gets at least `min_tokens`, so nothing becomes a meaningless stub.
  * A section never gets more than it already has (compression never grows text).
  * Code and headings are allocated their full size (they are not rewritten).
  * Any slack left after those caps is redistributed to the sections that still need it.
"""
from __future__ import annotations

from dataclasses import dataclass

from .segment import Section


@dataclass
class BudgetConfig:
    min_tokens: int = 12
    reserve_fraction: float = 0.08     # kept back for headers such as "Decisions:" blocks


def target_for(mode: str, input_tokens: int, ratio: float, target_tokens: int | None) -> int:
    if target_tokens:
        return max(1, int(target_tokens))
    return max(1, int(input_tokens * ratio))


def allocate(sections: list[Section], target_tokens: int, cfg: BudgetConfig | None = None) -> dict[int, int]:
    """Set `section.budget` on kept sections; returns {section_id: budget}."""
    cfg = cfg or BudgetConfig()
    kept = [s for s in sections if s.keep]
    if not kept:
        return {}
    total = max(1, int(target_tokens * (1 - cfg.reserve_fraction)))

    fixed = [s for s in kept if s.kind in ("code", "heading")]
    flexible = [s for s in kept if s not in fixed]
    for s in fixed:
        s.budget = s.tokens
    remaining = max(0, total - sum(s.budget for s in fixed))

    if not flexible:
        return {s.id: s.budget for s in kept}

    weights = [max(0.05, s.score) for s in flexible]
    wsum = sum(weights)
    alloc = {s.id: max(cfg.min_tokens, int(remaining * w / wsum)) for s, w in zip(flexible, weights)}

    # Cap at current size and redistribute slack, a few rounds.
    for _ in range(4):
        slack = 0
        needy = []
        for s in flexible:
            if alloc[s.id] > s.tokens:
                slack += alloc[s.id] - s.tokens
                alloc[s.id] = s.tokens
            elif alloc[s.id] < s.tokens:
                needy.append(s)
        if slack <= 0 or not needy:
            break
        nw = sum(max(0.05, s.score) for s in needy)
        for s in needy:
            alloc[s.id] += int(slack * max(0.05, s.score) / nw)

    for s in flexible:
        s.budget = max(cfg.min_tokens if s.tokens >= cfg.min_tokens else s.tokens, min(alloc[s.id], s.tokens))
    return {s.id: s.budget for s in kept}
