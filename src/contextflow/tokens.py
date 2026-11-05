"""Local token counting and would-have-cost estimates.

Uses tiktoken when its BPE file is available (first use downloads it once and caches it;
after that it is fully offline). If it is not available, falls back to a deterministic
word/punctuation heuristic and flags the counts as approximate.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Callable

_lock = threading.Lock()
_encoders: dict[str, "TokenCounter"] = {}
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass
class TokenCounter:
    name: str
    count: Callable[[str], int]
    approximate: bool = False


def _approx_count(text: str) -> int:
    # Roughly 1.3 tokens per word-like unit on English text, similar to BPE tokenizers.
    units = _WORD_RE.findall(text)
    return int(len(units) * 1.3 + 0.5)


def get_counter(name: str = "o200k_base") -> TokenCounter:
    """Return a TokenCounter. Thread-safe and cached."""
    with _lock:
        if name in _encoders:
            return _encoders[name]
        counter: TokenCounter
        try:
            import tiktoken
            enc = tiktoken.get_encoding(name)
            counter = TokenCounter(name=name, count=lambda s: len(enc.encode(s, disallowed_special=())))
        except Exception:
            counter = TokenCounter(name="approx", count=_approx_count, approximate=True)
        _encoders[name] = counter
        return counter


def count_tokens(text: str, name: str = "o200k_base") -> int:
    return get_counter(name).count(text)


def would_have_cost(tokens: int, price_per_million: float) -> float:
    """USD that `tokens` input tokens would cost at `price_per_million`."""
    return round(tokens / 1_000_000 * price_per_million, 6)
