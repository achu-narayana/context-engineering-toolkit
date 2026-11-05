"""Optional redaction of emails, phone numbers and API keys. Runs before anything else."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import RedactionConfig

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
PHONE_RE = re.compile(r"(?<![\w/.-])(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?![\w/.-])")
API_KEY_RES = [
    re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{16,}\b"),      # OpenAI / Anthropic style
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                           # AWS access key
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),                 # GitHub tokens
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),               # Slack tokens
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),                     # Google API keys
    re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password|passwd|pwd)\s*[:=]\s*['\"]?([A-Za-z0-9_\-./+=]{8,})['\"]?"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-.=]{16,}"),
    re.compile(r"\b[a-f0-9]{32,64}\b"),                            # long hex secrets / hashes
]


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)


def redact(text: str, cfg: RedactionConfig) -> RedactionResult:
    counts: dict[str, int] = {}
    if cfg.emails:
        text, n = EMAIL_RE.subn("[EMAIL]", text)
        counts["emails"] = n
    if cfg.phones:
        text, n = PHONE_RE.subn("[PHONE]", text)
        counts["phones"] = n
    if cfg.api_keys:
        total = 0
        for rx in API_KEY_RES:
            text, n = rx.subn("[REDACTED_KEY]", text)
            total += n
        counts["api_keys"] = total
    for pat in cfg.extra_patterns:
        try:
            text, n = re.subn(pat, "[REDACTED]", text)
            counts[pat] = n
        except re.error:
            continue
    return RedactionResult(text=text, counts=counts)
