"""ContextFlow: local, free context compression engine.

Public API:

    from contextflow import compress
    result = compress(text, mode="balanced", goal="what was decided?")
    print(result.text)
    print(result.report.to_text())
"""
from __future__ import annotations

from .engine.pipeline import compress, CompressResult
from .engine.report import Report
from .config import Config, Mode

__all__ = ["compress", "CompressResult", "Report", "Config", "Mode"]
__version__ = "0.1.0"
