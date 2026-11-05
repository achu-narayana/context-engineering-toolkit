"""Configuration: defaults, TOML loading and saving.

The config file lives at ~/.contextflow/config.toml (override with CONTEXTFLOW_HOME).
Nothing here talks to the network.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


class Mode(str, Enum):
    LIGHT = "light"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    PROMPT_READY = "prompt_ready"
    CUSTOM = "custom"


MODES = [m.value for m in Mode]


def home_dir() -> Path:
    p = Path(os.environ.get("CONTEXTFLOW_HOME", Path.home() / ".contextflow"))
    p.mkdir(parents=True, exist_ok=True)
    return p


# Target output size as a fraction of input tokens, per mode (custom uses target_tokens).
DEFAULT_RATIOS: dict[str, float] = {
    "light": 0.80,
    "balanced": 0.45,
    "aggressive": 0.25,
    "prompt_ready": 0.45,
}


@dataclass
class ModelConfig:
    backend: str = "auto"            # auto | llama_cpp | openai_compat | none
    repo_id: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    filename: str = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    path: str = ""                   # explicit local .gguf path (wins over repo/filename)
    server_url: str = "http://127.0.0.1:8080/v1"  # for openai_compat backend
    server_model: str = ""           # model name to send to the server (optional)
    n_ctx: int = 4096
    n_threads: int = 0               # 0 = auto
    temperature: float = 0.1
    max_chunk_tokens: int = 900      # map-reduce chunk size fed to the model
    timeout_s: float = 120.0
    idle_unload_s: float = 300.0     # free RAM after this many idle seconds


@dataclass
class RedactionConfig:
    enabled: bool = False
    emails: bool = True
    phones: bool = True
    api_keys: bool = True
    extra_patterns: list[str] = field(default_factory=list)


@dataclass
class Config:
    hotkey: str = "<ctrl>+<shift>+k"
    default_mode: str = "balanced"
    target_tokens: int = 0           # used by custom mode; 0 = use ratios
    ratios: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RATIOS))
    price_per_million: float = 3.0   # USD per 1M input tokens, for the would-have-cost estimate
    cost_tokenizer: str = "o200k_base"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    protected_patterns: list[str] = field(default_factory=list)  # extra regexes
    debug_logging: bool = False
    api_host: str = "127.0.0.1"
    api_port: int = 8765
    undo_history: int = 20
    memory_enabled: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)
    redaction: RedactionConfig = field(default_factory=RedactionConfig)

    # ---- persistence -------------------------------------------------
    @staticmethod
    def path() -> Path:
        return home_dir() / "config.toml"

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or cls.path()
        cfg = cls()
        if path.exists():
            with open(path, "rb") as f:
                data = tomllib.load(f)
            cfg = cls.from_dict(data)
        return cfg

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        cfg = cls()
        for k, v in data.items():
            if k == "model" and isinstance(v, dict):
                cfg.model = ModelConfig(**{kk: vv for kk, vv in v.items() if hasattr(ModelConfig, kk)})
            elif k == "redaction" and isinstance(v, dict):
                cfg.redaction = RedactionConfig(**{kk: vv for kk, vv in v.items() if hasattr(RedactionConfig, kk)})
            elif k == "ratios" and isinstance(v, dict):
                cfg.ratios.update({kk: float(vv) for kk, vv in v.items()})
            elif hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path | None = None) -> Path:
        import tomli_w
        path = path or self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            tomli_w.dump(self.to_dict(), f)
        return path

    def ratio_for(self, mode: str) -> float:
        return float(self.ratios.get(mode, DEFAULT_RATIOS.get(mode, 0.45)))
