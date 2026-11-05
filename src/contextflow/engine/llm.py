"""Local LLM backends. No hosted APIs, no keys.

* LlamaCppBackend   in-process llama.cpp via llama-cpp-python, GGUF model file.
* OpenAICompatBackend  any OpenAI-compatible server on localhost (llama-server, LM Studio, Jan, llamafile).

`get_backend(cfg)` returns None when nothing usable is available, and the engine then runs in no-LLM mode.
The only network call in this file is `download_model`, a one-time Hugging Face download the user triggers.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Protocol
from urllib.parse import urlparse

from ..config import Config, ModelConfig, home_dir

TokenCallback = Callable[[str], None]


class LLMError(RuntimeError):
    pass


class LLMBackend(Protocol):
    name: str

    def generate(self, system: str, user: str, max_tokens: int, on_token: TokenCallback | None = None,
                 temperature: float = 0.1, stop: list[str] | None = None) -> str: ...

    def count_tokens(self, text: str) -> int: ...

    def unload(self) -> None: ...


def models_dir() -> Path:
    p = home_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_model_path(mc: ModelConfig) -> Path | None:
    if mc.path:
        p = Path(mc.path).expanduser()
        return p if p.exists() else None
    p = models_dir() / mc.filename
    return p if p.exists() else None


def list_local_models() -> list[dict]:
    out = []
    for p in sorted(models_dir().glob("*.gguf")):
        out.append({"filename": p.name, "path": str(p), "size_mb": round(p.stat().st_size / 1e6, 1)})
    return out


# A curated list of small free models that run on CPU. Any .gguf works; these are just suggestions.
SUGGESTED_MODELS: list[dict] = [
    {"id": "qwen2.5-1.5b", "label": "Qwen2.5 1.5B Instruct (default, ~1.1 GB)", "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF", "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf", "size_mb": 1120},
    {"id": "qwen2.5-0.5b", "label": "Qwen2.5 0.5B Instruct (fastest, ~0.5 GB)", "repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf", "size_mb": 490},
    {"id": "qwen2.5-3b", "label": "Qwen2.5 3B Instruct (better, ~2.1 GB)", "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF", "filename": "qwen2.5-3b-instruct-q4_k_m.gguf", "size_mb": 2100},
    {"id": "llama-3.2-1b", "label": "Llama 3.2 1B Instruct (~0.8 GB)", "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF", "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "size_mb": 808},
    {"id": "llama-3.2-3b", "label": "Llama 3.2 3B Instruct (~2.0 GB)", "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF", "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf", "size_mb": 2020},
    {"id": "gemma-3-1b", "label": "Gemma 3 1B Instruct (~0.8 GB)", "repo_id": "unsloth/gemma-3-1b-it-GGUF", "filename": "gemma-3-1b-it-Q4_K_M.gguf", "size_mb": 806},
]


def download_model(repo_id: str, filename: str, progress: Callable[[int, int], None] | None = None) -> Path:
    """One-time model download from Hugging Face into ~/.contextflow/models."""
    from huggingface_hub import hf_hub_download
    target = models_dir()
    if progress:
        progress(0, 0)
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(target))
    p = Path(path)
    if progress:
        sz = p.stat().st_size
        progress(sz, sz)
    return p


# ---------------------------------------------------------------------------
class LlamaCppBackend:
    name = "llama_cpp"

    def __init__(self, model_path: Path, n_ctx: int = 4096, n_threads: int = 0, idle_unload_s: float = 300.0) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads or max(1, (os.cpu_count() or 4) - 1)
        self.idle_unload_s = idle_unload_s
        self._llm = None
        self._lock = threading.Lock()
        self._last_used = time.time()
        self.name = f"llama_cpp:{model_path.stem}"

    def _ensure(self):
        if self._llm is None:
            from llama_cpp import Llama
            self._llm = Llama(model_path=str(self.model_path), n_ctx=self.n_ctx, n_threads=self.n_threads,
                              n_batch=256, verbose=False, use_mlock=False)
            threading.Thread(target=self._idle_watch, daemon=True).start()
        self._last_used = time.time()
        return self._llm

    def _idle_watch(self) -> None:
        while self._llm is not None:
            time.sleep(15)
            if self.idle_unload_s > 0 and time.time() - self._last_used > self.idle_unload_s:
                self.unload()

    def unload(self) -> None:
        with self._lock:
            self._llm = None

    def count_tokens(self, text: str) -> int:
        llm = self._ensure()
        return len(llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))

    def generate(self, system: str, user: str, max_tokens: int, on_token: TokenCallback | None = None,
                 temperature: float = 0.1, stop: list[str] | None = None) -> str:
        with self._lock:
            llm = self._ensure()
            try:
                out: list[str] = []
                stream = llm.create_chat_completion(
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    max_tokens=max_tokens, temperature=temperature, top_p=0.9, repeat_penalty=1.1,
                    stop=stop or [], stream=True,
                )
                for chunk in stream:
                    delta = chunk["choices"][0].get("delta", {}).get("content")
                    if delta:
                        out.append(delta)
                        if on_token:
                            on_token(delta)
                self._last_used = time.time()
                return "".join(out).strip()
            except Exception as e:  # noqa: BLE001
                raise LLMError(f"llama.cpp generation failed: {e}") from e


# ---------------------------------------------------------------------------
class OpenAICompatBackend:
    """Talks to a local OpenAI-compatible /v1/chat/completions endpoint. Localhost only."""

    name = "openai_compat"

    def __init__(self, base_url: str, model: str = "", timeout_s: float = 120.0) -> None:
        host = urlparse(base_url).hostname or ""
        if host not in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
            raise LLMError(f"refusing non-local server {base_url}; only localhost is allowed")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.name = f"openai_compat:{model or 'default'}"

    def unload(self) -> None:
        return None

    def count_tokens(self, text: str) -> int:
        from ..tokens import count_tokens
        return count_tokens(text)

    def _models(self) -> list[str]:
        import httpx
        r = httpx.get(f"{self.base_url}/models", timeout=5)
        r.raise_for_status()
        return [m.get("id", "") for m in r.json().get("data", [])]

    def generate(self, system: str, user: str, max_tokens: int, on_token: TokenCallback | None = None,
                 temperature: float = 0.1, stop: list[str] | None = None) -> str:
        import httpx
        model = self.model
        if not model:
            try:
                ms = self._models()
                model = ms[0] if ms else "default"
            except Exception:
                model = "default"
        payload = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "max_tokens": max_tokens, "temperature": temperature, "stream": True}
        if stop:
            payload["stop"] = stop
        out: list[str] = []
        try:
            with httpx.stream("POST", f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout_s) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj["choices"][0].get("delta", {}).get("content")
                    except Exception:
                        continue
                    if delta:
                        out.append(delta)
                        if on_token:
                            on_token(delta)
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"local server request failed: {e}") from e
        return "".join(out).strip()


# ---------------------------------------------------------------------------
@dataclass
class Availability:
    available: bool
    backend: str
    detail: str
    model_path: str = ""


def check_availability(cfg: Config) -> Availability:
    mc = cfg.model
    if mc.backend == "none":
        return Availability(False, "none", "LLM disabled in config")
    if mc.backend in ("auto", "llama_cpp"):
        try:
            import llama_cpp  # noqa: F401
            p = resolve_model_path(mc)
            if p:
                return Availability(True, "llama_cpp", f"model {p.name}", str(p))
            if mc.backend == "llama_cpp":
                return Availability(False, "llama_cpp", f"model file not found: {mc.filename}. Run `contextflow setup` to download it.")
        except Exception as e:  # noqa: BLE001
            if mc.backend == "llama_cpp":
                return Availability(False, "llama_cpp", f"llama-cpp-python not importable: {e}")
    if mc.backend in ("auto", "openai_compat"):
        try:
            import httpx
            r = httpx.get(mc.server_url.rstrip("/") + "/models", timeout=1.5)
            if r.status_code == 200:
                ids = [m.get("id", "") for m in r.json().get("data", [])]
                return Availability(True, "openai_compat", f"server {mc.server_url} models: {', '.join(ids[:3]) or 'unknown'}")
        except Exception:
            pass
        if mc.backend == "openai_compat":
            return Availability(False, "openai_compat", f"no server at {mc.server_url}")
    p = resolve_model_path(mc)
    if not p:
        return Availability(False, "none", f"no model file ({mc.filename}) and no local server; run `contextflow setup`")
    return Availability(False, "none", "no usable backend")


_backend_cache: dict[str, LLMBackend] = {}


def get_backend(cfg: Config) -> LLMBackend | None:
    av = check_availability(cfg)
    if not av.available:
        return None
    key = f"{av.backend}:{av.model_path}:{cfg.model.server_url}:{cfg.model.n_ctx}"
    if key in _backend_cache:
        return _backend_cache[key]
    if av.backend == "llama_cpp":
        b: LLMBackend = LlamaCppBackend(Path(av.model_path), n_ctx=cfg.model.n_ctx, n_threads=cfg.model.n_threads, idle_unload_s=cfg.model.idle_unload_s)
    else:
        b = OpenAICompatBackend(cfg.model.server_url, cfg.model.server_model, cfg.model.timeout_s)
    _backend_cache[key] = b
    return b


def unload_all() -> None:
    for b in _backend_cache.values():
        b.unload()
