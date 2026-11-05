"""FastAPI server: POST /compress (JSON or NDJSON stream), status, models, samples, recall, and the web UI.

Everything is served from 127.0.0.1. There is no auth because nothing is reachable from outside the machine;
a loopback-only middleware rejects any request whose client address is not local.
"""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..config import Config, MODES
from ..engine import embeddings, llm
from ..engine.pipeline import compress as _compress

STATIC = Path(__file__).parent / "static"
SAMPLES = Path(__file__).resolve().parents[3] / "data" / "samples"

app = FastAPI(title="ContextFlow", version=__version__, docs_url="/docs")


@app.on_event("startup")
def _warm() -> None:
    """Load the embedding backend in the background so the first request is fast."""
    cfg = Config.load()
    threading.Thread(target=lambda: embeddings.warmup(cfg.embedding_model), daemon=True).start()


@app.middleware("http")
async def loopback_only(request: Request, call_next):
    host = request.client.host if request.client else ""
    if host not in ("127.0.0.1", "::1", "localhost", "testclient"):
        return JSONResponse({"error": "local requests only"}, status_code=403)
    return await call_next(request)


class CompressRequest(BaseModel):
    text: str
    mode: str = Field("balanced", description=f"one of {MODES}")
    goal: Optional[str] = None
    target_tokens: Optional[int] = None
    use_llm: Optional[bool] = None
    redact: Optional[bool] = None
    stream: bool = False


class RecallRequest(BaseModel):
    query: str
    k: int = 5


def _cfg() -> Config:
    return Config.load()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/status")
def status() -> dict[str, Any]:
    cfg = _cfg()
    av = llm.check_availability(cfg)
    try:
        from ..engine.memory import MemoryStore
        mem = MemoryStore(embedding_model=cfg.embedding_model).count()
    except Exception:  # noqa: BLE001
        mem = 0
    return {
        "version": __version__,
        "llm_available": av.available,
        "llm_backend": av.backend,
        "llm_detail": av.detail,
        "model": cfg.model.filename if cfg.model.backend != "openai_compat" else cfg.model.server_model or cfg.model.server_url,
        "embeddings_backend": embeddings.backend_name() or "not loaded yet",
        "embedding_model": cfg.embedding_model,
        "default_mode": cfg.default_mode,
        "price_per_million": cfg.price_per_million,
        "memory_items": mem,
        "modes": MODES,
    }


@app.post("/compress")
def compress_endpoint(req: CompressRequest):
    if req.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    if req.mode == "custom" and not req.target_tokens:
        raise HTTPException(400, "custom mode requires target_tokens")
    cfg = _cfg()
    if not req.stream:
        try:
            res = _compress(req.text, mode=req.mode, goal=req.goal, target_tokens=req.target_tokens, use_llm=req.use_llm, redact=req.redact, config=cfg)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"text": res.text, "report": res.report.to_dict()}

    q: "queue.Queue[dict[str, Any] | None]" = queue.Queue()

    def on_progress(stage: str, detail: str) -> None:
        q.put({"event": "progress", "stage": stage, "detail": detail})

    def on_token(tok: str) -> None:
        q.put({"event": "token", "text": tok})

    def worker() -> None:
        try:
            res = _compress(req.text, mode=req.mode, goal=req.goal, target_tokens=req.target_tokens, use_llm=req.use_llm,
                            redact=req.redact, config=cfg, on_progress=on_progress, on_token=on_token)
            q.put({"event": "result", "text": res.text, "report": res.report.to_dict()})
        except Exception as e:  # noqa: BLE001
            q.put({"event": "error", "message": str(e)})
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        while True:
            item = q.get()
            if item is None:
                break
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/recall")
def recall(req: RecallRequest) -> dict[str, Any]:
    from ..engine.memory import MemoryStore
    cfg = _cfg()
    store = MemoryStore(embedding_model=cfg.embedding_model)
    hits = store.recall(req.query, req.k)
    return {"count": store.count(), "hits": [h.__dict__ for h in hits]}


@app.get("/models")
def models() -> dict[str, Any]:
    cfg = _cfg()
    return {"local": llm.list_local_models(), "suggested": llm.SUGGESTED_MODELS, "current": cfg.model.filename, "backend": cfg.model.backend}


class ModelSelect(BaseModel):
    filename: Optional[str] = None
    suggested_id: Optional[str] = None
    download: bool = False


@app.post("/models/select")
def select_model(req: ModelSelect):
    cfg = _cfg()
    if req.suggested_id:
        sug = next((m for m in llm.SUGGESTED_MODELS if m["id"] == req.suggested_id), None)
        if not sug:
            raise HTTPException(404, "unknown suggested model")
        cfg.model.repo_id, cfg.model.filename, cfg.model.path = sug["repo_id"], sug["filename"], ""
    elif req.filename:
        cfg.model.filename, cfg.model.path = req.filename, ""
    cfg.save()
    if req.download and not llm.resolve_model_path(cfg.model):
        def gen():
            yield json.dumps({"event": "progress", "detail": f"downloading {cfg.model.filename} from Hugging Face (one-time)"}) + "\n"
            try:
                p = llm.download_model(cfg.model.repo_id, cfg.model.filename)
                yield json.dumps({"event": "done", "path": str(p)}) + "\n"
            except Exception as e:  # noqa: BLE001
                yield json.dumps({"event": "error", "message": str(e)}) + "\n"
        return StreamingResponse(gen(), media_type="application/x-ndjson")
    return {"ok": True, "model": cfg.model.filename, "available": llm.check_availability(cfg).available}


@app.get("/samples")
def samples() -> list[dict[str, Any]]:
    out = []
    if SAMPLES.exists():
        for p in sorted(SAMPLES.iterdir()):
            if p.suffix in (".txt", ".md", ".json", ".log") and p.name != "questions.json":
                out.append({"name": p.name, "size": p.stat().st_size})
    return out


@app.get("/samples/{name}")
def sample(name: str) -> dict[str, str]:
    p = SAMPLES / Path(name).name
    if not p.exists() or p.name == "questions.json":
        raise HTTPException(404, "no such sample")
    return {"name": p.name, "text": p.read_text(encoding="utf-8")}


class SettingsUpdate(BaseModel):
    default_mode: Optional[str] = None
    price_per_million: Optional[float] = None
    redaction_enabled: Optional[bool] = None
    memory_enabled: Optional[bool] = None
    hotkey: Optional[str] = None


@app.get("/settings")
def get_settings() -> dict[str, Any]:
    return _cfg().to_dict()


@app.post("/settings")
def update_settings(req: SettingsUpdate) -> dict[str, Any]:
    cfg = _cfg()
    if req.default_mode:
        if req.default_mode not in MODES:
            raise HTTPException(400, "bad mode")
        cfg.default_mode = req.default_mode
    if req.price_per_million is not None:
        cfg.price_per_million = float(req.price_per_million)
    if req.redaction_enabled is not None:
        cfg.redaction.enabled = req.redaction_enabled
    if req.memory_enabled is not None:
        cfg.memory_enabled = req.memory_enabled
    if req.hotkey:
        cfg.hotkey = req.hotkey
    cfg.save()
    return cfg.to_dict()


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
