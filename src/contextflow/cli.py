"""ContextFlow CLI.

  contextflow compress < file.txt
  contextflow compress file.txt --mode aggressive --goal "what was decided" --debug
  contextflow recall "pricing decision"
  contextflow setup            # check embeddings / model, offer the one-time model download
  contextflow models           # list local .gguf files and suggestions
  contextflow serve            # local API + web UI on 127.0.0.1:8765
  contextflow app              # desktop hotkey app
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .config import Config, MODES

app = typer.Typer(add_completion=False, help="Local, free context compression.", no_args_is_help=True)


def _stdio_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name)
        try:
            s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass


def _read_input(file: Optional[Path]) -> str:
    if file is not None:
        return file.read_text(encoding="utf-8", errors="replace")
    if sys.stdin.isatty():
        typer.echo("Reading from stdin... (paste text, then Ctrl+Z/Ctrl+D)", err=True)
    data = sys.stdin.buffer.read()
    return data.decode("utf-8", errors="replace")


@app.command()
def compress(
    file: Optional[Path] = typer.Argument(None, help="Input file (default: stdin)"),
    mode: str = typer.Option(None, "--mode", "-m", help=f"One of {', '.join(MODES)}"),
    goal: Optional[str] = typer.Option(None, "--goal", "-g", help="Optional goal/question to focus on"),
    target: Optional[int] = typer.Option(None, "--target", "-t", help="Target token count (implies custom mode if no mode given)"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Classical methods only"),
    redact: bool = typer.Option(False, "--redact", help="Redact emails, phones and API keys first"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Print the context debugger report to stderr"),
    as_json: bool = typer.Option(False, "--json", help="Print JSON {text, report} to stdout"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write compressed text to a file"),
) -> None:
    """Compress text from a file or stdin."""
    _stdio_utf8()
    from .engine.pipeline import compress as _compress

    cfg = Config.load()
    text = _read_input(file)
    if target and not mode:
        mode = "custom"
    mode = mode or cfg.default_mode

    def progress(stage: str, detail: str) -> None:
        if debug:
            typer.echo(f"[{stage}] {detail}", err=True)

    try:
        res = _compress(text, mode=mode, goal=goal, target_tokens=target, use_llm=not no_llm, redact=redact, config=cfg, on_progress=progress)
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)
    if as_json:
        typer.echo(json.dumps({"text": res.text, "report": res.report.to_dict()}, ensure_ascii=False, indent=2))
    else:
        if output:
            output.write_text(res.text, encoding="utf-8")
        else:
            typer.echo(res.text)
    if debug or not res.ok:
        typer.echo("\n" + res.report.to_text(), err=True)
    if not res.ok:
        raise typer.Exit(1)


@app.command()
def recall(query: str = typer.Argument(..., help="What to look for in local memory"), k: int = typer.Option(5, "--k")) -> None:
    """Search the local long-term memory (SQLite + embeddings)."""
    _stdio_utf8()
    from .engine.memory import MemoryStore
    cfg = Config.load()
    store = MemoryStore(embedding_model=cfg.embedding_model)
    hits = store.recall(query, k)
    if not hits:
        typer.echo(f"no memories ({store.count()} items stored)")
        return
    for h in hits:
        typer.echo(f"[{h.score:.2f}] ({h.kind} from {h.source}) {h.summary or h.text[:200]}")


@app.command()
def setup(
    download: bool = typer.Option(False, "--download", help="Download the configured model now (one-time, from Hugging Face)"),
    model: Optional[str] = typer.Option(None, "--model", help="Suggested model id (see `contextflow models`) to set as default"),
) -> None:
    """First-run check: embeddings, tokenizer, llama.cpp wheel, model file."""
    _stdio_utf8()
    from .engine import embeddings, llm
    from .tokens import get_counter

    cfg = Config.load()
    if model:
        sug = next((m for m in llm.SUGGESTED_MODELS if m["id"] == model), None)
        if not sug:
            typer.echo(f"unknown model id {model}; run `contextflow models`", err=True)
            raise typer.Exit(2)
        cfg.model.repo_id, cfg.model.filename = sug["repo_id"], sug["filename"]
        cfg.save()
        typer.echo(f"default model set to {sug['label']}")
    typer.echo(f"ContextFlow {__version__}  config: {Config.path()}")
    c = get_counter(cfg.cost_tokenizer)
    typer.echo(f"tokenizer: {c.name}{' (approximate fallback)' if c.approximate else ''}")
    m = embeddings.get_model(cfg.embedding_model, allow_download=True)
    typer.echo(f"embeddings: {'ok ' + cfg.embedding_model if m is not None else 'FALLBACK tf-idf (sentence-transformers model not available)'}")
    try:
        import llama_cpp  # noqa: F401
        typer.echo("llama.cpp: installed")
    except Exception as e:  # noqa: BLE001
        typer.echo(f"llama.cpp: NOT installed ({e}). Install: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu")
    av = llm.check_availability(cfg)
    typer.echo(f"model: {'ready' if av.available else 'not available'} - {av.detail}")
    if not av.available and av.backend != "openai_compat":
        typer.echo(f"\nDefault model: {cfg.model.repo_id} / {cfg.model.filename}")
        if download or typer.confirm("Download it now (one-time, ~1 GB)?", default=False):
            def prog(done: int, total: int) -> None:
                typer.echo("downloading..." if not total else f"done: {done / 1e6:.0f} MB", err=True)
            p = llm.download_model(cfg.model.repo_id, cfg.model.filename, prog)
            typer.echo(f"saved to {p}")
        else:
            typer.echo("Skipped. ContextFlow works in no-LLM mode until a model is present.")
    if not cfg.path().exists():
        cfg.save()
        typer.echo(f"wrote default config to {cfg.path()}")


@app.command()
def models() -> None:
    """List local .gguf models and suggested downloads."""
    _stdio_utf8()
    from .engine import llm
    cfg = Config.load()
    typer.echo(f"models dir: {llm.models_dir()}")
    local = llm.list_local_models()
    typer.echo("local:" if local else "local: (none)")
    for m in local:
        mark = " *" if m["filename"] == cfg.model.filename or m["path"] == cfg.model.path else ""
        typer.echo(f"  {m['filename']}  {m['size_mb']} MB{mark}")
    typer.echo("suggested (contextflow setup --model <id> --download):")
    for s in llm.SUGGESTED_MODELS:
        typer.echo(f"  {s['id']:14s} {s['label']}")


@app.command()
def serve(host: Optional[str] = typer.Option(None), port: Optional[int] = typer.Option(None), reload: bool = typer.Option(False)) -> None:
    """Run the local HTTP API and web UI (127.0.0.1 only)."""
    _stdio_utf8()
    import uvicorn
    cfg = Config.load()
    host = host or cfg.api_host
    if host not in ("127.0.0.1", "localhost", "::1"):
        typer.echo("refusing to bind to a non-local address", err=True)
        raise typer.Exit(2)
    port = port or cfg.api_port
    typer.echo(f"ContextFlow UI: http://{host}:{port}/   API: POST http://{host}:{port}/compress")
    uvicorn.run("contextflow.api.server:app", host=host, port=port, reload=reload, log_level="warning")


@app.command("app")
def desktop_app() -> None:
    """Run the desktop hotkey app (tray icon + overlay)."""
    _stdio_utf8()
    from .app.main import run
    run()


@app.command()
def bench(
    out: Optional[Path] = typer.Option(None, "--out", help="Write results JSON/markdown here"),
    no_llm_judge: bool = typer.Option(False, "--no-judge"),
) -> None:
    """Run the benchmark (original vs no-LLM vs LLM compression)."""
    _stdio_utf8()
    from bench.benchmark import main as bench_main  # type: ignore
    bench_main(out=out, judge=not no_llm_judge)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
