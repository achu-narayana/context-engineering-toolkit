# ContextFlow - Stage 1 Plan

## Machine facts (checked 2026-11-05)
- Windows 11, Ryzen 5 7530U (6 cores, integrated Radeon, no CUDA), 7.3 GB RAM -> CPU-only inference, model must be <= ~2 GB.
- Python 3.13.3. Already installed: sentence-transformers 5.5.1, torch 2.12 (CPU), tiktoken, fastapi, scikit-learn, nltk, numpy.
- all-MiniLM-L6-v2 already in the Hugging Face cache (no download needed).
- Ollama is NOT used (user decision). Runtime = llama.cpp via llama-cpp-python 0.3.36; a prebuilt CPU wheel installs on Python 3.13 with no compiler (verified with pip --dry-run against the abetlen CPU wheel index).

## Design principles taken from the references
- Anthropic: context is a finite "attention budget"; keep the smallest set of high-signal tokens; compaction = keep decisions/open issues/constraints, drop redundant tool output; tune compaction prompt for recall first, then precision; structured notes outside the window; just-in-time retrieval by identifiers.
- LangChain: four moves — Write (scratchpad/memory), Select (embedding retrieval), Compress (summarize, trim, heuristics on tool outputs), Isolate. The engine's pipeline maps to Select -> Compress, memory manager to Write/Select.
- Ollama API docs were read for design only (streaming chunks, num_ctx/num_predict/temperature, JSON-schema output, eval_count/prompt_eval_count timings). The same ideas are implemented on llama.cpp: create_chat_completion(stream=True), response_format json_schema / GBNF grammar, usage.prompt_tokens/completion_tokens, and llama.cpp's own tokenizer for exact local counts.
- SBERT: model.encode + model.similarity; all-MiniLM-L6-v2 is ~5x faster than mpnet with good quality.

## Architecture
  selection / stdin / HTTP / extension
            |
   contextflow.engine.pipeline.compress(text, mode, goal, target_tokens)
            |
   1 Detector        -> kind: prose | chat | email | json | log | code
   2 Protector       -> extract protected spans (numbers, dates, names, URLs, code, instructions) -> placeholders
   3 Segmenter       -> sections / turns / records
   4 Selector        -> embeddings (MiniLM) rank vs goal + centrality (TextRank), drop low value
   5 Budget          -> allocate target tokens across kept sections by priority
   6 Compressor      -> Light: rules only. Balanced/Aggressive/Prompt-ready: extractive pass, then local LLM
                        (map-reduce over chunks, short structured prompts, streamed). No-LLM: extractive only.
   7 Verifier        -> every protected item present? if not, re-insert
   8 Report          -> tokens before/after, % saved, dropped sections + reasons, latency, CPU/RAM, would-have-cost
            |
   result -> overlay / CLI / API / memory store (SQLite + embeddings)

## Folder structure
contextflow/
  pyproject.toml, README.md, config.example.toml
  src/contextflow/
    __init__.py            # from contextflow import compress
    config.py              # TOML config, defaults, protected patterns, redaction rules
    tokens.py              # tiktoken counting + would-have-cost
    engine/
      detect.py segment.py protect.py select.py budget.py extractive.py
      llm.py               # LLM backends: llama-cpp-python (in-process GGUF) + OpenAI-compatible localhost server; model download, streaming, timeouts
      compress.py          # modes, map-reduce, verifier
      conversation.py      # chat / email threads
      tooloutput.py        # JSON / logs / API responses (rule-based)
      memory.py            # SQLite + embeddings, keep/summarize/remove/store, recall
      redact.py report.py pipeline.py
    cli.py                 # contextflow compress|recall|setup|debug|bench
    api/server.py          # FastAPI on 127.0.0.1 only
    app/                   # desktop: hotkey.py clipboard.py overlay.py tray.py undo.py main.py
  extension/               # Chrome MV3: manifest.json background.js content.js
  data/samples/            # document, chat thread, email chain, api_response.json, questions.json
  bench/benchmark.py
  tests/                   # test_select test_budget test_compress test_protect test_tooloutput ...

## Library choices
| Need | Choice | Why / tradeoff |
|---|---|---|
| LLM | llama-cpp-python (in-process, GGUF) as the primary backend | pip-installable prebuilt CPU wheel, no daemon, no server, streaming, JSON-schema/grammar output, exact token counts. Second backend: any OpenAI-compatible localhost server (llama-server.exe, LM Studio, Jan, llamafile) selectable in settings for people who already run one |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 (already cached) | 22M params, fast on CPU; any GGUF embedding model via llama.cpp as option |
| Extractive | own TextRank on embedding similarity (numpy/sklearn) | no extra deps, uses same embeddings |
| Tokens | tiktoken o200k_base/cl100k_base for would-have-cost; llama.cpp model.tokenize() for the local model's real counts | one-time BPE download, then offline |
| Fuzzy scoring | rapidfuzz | fast, local |
| API | fastapi + uvicorn | already installed |
| CLI | typer or argparse | small |
| Global hotkey | pynput | works Win + macOS (needs Accessibility permission on mac). `keyboard` lib needs root on macOS; `global-hotkeys` is Windows-only. |
| Clipboard | pyperclip for text; pywin32 on Windows / pyobjc NSPasteboard on macOS to save & restore non-text formats | pyperclip alone cannot restore images/files |
| Copy/paste simulation | pynput Controller (Ctrl+C / Cmd+C) | cross-platform |
| Overlay | Tkinter (stdlib): frameless, always-on-top, near cursor, streams text | zero deps, fine on both OS. PySide6 is prettier but ~200 MB; customtkinter optional skin later |
| Tray | pystray + Pillow | Win/mac/Linux |
| Settings/undo | TOML config + JSON undo file (last 20) | simple |

## Model choice (CPU, 7.3 GB RAM) — GGUF files from Hugging Face, downloaded once with huggingface_hub
Default: Qwen2.5-1.5B-Instruct, Q4_K_M (official repo Qwen/Qwen2.5-1.5B-Instruct-GGUF, ~1.1 GB, Apache-2.0, 32K ctx). Best balance of instruction following, JSON reliability and speed on this CPU (expect ~15-25 tok/s, ~1.5 GB RAM at 4K ctx).
Options in settings (any local .gguf path or HF repo+file is accepted):
- Qwen2.5-0.5B-Instruct Q4_K_M (~0.4 GB): fastest, noticeably weaker.
- Llama-3.2-1B-Instruct Q4_K_M (~0.8 GB): fast, decent prose; Llama licence.
- Qwen3-1.7B Q4_K_M (~1.1 GB, Qwen/Qwen3-1.7B-GGUF): better quality; thinking disabled via "/no_think" / chat-template flag.
- Qwen3-4B / Gemma-3-4B-it Q4_K_M (~2.5 GB): quality mode, slow on this laptop (~5-8 tok/s), fine on a GPU box.
Model manager: lists installed .gguf files, downloads the chosen one with progress, verifies size, loads lazily and unloads on idle to free RAM.
Embedding model for recall: MiniLM default.

## How it works per OS
Windows: pynput global listener; copy via Ctrl+C; pywin32 snapshots clipboard formats, restores after; Tk overlay topmost at cursor; pystray tray; run as `contextflow app` or a startup shortcut.
macOS: same code; Cmd+C / Cmd+Shift+K; user must grant Accessibility (hotkey + paste) and Input Monitoring in System Settings; overlay via Tk (ships with python.org build); tray via pystray (AppKit). Secure-input fields (passwords) block simulated keys -> graceful "no selection" message.
Linux (optional): X11 works with pynput; Wayland does not.

## Safety / privacy
127.0.0.1 only; no outbound calls except the one-time model/BPE download from Hugging Face; no text logging unless debug=true; always preview; always restore clipboard; undo history of 20; redaction option; on any error original text untouched.

## Stages
1 plan (this) -> 2 engine no-LLM + CLI -> 3 llama.cpp compression -> 4 tests + benchmark -> 5 API -> 6 desktop app -> 7 Chrome extension -> 8 README.
Each stage: run it, report what works and what doesn't.

llama.cpp fallback behaviour: if the wheel fails to import or no .gguf is present, the engine runs in no-LLM mode and the setup command explains what to download.
