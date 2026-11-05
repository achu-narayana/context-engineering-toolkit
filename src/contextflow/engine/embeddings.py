"""Local embeddings with graceful fallbacks, fastest path first.

1. ONNX Runtime + `tokenizers` running the MiniLM export (starts in ~1 s; no torch import).
2. sentence-transformers (same model; slow to import on small machines but always correct).
3. TF-IDF from scikit-learn when neither model is available offline.

Either way the rest of the engine only sees unit-normalised float32 rows.
The only network call is the one-time download of the model files into the Hugging Face cache.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Iterable

import numpy as np

_lock = threading.Lock()
_backend = "none"
_onnx = None           # (session, tokenizer, input_names)
_st_model = None
_loaded_name = ""


def backend_name() -> str:
    return _backend


# ---------------------------------------------------------------------------
def _hf_file(repo: str, filename: str, allow_download: bool) -> Path | None:
    from huggingface_hub import hf_hub_download
    try:
        return Path(hf_hub_download(repo, filename, local_files_only=True))
    except Exception:
        if not allow_download:
            return None
        try:
            return Path(hf_hub_download(repo, filename))
        except Exception:
            return None


def _load_onnx(model_name: str, allow_download: bool) -> bool:
    global _onnx, _backend, _loaded_name
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except Exception:
        return False
    model_file = _hf_file(model_name, "onnx/model.onnx", allow_download)
    tok_file = _hf_file(model_name, "tokenizer.json", allow_download)
    if not model_file or not tok_file:
        return False
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 4) - 1))
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(model_file), so, providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(str(tok_file))
    tok.enable_truncation(max_length=256)
    tok.enable_padding(pad_id=0, pad_token="[PAD]")
    _onnx = (sess, tok, [i.name for i in sess.get_inputs()])
    _backend = "onnx"
    _loaded_name = model_name
    return True


def _encode_onnx(texts: list[str]) -> np.ndarray:
    assert _onnx is not None
    sess, tok, names = _onnx
    out: list[np.ndarray] = []
    for i in range(0, len(texts), 32):
        batch = tok.encode_batch(texts[i:i + 32])
        ids = np.array([b.ids for b in batch], dtype=np.int64)
        mask = np.array([b.attention_mask for b in batch], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = sess.run(None, feeds)[0]                       # (b, seq, dim)
        m = mask[:, :, None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        out.append(pooled)
    arr = np.concatenate(out, axis=0).astype(np.float32)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)


def _load_st(model_name: str, allow_download: bool) -> bool:
    global _st_model, _backend, _loaded_name
    try:
        from sentence_transformers import SentenceTransformer  # heavy import, lazy
        try:
            _st_model = SentenceTransformer(model_name, device="cpu", local_files_only=True)
        except Exception:
            if not allow_download:
                return False
            _st_model = SentenceTransformer(model_name, device="cpu")
        _backend = "sentence-transformers"
        _loaded_name = model_name
        return True
    except Exception:
        return False


def _tfidf(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english")
    try:
        m = vec.fit_transform(texts)
    except ValueError:  # all stop words
        vec = TfidfVectorizer(ngram_range=(1, 1), min_df=1)
        m = vec.fit_transform(texts)
    arr = m.toarray().astype(np.float32)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)


def warmup(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", allow_download: bool = True) -> str:
    """Load the embedding backend now (used by the server/app at startup). Returns the backend name."""
    global _backend
    with _lock:
        if _loaded_name == model_name and _backend in ("onnx", "sentence-transformers"):
            return _backend
        if _load_onnx(model_name, allow_download):
            return _backend
        if _load_st(model_name, allow_download):
            return _backend
        _backend = "tfidf"
        return _backend


def get_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", allow_download: bool = True):
    """Compatibility helper: returns a truthy object when a neural backend is loaded, else None."""
    b = warmup(model_name, allow_download)
    return _onnx or _st_model if b != "tfidf" else None


def embed(texts: Iterable[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2", allow_download: bool = True) -> np.ndarray:
    """Embed texts into unit-normalised float32 rows. Never raises; falls back to TF-IDF."""
    texts = [t if t.strip() else "." for t in texts]
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    b = warmup(model_name, allow_download)
    try:
        if b == "onnx":
            return _encode_onnx(texts)
        if b == "sentence-transformers" and _st_model is not None:
            vecs = _st_model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            return np.asarray(vecs, dtype=np.float32)
    except Exception:
        pass
    return _tfidf(texts)


def cosine_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    b = a if b is None else b
    return np.clip(a @ b.T, -1.0, 1.0)
