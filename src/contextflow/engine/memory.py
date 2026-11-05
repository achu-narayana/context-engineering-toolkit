"""Memory manager: per-item keep / summarize / remove / store decisions and a local long-term store.

Store = SQLite file in ~/.contextflow/memory.db with embeddings (numpy float32 blobs).
Nothing leaves the machine.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import home_dir
from .embeddings import embed
from .segment import Section


@dataclass
class MemoryHit:
    id: int
    score: float
    text: str
    summary: str
    kind: str
    source: str
    created: float


def decide(s: Section, target_ratio: float) -> str:
    """keep | summarize | store | remove for one section after selection/budgeting."""
    if s.kind in ("code", "heading"):
        return "keep"
    if s.keep:
        return "keep" if s.budget >= s.tokens * 0.9 else "summarize"
    if s.protected and (len(s.protected) >= 2 or s.score > 0.25):
        return "store"   # dropped from context but worth remembering
    return "remove"


class MemoryStore:
    def __init__(self, path: Path | None = None, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.path = path or home_dir() / "memory.db"
        self.embedding_model = embedding_model
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, created REAL, source TEXT, kind TEXT, "
            "text TEXT, summary TEXT, embedding BLOB)"
        )
        self._conn.commit()

    def add(self, text: str, summary: str = "", kind: str = "section", source: str = "") -> int:
        vec = embed([summary or text], self.embedding_model)[0].astype(np.float32)
        cur = self._conn.execute(
            "INSERT INTO items (created, source, kind, text, summary, embedding) VALUES (?,?,?,?,?,?)",
            (time.time(), source, kind, text, summary, vec.tobytes()),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def add_many(self, rows: list[tuple[str, str, str, str]]) -> int:
        if not rows:
            return 0
        vecs = embed([r[1] or r[0] for r in rows], self.embedding_model).astype(np.float32)
        now = time.time()
        self._conn.executemany(
            "INSERT INTO items (created, source, kind, text, summary, embedding) VALUES (?,?,?,?,?,?)",
            [(now, r[3], r[2], r[0], r[1], v.tobytes()) for r, v in zip(rows, vecs)],
        )
        self._conn.commit()
        return len(rows)

    def recall(self, query: str, k: int = 5) -> list[MemoryHit]:
        rows = self._conn.execute("SELECT id, created, source, kind, text, summary, embedding FROM items").fetchall()
        if not rows:
            return []
        mats = np.stack([np.frombuffer(r[6], dtype=np.float32) for r in rows])
        q = embed([query], self.embedding_model)[0].astype(np.float32)
        if mats.shape[1] != q.shape[0]:   # embedding backend changed (e.g. tf-idf fallback); fall back to lexical overlap
            ql = set(query.lower().split())
            scores = np.array([len(ql & set((r[5] or r[4]).lower().split())) / (len(ql) or 1) for r in rows])
        else:
            scores = mats @ q
        idx = np.argsort(-scores)[:k]
        return [MemoryHit(id=rows[i][0], score=float(scores[i]), text=rows[i][4], summary=rows[i][5], kind=rows[i][3], source=rows[i][2], created=rows[i][1]) for i in idx]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM items").fetchone()[0])

    def clear(self) -> None:
        self._conn.execute("DELETE FROM items")
        self._conn.commit()
