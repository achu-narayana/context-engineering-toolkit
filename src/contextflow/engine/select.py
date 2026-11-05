"""Context selector: rank sections by relevance and centrality, drop low-value ones.

Score per section (all in 0..1):
  goal_sim   cosine similarity to the optional goal (embeddings)
  centrality TextRank score over the section similarity graph (what the document is "about")
  position   small lead/tail prior (first and last parts carry setup and conclusions)
  protected  bonus for sections containing protected items (numbers, dates, names, code...)
  structure  headings/code/lists get a floor so structure survives

Sections are dropped only when they fall outside the selection budget or duplicate a kept section.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embeddings import embed, cosine_matrix
from .segment import Section


@dataclass
class SelectionConfig:
    w_goal: float = 0.45
    w_central: float = 0.35
    w_position: float = 0.08
    w_protected: float = 0.12
    duplicate_threshold: float = 0.92
    min_keep_fraction: float = 0.25        # never drop more than 75% of sections in one pass
    selection_headroom: float = 1.6        # select up to this x target tokens before compression


def textrank(sim: np.ndarray, damping: float = 0.85, iters: int = 50) -> np.ndarray:
    n = sim.shape[0]
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)
    w = sim.copy()
    np.fill_diagonal(w, 0.0)
    w = np.clip(w, 0.0, None)
    col = w.sum(axis=0, keepdims=True) + 1e-9
    m = w / col
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        r_new = (1 - damping) / n + damping * (m @ r)
        if np.abs(r_new - r).sum() < 1e-6:
            r = r_new
            break
        r = r_new
    r = r - r.min()
    return r / (r.max() + 1e-9)


def select_sections(
    sections: list[Section],
    goal: str | None,
    target_tokens: int,
    mode: str,
    cfg: SelectionConfig | None = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    allow_download: bool = True,
) -> list[Section]:
    """Scores every section, marks keep/drop with a reason, returns sections (same objects)."""
    cfg = cfg or SelectionConfig()
    if not sections:
        return sections
    texts = [s.text for s in sections]
    vecs = embed(texts + ([goal] if goal else []), embedding_model, allow_download)
    sec_vecs = vecs[: len(sections)]
    sim = cosine_matrix(sec_vecs)
    central = textrank(sim)
    if goal:
        goal_vec = vecs[len(sections):]
        gsim = cosine_matrix(sec_vecs, goal_vec)[:, 0]
        gsim = (gsim - gsim.min()) / (gsim.max() - gsim.min() + 1e-9)
        w_goal, w_central = cfg.w_goal, cfg.w_central
    else:
        gsim = np.zeros(len(sections))
        w_goal, w_central = 0.0, cfg.w_goal + cfg.w_central

    for i, s in enumerate(sections):
        pos_prior = 1.0 if s.position <= 0.1 or s.position >= 0.9 else 0.3
        prot = min(1.0, 0.25 * len(s.protected))
        structure = 0.0
        if s.kind in ("heading", "code"):
            structure = 0.6
        elif s.kind == "list":
            structure = 0.3
        score = w_goal * gsim[i] + w_central * central[i] + cfg.w_position * pos_prior + cfg.w_protected * prot
        s.score = float(min(1.0, score + structure * 0.5))
        s.meta["goal_sim"] = float(gsim[i])
        s.meta["centrality"] = float(central[i])

    # 1) duplicates: drop a section nearly identical to a higher-scored one
    order = sorted(range(len(sections)), key=lambda i: -sections[i].score)
    kept_idx: list[int] = []
    for i in order:
        dup_of = next((j for j in kept_idx if sim[i, j] >= cfg.duplicate_threshold and sections[j].kind == sections[i].kind), None)
        if dup_of is not None and sections[i].kind not in ("code", "heading"):
            sections[i].keep = False
            sections[i].reason = f"near-duplicate of section {dup_of} (sim {sim[i, dup_of]:.2f})"
        else:
            kept_idx.append(i)

    if mode == "light":
        for i in kept_idx:
            sections[i].keep = True
            sections[i].reason = "kept (light mode keeps all unique sections)"
        return sections

    # 2) budgeted selection: keep highest scores until the selection budget is hit
    budget = int(target_tokens * cfg.selection_headroom)
    min_keep = max(1, int(len(sections) * cfg.min_keep_fraction))
    used = 0
    n_kept = 0
    for i in kept_idx:
        s = sections[i]
        must = s.kind in ("code", "heading") or len(s.protected) >= 3
        if used + s.tokens <= budget or n_kept < min_keep or must:
            s.keep = True
            s.reason = _why_kept(s, goal)
            used += s.tokens
            n_kept += 1
        else:
            s.keep = False
            s.reason = f"low value (score {s.score:.2f}, centrality {s.meta['centrality']:.2f}" + (
                f", goal match {s.meta['goal_sim']:.2f}" if goal else "") + ") and over selection budget"
    return sections


def _why_kept(s: Section, goal: str | None) -> str:
    bits = []
    if goal and s.meta.get("goal_sim", 0) > 0.6:
        bits.append("matches goal")
    if s.meta.get("centrality", 0) > 0.6:
        bits.append("central to the text")
    if s.protected:
        bits.append(f"{len(s.protected)} protected item(s)")
    if s.kind == "code":
        bits.append("code block")
    if s.kind == "heading":
        bits.append("heading")
    if s.position <= 0.1:
        bits.append("opening")
    elif s.position >= 0.9:
        bits.append("closing")
    return "kept: " + (", ".join(bits) if bits else f"score {s.score:.2f}")
