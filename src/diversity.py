"""Variant collapsing (Jaccard over title shingles) + MMR re-ranking (Carbonell & Goldstein 1998).

Lecture mapping:
  shingles()/variant_groups() -> Jaccard coefficient over word k-shingles; near-duplicate listings
                                 (same product, different colour/size) are merged into one group
  mmr_rerank()                -> greedy trade-off: lam * relevance - (1 - lam) * max cosine to chosen items
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .recommender import Config
from .tfidf import tokenize


def shingles(text: str, k: int = 2) -> set:
    t = tokenize(text)
    if len(t) <= k:
        return {" ".join(t)} if t else set()
    return {" ".join(t[i:i + k]) for i in range(len(t) - k + 1)}


def variant_groups(products, k: int = 2, thr: float = 0.6, max_bucket: int = 60):
    """Union-find over pairs with Jaccard(shingles(title)) >= thr. Candidate pairs come from an
    inverted index shingle -> items (rare shingles only), so no O(N^2) scan. Returns group id per row."""
    sh = [shingles(t, k) for t in products["title"]]
    inv = defaultdict(list)
    for i, s in enumerate(sh):
        for g in s:
            inv[g].append(i)
    parent = list(range(len(sh)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen = set()
    for items in inv.values():
        if len(items) > max_bucket:                      # very common shingle: skip (stop-shingle)
            continue
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                i, j = items[a], items[b]
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                u = len(sh[i] | sh[j])
                if u and len(sh[i] & sh[j]) / u >= thr:
                    parent[find(i)] = find(j)
    return np.array([find(i) for i in range(len(sh))])


def mmr_rerank(rec, hist_rows, k=10, cfg: Config | None = None, lam=0.7, pool=100, groups=None):
    """Return [(row, net)] of length k. If groups is given, keep only the best row of each variant group."""
    cfg = cfg or Config()
    cand = rec.rank_rows(hist_rows, pool, cfg)
    if groups is not None:
        seen, kept = set(), []
        for r, s in cand:
            if groups[r] not in seen:
                seen.add(groups[r])
                kept.append((r, s))
        cand = kept
    if not cand:
        return []
    rows = np.array([r for r, _ in cand])
    net = np.array([s for _, s in cand], dtype=float)
    rel = (net - net.min()) / (net.max() - net.min() + 1e-9)
    S = (rec.index.D[rows] @ rec.index.D[rows].T).toarray()
    chosen, rest = [], list(range(len(rows)))
    while rest and len(chosen) < k:
        if not chosen:
            pick = max(rest, key=lambda i: rel[i])
        else:
            pick = max(rest, key=lambda i: lam * rel[i] - (1 - lam) * S[i, chosen].max())
        chosen.append(pick)
        rest.remove(pick)
    return [(int(rows[i]), float(net[i])) for i in chosen]
