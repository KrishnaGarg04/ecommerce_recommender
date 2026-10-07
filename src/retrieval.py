"""Heap-based top-K, champion lists and cluster pruning (leaders/followers)."""
from __future__ import annotations

import heapq

import numpy as np


def topk(scores: np.ndarray, k: int, mask: np.ndarray | None = None):
    """Heap-based top-K over finite scores (-inf = excluded). Returns [(row, score)]."""
    idx = np.flatnonzero(np.isfinite(scores))
    if mask is not None:
        idx = idx[mask[idx]]
    best = heapq.nlargest(k, idx.tolist(), key=scores.__getitem__)
    return [(int(i), float(scores[i])) for i in best]


class ChampionLists:
    """For each term keep the r docs with the highest weight; candidates = union over query terms."""

    def __init__(self, index, r: int = 200):
        Dc = index.Dc
        self.lists = []
        for j in range(index.V):
            s, e = Dc.indptr[j], Dc.indptr[j + 1]
            rows, w = Dc.indices[s:e], Dc.data[s:e]
            if e - s > r:
                rows = rows[np.argpartition(-w, r)[:r]]
            self.lists.append(np.array(rows))

    def candidates(self, q: np.ndarray, n_terms: int = 30) -> np.ndarray:
        nz = np.flatnonzero(q)
        nz = nz[np.argsort(-q[nz])][:n_terms]
        if len(nz) == 0:
            return np.array([], dtype=int)
        return np.unique(np.concatenate([self.lists[j] for j in nz]))


class ClusterPruning:
    """sqrt(N) random leaders; every doc follows its nearest leader. Query -> b nearest leaders."""

    def __init__(self, index, seed: int = 0):
        rng = np.random.default_rng(seed)
        n_lead = max(2, int(np.sqrt(index.N)))
        self.leaders = rng.choice(index.N, n_lead, replace=False)
        self.Lm = index.D[self.leaders]
        sims = (index.D @ self.Lm.T).toarray()
        assign = sims.argmax(1)
        assign[sims.max(1) == 0] = rng.integers(0, n_lead, size=int((sims.max(1) == 0).sum()))
        self.followers = [np.flatnonzero(assign == l) for l in range(n_lead)]

    def candidates(self, q: np.ndarray, b: int = 3) -> np.ndarray:
        ls = np.asarray(self.Lm @ q).ravel()
        top = np.argsort(-ls)[:b]
        return np.concatenate([self.followers[l] for l in top])
