"""Behavior signal (Linden et al. 2003 style item-to-item CF) + static quality g(d).

IMPORTANT: build both from TRAIN interactions only (no test leakage).

Lecture mapping:
  CoPurchase   -> items as vectors over reviewers; cosine or Jaccard over reviewer sets
  StaticQuality-> g(d): popularity, shrunk rating quality, recency; combined in net score
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


class CoPurchase:
    def __init__(self, train: pd.DataFrame, n_items: int):
        users = pd.factorize(train["user_id"])[0]
        rows = train["row"].to_numpy()
        R = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, users)), shape=(n_items, users.max() + 1))
        R.data[:] = 1.0
        self.n_items = n_items
        self.R = R
        self.RT = R.T.tocsr()
        self.count = np.asarray(R.sum(1)).ravel()              # interactions per item
        self.norm = np.sqrt(np.maximum(self.count, 1))
        self._cache: dict = {}

    def item_sims(self, h: int, kind: str = "cosine"):
        """(neighbour rows, similarity, #shared reviewers) of item h; self excluded."""
        key = (h, kind)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        inter = (self.R[h] @ self.RT).tocoo()
        j, v = inter.col, inter.data
        if kind == "jaccard":
            s = v / (self.count[h] + self.count[j] - v)
        else:
            s = v / (self.norm[h] * self.norm[j])
        keep = j != h
        res = (j[keep], s[keep].astype(np.float32), v[keep])
        if len(self._cache) > 20000:
            self._cache.clear()
        self._cache[key] = res
        return res

    def behavior_scores(self, hist, kind: str = "cosine") -> np.ndarray:
        out = np.zeros(self.n_items, dtype=np.float32)
        for h in hist:
            j, s, _ = self.item_sims(int(h), kind)
            out[j] += s
        return out

    def pair_sim(self, h: int, r: int, kind: str = "cosine"):
        j, s, v = self.item_sims(h, kind)
        m = np.flatnonzero(j == r)
        return (float(s[m[0]]), int(v[m[0]])) if len(m) else (0.0, 0)


class StaticQuality:
    def __init__(self, train: pd.DataFrame, n_items: int, m: float = 5.0, half_life_days: float = 365.0):
        row = train["row"].to_numpy()
        n = np.bincount(row, minlength=n_items).astype(float)
        sr = np.bincount(row, weights=train["rating"].to_numpy(), minlength=n_items)
        gm = train["rating"].mean()
        bayes = (sr + m * gm) / (n + m)                           # shrunk mean: few ratings -> global mean
        self.rating = ((bayes - 1) / 4).astype(np.float32)
        self.pop = (np.log1p(n) / np.log1p(n.max())).astype(np.float32)
        last = np.full(n_items, np.nan)
        g = train.groupby("row")["ts"].max()
        last[g.index.to_numpy()] = g.to_numpy()
        age = (np.nanmax(last) - last) / 86400.0
        self.rec = np.where(np.isnan(age), 0.0, 0.5 ** (age / half_life_days)).astype(np.float32)
