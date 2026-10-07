"""Hybrid recommender: content cosine + co-purchase blend + static-quality net score.

  rel_c   = cosine(profile, doc) / max            (substitutes: same kind of product)
  rel_b   = sum_h cos_reviewers(h, c) / max       (complements: bought together)
  w_c     = n_c / (n_c + tau)                     n_c = train interactions of candidate c
  rel     = (1 - w_c) * rel_c + w_c * rel_b       cold items fall back to text only
  net     = rel + alpha*pop + beta*rating + gamma*recency        (static quality g(d))
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .copurchase import CoPurchase, StaticQuality
from .retrieval import ChampionLists, ClusterPruning, topk


@dataclass
class Config:
    mode: str = "blend"            # content | behavior | blend
    alpha: float = 0.0             # popularity weight
    beta: float = 0.0              # rating-quality weight
    gamma: float = 0.0             # recency weight
    tau: float = 5.0               # blend: w = n / (n + tau)
    w_fixed: Optional[float] = None
    w_max: Optional[float] = None  # cap on the adaptive blend weight (keeps some text signal for popular items)
    sim: str = "cosine"            # cosine | jaccard (reviewer sets)
    prune: str = "none"            # none | champion | cluster
    r: int = 200                   # champion list length
    n_terms: int = 30              # query terms used for champion candidates
    probe: int = 3                 # leaders probed for cluster pruning
    min_idf: float = 0.0           # index elimination threshold
    filters: Optional[dict] = None  # parametric: category, brand, min_price, max_price

    def replace(self, **kw) -> "Config":
        return dataclasses.replace(self, **kw)


class Recommender:
    def __init__(self, index, products, train_inter):
        self.index = index
        self.products = products.reset_index(drop=True)
        self.cp = CoPurchase(train_inter, index.N)
        self.sq = StaticQuality(train_inter, index.N)
        self._champ: dict = {}
        self._clus = None
        self._masks: dict = {}
        self.last_ncand = index.N

    # ---- pruning structures (built lazily) ---------------------------------------------
    def champions(self, r):
        if r not in self._champ:
            self._champ[r] = ChampionLists(self.index, r)
        return self._champ[r]

    def clusters(self):
        if self._clus is None:
            self._clus = ClusterPruning(self.index)
        return self._clus

    # ---- parametric filters ------------------------------------------------------------
    def _mask(self, f):
        key = tuple(sorted(f.items()))
        if key not in self._masks:
            p, m = self.products, np.ones(self.index.N, bool)
            if f.get("category"):
                m &= p["category"].str.lower().str.contains(f["category"].lower(), regex=False).to_numpy()
            if f.get("brand"):
                m &= (p["brand"].str.lower() == f["brand"].lower()).to_numpy()
            if f.get("min_price") is not None:
                m &= (p["price"] >= f["min_price"]).fillna(False).to_numpy()
            if f.get("max_price") is not None:
                m &= (p["price"] <= f["max_price"]).fillna(False).to_numpy()
            self._masks[key] = m
        return self._masks[key]

    # ---- scoring -----------------------------------------------------------------------
    def _score(self, hist, cfg: Config) -> dict:
        idx, N = self.index, self.index.N
        hist = np.asarray(hist, dtype=int)
        q = idx.profile_vector(hist)
        b = np.zeros(N, np.float32) if cfg.mode == "content" else self.cp.behavior_scores(hist, cfg.sim)

        cand = None
        if cfg.prune != "none" and cfg.mode != "behavior":
            cand = (self.champions(cfg.r).candidates(q, cfg.n_terms) if cfg.prune == "champion"
                    else self.clusters().candidates(q, cfg.probe))
            if cfg.mode == "blend":
                cand = np.union1d(cand, np.flatnonzero(b))
        valid = np.ones(N, bool)
        if cfg.mode == "behavior":
            c = np.zeros(N, np.float32)
        elif cand is None:
            c = idx.cosine_taat(q, cfg.min_idf)
        else:
            c = np.zeros(N, np.float32)
            c[cand] = np.asarray(idx.D[cand] @ q).ravel()
            valid = np.zeros(N, bool)
            valid[cand] = True
        self.last_ncand = N if cand is None else len(cand)

        cn = c / c[valid].max() if c[valid].max() > 0 else c
        bn = b / b.max() if b.max() > 0 else b
        if cfg.mode == "content":
            w = np.zeros(N, np.float32)
        elif cfg.mode == "behavior":
            w = np.ones(N, np.float32)
        elif cfg.w_fixed is not None:
            w = np.full(N, cfg.w_fixed, np.float32)
        else:
            w = (self.cp.count / (self.cp.count + cfg.tau)).astype(np.float32)
            if cfg.w_max is not None:
                w = np.minimum(w, cfg.w_max)
        rel = (1 - w) * cn + w * bn
        net = rel + cfg.alpha * self.sq.pop + cfg.beta * self.sq.rating + cfg.gamma * self.sq.rec
        net = net.astype(np.float64)
        net[~valid] = -np.inf
        net[hist] = -np.inf
        if cfg.filters:
            net[~self._mask(cfg.filters)] = -np.inf
        return dict(net=net, c=c, b=b, w=w, q=q)

    def rank_rows(self, hist, k=10, cfg: Config | None = None):
        s = self._score(hist, cfg or Config())
        return topk(s["net"], k)

    # ---- public API (the "contracts" in the plan) ---------------------------------------
    def recommend(self, history, k=10, cfg: Config | None = None):
        """final_rank(): history = list of asins. Returns dicts with the net-score parts."""
        cfg = cfg or Config()
        rows = [self.index.asin2row[a] for a in history]
        s = self._score(rows, cfg)
        out = []
        for r, sc in topk(s["net"], k):
            out.append(dict(asin=self.index.asins[r], title=self.products.at[r, "title"], net=sc,
                            cosine=float(s["c"][r]), behavior=float(s["b"][r]), w=float(s["w"][r]),
                            pop=float(self.sq.pop[r]), rating=float(self.sq.rating[r]),
                            recency=float(self.sq.rec[r])))
        return out

    def recommend_content(self, history, k=10):
        return self.recommend(history, k, Config(mode="content"))

    def recommend_behavior(self, history, k=10):
        return self.recommend(history, k, Config(mode="behavior"))