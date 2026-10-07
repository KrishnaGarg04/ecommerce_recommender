"""Evaluation harness: temporal leave-last-out split, baselines, metrics, experiments.

Protocol: per user (>= 3 interactions) sort by time; last item = target, the rest = history.
Co-purchase statistics and static quality are built from TRAIN interactions only (everything except
each user's last item), so the target is never seen by the behavior signal.
Metrics (single target per user): Hit@k = Recall@k, P@k = Hit@k / k, NDCG@k = 1/log2(rank+1) if rank <= k.
Also: catalogue coverage@k and intra-list diversity (1 - mean pairwise cosine of top-k text vectors).
"""
from __future__ import annotations

import time

import numpy as np

from .recommender import Config, Recommender
from .retrieval import topk
from .tfidf import TfidfIndex

KS = (5, 10, 20)


def temporal_split(inter, min_len=3):
    """Returns (train_inter, users) with users = list of (history_rows, target_row)."""
    g = inter.groupby("user_id")["row"]
    size = g.transform("size")
    pos = g.cumcount(ascending=False)                      # 0 = last interaction
    ok = size >= min_len
    train = inter[ok & (pos > 0) | ~ok]
    test = inter[ok & (pos == 0)]
    hist = train[train["user_id"].isin(test["user_id"])].groupby("user_id")["row"].apply(list)
    users = [(hist[u], int(r)) for u, r in zip(test["user_id"], test["row"])]
    return train, users


def build(products, inter, **kw):
    train, users = temporal_split(inter)
    index = TfidfIndex(products, **kw)
    return Recommender(index, products, train), users, train


def popularity_ranker(rec):
    def f(hist, k):
        s = rec.sq.pop.astype(np.float64).copy()
        s[np.asarray(hist, dtype=int)] = -np.inf
        return topk(s, k)
    return f


def model_ranker(rec, cfg):
    return lambda hist, k: rec.rank_rows(hist, k, cfg)


def metrics(ranker, rec, users, ks=KS, n_eval=None, seed=0, cold_bins=None, groups=None):
    """Run a ranker over (a sample of) users. Returns dict of metrics + per-user target ranks."""
    rng = np.random.default_rng(seed)
    if n_eval and n_eval < len(users):
        users = [users[i] for i in rng.choice(len(users), n_eval, replace=False)]
    K = max(ks)
    ranks, seen, ild, dist = [], [set() for _ in ks], [], []
    t0 = time.perf_counter()
    for hist, tgt in users:
        top = ranker(hist, K)
        rows = [r for r, _ in top]
        ranks.append(rows.index(tgt) + 1 if tgt in rows else K + 1)
        if groups is not None:                      # distinct variant groups in the top 10
            dist.append(len(set(groups[rows[:10]].tolist())))
        for i, k in enumerate(ks):
            seen[i].update(rows[:k])
        if len(rows) >= 2:
            M = rec.index.D[rows[:10]]
            S = (M @ M.T).toarray()
            n = S.shape[0]
            ild.append(1 - (S.sum() - np.trace(S)) / (n * (n - 1)))
    dt = (time.perf_counter() - t0) / len(users)
    ranks = np.array(ranks)
    out = {"ms_per_user": 1000 * dt, "ILD@10": float(np.mean(ild)) if ild else 0.0}
    if groups is not None:
        out["Distinct@10"] = float(np.mean(dist))
    for i, k in enumerate(ks):
        hit = ranks <= k
        out[f"Hit@{k}"] = float(hit.mean())
        out[f"P@{k}"] = float(hit.mean() / k)
        out[f"Recall@{k}"] = float(hit.mean())
        out[f"NDCG@{k}"] = float(np.where(hit, 1 / np.log2(ranks + 1), 0).mean())
        out[f"Coverage@{k}"] = len(seen[i]) / rec.index.N
    out["_ranks"] = ranks
    out["_targets"] = np.array([t for _, t in users])
    return out


def pruning_experiment(rec, users, n_eval=300, base=None, seed=0):
    """Speed vs recall of pruned candidate generation compared with brute force (same config)."""
    base = base or Config(mode="content")
    rng = np.random.default_rng(seed)
    sample = [users[i] for i in rng.choice(len(users), min(n_eval, len(users)), replace=False)]
    brute = [set(r for r, _ in rec.rank_rows(h, 10, base)) for h, _ in sample]
    rows = []
    variants = [("brute force", base)]
    variants += [(f"champion r={r}", base.replace(prune="champion", r=r)) for r in (20, 50, 200)]
    variants += [(f"cluster b={b}", base.replace(prune="cluster", probe=b)) for b in (1, 3, 5)]
    variants += [("index elimination idf>=1", base.replace(min_idf=1.0))]
    for name, cfg in variants:
        t0, ov, nc, hit = time.perf_counter(), [], [], 0
        for (h, tgt), b in zip(sample, brute):
            top = [r for r, _ in rec.rank_rows(h, 10, cfg)]
            nc.append(rec.last_ncand)
            ov.append(len(b & set(top)) / 10)
            hit += tgt in top
        rows.append(dict(method=name, ms_per_query=1000 * (time.perf_counter() - t0) / len(sample),
                         overlap_with_brute_top10=float(np.mean(ov)), avg_candidates=float(np.mean(nc)),
                         Hit10=hit / len(sample)))
    return rows


def cold_start_curve(rec, results: dict, bins=((0, 0), (1, 2), (3, 5), (6, 10), (11, 10**9)), k=10):
    """results: name -> metrics() output over the SAME users. Hit@k by train popularity of the target."""
    out = []
    for lo, hi in bins:
        row = {"bin": f"{lo}" if lo == hi else (f"{lo}+" if hi > 10**8 else f"{lo}-{hi}")}
        for name, res in results.items():
            cnt = rec.cp.count[res["_targets"]]
            m = (cnt >= lo) & (cnt <= hi)
            row[name] = float((res["_ranks"][m] <= k).mean()) if m.any() else float("nan")
            row["n"] = int(m.sum())
        out.append(row)
    return out
