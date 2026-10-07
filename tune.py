"""Honest tuning: tune on one half of the users, report on the OTHER half, with bootstrap 95% CIs.
Two weight families are tuned separately: adaptive w = n/(n+tau)  vs  a single fixed w.
   python tune.py --n-tune 1500 --n-test 6000
Writes results/tuned.json, which run_experiments.py reads.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_dataset
from src.evaluate import build, metrics, model_ranker, popularity_ranker
from src.recommender import Config

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="results")
ap.add_argument("--n-tune", type=int, default=1500)
ap.add_argument("--n-test", type=int, default=6000)
ap.add_argument("--taus", type=float, nargs="+", default=[1, 5, 20, 50])
ap.add_argument("--wfix", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
args = ap.parse_args()
out = Path(args.out); out.mkdir(exist_ok=True)

products, inter = load_dataset(args.data)
rec, users, _ = build(products, inter)
rng = np.random.default_rng(123)
perm = rng.permutation(len(users))
half = len(users) // 2
tune_u = [users[i] for i in perm[:half]][: args.n_tune]
test_u = [users[i] for i in perm[half:]][: args.n_test]
print(f"tune on {len(tune_u)} users, report on {len(test_u)} DIFFERENT users")

gd = [(0, 0, 0), (.1, 0, 0), (.1, .1, .1), (.2, .1, .1), (.3, .1, .1), (.5, .1, .1)]
grid = []
for (tau, wf), (a, b, g), sim in itertools.product(
        [(t, None) for t in args.taus] + [(5.0, w) for w in args.wfix], gd, ["cosine", "jaccard"]):
    cfg = Config(tau=tau, w_fixed=wf, alpha=a, beta=b, gamma=g, sim=sim)
    m = metrics(model_ranker(rec, cfg), rec, tune_u)
    grid.append(dict(family="fixed" if wf is not None else "adaptive", tau=tau, w_fixed=wf, alpha=a, beta=b,
                     gamma=g, sim=sim, **{k: m[k] for k in ("Hit@10", "NDCG@10")}))
G = pd.DataFrame(grid).sort_values("NDCG@10", ascending=False)
G.to_csv(out / "tuning_grid.csv", index=False)
for fam in ("adaptive", "fixed"):
    print(f"\n== top 5 {fam}-weight configs on the TUNE half (objective NDCG@10) ==")
    print(G[G.family == fam].head(5).round(4).to_string(index=False))


def to_cfg(row):
    wf = None if row["family"] == "adaptive" else float(row["w_fixed"])
    return Config(tau=float(row["tau"]), w_fixed=wf, alpha=float(row["alpha"]), beta=float(row["beta"]),
                  gamma=float(row["gamma"]), sim=row["sim"])


best = {fam: G[G.family == fam].iloc[0] for fam in ("adaptive", "fixed")}
models = {
    "popularity": popularity_ranker(rec),
    "centroid tf-idf (content)": model_ranker(rec, Config(mode="content")),
    "blend default (tau=5, no g(d))": model_ranker(rec, Config()),
    "blend tuned, adaptive w": model_ranker(rec, to_cfg(best["adaptive"])),
    "blend tuned, fixed w": model_ranker(rec, to_cfg(best["fixed"])),
}
res = {n: metrics(f, rec, test_u) for n, f in models.items()}


def per_user(res_, k=10):
    r = res_["_ranks"]
    return (r <= k).astype(float), np.where(r <= k, 1 / np.log2(r + 1), 0.0)


def boot(x, B=2000, seed=0):
    g = np.random.default_rng(seed)
    n = len(x)
    s = np.array([x[g.integers(0, n, n)].mean() for _ in range(B)])
    return float(x.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


rows = []
for n, r in res.items():
    h, nd = per_user(r)
    (hm, hl, hh), (nm, nl, nh) = boot(h), boot(nd)
    rows.append({"model": n, "Hit@10": hm, "Hit@10 CI": f"[{hl:.4f}, {hh:.4f}]", "NDCG@10": nm,
                 "NDCG@10 CI": f"[{nl:.4f}, {nh:.4f}]", "Coverage@10": r["Coverage@10"], "ILD@10": r["ILD@10"]})
T = pd.DataFrame(rows)
T.to_csv(out / "final_test_with_ci.csv", index=False)
print("\n== held-out test users, bootstrap 95% CI ==\n", T.round(4).to_string(index=False))

print("\n== paired differences (same users), 95% CI ==")
diffs = {}
A, F, D, C, P = ("blend tuned, adaptive w", "blend tuned, fixed w", "blend default (tau=5, no g(d))",
                 "centroid tf-idf (content)", "popularity")
for a, b in [(F, A), (A, C), (F, C), (A, D), (D, C), (A, P)]:
    ha, na = per_user(res[a]); hb, nb = per_user(res[b])
    for metric, x in (("Hit@10", ha - hb), ("NDCG@10", na - nb)):
        m, lo, hi = boot(x)
        sig = "significant" if lo > 0 or hi < 0 else "NOT significant"
        print(f"  {a} - {b} | {metric}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] {sig}")
        diffs[f"{a} - {b} | {metric}"] = [m, lo, hi]


def cfg_json(row):
    return {k: (None if (k == "w_fixed" and row["family"] == "adaptive") else
                (row[k] if isinstance(row[k], str) else float(row[k])))
            for k in ("tau", "w_fixed", "alpha", "beta", "gamma", "sim")}


json.dump(dict(adaptive=cfg_json(best["adaptive"]), fixed=cfg_json(best["fixed"]), diffs=diffs),
          open(out / "tuned.json", "w"), indent=2, default=float)
