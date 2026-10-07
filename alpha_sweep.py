"""Popularity weight (alpha) and blend-weight cap (w_max): accuracy vs relevance trade-off, on HELD-OUT users.
   python3 alpha_sweep.py [--n 3000]          -> results/alpha_sweep.csv
PopDriven% = share of top-10 items whose static quality g(d) exceeds their relevance (what demo.py labels 'popular item').
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_dataset
from src.evaluate import build, metrics, model_ranker
from src.recommender import Config

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--n-pop", type=int, default=500, help="users used for the PopDriven% estimate")
a = ap.parse_args()

products, inter = load_dataset(a.data)
rec, users, _ = build(products, inter)
perm = np.random.default_rng(123).permutation(len(users))        # same split as tune.py
test_u = [users[i] for i in perm[len(users) // 2:]][: a.n]

base = Config(tau=5.0, alpha=0.5, beta=0.1, gamma=0.1, sim="jaccard")
tj = Path("results/tuned.json")
if tj.exists():
    d = json.load(open(tj))["adaptive"]
    base = Config(tau=d["tau"], alpha=d["alpha"], beta=d["beta"], gamma=d["gamma"], sim=d["sim"])
print("base config:", base, f"| {len(test_u)} held-out users")

rows = []
for w_max in (None, 0.8, 0.6):
    for alpha in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8):
        cfg = base.replace(alpha=alpha, w_max=w_max)
        m = metrics(model_ranker(rec, cfg), rec, test_u)
        drv = []
        for hist, _ in test_u[: a.n_pop]:
            for x in rec.recommend([rec.index.asins[r] for r in hist], 10, cfg):
                gd = cfg.alpha * x["pop"] + cfg.beta * x["rating"] + cfg.gamma * x["recency"]
                drv.append(gd > x["net"] - gd)
        rows.append(dict(w_max=w_max if w_max is not None else "none", alpha=alpha, Hit10=m["Hit@10"],
                         NDCG10=m["NDCG@10"], Coverage10=m["Coverage@10"], ILD10=m["ILD@10"],
                         PopDriven_pct=100 * float(np.mean(drv))))
        print(rows[-1])
df = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/alpha_sweep.csv", index=False)
print("\n", df.round(4).to_string(index=False))
