"""Runs every experiment in the brief and writes tables (CSV/JSON) + graphs to results/.
   python run_experiments.py --data data --n-eval 2000
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_dataset
from src.diversity import mmr_rerank, variant_groups
from src.evaluate import (build, cold_start_curve, metrics, model_ranker, popularity_ranker,
                          pruning_experiment)
from src.explain import faithfulness
from src.recommender import Config
from src.tfidf import FLAT_ZONES

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="results")
ap.add_argument("--n-eval", type=int, default=6000)
args = ap.parse_args()

# Chosen by tune.py on a DISJOINT half of the users (tau=5, alpha=.3, beta=.1, gamma=.1, jaccard)
TUNED = Config(tau=5.0, alpha=0.3, beta=0.1, gamma=0.1, sim="jaccard")
TUNED_FIXED = TUNED.replace(w_fixed=0.5)
_tj = Path(args.out) / "tuned.json"
if _tj.exists():                                   # written by tune.py (newer version)
    _j = json.load(open(_tj))
    if "adaptive" in _j:
        _c = lambda d: Config(tau=float(d["tau"]), w_fixed=None if d["w_fixed"] is None else float(d["w_fixed"]),
                              alpha=float(d["alpha"]), beta=float(d["beta"]), gamma=float(d["gamma"]), sim=d["sim"])
        TUNED, TUNED_FIXED = _c(_j["adaptive"]), _c(_j["fixed"])
print("adaptive config:", TUNED, "\nfixed config:   ", TUNED_FIXED)
out = Path(args.out)
out.mkdir(exist_ok=True)

products, inter = load_dataset(args.data)
rec, users, train = build(products, inter)
print(f"{rec.index.N} products, V={rec.index.V}, {len(train)} train interactions, {len(users)} test users")
N = args.n_eval
SHOW = ["Hit@10", "P@10", "Recall@10", "NDCG@10", "Coverage@10", "ILD@10", "ms_per_user"]


def table(rows: dict, name, show=SHOW):
    df = pd.DataFrame({k: {m: v[m] for m in show} for k, v in rows.items()}).T
    df.to_csv(out / f"{name}.csv")
    print(f"\n== {name} ==\n{df.round(4).to_string()}")
    return df


# 1) baselines vs ours -------------------------------------------------------------------
main = {
    "popularity": metrics(popularity_ranker(rec), rec, users, n_eval=N),
    "centroid tf-idf (content)": metrics(model_ranker(rec, Config(mode="content")), rec, users, n_eval=N),
    "behavior only (item-item)": metrics(model_ranker(rec, Config(mode="behavior")), rec, users, n_eval=N),
    "blend default (tau=5)": metrics(model_ranker(rec, Config()), rec, users, n_eval=N),
    "blend tuned (+g(d), jaccard)": metrics(model_ranker(rec, TUNED), rec, users, n_eval=N),
    "blend tuned, fixed w": metrics(model_ranker(rec, TUNED_FIXED), rec, users, n_eval=N),
}
table(main, "main_comparison")
FULL = main["blend tuned (+g(d), jaccard)"]

# 2) ablation: each component on/off -----------------------------------------------------
rec_flat, _, _ = build(products, inter, zone_weights=FLAT_ZONES)
mr = lambda cfg, r=rec: metrics(model_ranker(r, cfg), r, users, n_eval=N)
abl = {
    "full (tuned blend)": FULL,
    "- all g(d) (alpha=beta=gamma=0)": mr(TUNED.replace(alpha=0, beta=0, gamma=0)),
    "- popularity (alpha=0)": mr(TUNED.replace(alpha=0)),
    "- rating quality (beta=0)": mr(TUNED.replace(beta=0)),
    "- recency (gamma=0)": mr(TUNED.replace(gamma=0)),
    "- jaccard (use cosine)": mr(TUNED.replace(sim="cosine")),
    "- zones (flat title/cat/desc)": metrics(model_ranker(rec_flat, TUNED), rec_flat, users, n_eval=N),
    "- adaptive w (-> tuned fixed w)": main["blend tuned, fixed w"],
    "- behavior (content + g(d) only)": mr(TUNED.replace(mode="content")),
    "- content (behavior + g(d) only)": mr(TUNED.replace(mode="behavior")),
}
table(abl, "ablation")

# 3) cluster pruning / champion lists vs brute force ---------------------------------------
pr = pd.DataFrame(pruning_experiment(rec, users))
pr.to_csv(out / "pruning.csv", index=False)
print("\n== pruning vs brute force ==\n", pr.round(4).to_string())
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.scatter(pr["ms_per_query"], pr["overlap_with_brute_top10"])
for _, r in pr.iterrows():
    ax.annotate(r["method"], (r["ms_per_query"], r["overlap_with_brute_top10"]), fontsize=7)
ax.set_xlabel("ms / query"); ax.set_ylabel("top-10 overlap with brute force"); ax.set_title("Pruning: speed vs recall")
fig.tight_layout(); fig.savefig(out / "pruning.png", dpi=150); plt.close(fig)

# 4) cold-start curve ---------------------------------------------------------------------
cs_in = {k: main[k] for k in ["popularity", "centroid tf-idf (content)", "behavior only (item-item)",
                              "blend tuned (+g(d), jaccard)", "blend tuned, fixed w"]}
# metrics() subsamples with a fixed seed, so ranks line up across models
cs = pd.DataFrame(cold_start_curve(rec, cs_in))
cs.to_csv(out / "cold_start.csv", index=False)
print("\n== cold-start (Hit@10 by target's train interactions) ==\n", cs.round(4).to_string())
fig, ax = plt.subplots(figsize=(6, 4))
for c in cs.columns[1:-1]:
    ax.plot(cs["bin"], cs[c], marker="o", label=c)
ax.set_xlabel("train interactions of the target item"); ax.set_ylabel("Hit@10")
ax.set_title("Cold-start curve"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(out / "cold_start.png", dpi=150); plt.close(fig)

# 5) explanation faithfulness (counterfactual) --------------------------------------------
hs = [[rec.index.asins[r] for r in h] for h, _ in users if len(h) >= 2][:200]
faith = faithfulness(rec, hs, TUNED, top_n=3, K=50)
json.dump(faith, open(out / "faithfulness.json", "w"), indent=2)
print("\n== faithfulness ==\n", json.dumps(faith, indent=2))

# 6) MMR + variant collapsing: coverage vs accuracy ----------------------------------------
groups = variant_groups(products)
print(f"\nvariant groups: {len(set(groups))} for {len(groups)} listings")
rows = {"no MMR": metrics(model_ranker(rec, TUNED), rec, users, n_eval=N, groups=groups)}
for collapse in (False, True):
    for lam in (0.8, 0.6, 0.4):
        name = f"MMR lam={lam}" + (" + variant collapse" if collapse else "")
        g = groups if collapse else None
        rows[name] = metrics(lambda h, k, g=g, lam=lam: mmr_rerank(rec, h, k, TUNED, lam=lam, groups=g),
                             rec, users, n_eval=N, groups=groups)
mm = table(rows, "mmr_diversity", SHOW + ["Distinct@10"])
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.scatter(mm["Hit@10"], mm["ILD@10"])
for n, r in mm.iterrows():
    ax.annotate(n, (r["Hit@10"], r["ILD@10"]), fontsize=6)
ax.set_xlabel("Hit@10 (accuracy)"); ax.set_ylabel("ILD@10 (diversity)"); ax.set_title("Accuracy vs diversity")
fig.tight_layout(); fig.savefig(out / "mmr.png", dpi=150); plt.close(fig)
print("\nall results written to", out)
