"""Live demo for the video: postings, weights, scores and explanations for one user.
   python3 demo.py --user 3 --tuned --mmr 0.8 --k 5
   parametric filters: --category "Dishwasher" --brand "GE" --min-price 10 --max-price 60
   --tuned : use the config chosen by tune.py (results/tuned.json)   --mmr LAM : MMR re-rank + collapse variant listings
"""
import argparse
import json
from pathlib import Path

from src.data import load_dataset
from src.diversity import mmr_rerank, variant_groups
from src.evaluate import build
from src.explain import explain
from src.recommender import Config

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--user", type=int, default=0, help="index into the test users")
ap.add_argument("--mode", default="blend", choices=["content", "behavior", "blend"])
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--word", default="filter", help="show this term's postings list")
ap.add_argument("--tuned", action="store_true", help="use results/tuned.json (adaptive config)")
ap.add_argument("--mmr", type=float, default=None, help="MMR lambda; also collapses near-duplicate variants")
ap.add_argument("--alpha", type=float, default=None, help="override the popularity weight (e.g. 0.1) to see relevance-driven lists")
ap.add_argument("--category", default=None, help="parametric filter: category text contains this")
ap.add_argument("--brand", default=None, help="parametric filter: exact brand/store name")
ap.add_argument("--min-price", type=float, default=None)
ap.add_argument("--max-price", type=float, default=None)
a = ap.parse_args()

products, inter = load_dataset(a.data)
rec, users, _ = build(products, inter)
idx = rec.index
print(f"index: {idx.N} docs, {idx.V} terms | postings('{a.word}') top weights:", idx.postings(a.word, 3))
hist, target = users[a.user]
hist_asins = [idx.asins[r] for r in hist]
print("\nHISTORY:")
for r in hist:
    print("  -", products.at[r, "title"][:120])
print("HELD-OUT NEXT ITEM:", products.at[target, "title"][:120])

cfg = Config(mode=a.mode)
tj = Path("results/tuned.json")
if a.tuned and tj.exists():
    d = json.load(open(tj))["adaptive"]
    cfg = Config(tau=d["tau"], alpha=d["alpha"], beta=d["beta"], gamma=d["gamma"], sim=d["sim"])
    print("config:", cfg)
if a.alpha is not None:
    cfg = cfg.replace(alpha=a.alpha)
    print("popularity weight alpha overridden to", a.alpha)
filters = {k: v for k, v in dict(category=a.category, brand=a.brand, min_price=a.min_price,
                                 max_price=a.max_price).items() if v is not None}
if filters:
    cfg = cfg.replace(filters=filters)
    print("parametric filters (cosine ranking restricted to matching products):", filters)
if a.mmr is not None:
    pool = {x["asin"]: x for x in rec.recommend(hist_asins, 100, cfg)}
    picks = mmr_rerank(rec, hist, a.k, cfg, lam=a.mmr, groups=variant_groups(products))
    recs = [pool[idx.asins[r]] for r, _ in picks if idx.asins[r] in pool]
    print(f"\nTOP-{a.k} (blend + MMR lam={a.mmr} + variant collapsing):")
else:
    recs = rec.recommend(hist_asins, a.k, cfg)
    print(f"\nTOP-{a.k} ({cfg.mode}):")
for x in recs:
    e = explain(rec, hist_asins, x["asin"], cfg)
    hit = "  <== HELD-OUT ITEM" if x["asin"] == idx.asins[target] else ""
    gd = cfg.alpha * x["pop"] + cfg.beta * x["rating"] + cfg.gamma * x["recency"]   # static quality part of net
    rel = x["net"] - gd
    why = e["text"]
    if gd > rel:   # the text/behavior evidence is weaker than the static-quality boost: say so honestly
        why = (f"Mostly a popular item (static quality {gd:.2f} vs relevance {rel:.2f}); "
               f"text match is weak (cos={x['cosine']:.2f}).")
    print(f"  {x['title'][:120]}{hit}\n     net={x['net']:.3f} cos={x['cosine']:.3f} beh={x['behavior']:.3f} "
          f"w={x['w']:.2f} | relevance={rel:.3f} static g(d)={gd:.3f}\n     {why[:260]}  top terms: {e['top_terms'][:3]}")
