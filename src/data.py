"""Data preparation.

Outputs (same schema for real and synthetic data):
  data/products.parquet      asin,title,description,category,brand,price,avg_rating,rating_count
  data/interactions.parquet  user_id,asin,ts,rating          (ts in seconds)

Usage:
  python -m src.data prepare --meta META.jsonl[.gz] --reviews REVIEWS.jsonl[.gz]|.csv[.gz] [--max-products 30000]
  python -m src.data synthetic            # SMOKE-TEST DATA ONLY, never report its numbers
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _open(path):
    p = str(path)
    return gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, "r", encoding="utf-8")


def _filter_users(inter: pd.DataFrame, min_user: int) -> pd.DataFrame:
    c = inter.groupby("user_id")["asin"].transform("size")
    return inter[c >= min_user]


def prepare_amazon2023(meta_path, review_path, out_dir="data", max_products=30000, min_user=3, seed=0, user_frac=1.0):
    """Amazon Reviews 2023 (Hou et al. 2024). Reviews are used as purchase proxies."""
    cols = ["user_id", "parent_asin", "rating", "timestamp"]
    p = str(review_path)

    def keep_frac(c):
        """Stream-friendly user sampling: hash(user_id) decides, so a sampled user keeps ALL their reviews."""
        c = c[cols]
        if user_frac < 1.0:
            h = pd.util.hash_pandas_object(c["user_id"], index=False).to_numpy() % 10_000
            c = c[h < user_frac * 10_000]
        return c

    if ".csv" in p:
        reader = pd.read_csv(review_path, usecols=cols, chunksize=1_000_000)
    else:
        reader = pd.read_json(review_path, lines=True, chunksize=500_000, convert_dates=False, dtype=False)
    inter = pd.concat([keep_frac(c) for c in reader], ignore_index=True)
    inter = inter.rename(columns={"parent_asin": "asin", "timestamp": "ts"}).dropna()
    inter["ts"] = inter["ts"].astype("float64")
    inter["ts"] = np.where(inter["ts"] > 1e11, inter["ts"] / 1000.0, inter["ts"])  # ms -> s
    inter = inter.drop_duplicates(["user_id", "asin"])

    # Sample USERS (not items): every kept user keeps their full history, so the behavior signal survives,
    # and items seen once or twice stay in the catalogue as genuine cold items.
    inter = _filter_users(inter, min_user)
    if inter["asin"].nunique() > max_products:
        rng = np.random.default_rng(seed)
        users = np.asarray(inter["user_id"].unique(), dtype=object)
        rng.shuffle(users)
        by_user = inter.groupby("user_id")["asin"].apply(set)
        items, keep_u = set(), []
        for u in users:
            keep_u.append(u)
            items |= by_user[u]
            if len(items) >= max_products:
                break
        inter = inter[inter["user_id"].isin(set(keep_u))]
    need = set(inter["asin"])

    rows = []
    with _open(meta_path) as f:
        for line in f:
            m = json.loads(line)
            a = m.get("parent_asin")
            if a in need and m.get("title"):
                desc = " ".join(m.get("description") or []) + " " + " ".join(m.get("features") or [])
                cats = " ".join([m.get("main_category") or ""] + list(m.get("categories") or []))
                rows.append(dict(asin=a, title=m["title"], description=desc.strip(), category=cats.strip(),
                                 brand=m.get("store") or "", price=pd.to_numeric(m.get("price"), errors="coerce"),
                                 avg_rating=m.get("average_rating"), rating_count=m.get("rating_number")))
    products = pd.DataFrame(rows).drop_duplicates("asin")
    inter = inter[inter["asin"].isin(products["asin"])]
    inter = _filter_users(inter, min_user)
    products = products[products["asin"].isin(inter["asin"])]
    _save(products, inter, out_dir)


def make_synthetic(out_dir="data", n_users=6000, n_families=10, seed=0):
    """Toy world: users pick a 'family', buy 2-4 phones of that family, then 1-2 accessories.
    Accessory text carries NO family vocabulary, so only behavior can link phone -> accessory
    (the complement/substitute effect). Zipf popularity creates cold items."""
    rng = np.random.default_rng(seed)
    main_w = "smartphone display camera battery processor screen storage 5g wireless amoled".split()
    common = "premium quality durable design compatible new original lightweight".split()
    acc = {"Case": "protective shockproof case cover shell bumper slim silicone".split(),
           "Charger": "charger adapter fast charging usb power wall plug cable".split(),
           "Protector": "screen protector tempered glass film scratch shatter clear".split()}
    prod, fam_main, fam_acc = [], {}, {}
    for f in range(n_families):
        fw = [f"f{f}w{i}" for i in range(8)]
        fam_main[f], fam_acc[f] = [], []
        for i in range(80):
            a = f"M{f:02d}{i:03d}"
            title = f"brand{f} x{rng.integers(100, 999)} smartphone " + " ".join(rng.choice(main_w, 2))
            desc = " ".join(list(rng.choice(fw, 6)) + list(rng.choice(main_w, 8)) + list(rng.choice(common, 4)))
            prod.append(dict(asin=a, title=title, description=desc, category="Electronics Cell Phones",
                             brand=f"brand{f}", price=float(rng.integers(150, 900))))
            fam_main[f].append(a)
        for i in range(50):
            t = list(acc)[i % 3]
            a = f"A{f:02d}{i:03d}"
            title = f"{rng.choice(acc[t])} {t.lower()} for phones m{rng.integers(100, 999)}"
            desc = " ".join(list(rng.choice(acc[t], 8)) + list(rng.choice(common, 4)))
            prod.append(dict(asin=a, title=title, description=desc, category=f"Electronics Accessories {t}",
                             brand="generic", price=float(rng.integers(5, 60))))
            fam_acc[f].append(a)

    def zipf_pick(pool, n):
        p = 1.0 / (np.arange(len(pool)) + 1) ** 0.8
        p /= p.sum()
        return list(rng.choice(pool, size=n, replace=False, p=p))

    rows = []
    for u in range(n_users):
        f = int(rng.integers(n_families))
        seq = zipf_pick(fam_main[f], int(rng.integers(2, 5))) + zipf_pick(fam_acc[f], int(rng.integers(1, 3)))
        if rng.random() < 0.3:
            rng.shuffle(seq)
        t0 = rng.integers(1_640_000_000, 1_690_000_000)
        for j, a in enumerate(seq):
            rows.append(dict(user_id=f"U{u}", asin=a, ts=float(t0 + j * 86400 * rng.integers(1, 30)),
                             rating=float(rng.integers(3, 6))))
    inter = pd.DataFrame(rows)
    products = pd.DataFrame(prod)
    products = products[products["asin"].isin(inter["asin"])]
    g = inter.groupby("asin")["rating"].agg(["mean", "size"])
    products = products.merge(g, left_on="asin", right_index=True)
    products = products.rename(columns={"mean": "avg_rating", "size": "rating_count"})
    _save(products, inter, out_dir)


def _save(products, inter, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    products.reset_index(drop=True).to_parquet(out / "products.parquet")
    inter.reset_index(drop=True).to_parquet(out / "interactions.parquet")
    print(f"saved {len(products)} products, {len(inter)} interactions, "
          f"{inter['user_id'].nunique()} users -> {out}/")


def load_dataset(data_dir="data"):
    """Returns (products, interactions); interactions get a 'row' column = index row of the product."""
    d = Path(data_dir)
    products = pd.read_parquet(d / "products.parquet").reset_index(drop=True)
    for c in ["title", "description", "category", "brand"]:
        products[c] = products[c].fillna("").astype(str)
    inter = pd.read_parquet(d / "interactions.parquet")
    a2r = {a: i for i, a in enumerate(products["asin"])}
    inter = inter[inter["asin"].isin(a2r)].copy()
    inter["row"] = inter["asin"].map(a2r).astype(int)
    return products, inter.sort_values(["user_id", "ts"]).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--meta", required=True)
    a.add_argument("--reviews", required=True)
    a.add_argument("--out", default="data")
    a.add_argument("--max-products", type=int, default=30000)
    a.add_argument("--user-frac", type=float, default=1.0, help="stream-sample this fraction of users while reading (use for huge files)")
    s = sub.add_parser("synthetic")
    s.add_argument("--out", default="data")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare_amazon2023(args.meta, args.reviews, args.out, args.max_products, user_frac=args.user_frac)
    else:
        make_synthetic(args.out)
