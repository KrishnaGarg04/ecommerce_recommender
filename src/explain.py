"""Term-level explanations + counterfactual faithfulness test.

explain(): top terms = per-term contribution q_t * d_t to the cosine; cited history item =
largest share of the (content, behavior) blend, using the same weight w as the ranker.
faithfulness(): remove the cited item from the history, re-rank; a faithful explanation means the
recommendation falls. Control: remove a random *other* history item.
"""
from __future__ import annotations

import numpy as np

from .recommender import Config
from .tfidf import tokenize


def explain(rec, history, asin, cfg: Config | None = None, n_terms=5):
    cfg = cfg or Config()
    idx = rec.index
    hist = [idx.asin2row[a] for a in history]
    r = idx.asin2row[asin]
    q = idx.profile_vector(hist)
    d = idx.D[r]
    contrib = q[d.indices] * d.data
    # drop terms that only come from the seller/brand name (e.g. 'supplying demand'): they match but explain nothing
    brand = set(tokenize(rec.products.at[r, "brand"]))
    order = [o for o in np.argsort(-contrib) if idx.terms[int(d.indices[o])] not in brand][:n_terms]
    top_terms = [(idx.surface_word(int(d.indices[o])), round(float(contrib[o]), 4))
                 for o in order if contrib[o] > 0]

    c_h = np.array([float(idx.profile_vector([h])[d.indices] @ d.data) for h in hist])
    sb = [rec.cp.pair_sim(h, r, cfg.sim) for h in hist]
    b_h = np.array([x[0] for x in sb])
    n = rec.cp.count[r]
    w = cfg.w_fixed if cfg.w_fixed is not None else n / (n + cfg.tau)
    if cfg.w_fixed is None and cfg.w_max is not None:
        w = min(w, cfg.w_max)
    if cfg.mode == "content":
        w = 0.0
    elif cfg.mode == "behavior":
        w = 1.0
    share_c = (1 - w) * c_h / (c_h.sum() + 1e-9)
    share_b = w * b_h / (b_h.sum() + 1e-9)
    share = share_c + share_b
    best = int(np.argmax(share))
    reason = "behavior" if share_b[best] > share_c[best] and b_h[best] > 0 else "content"
    title = lambda row: rec.products.at[row, "title"]
    if reason == "behavior":
        text = (f"Because you bought '{title(hist[best])}': {sb[best][1]} shoppers who bought it "
                f"also bought this.")
    else:
        words = ", ".join(t for t, _ in top_terms[:3])
        text = f"Because you bought '{title(hist[best])}': similar product text ({words})."
    return dict(asin=asin, title=title(r), top_terms=top_terms, because_asin=idx.asins[hist[best]],
                because_title=title(hist[best]), reason=reason, co_buyers=sb[best][1], text=text)


def _rank(recs, asin, K):
    for i, x in enumerate(recs):
        if x["asin"] == asin:
            return i + 1
    return K + 1                                               # fell out of the list


def faithfulness(rec, histories, cfg: Config | None = None, top_n=3, K=50, seed=0):
    """histories: list of asin lists (len >= 2)."""
    cfg = cfg or Config()
    rng = np.random.default_rng(seed)
    cited_shift, ctrl_shift, cited_drop, ctrl_drop = [], [], [], []
    for hist in histories:
        recs = rec.recommend(hist, K, cfg)
        for pos, x in enumerate(recs[:top_n], start=1):
            cited = explain(rec, hist, x["asin"], cfg)["because_asin"]
            others = [a for a in hist if a != cited]
            if not others:
                continue
            ctrl = others[int(rng.integers(len(others)))]
            r_c = _rank(rec.recommend([a for a in hist if a != cited], K, cfg), x["asin"], K)
            r_r = _rank(rec.recommend([a for a in hist if a != ctrl], K, cfg), x["asin"], K)
            cited_shift.append(r_c - pos)
            ctrl_shift.append(r_r - pos)
            cited_drop.append(r_c > pos)
            ctrl_drop.append(r_r > pos)
    return dict(n=len(cited_shift), cited_drop_rate=float(np.mean(cited_drop)),
                control_drop_rate=float(np.mean(ctrl_drop)), cited_mean_rank_shift=float(np.mean(cited_shift)),
                control_mean_rank_shift=float(np.mean(ctrl_shift)))