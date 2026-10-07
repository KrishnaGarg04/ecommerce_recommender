"""Streamlit dashboard for the video:  python3 -m streamlit run app.py
Tabs: Recommend (scores + explanations) | Search the index (postings, tf-idf, build a history) |
      Explanation test (counterfactual) | Results (tables and graphs from results/).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data import load_dataset
from src.diversity import mmr_rerank, variant_groups
from src.evaluate import build
from src.explain import explain
from src.recommender import Config
from src.retrieval import topk

st.set_page_config(page_title="Explainable e-commerce recommender (IR)", layout="wide")
DATA = "data"


@st.cache_resource(show_spinner="Building tf-idf index and co-purchase matrix...")
def load():
    products, inter = load_dataset(DATA)
    rec, users, _ = build(products, inter)
    return products, rec, users


@st.cache_resource(show_spinner="Detecting near-duplicate listings (Jaccard over title shingles)...")
def get_groups():
    return variant_groups(load()[0])


def tuned_cfg():
    p = Path("results/tuned.json")
    if p.exists():
        d = json.load(open(p))["adaptive"]
        return Config(tau=d["tau"], alpha=d["alpha"], beta=d["beta"], gamma=d["gamma"], sim=d["sim"])
    return Config(tau=5.0, alpha=0.5, beta=0.1, gamma=0.1, sim="jaccard")


products, rec, users = load()
idx = rec.index
if "custom" not in st.session_state:
    st.session_state.custom = []

# ------------------------------------------------------------------ sidebar
PRESETS = {
    "Blend, tuned": tuned_cfg(),
    "Blend, popularity-light (alpha 0.2)": tuned_cfg().replace(alpha=0.2),
    "Blend, default (no g(d))": Config(),
    "Content only (centroid tf-idf)": Config(mode="content"),
    "Behavior only (item-item co-purchase)": Config(mode="behavior"),
}
st.sidebar.header("Model")
preset = st.sidebar.selectbox("Preset", list(PRESETS))
base = PRESETS[preset]
with st.sidebar.expander("Fine-tune the net score", expanded=False):
    alpha = st.slider("alpha: popularity", 0.0, 1.0, float(base.alpha), 0.05, key=f"a{preset}")
    beta = st.slider("beta: rating quality", 0.0, 0.5, float(base.beta), 0.05, key=f"b{preset}")
    gamma = st.slider("gamma: recency", 0.0, 0.5, float(base.gamma), 0.05, key=f"g{preset}")
    tau = st.slider("tau: w = n/(n+tau)", 1.0, 50.0, float(base.tau), 1.0, key=f"t{preset}")
    cap = st.checkbox("cap blend weight w_max", value=False, key=f"c{preset}")
    wmax = st.slider("w_max", 0.3, 1.0, 0.6, 0.05, key=f"w{preset}") if cap else None
    sim = st.radio("reviewer-set similarity", ["cosine", "jaccard"], index=["cosine", "jaccard"].index(base.sim),
                   horizontal=True, key=f"s{preset}")
cfg = base.replace(alpha=alpha, beta=beta, gamma=gamma, tau=tau, w_max=wmax, sim=sim)

st.sidebar.header("List options")
K = st.sidebar.slider("Top-K", 3, 15, 5)
use_mmr = st.sidebar.checkbox("MMR re-ranking (diversity)", value=False)
lam = st.sidebar.slider("MMR lambda (1 = relevance only)", 0.3, 1.0, 0.8, 0.05) if use_mmr else 1.0
collapse = st.sidebar.checkbox("Collapse near-duplicate variants (Jaccard over shingles)", value=False)
st.sidebar.header("Parametric filters")
f_cat = st.sidebar.text_input("Category contains")
f_brand = st.sidebar.text_input("Brand / seller (exact)")
f_price = st.sidebar.checkbox("Price range")
if f_price:
    pmax = float(np.nanmax(products["price"])) if products["price"].notna().any() else 1000.0
    lo, hi = st.sidebar.slider("Price ($)", 0.0, min(pmax, 500.0), (0.0, 60.0), 1.0)
filters = {}
if f_cat:
    filters["category"] = f_cat
if f_brand:
    filters["brand"] = f_brand
if f_price:
    filters.update(min_price=lo, max_price=hi)
if filters:
    cfg = cfg.replace(filters=filters)


# ------------------------------------------------------------------ helpers
def get_recs(hist_rows, k, cfg):
    asins = [idx.asins[r] for r in hist_rows]
    if use_mmr or collapse:
        pool = {x["asin"]: x for x in rec.recommend(asins, 100, cfg)}
        picks = mmr_rerank(rec, hist_rows, k, cfg, lam=lam, groups=get_groups() if collapse else None)
        return [pool[idx.asins[r]] for r, _ in picks if idx.asins[r] in pool]
    return rec.recommend(asins, k, cfg)


def parts(x, cfg):
    gd = cfg.alpha * x["pop"] + cfg.beta * x["rating"] + cfg.gamma * x["recency"]
    return x["net"] - gd, gd


st.title("Cold-start-aware, explainable e-commerce recommender")
st.caption(f"CSD358 IR hackathon, Track T3 | {idx.N:,} products, {idx.V:,} terms, "
           f"{int(rec.cp.count.sum()):,} train interactions | products = documents, history = query")
t1, t2, t3, t4 = st.tabs(["1. Recommend", "2. Search the index", "3. Explanation test", "4. Results"])

# ------------------------------------------------------------------ tab 1
with t1:
    src = st.radio("History source", ["Test user (held-out next item is known)", "My own history (build it in tab 2)"],
                   horizontal=True)
    target_asin = None
    if src.startswith("Test"):
        c1, c2 = st.columns([3, 1])
        if c2.button("Random user"):
            st.session_state.uidx = int(np.random.default_rng().integers(len(users)))
        uidx = c1.number_input("Test user index", 0, len(users) - 1, st.session_state.get("uidx", 3), key="uidx")
        hist_rows, target = users[int(uidx)]
        target_asin = idx.asins[target]
        st.markdown("**History (what the shopper already bought)**")
        for r in hist_rows:
            st.write("-", products.at[r, "title"][:150])
        st.markdown(f"**Held-out next item:** {products.at[target, 'title'][:150]}")
    else:
        hist_rows = [idx.asin2row[a] for a in st.session_state.custom]
        st.markdown("**Your history**")
        for r in hist_rows:
            st.write("-", products.at[r, "title"][:150])
        if not hist_rows:
            st.info("Empty. Go to tab 2, search the catalogue, and add a few products.")

    if hist_rows:
        recs = get_recs(hist_rows, K, cfg)
        if not recs:
            st.warning("No products match the filters.")
        else:
            hist_asins = [idx.asins[r] for r in hist_rows]
            rows = []
            for i, x in enumerate(recs, 1):
                rel, gd = parts(x, cfg)
                e = explain(rec, hist_asins, x["asin"], cfg)
                why = (f"Popular item (static quality {gd:.2f} > relevance {rel:.2f})" if gd > rel else e["text"])
                rows.append({"#": i, "Product": x["title"][:110], "net": round(x["net"], 3),
                             "relevance": round(rel, 3), "g(d)": round(gd, 3), "cosine": round(x["cosine"], 3),
                             "behavior": round(x["behavior"], 3), "blend w": round(x["w"], 2), "Why": why,
                             "": "HIT: held-out item" if x["asin"] == target_asin else ""})
            df = pd.DataFrame(rows)
            st.subheader(f"Top-{K} ({preset}" + (f", MMR lambda={lam}" if use_mmr else "")
                         + (", variants collapsed" if collapse else "") + ")")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown("**Net score = relevance (text/behavior blend) + g(d) (popularity, rating, recency)**")
            st.bar_chart(df.set_index("#")[["relevance", "g(d)"]], height=220)
            n_pop = sum(parts(x, cfg)[1] > parts(x, cfg)[0] for x in recs)
            if n_pop:
                st.warning(f"{n_pop} of {len(recs)} items are popularity-driven (g(d) > relevance). "
                           "Lower alpha in the sidebar to see relevance-driven lists.")
            if target_asin and any(x["asin"] == target_asin for x in recs):
                st.success("The held-out item is in this list.")
            st.caption("blend w = n/(n+tau) per candidate: 0 = brand-new item, ranked on text only; "
                       "near 1 = well-observed item, ranked mostly on co-purchase.")

# ------------------------------------------------------------------ tab 2
with t2:
    st.markdown("Type a query: it becomes an ltc vector, and the cosine is computed over the **postings lists** of its terms.")
    q = st.text_input("Query", "dishwasher drain pump")
    if q:
        qv = idx.text_query_vector(q)
        nz = np.flatnonzero(qv)
        if len(nz) == 0:
            st.warning("No query term is in the vocabulary.")
        else:
            st.markdown("**Query terms, idf and postings**")
            st.dataframe(pd.DataFrame([{"term (stem)": idx.terms[j], "surface": idx.surface_word(int(j)),
                                        "df": int(idx.df[j]), "idf": round(float(idx.idf[j]), 2),
                                        "query weight (ltc)": round(float(qv[j]), 3),
                                        "top posting": idx.postings(idx.surface_word(int(j)), 1)[:1]}
                                       for j in nz]), use_container_width=True, hide_index=True)
            s = idx.cosine_taat(qv)
            top = topk(s, 10)
            res = pd.DataFrame([{"asin": idx.asins[r], "cosine": round(sc, 3), "Product": products.at[r, "title"][:110]}
                                for r, sc in top])
            st.markdown("**Top-10 by term-at-a-time cosine (heap top-K)**")
            st.dataframe(res, use_container_width=True, hide_index=True)
            labels = {f"{products.at[r, 'title'][:90]} [{idx.asins[r]}]": idx.asins[r] for r, _ in top}
            pick = st.multiselect("Add products to my history", list(labels))
            c1, c2 = st.columns(2)
            if c1.button("Add to history"):
                for p in pick:
                    if labels[p] not in st.session_state.custom:
                        st.session_state.custom.append(labels[p])
                st.rerun()
            if c2.button("Clear my history"):
                st.session_state.custom = []
                st.rerun()
    st.write("My history:", [products.at[idx.asin2row[a], "title"][:60] for a in st.session_state.custom] or "empty")

# ------------------------------------------------------------------ tab 3
with t3:
    st.markdown("**Counterfactual faithfulness test, live:** remove the history item an explanation cites and see "
                "whether the recommendation falls. A control removes a different history item.")
    if not hist_rows or len(hist_rows) < 2:
        st.info("Needs a history of at least 2 items (pick a test user in tab 1).")
    else:
        recs50 = rec.recommend([idx.asins[r] for r in hist_rows], 50, cfg)
        hist_asins = [idx.asins[r] for r in hist_rows]
        titles = [f"{i + 1}. {x['title'][:100]}" for i, x in enumerate(recs50[:10])]
        choice = st.selectbox("Recommendation to test", titles)
        pos = titles.index(choice)
        x = recs50[pos]
        e = explain(rec, hist_asins, x["asin"], cfg)
        st.write("Explanation:", e["text"])
        st.write("Top contributing terms (q_t x d_t):", e["top_terms"])

        def rank_of(h):
            r = rec.recommend(h, 50, cfg)
            return next((i + 1 for i, y in enumerate(r) if y["asin"] == x["asin"]), 51)

        cited = e["because_asin"]
        others = [a for a in hist_asins if a != cited]
        r_cited = rank_of(others)
        r_ctrl = rank_of([a for a in hist_asins if a != others[0]]) if others else None
        m1, m2, m3 = st.columns(3)
        m1.metric("Rank with full history", pos + 1)
        m2.metric("Rank without the CITED item", r_cited if r_cited <= 50 else ">50", delta=r_cited - (pos + 1),
                  delta_color="inverse")
        if r_ctrl is not None:
            m3.metric("Rank without a CONTROL item", r_ctrl if r_ctrl <= 50 else ">50", delta=r_ctrl - (pos + 1),
                      delta_color="inverse")
        st.caption("Over 600 test items (results tab): removing the cited item dropped the recommendation 86% of the time "
                   "vs 31% for a random control.")

# ------------------------------------------------------------------ tab 4
with t4:
    R = Path("results")

    def show(name, title, note=""):
        p = R / name
        if p.exists():
            st.subheader(title)
            st.dataframe(pd.read_csv(p), use_container_width=True, hide_index=True)
            if note:
                st.caption(note)

    show("final_test_with_ci.csv", "Held-out users: bootstrap 95% CI", "Tuned on one half of the users, reported on the other.")
    show("ablation.csv", "Ablation: one component removed at a time")
    p = R / "cold_start.csv"
    if p.exists():
        st.subheader("Cold-start curve: Hit@10 by the target item's training interactions")
        cs = pd.read_csv(p)
        st.line_chart(cs.set_index("bin").drop(columns=["n"], errors="ignore"))
        st.dataframe(cs, use_container_width=True, hide_index=True)
    show("pruning.csv", "Champion lists / cluster pruning vs brute force")
    show("mmr_diversity.csv", "MMR and variant collapsing", "Distinct@10 = distinct variant groups in the top 10.")
    show("alpha_sweep.csv", "Popularity weight sweep and blend-weight cap")
    if (R / "faithfulness.json").exists():
        st.subheader("Explanation faithfulness")
        st.json(json.load(open(R / "faithfulness.json")))