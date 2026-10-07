# Cold-Start-Aware, Explainable E-Commerce Recommendation
CSD358 IR Hackathon, Track T3 (Recommender systems and personalization)

Products are **documents**, a shopper's history is a **query**. Text (tf-idf cosine) finds substitutes, reviewer
overlap (item-item CF, Linden et al. 2003) finds complements. The blend weight depends on how many interactions a
candidate has (`w = n/(n+tau)`), so new items fall back to text. Each recommendation is explained with the terms
that drive the cosine and, where behavior drove it, the number of co-buyers; explanations are tested
counterfactually. MMR re-ranking with variant collapsing (Jaccard over title shingles) reduces duplicate listings.

## Setup
    pip install -r requirements.txt        # Python 3.10+

## Data
Amazon Reviews 2023 (Hou et al., 2024, McAuley Lab), Appliances category, from Hugging Face:
https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 (files `raw/review_categories/Appliances.jsonl`
and `raw/meta_categories/meta_Appliances.jsonl`).

    python3 -m src.data prepare --meta meta_Appliances.jsonl --reviews Appliances.jsonl --max-products 30000

Users with >= 3 reviews are kept, then users are sampled (full histories) until about 30k products are covered:
30,000 products, 130,000 interactions, 35,208 users. For huge files add `--user-frac 0.05` (streaming user sample).
Reviews are purchase proxies; only anonymised ids are used. `python3 -m src.data synthetic` makes a toy world for
smoke tests only; never report its numbers.

## Reproduce
    python3 tune.py --n-tune 1500 --n-test 6000     # tunes on one half of users, reports on the other half (bootstrap CIs) -> results/tuned.json
    python3 run_experiments.py                      # ablation, cold-start, pruning, faithfulness, MMR -> results/*.csv, *.png
    python3 alpha_sweep.py --n 3000                 # popularity weight / blend-weight cap trade-off -> results/alpha_sweep.csv
    python3 demo.py --user 3 --tuned --mmr 0.8      # CLI demo: postings, weights, scores, explanations
    python3 demo.py --user 3 --tuned --category Dishwasher --max-price 60   # parametric filters + cosine ranking
    python3 demo.py --user 3 --tuned --alpha 0.1    # lower popularity weight: more relevance-driven lists
    python3 inspect_variants.py                     # hand-check merged near-duplicate listings
    python3 -m streamlit run app.py                 # dashboard: recommend, search the index, explanation test, results

`app.py` is a Streamlit dashboard over the same code (frontend is not graded; it is only for demonstration). Tabs:
Recommend (net score split into relevance and g(d), explanations, filters, MMR), Search the index (query terms, idf,
postings, term-at-a-time cosine), Explanation test (live counterfactual), Results (tables and graphs from `results/`).

Evaluation protocol: per user, the last interaction is the target (temporal split); co-purchase statistics and g(d)
use train interactions only. Baselines: popularity and single-centroid tf-idf.

## Results (held-out 6,000 users, Hit@10 with 95% bootstrap CI)
| Model | Hit@10 | NDCG@10 |
|---|---|---|
| Popularity | 0.0202 [0.0167, 0.0237] | 0.0104 |
| Content only (centroid tf-idf) | 0.0400 [0.0352, 0.0450] | 0.0222 |
| Blend default (tau=5) | 0.0520 [0.0467, 0.0577] | 0.0337 |
| Blend tuned, adaptive weight | 0.0748 [0.0683, 0.0818] | 0.0455 |
| Blend tuned, fixed weight | 0.0792 [0.0725, 0.0860] | 0.0466 |

Text + behavior significantly beats either alone and popularity; adaptive vs fixed weight overall is not
significant, but only the adaptive weight works on items with no interactions. Full tables and graphs are in
`results/` and the report.

## Map to lecture concepts
| Concept | Where |
|---|---|
| Term vocabulary (case folding, stop words, Porter, model codes kept intact) | `src/tfidf.py: tokenize` |
| Zone index (title x3, category x2, description x1) | `src/tfidf.py: ZONES` |
| lnc documents, ltc query (idf-weighted centroid profile), cosine | `src/tfidf.py: TfidfIndex` |
| Postings, term-at-a-time scoring, index elimination | `Dc`, `cosine_taat` |
| Heap top-K | `src/retrieval.py: topk` |
| Champion lists, cluster pruning | `src/retrieval.py` |
| Jaccard / cosine over reviewer sets | `src/copurchase.py` |
| Static quality g(d), net score | `src/copurchase.py: StaticQuality`, `src/recommender.py` |
| Parametric filters (category, brand, price) | `Recommender._mask`, `demo.py --category/--brand/--min-price/--max-price` |
| Jaccard over shingles (variant collapsing) + MMR | `src/diversity.py` |
| Term-level explanations + counterfactual faithfulness | `src/explain.py` |
| Temporal split, Hit/P/Recall/NDCG, coverage, ILD | `src/evaluate.py` |

### Popularity trade-off (`alpha_sweep.py`, 3,000 held-out users)
Raising the popularity weight alpha raises Hit@10 up to alpha = 0.5 (0.0737), then it falls (0.0667 at 0.8) while
catalogue coverage collapses (0.25 at 0.5, 0.09 at 0.8). At alpha = 0.5, 45% of top-10 items are popularity-driven
(static quality exceeds relevance); at alpha = 0.3 it is 11% with Hit@10 of 0.0683. A cap on the blend weight
(`w_max`) is supported in the code; in the sweep it looked helpful but was not re-tuned on a separate split.

## What works / what is planned
Works: everything above, end to end on the real data (Amazon Appliances), including the dashboard. Not done: BM25 / dense-retrieval comparison channel and target
attention over history (optional extras). Planned if continued: denser category (Cell Phones and Accessories),
re-tuning with a blend-weight cap, a compatibility filter for appliance parts, popularity-aware explanations.

## Known limitations
Reviews are a proxy for purchases; offline metrics only approximate shopper satisfaction; product text quality
varies; sparse data (about 3.7 interactions per user, one category); popularity raises accuracy but lowers catalogue
coverage; the counterfactual test removes single history items; variant collapsing uses one fixed threshold.

## AI-use declaration
We used Claude (Anthropic) as an AI coding assistant. It generated most of the code in this repository and drafted
the report text, working from our design and requirements; we ran all experiments on our own machines and reviewed
the results. No pretrained models or LLM APIs are part of the recommender itself. Libraries: numpy, scipy, pandas,
scikit-learn (stop words, normalisation), NLTK (Porter stemmer), matplotlib, pyarrow, streamlit.