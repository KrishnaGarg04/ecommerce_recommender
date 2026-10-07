"""Term vocabulary, zone-weighted lnc document vectors, postings, ltc query vectors.

Lecture mapping:
  tokenize()          -> term vocabulary (case folding, stop words, Porter stemming; tokens with digits
                         such as '128gb', 'usb-c', 'x450' are kept intact: model codes and units)
  ZONES               -> zone index: title x3, category x2, description x1 (weighted term frequency)
  TfidfIndex.D        -> documents as lnc vectors (log tf, no idf, cosine normalised)
  TfidfIndex.Dc       -> the same matrix column-wise = postings lists (term -> (doc, weight))
  profile_vector()    -> ltc query (user profile = idf-weighted centroid of history docs, Rocchio-style)
  cosine_taat()       -> term-at-a-time cosine scoring over postings, optional index elimination
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize

ZONES = {"title": 3.0, "category": 2.0, "description": 1.0}
FLAT_ZONES = {"title": 1.0, "category": 1.0, "description": 1.0}

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
_STEMMER = PorterStemmer()
_STEM_CACHE: dict[str, str] = {}


def _stem(w: str) -> str:
    s = _STEM_CACHE.get(w)
    if s is None:
        s = _STEM_CACHE[w] = _STEMMER.stem(w)
    return s


def tokenize(text: str, surface=None) -> list[str]:
    out = []
    for m in _TOKEN.findall((text or "").lower()):
        if len(m) < 2 or m in ENGLISH_STOP_WORDS:
            continue
        t = _stem(m) if m.isalpha() else m
        out.append(t)
        if surface is not None:
            surface[t][m] += 1
    return out


class TfidfIndex:
    def __init__(self, products, zone_weights=None):
        self.zone_weights = zone_weights or ZONES
        self.asins = products["asin"].tolist()
        self.asin2row = {a: i for i, a in enumerate(self.asins)}
        self.N = len(self.asins)
        surface = defaultdict(Counter)
        vocab: dict[str, int] = {}
        r, c, v = [], [], []
        for i, rec in enumerate(products.itertuples(index=False)):
            tf = Counter()
            for zone, w in self.zone_weights.items():
                for t in tokenize(getattr(rec, zone), surface):
                    tf[t] += w
            for t, f in tf.items():
                j = vocab.setdefault(t, len(vocab))
                r.append(i)
                c.append(j)
                v.append(1 + math.log10(f) if f >= 1 else f)          # log-frequency tf
        self.vocab = vocab
        self.V = len(vocab)
        self.terms = [None] * self.V
        for t, j in vocab.items():
            self.terms[j] = t
        D = sp.csr_matrix((v, (r, c)), shape=(self.N, self.V), dtype=np.float32)
        self.D = normalize(D, norm="l2", axis=1).astype(np.float32).tocsr()   # 'lnc' + cosine norm
        self.Dc = self.D.tocsc()                                              # postings by term
        self.df = np.diff(self.Dc.indptr)
        self.idf = np.log10(self.N / np.maximum(self.df, 1)).astype(np.float32)
        self._surface = surface
        self._surface_cache: dict[int, str] = {}

    # ---- helpers -------------------------------------------------------------------------
    def surface_word(self, j: int) -> str:
        w = self._surface_cache.get(j)
        if w is None:
            w = self._surface_cache[j] = self._surface[self.terms[j]].most_common(1)[0][0]
        return w

    def postings(self, word: str, top: int = 5):
        """Debug/demo view of a postings list: [(asin, weight)] sorted by weight."""
        toks = tokenize(word)
        if not toks or toks[0] not in self.vocab:
            return []
        j = self.vocab[toks[0]]
        s, e = self.Dc.indptr[j], self.Dc.indptr[j + 1]
        rows, w = self.Dc.indices[s:e], self.Dc.data[s:e]
        order = np.argsort(-w)[:top]
        return [(self.asins[rows[o]], float(w[o])) for o in order]

    # ---- query side: ltc -----------------------------------------------------------------
    def profile_vector(self, rows, weights=None) -> np.ndarray:
        """User profile / query: idf-weighted centroid of history doc vectors, cosine-normalised."""
        rows = np.asarray(rows, dtype=int)
        w = np.ones(len(rows), dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
        v = np.asarray(self.D[rows].T @ w).ravel() * self.idf
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)

    def text_query_vector(self, text: str) -> np.ndarray:
        tf = Counter(t for t in tokenize(text) if t in self.vocab)
        v = np.zeros(self.V, dtype=np.float32)
        for t, f in tf.items():
            j = self.vocab[t]
            v[j] = (1 + math.log10(f)) * self.idf[j]
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    # ---- efficient cosine: term-at-a-time over postings -----------------------------------
    def cosine_taat(self, q: np.ndarray, min_idf: float = 0.0) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float32)
        nz = np.flatnonzero(q)
        if min_idf > 0:                                   # index elimination: skip low-idf terms
            nz = nz[self.idf[nz] >= min_idf]
        ind, dat, ptr = self.Dc.indices, self.Dc.data, self.Dc.indptr
        for j in nz:
            s, e = ptr[j], ptr[j + 1]
            scores[ind[s:e]] += dat[s:e] * q[j]
        return scores
