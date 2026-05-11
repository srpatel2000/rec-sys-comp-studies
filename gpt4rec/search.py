"""BM25 search components for GPT4Rec."""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


class BM25SearchIndex:
    def __init__(self, item_text_by_item_id: Dict[int, str], k1: float = 1.2, b: float = 0.75):
        self.item_text = item_text_by_item_id
        self.k1 = k1
        self.b = b
        self.doc_tokens: Dict[int, List[str]] = {}
        self.doc_len: Dict[int, int] = {}
        self.df = defaultdict(int)
        self.N = 0
        self.avgdl = 0.0
        self._build()

    def _build(self):
        total_len = 0
        for iid, txt in self.item_text.items():
            toks = tokenize(txt)
            self.doc_tokens[iid] = toks
            self.doc_len[iid] = len(toks)
            total_len += len(toks)
            for t in set(toks):
                self.df[t] += 1
        self.N = max(1, len(self.doc_tokens))
        self.avgdl = total_len / self.N if self.N else 1.0

    def set_params(self, k1: float, b: float):
        self.k1 = float(k1)
        self.b = float(b)

    def _idf(self, term: str) -> float:
        nt = self.df.get(term, 0)
        return math.log((self.N - nt + 0.5) / (nt + 0.5) + 1e-12)

    def score(self, query: str, item_id: int) -> float:
        q_terms = tokenize(query)
        if not q_terms:
            return 0.0
        tf = Counter(self.doc_tokens.get(item_id, []))
        dl = self.doc_len.get(item_id, 0)
        norm = 1 - self.b + self.b * (dl / self.avgdl if self.avgdl > 0 else 1.0)
        s = 0.0
        for t in q_terms:
            f = tf.get(t, 0)
            if f == 0:
                continue
            tf_weight = f / (f + self.k1 * norm)
            s += self._idf(t) * tf_weight
        return s

    def search(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        scored = [(iid, self.score(query, iid)) for iid in self.doc_tokens.keys()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [x for x in scored[:top_k] if x[1] > 0]


def aggregate_candidates(
    index: BM25SearchIndex, queries: Iterable[str], top_k: int
) -> Tuple[List[int], List[float]]:
    scores = defaultdict(float)
    for q in queries:
        for iid, s in index.search(q, top_k=top_k):
            scores[iid] = max(scores[iid], s)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked], [s for _, s in ranked]

