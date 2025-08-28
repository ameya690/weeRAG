import math
import re
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Iterable

_WORD = re.compile(r"\b\w+\b", re.UNICODE)

def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(text)]

class BM25:
    """
    Minimal BM25 index (Okapi BM25).
    - add(corpus) to build from a list of strings
    - search(query, k) to get top-k (doc_id, score)
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.N: int = 0
        self.inverted: Dict[str, List[Tuple[int, int]]] = defaultdict(list)  # term -> List[(doc_id, tf)]
        self.docs: List[str] = []

    def add(self, corpus: Iterable[str]):
        for doc in corpus:
            doc_id = self.N
            self.N += 1
            self.docs.append(doc)
            tokens = _tokenize(doc)
            self.doc_len.append(len(tokens))
            tf = Counter(tokens)
            for term, f in tf.items():
                self.inverted[term].append((doc_id, f))
            for term in tf.keys():
                self.doc_freq[term] += 1
        self.avgdl = sum(self.doc_len) / max(self.N, 1)

    def _idf(self, term: str) -> float:
        # BM25 idf with +0.5 smoothing
        n_qi = self.doc_freq.get(term, 0)
        if n_qi == 0:
            return 0.0
        return math.log((self.N - n_qi + 0.5) / (n_qi + 0.5) + 1.0)

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        q_terms = _tokenize(query)
        scores: Dict[int, float] = defaultdict(float)
        for term in q_terms:
            postings = self.inverted.get(term, [])
            idf = self._idf(term)
            for doc_id, tf in postings:
                dl = self.doc_len[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                score = idf * (tf * (self.k1 + 1)) / max(denom, 1e-9)
                scores[doc_id] += score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return ranked
