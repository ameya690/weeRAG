from __future__ import annotations

import numpy as np

from .bm25 import BM25
from .vectorstore import VectorStore


def _mmr(q: np.ndarray, candidates: np.ndarray, k: int, lambda_: float = 0.5) -> list[int]:
    selected = []
    remaining = list(range(candidates.shape[0]))
    q = q.reshape(1, -1)
    doc_sims = (candidates @ q.T).ravel()
    cand_sims = candidates @ candidates.T
    for _ in range(min(k, candidates.shape[0])):
        best_idx, best_score = -1, -1e9
        for i in remaining:
            diversity = 0.0 if not selected else np.max(cand_sims[i, selected])
            score = lambda_ * doc_sims[i] - (1 - lambda_) * diversity
            if score > best_score:
                best_score, best_idx = score, i
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected

def _rrf(ranks: list[list[int]], k: int, K: int = 60) -> list[int]:
    N = max((max(r) if r else -1) for r in ranks) + 1 if ranks else 0
    scores = np.zeros(N, dtype=float)
    for r in ranks:
        for rank_pos, doc_idx in enumerate(r[:k], start=1):
            scores[doc_idx] += 1.0 / (K + rank_pos)
    return list(np.argsort(-scores)[:k])

class Retriever:
    def __init__(self, method: str = "topk", k: int = 5, mmr_lambda: float = 0.5):
        assert method in ("topk", "mmr", "rrf")
        self.method = method
        self.k = k
        self.mmr_lambda = mmr_lambda
        self.vs: VectorStore | None = None
        self.bm25: BM25 | None = None
        self.embed_fn = None

    def attach_vectorstore(self, vs: VectorStore, embed_fn):
        self.vs = vs
        self.embed_fn = embed_fn
        return self

    def attach_bm25(self, bm: BM25):
        self.bm25 = bm
        return self

    def _id_to_index(self) -> dict[str, int]:
        return self.vs._id_index

    def search(self, query: str):
        assert self.vs is not None or self.bm25 is not None, "Attach at least one backend"
        if self.method == "topk":
            assert self.vs is not None and self.embed_fn is not None
            return self.vs.search_text(query, self.embed_fn, k=self.k)

        if self.method == "mmr":
            assert self.vs is not None and self.embed_fn is not None
            q = self.embed_fn(query)
            id_idx = self._id_to_index()
            pool = min(max(10, self.k * 5), len(self.vs.ids))
            cand = self.vs.search(q, k=pool)
            idxs = [id_idx[doc_id] for doc_id, _, _ in cand]
            cand_vecs = self.vs.vectors[idxs]
            selected_local = _mmr(q, cand_vecs, k=self.k, lambda_=self.mmr_lambda)
            selected = [idxs[i] for i in selected_local]
            return [(self.vs.ids[i], float((self.vs.vectors[i] @ q).item()), self.vs.metadata[i]) for i in selected]

        if self.method == "rrf":
            assert self.vs is not None and self.embed_fn is not None and self.bm25 is not None
            q = self.embed_fn(query)
            id_idx = self._id_to_index()
            dense_top = self.vs.search(q, k=min(max(10, self.k * 5), len(self.vs.ids)))
            dense_indices = [id_idx[doc_id] for doc_id, _, _ in dense_top]
            bm_hits = self.bm25.search(query, k=min(max(10, self.k * 5), self.bm25.N))
            bm_indices = [doc_id for doc_id, _ in bm_hits]
            fused_indices = _rrf([dense_indices, bm_indices], k=self.k)
            return [(self.vs.ids[i], float(self.vs.vectors[i] @ q), self.vs.metadata[i]) for i in fused_indices]

        raise ValueError("Unknown method")
