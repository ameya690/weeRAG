from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Iterable
import numpy as np
from .bm25 import BM25

class Reranker:
    def __init__(self, mode: str = "dense", alpha: float = 0.5):
        assert mode in ("dense", "lexical", "hybrid")
        self.mode = mode
        self.alpha = alpha
        self.embed_fn = None
        self.bm25 = None

    def attach_dense(self, embed_fn):
        self.embed_fn = embed_fn
        return self

    def attach_lexical(self, corpus_texts: Iterable[str]):
        bm = BM25()
        bm.add(list(corpus_texts))
        self.bm25 = bm
        return self

    def rerank(self, query: str, candidates: List[Tuple[str, float, Dict]], texts_by_id: Dict[str, str], k: int = 5):
        ids = [cid for cid, _, _ in candidates]
        if self.mode == "dense":
            assert self.embed_fn is not None
            q = self.embed_fn(query)
            X = np.vstack([self.embed_fn(texts_by_id[i]) for i in ids]).astype(np.float32)
            q = q.reshape(1, -1)
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
            q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
            scores = (X @ q.T).ravel()
            order = np.argsort(-scores)[:k]
            return [(ids[i], float(scores[i]), candidates[i][2]) for i in order]

        if self.mode == "lexical":
            bm = BM25()
            bm.add([texts_by_id[i] for i in ids])
            local_hits = bm.search(query, k=len(ids))
            order = [doc_id for doc_id, _ in local_hits][:k]
            return [candidates[i] for i in order]

        if self.mode == "hybrid":
            assert self.embed_fn is not None
            q = self.embed_fn(query)
            X = np.vstack([self.embed_fn(texts_by_id[i]) for i in ids]).astype(np.float32)
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
            qn = q.reshape(1, -1) / (np.linalg.norm(q) + 1e-9)
            dense_scores = (X @ qn.T).ravel()
            dense_rank = np.argsort(-dense_scores)
            dense_rank_pos = {i: r for r, i in enumerate(dense_rank)}
            bm = BM25()
            bm.add([texts_by_id[i] for i in ids])
            lex_hits = bm.search(query, k=len(ids))
            lex_rank = [doc_id for doc_id, _ in lex_hits]
            lex_rank_pos = {i: r for r, i in enumerate(lex_rank)}
            fused = []
            n = len(ids)
            for i in range(n):
                r_dense = dense_rank_pos[i] / max(n - 1, 1)
                r_lex = lex_rank_pos[i] / max(n - 1, 1)
                score = -(self.alpha * r_dense + (1 - self.alpha) * r_lex)
                fused.append((i, score))
            order = [i for i, _ in sorted(fused, key=lambda x: -x[1])][:k]
            return [candidates[i] for i in order]

        raise ValueError("Unknown mode")
