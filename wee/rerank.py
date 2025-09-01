from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Iterable
import numpy as np
from .bm25 import BM25

class Reranker:
    """
    Minimal reranker:
    - mode="dense": cosine between query and chunk embeddings
    - mode="lexical": BM25 over chunk texts
    - mode="hybrid": weighted sum of dense + lexical normalized ranks
    """
    def __init__(self, mode: str = "dense", alpha: float = 0.5):
        assert mode in ("dense", "lexical", "hybrid")
        self.mode = mode
        self.alpha = alpha
        self.embed_fn = None  # for dense
        self.bm25 = None      # for lexical (not required; we reindex candidates)

    def attach_dense(self, embed_fn):
        self.embed_fn = embed_fn
        return self

    def attach_lexical(self, corpus_texts: Iterable[str]):
        bm = BM25()
        bm.add(list(corpus_texts))
        self.bm25 = bm
        return self

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float, Dict]],  # (id, score, meta) where meta may include 'text'
        texts_by_id: Dict[str, str],
        k: int = 5,
    ) -> List[Tuple[str, float, Dict]]:
        ids = [cid for cid, _, _ in candidates]
        n = len(ids)
        if n == 0:
            return []

        if self.mode == "dense":
            assert self.embed_fn is not None
            q = self.embed_fn(query)
            X = np.vstack([self.embed_fn(texts_by_id[i]) for i in ids]).astype(np.float32)
            # cosine
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
            q = q.reshape(1, -1) / (np.linalg.norm(q) + 1e-9)
            scores = (X @ q.T).ravel()
            order = np.argsort(-scores)[:k]
            return [(ids[i], float(scores[i]), candidates[i][2]) for i in order]

        if self.mode == "lexical":
            # Build a local BM25 only over these candidate texts
            bm = BM25()
            bm.add([texts_by_id[i] for i in ids])
            local_hits = bm.search(query, k=n)  # may return fewer than n
            # Map: missing docs (no term match) → worst rank at the tail
            seen = {doc_id for doc_id, _ in local_hits}
            missing = [i for i in range(n) if i not in seen]
            order = [doc_id for doc_id, _ in local_hits] + missing
            order = order[:k]
            return [candidates[i] for i in order]

        if self.mode == "hybrid":
            assert self.embed_fn is not None
            # Dense ranks
            q = self.embed_fn(query)
            X = np.vstack([self.embed_fn(texts_by_id[i]) for i in ids]).astype(np.float32)
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
            qn = q.reshape(1, -1) / (np.linalg.norm(q) + 1e-9)
            dense_scores = (X @ qn.T).ravel()
            dense_rank = np.argsort(-dense_scores)
            dense_rank_pos = {i: r for r, i in enumerate(dense_rank)}

            # Lexical ranks over candidates
            bm = BM25()
            bm.add([texts_by_id[i] for i in ids])
            lex_hits = bm.search(query, k=n)  # may be < n
            lex_rank_pos = {doc_id: r for r, (doc_id, _) in enumerate(lex_hits)}

            # Normalize ranks and fuse; unseen lexical docs -> worst rank (n-1)
            denom = max(n - 1, 1)
            fused = []
            for i in range(n):
                r_dense = dense_rank_pos[i] / denom
                r_lex = lex_rank_pos.get(i, n - 1) / denom  # default to worst
                score = -(self.alpha * r_dense + (1 - self.alpha) * r_lex)
                fused.append((i, score))

            order = [i for i, _ in sorted(fused, key=lambda x: -x[1])][:k]
            return [candidates[i] for i in order]

        raise ValueError("Unknown mode")
