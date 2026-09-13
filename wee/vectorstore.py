from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np


def _l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

class VectorStore:
    def __init__(self, dim: int | None = None, normalize: bool = True):
        self.dim = dim
        self.normalize = normalize
        self.vectors: np.ndarray | None = None
        self.ids: list[str] = []
        self.metadata: list[dict] = []
        self._id_index: dict[str, int] = {}
        self._pending: list[np.ndarray] = []

    def _flush(self):
        if not self._pending:
            return
        new = np.vstack(self._pending)
        self._pending.clear()
        if self.vectors is None:
            self.vectors = new
        else:
            self.vectors = np.vstack([self.vectors, new])

    def add(self, vectors: np.ndarray, ids: Iterable[str] | None = None, metadata: Iterable[dict] | None = None):
        vectors = np.asarray(vectors, dtype=np.float32)
        if self.dim is None:
            self.dim = vectors.shape[1]
        assert vectors.shape[1] == self.dim, "Dim mismatch"
        if self.normalize:
            vectors = _l2_normalize(vectors)
        if ids is None:
            start = len(self.ids)
            ids = [str(i) for i in range(start, start + vectors.shape[0])]
        else:
            ids = list(ids)
        if metadata is None:
            metadata = [{} for _ in range(vectors.shape[0])]
        else:
            metadata = list(metadata)

        base = len(self.ids)
        for i, doc_id in enumerate(ids):
            self._id_index[doc_id] = base + i
        self.ids.extend(ids)
        self.metadata.extend(metadata)
        self._pending.append(vectors)

    def add_texts(self, texts: Iterable[str], embed_fn, ids: Iterable[str] | None = None, metadata: Iterable[dict] | None = None):
        vecs = np.vstack([embed_fn(t) for t in texts]).astype(np.float32)
        md = list(metadata) if metadata is not None else [{} for _ in range(vecs.shape[0])]
        self.add(vecs, ids=ids, metadata=md)

    def delete(self, ids: Iterable[str]) -> int:
        """Remove vectors by ID. Returns the count of actually deleted entries."""
        self._flush()
        to_delete = {doc_id for doc_id in ids if doc_id in self._id_index}
        if not to_delete:
            return 0

        keep = [i for i, doc_id in enumerate(self.ids) if doc_id not in to_delete]
        self.ids = [self.ids[i] for i in keep]
        self.metadata = [self.metadata[i] for i in keep]
        if self.vectors is not None and len(keep) > 0:
            self.vectors = self.vectors[keep]
        elif len(keep) == 0:
            self.vectors = None

        self._id_index = {doc_id: i for i, doc_id in enumerate(self.ids)}
        return len(to_delete)

    def _cosine_scores(self, q: np.ndarray) -> np.ndarray:
        self._flush()
        q = q.astype(np.float32).reshape(1, -1)
        if self.normalize:
            q = _l2_normalize(q)
        return (self.vectors @ q.T).ravel()

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        filter_fn: Callable[[dict], bool] | None = None,
    ):
        self._flush()
        assert self.vectors is not None and len(self.ids) > 0, "Empty index"
        scores = self._cosine_scores(query_vector)

        if filter_fn is not None:
            mask = np.array([filter_fn(m) for m in self.metadata], dtype=bool)
            scores = np.where(mask, scores, -np.inf)

        if k >= len(scores):
            idx = np.argsort(-scores)[:k]
        else:
            idx = np.argpartition(-scores, k)[:k]
            idx = idx[np.argsort(-scores[idx])]

        return [(self.ids[i], float(scores[i]), self.metadata[i]) for i in idx]

    def search_text(
        self,
        query_text: str,
        embed_fn,
        k: int = 5,
        filter_fn: Callable[[dict], bool] | None = None,
    ):
        q = embed_fn(query_text)
        return self.search(q, k=k, filter_fn=filter_fn)

    def save(self, path: str):
        self._flush()
        np.savez_compressed(path, vectors=self.vectors, dim=self.dim, normalize=self.normalize, ids=np.array(self.ids))
        import json
        import os
        with open(os.path.splitext(path)[0] + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> VectorStore:
        import json
        import os
        data = np.load(path, allow_pickle=True)
        vs = cls(dim=int(data["dim"]), normalize=bool(data["normalize"]))
        vs.vectors = data["vectors"]
        vs.ids = list(data["ids"])
        vs._id_index = {doc_id: i for i, doc_id in enumerate(vs.ids)}
        with open(os.path.splitext(path)[0] + ".meta.json", encoding="utf-8") as f:
            vs.metadata = json.load(f)
        return vs
