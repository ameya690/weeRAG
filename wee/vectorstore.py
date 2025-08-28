from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Iterable
import numpy as np

def _l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

class VectorStore:
    def __init__(self, dim: Optional[int] = None, normalize: bool = True):
        self.dim = dim
        self.normalize = normalize
        self.vectors: Optional[np.ndarray] = None
        self.ids: List[str] = []
        self.metadata: List[Dict] = []

    def add(self, vectors: np.ndarray, ids: Optional[Iterable[str]] = None, metadata: Optional[Iterable[Dict]] = None):
        vectors = np.asarray(vectors, dtype=np.float32)
        if self.dim is None:
            self.dim = vectors.shape[1]
        assert vectors.shape[1] == self.dim, "Dim mismatch"
        if self.normalize:
            vectors = _l2_normalize(vectors)
        if ids is None:
            start = len(self.ids)
            ids = [str(i) for i in range(start, start + vectors.shape[0])]
        if metadata is None:
            metadata = [{} for _ in range(vectors.shape[0])]
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.ids.extend(list(ids))
        self.metadata.extend(list(metadata))

    def add_texts(self, texts: Iterable[str], embed_fn, ids: Optional[Iterable[str]] = None, metadata: Optional[Iterable[Dict]] = None):
        vecs = np.vstack([embed_fn(t) for t in texts]).astype(np.float32)
        md = list(metadata) if metadata is not None else [{} for _ in range(vecs.shape[0])]
        self.add(vecs, ids=ids, metadata=md)

    def _cosine_scores(self, q: np.ndarray) -> np.ndarray:
        q = q.astype(np.float32).reshape(1, -1)
        if self.normalize:
            q = _l2_normalize(q)
        return (self.vectors @ q.T).ravel()

    def search(self, query_vector: np.ndarray, k: int = 5):
        assert self.vectors is not None and len(self.ids) > 0, "Empty index"
        scores = self._cosine_scores(query_vector)
        idx = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i]), self.metadata[i]) for i in idx]

    def search_text(self, query_text: str, embed_fn, k: int = 5):
        q = embed_fn(query_text)
        return self.search(q, k=k)

    def save(self, path: str):
        np.savez_compressed(path, vectors=self.vectors, dim=self.dim, normalize=self.normalize, ids=np.array(self.ids))
        import json, os
        with open(os.path.splitext(path)[0] + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        import json, os
        data = np.load(path, allow_pickle=True)
        vs = cls(dim=int(data["dim"]), normalize=bool(data["normalize"]))
        vs.vectors = data["vectors"]
        vs.ids = list(data["ids"])
        with open(os.path.splitext(path)[0] + ".meta.json", "r", encoding="utf-8") as f:
            vs.metadata = json.load(f)
        return vs
