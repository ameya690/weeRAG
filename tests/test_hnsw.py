"""Tests for wee.hnsw.HNSWIndex."""
import random

import numpy as np

from wee.hnsw import HNSWIndex


def dummy_embed(text, dim=32):
    np.random.seed(hash(text) % 2**31)
    v = np.random.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _cosine_sim(a, b):
    """Brute-force cosine similarity."""
    dot = float(np.dot(a, b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return dot / denom


# ------------------------------------------------------------------
# test_basic_search
# ------------------------------------------------------------------


def test_basic_search():
    """Add vectors, search, verify results are reasonable."""
    dim = 32
    rng = np.random.RandomState(42)
    random.seed(42)

    idx = HNSWIndex(dim=dim, M=8, ef_construction=50, max_level=3)

    n = 50
    vecs = rng.randn(n, dim).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    vecs = vecs / norms

    for i in range(n):
        idx.add(vecs[i], doc_id=f"d{i}")

    # Query is identical to vector 10
    query = vecs[10].copy()
    results = idx.search(query, k=5, ef=50)

    assert len(results) >= 1
    # The top result should be d10 (exact match)
    assert results[0][0] == "d10"
    assert results[0][1] > 0.99  # cosine sim ~1.0

    # All returned similarities should be positive and descending
    sims = [s for _, s in results]
    assert all(s > 0 for s in sims)
    assert sims == sorted(sims, reverse=True)


# ------------------------------------------------------------------
# test_recall
# ------------------------------------------------------------------


def test_recall():
    """Compare HNSW results with brute force; recall should be > 0.5."""
    dim = 32
    k = 10
    rng = np.random.RandomState(123)
    random.seed(123)

    idx = HNSWIndex(dim=dim, M=16, ef_construction=100, max_level=4)

    n = 100
    vecs = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    vecs = vecs / norms

    for i in range(n):
        idx.add(vecs[i], doc_id=f"d{i}")

    # Random query
    query = rng.randn(dim).astype(np.float32)
    query = query / (np.linalg.norm(query) + 1e-9)

    # Brute force exact top-k
    sims = [(_cosine_sim(query, vecs[i]), f"d{i}") for i in range(n)]
    sims.sort(key=lambda x: -x[0])
    exact_top_k = [(doc_id, sim) for sim, doc_id in sims[:k]]

    # HNSW recall
    recall = idx.recall_at_k(query, k=k, exact_results=exact_top_k)

    # With reasonable ef, recall should be well above 0.5
    assert recall > 0.5, f"Recall {recall} is too low"
