"""Tests for wee.colbert.ColBERTIndex."""
import numpy as np

from wee.colbert import ColBERTIndex


def _make_token_embeddings(n_tokens, dim, seed):
    """Generate random token embeddings for a single document."""
    rng = np.random.RandomState(seed)
    vecs = rng.randn(n_tokens, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs / norms


# ------------------------------------------------------------------
# test_add_and_search
# ------------------------------------------------------------------


def test_add_and_search():
    """Add token embeddings for several docs, search, verify ranking."""
    dim = 16
    idx = ColBERTIndex(dim=dim)

    # Add 5 documents with different random token embeddings
    doc_embs = {}
    for i in range(5):
        emb = _make_token_embeddings(n_tokens=8, dim=dim, seed=i * 10)
        idx.add(f"doc_{i}", emb)
        doc_embs[f"doc_{i}"] = emb

    # Query using tokens similar to doc_2
    query = doc_embs["doc_2"] + np.random.RandomState(999).randn(8, dim).astype(np.float32) * 0.01

    results = idx.search(query, k=3)

    assert len(results) == 3
    # doc_2 should be the best match since the query is nearly identical
    assert results[0][0] == "doc_2"

    # Scores should be in descending order
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


# ------------------------------------------------------------------
# test_maxsim
# ------------------------------------------------------------------


def test_maxsim():
    """Verify MaxSim score computation with known vectors."""
    dim = 4

    # Query: 2 tokens, identity-like vectors (normalized)
    q = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ], dtype=np.float32)

    # Document: 3 tokens
    d = np.array([
        [1.0, 0.0, 0.0, 0.0],  # identical to q[0]
        [0.0, 0.0, 1.0, 0.0],  # orthogonal to both q tokens
        [0.0, 0.7071, 0.7071, 0.0],  # partially aligned with q[1]
    ], dtype=np.float32)

    score = ColBERTIndex.maxsim(q, d)

    # For q[0]: max similarity with d[0]=1.0, d[1]=0.0, d[2]=0.0 -> max = 1.0
    # For q[1]: max similarity with d[0]=0.0, d[1]=0.0, d[2]=0.7071 -> max = 0.7071
    # Total = 1.0 + 0.7071 = 1.7071
    expected = 1.0 + 0.7071
    np.testing.assert_allclose(score, expected, atol=1e-3)

    # Self-similarity: MaxSim of a matrix with itself should equal number of tokens
    # (each query token's best match is itself with sim=1.0)
    identity_vecs = np.eye(dim, dtype=np.float32)
    self_score = ColBERTIndex.maxsim(identity_vecs, identity_vecs)
    np.testing.assert_allclose(self_score, dim, atol=1e-6)
