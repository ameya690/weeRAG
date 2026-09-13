"""Tests for wee.vectorstore.VectorStore."""
import numpy as np
import pytest

from wee.vectorstore import VectorStore


def dummy_embed(text, dim=32):
    np.random.seed(hash(text) % 2**31)
    v = np.random.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


# ------------------------------------------------------------------
# test_add_and_search
# ------------------------------------------------------------------


def test_add_and_search():
    """Add vectors directly, search, verify correct top-k ordering."""
    dim = 32
    rng = np.random.RandomState(42)

    vs = VectorStore(dim=dim)

    # Create 20 random vectors
    vecs = rng.randn(20, dim).astype(np.float32)
    ids = [f"doc_{i}" for i in range(20)]
    vs.add(vecs, ids=ids)

    # Query is close to the first vector
    query = vecs[0] + rng.randn(dim).astype(np.float32) * 0.01

    results = vs.search(query, k=5)
    assert len(results) == 5

    # The closest result should be doc_0
    result_ids = [r[0] for r in results]
    assert result_ids[0] == "doc_0"

    # Scores should be in descending order
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)


# ------------------------------------------------------------------
# test_add_texts
# ------------------------------------------------------------------


def test_add_texts():
    """Add documents via embed_fn, verify search works."""
    vs = VectorStore(dim=32)

    texts = ["the cat sat on the mat", "the dog ran in the park", "quantum computing research"]
    vs.add_texts(texts, embed_fn=dummy_embed, ids=["a", "b", "c"])

    assert len(vs.ids) == 3

    # Search for something similar to "cat"
    q = dummy_embed("the cat sat on the mat")
    results = vs.search(q, k=2)
    assert len(results) == 2
    # The exact text should be the top hit
    assert results[0][0] == "a"


# ------------------------------------------------------------------
# test_delete
# ------------------------------------------------------------------


def test_delete():
    """Add, delete some entries, verify they are gone, search still works."""
    dim = 16
    rng = np.random.RandomState(99)
    vs = VectorStore(dim=dim)

    vecs = rng.randn(10, dim).astype(np.float32)
    ids = [f"d{i}" for i in range(10)]
    vs.add(vecs, ids=ids)

    # Delete d3 and d7
    count = vs.delete(["d3", "d7"])
    assert count == 2

    assert "d3" not in vs.ids
    assert "d7" not in vs.ids
    assert len(vs.ids) == 8

    # Search should still work on the remaining entries
    results = vs.search(vecs[0], k=3)
    assert len(results) == 3
    result_ids = {r[0] for r in results}
    assert "d3" not in result_ids
    assert "d7" not in result_ids

    # Deleting non-existent id returns 0
    assert vs.delete(["nonexistent"]) == 0


# ------------------------------------------------------------------
# test_filter_fn
# ------------------------------------------------------------------


def test_filter_fn():
    """Metadata filtering during search."""
    dim = 16
    rng = np.random.RandomState(7)
    vs = VectorStore(dim=dim)

    vecs = rng.randn(6, dim).astype(np.float32)
    ids = [f"d{i}" for i in range(6)]
    metas = [{"category": "A" if i % 2 == 0 else "B"} for i in range(6)]
    vs.add(vecs, ids=ids, metadata=metas)

    # Filter to only category A
    results = vs.search(vecs[0], k=6, filter_fn=lambda m: m["category"] == "A")

    # Only even-indexed docs should appear with real scores
    for doc_id, score, meta in results:
        if score > -np.inf:
            assert meta["category"] == "A"


# ------------------------------------------------------------------
# test_save_load
# ------------------------------------------------------------------


def test_save_load(tmp_path):
    """Round-trip save/load preserves vectors, ids, and metadata."""
    dim = 16
    rng = np.random.RandomState(123)
    vs = VectorStore(dim=dim)

    vecs = rng.randn(5, dim).astype(np.float32)
    ids = ["x", "y", "z", "w", "v"]
    metas = [{"idx": i} for i in range(5)]
    vs.add(vecs, ids=ids, metadata=metas)

    path = str(tmp_path / "store.npz")
    vs.save(path)

    loaded = VectorStore.load(path)
    assert loaded.dim == dim
    assert loaded.ids == ids
    assert loaded.metadata == metas

    # Vectors should be identical (both normalized)
    vs._flush()
    np.testing.assert_allclose(loaded.vectors, vs.vectors, atol=1e-6)

    # Search should produce the same result
    q = rng.randn(dim).astype(np.float32)
    r1 = vs.search(q, k=3)
    r2 = loaded.search(q, k=3)
    assert [r[0] for r in r1] == [r[0] for r in r2]


# ------------------------------------------------------------------
# test_empty_search
# ------------------------------------------------------------------


def test_empty_search():
    """Searching an empty index should raise an assertion error."""
    vs = VectorStore(dim=8)
    q = np.random.randn(8).astype(np.float32)
    with pytest.raises(AssertionError, match="Empty index"):
        vs.search(q, k=1)
