"""Tests for wee.embed (Embedder, quantize, dequantize, binary search)."""
import numpy as np

from wee.embed import (
    Embedder,
    quantize_embeddings,
    dequantize_embeddings,
    binary_search,
)


def _raw_embed(text, dim=64):
    """Raw embedding function that returns a full-dimension vector."""
    np.random.seed(hash(text) % 2**31)
    return np.random.randn(dim).astype(np.float32)


# ------------------------------------------------------------------
# test_matryoshka_truncation
# ------------------------------------------------------------------


def test_matryoshka_truncation():
    """Embedder.truncate() produces a new Embedder with fewer dimensions."""
    full_dim = 64
    trunc_dim = 16

    emb_full = Embedder(_raw_embed, dim=full_dim)
    emb_trunc = emb_full.truncate(trunc_dim)

    vec_full = emb_full.embed("hello world")
    vec_trunc = emb_trunc.embed("hello world")

    assert vec_full.shape == (full_dim,)
    assert vec_trunc.shape == (trunc_dim,)

    # The truncated vector should be the first `trunc_dim` dims of the raw
    # output, re-normalized. Verify they point in a similar direction.
    raw = _raw_embed("hello world", dim=full_dim)
    prefix = raw[:trunc_dim]
    prefix_norm = prefix / (np.linalg.norm(prefix) + 1e-9)
    np.testing.assert_allclose(vec_trunc, prefix_norm, atol=1e-5)

    # Full embed should be L2-normalized
    np.testing.assert_allclose(np.linalg.norm(vec_full), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(vec_trunc), 1.0, atol=1e-5)


# ------------------------------------------------------------------
# test_quantize_int8
# ------------------------------------------------------------------


def test_quantize_int8():
    """Round-trip int8 quantize/dequantize, check compression ratio."""
    rng = np.random.RandomState(42)
    vecs = rng.randn(100, 64).astype(np.float32)

    q = quantize_embeddings(vecs, method="int8")

    assert q["method"] == "int8"
    assert q["vectors"].dtype == np.int8
    assert q["vectors"].shape == (100, 64)
    # float32 -> int8: 4x compression
    assert q["compression_ratio"] >= 3.9

    # Dequantize and check reconstruction is reasonable
    recovered = dequantize_embeddings(q)
    assert recovered.shape == vecs.shape
    assert recovered.dtype == np.float32

    # Reconstruction error should be small relative to the value range
    max_err = np.abs(recovered - vecs).max()
    value_range = vecs.max() - vecs.min()
    # Per-dimension quantization to 256 levels: max error ~ range/255
    assert max_err < value_range * 0.02, f"Max error {max_err} too large"


# ------------------------------------------------------------------
# test_quantize_binary
# ------------------------------------------------------------------


def test_quantize_binary():
    """Binary quantization: verify 32x compression ratio."""
    rng = np.random.RandomState(7)
    dim = 128
    vecs = rng.randn(50, dim).astype(np.float32)

    q = quantize_embeddings(vecs, method="binary")

    assert q["method"] == "binary"
    assert q["vectors"].dtype == np.uint8
    # 128 dims -> 16 bytes per vector
    assert q["vectors"].shape == (50, dim // 8)
    # float32 (4 bytes/dim) -> 1 bit/dim (packed in uint8): 32x
    assert q["compression_ratio"] >= 31.0

    # Dequantize: should recover sign information
    recovered = dequantize_embeddings(q)
    assert recovered.shape == vecs.shape

    # Each recovered value should be +1 or -1
    unique_vals = set(np.unique(recovered))
    assert unique_vals == {-1.0, 1.0}

    # Signs should match the original
    original_signs = np.sign(vecs)
    # Where original is exactly 0, sign is 0, but quantization maps >=0 to +1
    # So just check that positive values map to +1 and negative to -1
    positive_mask = vecs > 0
    negative_mask = vecs < 0
    assert np.all(recovered[positive_mask] == 1.0)
    assert np.all(recovered[negative_mask] == -1.0)


# ------------------------------------------------------------------
# test_binary_search
# ------------------------------------------------------------------


def test_binary_search():
    """Search with binary embeddings finds the nearest neighbor."""
    rng = np.random.RandomState(55)
    dim = 64
    n = 30

    # Create float vectors, then binary-quantize
    vecs = rng.randn(n, dim).astype(np.float32)
    q_data = quantize_embeddings(vecs, method="binary")
    index_binary = q_data["vectors"]

    # Use the first vector as query
    query_binary = index_binary[0]

    results = binary_search(query_binary, index_binary, k=5)

    assert len(results) == 5
    # The query is identical to index[0], so it should be the top hit with distance 0
    assert results[0][0] == 0
    assert results[0][1] == 0

    # Distances should be in ascending order
    distances = [d for _, d in results]
    assert distances == sorted(distances)
