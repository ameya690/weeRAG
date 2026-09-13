from __future__ import annotations
from typing import List, Tuple, Callable, Optional
import numpy as np


def _l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """L2-normalize along the last axis."""
    if x.ndim == 1:
        n = np.linalg.norm(x)
        return x / max(n, eps)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


class Embedder:
    """
    Adapter wrapping any embedding function with optional post-processing.

    Supports Matryoshka-style dimension truncation: modern embedding models
    (e.g. text-embedding-3-small) are trained so that the first N dimensions
    retain most of the representational quality.  Truncating from 1536 to 256
    dims gives ~6x storage compression with only a small quality loss.

    Usage::

        emb = Embedder(my_api_embed, dim=256)   # truncate to 256
        vec = emb.embed("hello world")           # shape (256,)
        small = emb.truncate(128)                # new Embedder at 128 dims
    """

    def __init__(self, embed_fn: Callable[[str], np.ndarray], dim: Optional[int] = None):
        """
        Args:
            embed_fn: callable mapping a single text string to a 1-D float
                      numpy array of shape ``(full_dim,)``.
            dim:      if set, truncate embeddings to the first *dim*
                      dimensions (Matryoshka truncation) before normalizing.
        """
        self._embed_fn = embed_fn
        self.dim = dim

    # ------------------------------------------------------------------
    # core API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Returns a 1-D float32 array of shape ``(dim,)`` that is
        L2-normalized after optional truncation.
        """
        vec = np.asarray(self._embed_fn(text), dtype=np.float32)
        if self.dim is not None:
            vec = vec[: self.dim]
        return _l2_normalize(vec)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts.

        Returns a float32 array of shape ``(N, dim)`` where every row is
        L2-normalized.
        """
        rows = [self.embed(t) for t in texts]
        return np.vstack(rows)

    # ------------------------------------------------------------------
    # Matryoshka helpers
    # ------------------------------------------------------------------

    def truncate(self, dim: int) -> Embedder:
        """Return a *new* Embedder whose output dimension is ``dim``.

        This is the Matryoshka trick: because the model was trained with
        nested objectives, simply keeping the first ``dim`` components of
        the full vector and re-normalizing gives a useful lower-dimensional
        embedding.
        """
        return Embedder(self._embed_fn, dim=dim)


# ======================================================================
# Quantized embeddings
# ======================================================================

def quantize_embeddings(vectors: np.ndarray, method: str = "int8") -> dict:
    """Quantize embedding vectors for storage efficiency.

    Args:
        vectors: ``(N, dim)`` float32 array.
        method:  ``"int8"``  -- scale to [-128, 127] per dimension, 4x compression.
                 ``"uint8"`` -- scale to [0, 255] per dimension, 4x compression.
                 ``"binary"`` -- sign bit per dimension packed into uint8,
                                 32x compression; enables Hamming-distance search.

    Returns:
        dict with keys ``"vectors"`` (quantized array), ``"params"`` (scale/offset
        needed for dequantization), ``"method"`` (str), ``"compression_ratio"``
        (float).
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    N, dim = vectors.shape
    orig_bytes = vectors.nbytes  # N * dim * 4

    if method == "int8":
        # per-dimension min/max -> affine mapping to [-128, 127]
        vmin = vectors.min(axis=0)  # (dim,)
        vmax = vectors.max(axis=0)  # (dim,)
        scale = (vmax - vmin) / 255.0
        scale = np.where(scale < 1e-9, 1.0, scale)  # avoid div-by-zero
        offset = vmin
        quantized = np.clip(
            np.round((vectors - offset) / scale) - 128, -128, 127
        ).astype(np.int8)
        q_bytes = quantized.nbytes
        return {
            "vectors": quantized,
            "params": {"scale": scale, "offset": offset},
            "method": "int8",
            "compression_ratio": orig_bytes / q_bytes,
        }

    if method == "uint8":
        vmin = vectors.min(axis=0)
        vmax = vectors.max(axis=0)
        scale = (vmax - vmin) / 255.0
        scale = np.where(scale < 1e-9, 1.0, scale)
        offset = vmin
        quantized = np.clip(
            np.round((vectors - offset) / scale), 0, 255
        ).astype(np.uint8)
        q_bytes = quantized.nbytes
        return {
            "vectors": quantized,
            "params": {"scale": scale, "offset": offset},
            "method": "uint8",
            "compression_ratio": orig_bytes / q_bytes,
        }

    if method == "binary":
        # 1 bit per dimension: sign(x) packed into uint8
        bits = (vectors >= 0).astype(np.uint8)  # (N, dim)
        # pack 8 bits per byte
        packed_dim = (dim + 7) // 8
        packed = np.zeros((N, packed_dim), dtype=np.uint8)
        for i in range(dim):
            byte_idx = i // 8
            bit_idx = i % 8
            packed[:, byte_idx] |= bits[:, i] << bit_idx
        q_bytes = packed.nbytes
        return {
            "vectors": packed,
            "params": {"original_dim": dim},
            "method": "binary",
            "compression_ratio": orig_bytes / q_bytes,
        }

    raise ValueError(f"Unknown quantization method: {method!r}. Use 'int8', 'uint8', or 'binary'.")


def dequantize_embeddings(quantized: dict) -> np.ndarray:
    """Reverse :func:`quantize_embeddings` to recover approximate float32 vectors."""
    method = quantized["method"]
    vecs = quantized["vectors"]
    params = quantized["params"]

    if method in ("int8", "uint8"):
        scale = params["scale"]
        offset = params["offset"]
        if method == "int8":
            return (vecs.astype(np.float32) + 128) * scale + offset
        return vecs.astype(np.float32) * scale + offset

    if method == "binary":
        original_dim = params["original_dim"]
        N = vecs.shape[0]
        unpacked = np.zeros((N, original_dim), dtype=np.float32)
        for i in range(original_dim):
            byte_idx = i // 8
            bit_idx = i % 8
            unpacked[:, i] = ((vecs[:, byte_idx] >> bit_idx) & 1).astype(np.float32) * 2 - 1
        return unpacked

    raise ValueError(f"Unknown method: {method!r}")


# ======================================================================
# Binary-embedding search utilities
# ======================================================================

def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Hamming distance between two binary-quantized vectors (packed uint8).

    Each vector is a 1-D uint8 array where every bit represents one
    embedding dimension.  The distance equals the number of differing bits.
    """
    xor = np.bitwise_xor(a, b)
    # popcount via lookup -- fast enough for educational use
    return int(sum(bin(byte).count("1") for byte in xor))


def binary_search(
    query_binary: np.ndarray,
    index_binary: np.ndarray,
    k: int = 5,
) -> List[Tuple[int, int]]:
    """Fast approximate nearest-neighbor search using binary embeddings.

    Args:
        query_binary:  packed uint8 vector, shape ``(packed_dim,)``.
        index_binary:  packed uint8 array, shape ``(N, packed_dim)``.
        k:             number of nearest neighbours to return.

    Returns:
        list of ``(index, hamming_distance)`` tuples sorted by distance
        ascending.
    """
    N = index_binary.shape[0]
    # XOR each index row with query, then popcount
    xor = np.bitwise_xor(index_binary, query_binary)  # (N, packed_dim)
    # vectorized popcount using a lookup table
    _popcount_table = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)
    distances = _popcount_table[xor].sum(axis=1)  # (N,)

    if k >= N:
        idx = np.argsort(distances)
    else:
        idx = np.argpartition(distances, k)[:k]
        idx = idx[np.argsort(distances[idx])]

    return [(int(i), int(distances[i])) for i in idx]
