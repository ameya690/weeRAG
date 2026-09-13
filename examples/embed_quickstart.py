"""
Embedding adapter quickstart -- Matryoshka truncation, quantization, and binary search.
"""
import numpy as np
from wee.embed import Embedder, quantize_embeddings, dequantize_embeddings, binary_search

FULL_DIM = 1536  # e.g. text-embedding-3-small

# --- 1. Create an Embedder with a dummy embed_fn (deterministic random) ---

rng = np.random.default_rng(42)

def dummy_embed(text: str) -> np.ndarray:
    """Deterministic pseudo-embedding: hash the text to seed a random vector."""
    seed = hash(text) % (2**31)
    return np.random.default_rng(seed).standard_normal(FULL_DIM).astype(np.float32)

embedder = Embedder(dummy_embed)
vec_full = embedder.embed("Matryoshka embeddings are neat")
print(f"Full embedding  : dim={vec_full.shape[0]}, norm={np.linalg.norm(vec_full):.4f}")

# --- 2. Matryoshka truncation ---

embedder_256 = embedder.truncate(256)
vec_256 = embedder_256.embed("Matryoshka embeddings are neat")
print(f"Truncated (256) : dim={vec_256.shape[0]}, norm={np.linalg.norm(vec_256):.4f}")

embedder_128 = embedder.truncate(128)
vec_128 = embedder_128.embed("Matryoshka embeddings are neat")
print(f"Truncated (128) : dim={vec_128.shape[0]}, norm={np.linalg.norm(vec_128):.4f}")

print(f"\nSize comparison  : {FULL_DIM}d -> {FULL_DIM*4} bytes  |  "
      f"256d -> {256*4} bytes  |  128d -> {128*4} bytes")

# --- 3. Quantize embeddings (int8 and binary) ---

corpus = [
    "Transformers use self-attention to model long-range dependencies.",
    "BM25 is a classic lexical retrieval baseline.",
    "Matryoshka embeddings allow flexible dimensionality.",
    "Binary quantization gives 32x compression for fast search.",
    "ColBERT uses late interaction for token-level matching.",
    "Retrieval-augmented generation grounds LLMs in evidence.",
    "Cosine similarity measures angle between unit vectors.",
    "Hamming distance counts differing bits between binary codes.",
]

vecs = embedder.embed_batch(corpus)
print(f"\nCorpus: {vecs.shape[0]} docs, {vecs.shape[1]}d, {vecs.nbytes} bytes (float32)")

q_int8 = quantize_embeddings(vecs, method="int8")
print(f"int8 quantized  : {q_int8['vectors'].nbytes} bytes, "
      f"compression ratio: {q_int8['compression_ratio']:.1f}x")

q_binary = quantize_embeddings(vecs, method="binary")
print(f"binary quantized: {q_binary['vectors'].nbytes} bytes, "
      f"compression ratio: {q_binary['compression_ratio']:.1f}x")

# verify round-trip for int8
restored = dequantize_embeddings(q_int8)
mse = np.mean((vecs - restored) ** 2)
print(f"int8 round-trip MSE: {mse:.6f}")

# --- 4. Binary search ---

query = "How does attention work in transformers?"
q_vec = embedder.embed(query)
q_bin = quantize_embeddings(q_vec.reshape(1, -1), method="binary")

results = binary_search(
    query_binary=q_bin["vectors"][0],
    index_binary=q_binary["vectors"],
    k=3,
)

print(f"\nBinary search top-3 for: {query!r}")
for rank, (idx, dist) in enumerate(results, 1):
    print(f"  {rank}. [{dist:3d} bits] {corpus[idx]}")

# --- 5. Compression ratio summary ---

print("\n--- Compression summary ---")
for method in ("int8", "uint8", "binary"):
    q = quantize_embeddings(vecs, method=method)
    print(f"  {method:6s}: {q['compression_ratio']:5.1f}x  "
          f"({vecs.nbytes:,} -> {q['vectors'].nbytes:,} bytes)")
