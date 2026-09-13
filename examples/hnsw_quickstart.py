"""HNSW quickstart -- build an index, search, and compare with brute-force."""

import time
import numpy as np
from wee.hnsw import HNSWIndex

# -----------------------------------------------------------------
# 1. Build an index with random vectors
# -----------------------------------------------------------------
DIM = 128
N = 5_000
K = 10

rng = np.random.default_rng(42)
data = rng.standard_normal((N, DIM)).astype(np.float32)

print(f"Building HNSW index: {N} vectors, dim={DIM}, M=16, ef_construction=200")
index = HNSWIndex(dim=DIM, M=16, ef_construction=200, max_level=5)

t0 = time.perf_counter()
for i in range(N):
    index.add(data[i], doc_id=f"doc-{i}")
build_time = time.perf_counter() - t0
print(f"  Build time: {build_time:.2f}s\n")

# -----------------------------------------------------------------
# 2. Brute-force exact search (for comparison)
# -----------------------------------------------------------------
query = rng.standard_normal(DIM).astype(np.float32)

def brute_force_search(q: np.ndarray, vectors: np.ndarray, k: int):
    """Cosine similarity brute-force."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
    normed = vectors / norms
    q_norm = q / max(np.linalg.norm(q), 1e-12)
    sims = (normed @ q_norm).ravel()
    top_k = np.argpartition(-sims, k)[:k]
    top_k = top_k[np.argsort(-sims[top_k])]
    return [(f"doc-{i}", float(sims[i])) for i in top_k]

t0 = time.perf_counter()
exact = brute_force_search(query, data, K)
bf_time = time.perf_counter() - t0

# -----------------------------------------------------------------
# 3. HNSW search at various ef values -- recall vs speed tradeoff
# -----------------------------------------------------------------
print(f"{'ef':>6}  {'recall@' + str(K):>10}  {'HNSW (ms)':>10}  {'speedup':>8}")
print("-" * 42)

exact_ids = {r[0] for r in exact}

for ef in [10, 20, 50, 100, 200, 500]:
    t0 = time.perf_counter()
    results = index.search(query, k=K, ef=ef)
    hnsw_time = time.perf_counter() - t0

    found_ids = {r[0] for r in results}
    recall = len(found_ids & exact_ids) / K

    speedup = bf_time / hnsw_time if hnsw_time > 0 else float("inf")
    print(f"{ef:>6}  {recall:>10.2f}  {hnsw_time * 1000:>9.2f}ms  {speedup:>7.1f}x")

print(f"\nBrute-force time: {bf_time * 1000:.2f}ms")
print("\nTakeaway: higher ef => better recall, but slower search.")
