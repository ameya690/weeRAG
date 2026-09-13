"""ColBERT late-interaction retrieval quickstart.

ColBERT keeps a separate embedding vector for every *token* in a document,
rather than compressing the whole document into a single vector.  At query
time it uses **MaxSim** scoring: for each query-token vector, find the
maximum cosine similarity with any document-token vector, then sum those
maxima.  This captures fine-grained token-level matches while still
allowing document embeddings to be pre-computed.

This demo uses a deterministic random encoder (seeded on the text hash) so
it runs without any model download.  Swap ``token_encode_fn`` with a real
encoder (e.g. a ColBERT checkpoint) for meaningful retrieval.
"""
import numpy as np

from wee.colbert import ColBERTIndex

DIM = 128          # embedding dimension per token
TOKENS_PER_TEXT = 8  # fixed number of "tokens" our dummy encoder produces


# ---------------------------------------------------------------------------
# Dummy token encoder: deterministic random vectors seeded on text hash.
# A real encoder would run a transformer and return one vector per wordpiece.
# ---------------------------------------------------------------------------
def token_encode_fn(text: str) -> np.ndarray:
    """Return shape (TOKENS_PER_TEXT, DIM) of fake per-token embeddings."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    vecs = rng.standard_normal((TOKENS_PER_TEXT, DIM)).astype(np.float32)
    return vecs  # ColBERTIndex.add() handles L2-normalisation


def main():
    # --- Build the index ---
    docs = {
        "doc-rag": "Retrieval-augmented generation grounds LLM answers in external knowledge.",
        "doc-bm25": "BM25 is a strong lexical baseline using term frequency and inverse document frequency.",
        "doc-colbert": "ColBERT scores passages with late interaction between query and document token embeddings.",
        "doc-vec": "Dense vector search encodes full passages into single embeddings for fast ANN lookup.",
        "doc-cross": "Cross-encoders score query-document pairs jointly but are expensive at scale.",
    }

    index = ColBERTIndex(dim=DIM)
    index.add_texts(
        doc_ids=list(docs.keys()),
        texts=list(docs.values()),
        encode_fn=token_encode_fn,
    )
    print(f"Indexed {len(docs)} documents ({TOKENS_PER_TEXT} token vectors each, dim={DIM})\n")

    # --- Search ---
    query = "How does ColBERT score documents?"
    print(f"Query: {query!r}\n")

    results = index.search_text(query, encode_fn=token_encode_fn, k=3)

    print("Top-3 results (doc_id, MaxSim score):")
    for doc_id, score in results:
        print(f"  {doc_id:12s}  score={score:.3f}  {docs[doc_id][:60]}...")

    # --- Peek at the raw MaxSim matrix for the top hit ---
    q_emb = token_encode_fn(query)
    # Normalise like the index does
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
    top_doc_id = results[0][0]
    # Find the stored (normalised) doc embeddings
    doc_emb = next(emb for did, emb in index._docs if did == top_doc_id)
    sim_matrix = q_emb @ doc_emb.T  # (Q, D) cosine similarities
    print(f"\nSimilarity matrix shape (query tokens x doc tokens): {sim_matrix.shape}")
    print(f"Per-query-token max sims: {sim_matrix.max(axis=1).round(3)}")
    print(f"Sum (= MaxSim score):     {sim_matrix.max(axis=1).sum():.3f}")


if __name__ == "__main__":
    main()
