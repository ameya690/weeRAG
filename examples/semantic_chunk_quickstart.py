"""Semantic chunking & parent-document retrieval quickstart.

Demonstrates:
  1. ``chunk_by_semantic`` with a deterministic dummy embedding function.
  2. ``ParentRetriever`` end-to-end with the same dummy embeddings.

Swap ``dummy_embed`` for a real embedding model (e.g. sentence-transformers)
to get meaningful similarity scores.
"""

import numpy as np
from wee.chunk import chunk_by_semantic, chunk_with_parents
from wee.parent_retriever import ParentRetriever


# ---------------------------------------------------------------------------
# Dummy embedding: hash-based, deterministic, not semantically meaningful.
# Replace with a real model for production use.
# ---------------------------------------------------------------------------
_DIM = 64


def dummy_embed(text: str) -> np.ndarray:
    """Deterministic pseudo-embedding based on character hashing."""
    rng = np.random.RandomState(abs(hash(text)) % (2**31))
    vec = rng.randn(_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-9
    return vec


# ---------------------------------------------------------------------------
# Part 1: Semantic chunking
# ---------------------------------------------------------------------------
def demo_semantic_chunking():
    print("=" * 60)
    print("Part 1: Semantic Chunking")
    print("=" * 60)

    doc = (
        "Retrieval-Augmented Generation (RAG) combines a retriever with a "
        "language model. The retriever fetches relevant passages from a "
        "knowledge base. The language model then conditions on those passages "
        "to produce a grounded answer. "
        "Transformers use self-attention to weigh token relationships. "
        "Multi-head attention allows the model to focus on different parts of "
        "the input simultaneously. "
        "Chunking the knowledge base is critical. Each chunk must be "
        "self-contained enough for accurate retrieval. Small chunks improve "
        "precision but may lose surrounding context."
    )

    chunks = chunk_by_semantic(doc, embed_fn=dummy_embed, threshold=0.5, min_chunk_size=50)
    print(f"\nInput length : {len(doc)} chars")
    print(f"Chunks found : {len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        print(f"  [{i}] ({len(chunk)} chars) {chunk[:90]}...")
    print()


# ---------------------------------------------------------------------------
# Part 2: Parent-document retrieval
# ---------------------------------------------------------------------------
def demo_parent_retrieval():
    print("=" * 60)
    print("Part 2: Parent-Document Retrieval")
    print("=" * 60)

    documents = [
        (
            "RAG systems retrieve passages from a knowledge base and feed "
            "them to a language model. This reduces hallucination and keeps "
            "answers grounded in source material. Chunking strategy matters: "
            "too large and retrieval is imprecise, too small and context is "
            "lost. A two-level hierarchy solves this trade-off."
        ),
        (
            "Transformers revolutionised NLP with self-attention. Each layer "
            "computes attention scores between all token pairs. Positional "
            "encodings inject order information. Modern variants use rotary "
            "position embeddings and grouped-query attention for efficiency."
        ),
    ]

    # Show the raw hierarchy
    print("\nRaw hierarchy for document 0:")
    hierarchy = chunk_with_parents(documents[0])
    for entry in hierarchy:
        print(f"  Parent {entry['parent_id']}: {entry['parent'][:70]}...")
        for child in entry["children"]:
            print(f"    Child {child['child_id']}: {child['text'][:60]}...")
    print()

    # Build retriever
    retriever = ParentRetriever.from_documents(documents, embed_fn=dummy_embed)

    query = "How does chunking affect retrieval quality?"
    print(f'Query: "{query}"\n')

    results = retriever.search(query, k=3)
    for i, r in enumerate(results):
        print(f"  Result {i} (score={r['score']:.4f})")
        print(f"    Child : {r['child'][:70]}...")
        print(f"    Parent: {r['parent'][:70]}...")
    print()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_semantic_chunking()
    demo_parent_retrieval()
