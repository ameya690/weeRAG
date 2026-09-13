"""Contextual retrieval quickstart.

Demonstrates ``ContextualChunker`` with a dummy generate_fn that doesn't
call a real LLM — swap it out with your own model wrapper for production.
"""

from wee import ContextualChunker


def dummy_generate_fn(document: str, chunk: str) -> str:
    """Placeholder that summarises the chunk's first 10 words."""
    first_10 = " ".join(chunk.split()[:10])
    return f"This chunk discusses: {first_10}"


def main():
    doc = (
        "Retrieval-Augmented Generation (RAG) combines a retriever with a "
        "language model. The retriever fetches relevant passages from a "
        "knowledge base. The language model then conditions on those passages "
        "to produce a grounded answer. This two-stage approach reduces "
        "hallucination and lets the model access up-to-date information "
        "without retraining. Chunking the knowledge base is critical — each "
        "chunk must be self-contained enough for accurate retrieval, yet small "
        "enough to fit within the model's context window."
    )

    chunker = ContextualChunker(generate_fn=dummy_generate_fn)
    results = chunker.contextualize(doc)

    for i, r in enumerate(results):
        print(f"--- Chunk {i} ---")
        print(f"Original:        {r['original'][:80]}...")
        print(f"Context:         {r['context']}")
        print(f"Contextualized:  {r['contextualized'][:120]}...")
        print()

    # Batch mode
    docs = [doc, "Transformers use self-attention to weigh token relationships."]
    batch = chunker.contextualize_batch(docs)
    print(f"Batch processed {len(batch)} documents, "
          f"yielding {sum(len(d) for d in batch)} total chunks.")


if __name__ == "__main__":
    main()
