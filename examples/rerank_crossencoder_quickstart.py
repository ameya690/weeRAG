"""Cross-encoder reranking quickstart.

Demonstrates mode="cross_encoder" and mode="cross_encoder_hybrid" using a
dummy cross-encoder (word-overlap score) so no model download is needed.
"""
import numpy as np
from wee import BM25, VectorStore, Retriever, Reranker, chunk_by_sentences


# ---------------------------------------------------------------------------
# Dummy cross-encoder: word-overlap ratio as a stand-in for a real model.
# In production, swap this with e.g. a HuggingFace cross-encoder or an LLM call.
# ---------------------------------------------------------------------------
def dummy_cross_encoder(query: str, doc: str) -> float:
    q_tokens = set(query.lower().split())
    d_tokens = set(doc.lower().split())
    if not q_tokens:
        return 0.0
    return len(q_tokens & d_tokens) / len(q_tokens)


def hash_embed(dim=128):
    def _emb(text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(dim).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        return v
    return _emb


def main():
    docs = [
        "RAG retrieves relevant context from a knowledge base.",
        "Transformers use attention to model token dependencies.",
        "BM25 provides a strong lexical baseline for retrieval.",
        "Vector stores enable fast similarity search over embeddings.",
        "Cross-encoder models score query-document pairs jointly for high accuracy.",
    ]

    # Build chunks and metadata
    chunks, meta = [], []
    for i, d in enumerate(docs):
        for j, c in enumerate(chunk_by_sentences(d, max_chars=200)):
            chunks.append(c)
            meta.append({"doc_id": i, "chunk_id": j, "text": c})

    emb = hash_embed(128)
    vs = VectorStore(dim=128, normalize=True)
    vs.add_texts(
        [m["text"] for m in meta],
        embed_fn=emb,
        ids=[f"{m['doc_id']}-{m['chunk_id']}" for m in meta],
        metadata=meta,
    )

    texts_by_id = {f"{m['doc_id']}-{m['chunk_id']}": m["text"] for m in meta}
    q = "How does cross-encoder reranking improve retrieval?"

    # Initial dense retrieval
    rt = Retriever(method="topk", k=5).attach_vectorstore(vs, emb)
    hits = rt.search(q)
    print("Initial dense hits:")
    for cid, score, m in hits:
        print(f"  {cid}  score={score:.3f}  {texts_by_id[cid][:60]}...")

    # --- Cross-encoder rerank ---
    reranker_ce = Reranker(mode="cross_encoder").attach_cross_encoder(dummy_cross_encoder)
    reranked_ce = reranker_ce.rerank(q, hits, texts_by_id, k=3)
    print("\nCross-encoder reranked (top 3):")
    for cid, score, m in reranked_ce:
        print(f"  {cid}  score={score:.3f}  {texts_by_id[cid][:60]}...")

    # --- Cross-encoder + BM25 hybrid rerank ---
    reranker_hybrid = (
        Reranker(mode="cross_encoder_hybrid", alpha=0.7)
        .attach_cross_encoder(dummy_cross_encoder)
    )
    reranked_hybrid = reranker_hybrid.rerank(q, hits, texts_by_id, k=3)
    print("\nCross-encoder hybrid reranked (alpha=0.7, top 3):")
    for cid, score, m in reranked_hybrid:
        print(f"  {cid}  score={score:.3f}  {texts_by_id[cid][:60]}...")


if __name__ == "__main__":
    main()
