import numpy as np
from wee import Tokenizer, BM25, VectorStore, Retriever, Reranker, pack_context, chunk_by_sentences

def hash_embed(dim=128):
    def _emb(text: str) -> np.ndarray:
        import numpy as np
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
        "Context packing fits the best chunks into a token budget."
    ]

    bm = BM25(); bm.add(docs)

    chunks = []; meta = []
    for i, d in enumerate(docs):
        for j, c in enumerate(chunk_by_sentences(d, max_chars=200)):
            chunks.append(c)
            meta.append({"doc_id": i, "chunk_id": j, "text": c})

    emb = hash_embed(128)
    vs = VectorStore(dim=128, normalize=True)
    vs.add_texts([m["text"] for m in meta], embed_fn=emb, ids=[f"{m['doc_id']}-{m['chunk_id']}" for m in meta], metadata=meta)

    q = "How do I get relevant passages for a question?"
    rt = Retriever(method="topk", k=3).attach_vectorstore(vs, emb)
    hits = rt.search(q); print("Dense hits:", hits)

    mmr = Retriever(method="mmr", k=3, mmr_lambda=0.6).attach_vectorstore(vs, emb)
    hits_mmr = mmr.search(q); print("MMR hits:", hits_mmr)

    rrf = Retriever(method="rrf", k=3).attach_vectorstore(vs, emb).attach_bm25(bm)
    hits_rrf = rrf.search(q); print("RRF hits:", hits_rrf)

    texts_by_id = {f"{m['doc_id']}-{m['chunk_id']}": m["text"] for m in meta}
    reranker = Reranker(mode="hybrid", alpha=0.5).attach_dense(emb)
    reranked = reranker.rerank(q, hits_rrf, texts_by_id, k=3); print("Reranked:", reranked)

    tokenizer = Tokenizer(); tokenizer.train(docs, vocab_size=500)
    ctx = pack_context([texts_by_id[i] for i,_,_ in reranked], scores=[s for _,s,_ in reranked], max_tokens=80, tokenizer=tokenizer)
    print("--- Packed Context ---"); print(ctx)

if __name__ == "__main__":
    main()
