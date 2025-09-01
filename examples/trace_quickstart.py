import numpy as np
from wee import Tracer, Retriever, VectorStore, BM25, Reranker, Tokenizer, pack_context, chunk_by_sentences
def hash_embed(dim=64):
    def _emb(text: str):
        rng=np.random.default_rng(abs(hash(text))%(2**32)); v=rng.standard_normal(dim).astype(np.float32); v/= (np.linalg.norm(v)+1e-9); return v
    return _emb
def main():
    tracer=Tracer()
    with tracer.span('pipeline'):
        docs=[
            'RAG retrieves relevant context from a knowledge base.',
            'Transformers use attention to model token dependencies.',
            'BM25 provides a strong lexical baseline for retrieval.',
            'Vector stores enable fast similarity search over embeddings.',
            'Context packing fits the best chunks into a token budget.'
        ]
        emb=hash_embed(64)
        with tracer.span('index'):
            vs=VectorStore(dim=64, normalize=True); meta=[{"id":str(i),"text":d} for i,d in enumerate(docs)]
            vs.add_texts([m['text'] for m in meta], embed_fn=emb, ids=[m['id'] for m in meta], metadata=meta)
            bm=BM25(); bm.add(docs)
        q='How do I get relevant passages for a question?'
        with tracer.span('retrieve', query=q):
            rt=Retriever(method='rrf', k=3).attach_vectorstore(vs, emb).attach_bm25(bm); hits=rt.search(q)
        with tracer.span('rerank'):
            texts={m['id']:m['text'] for m in meta}; rr=Reranker(mode='hybrid', alpha=0.5).attach_dense(emb); hits=rr.rerank(q, hits, texts, k=3)
        with tracer.span('pack'):
            tok=Tokenizer(); tok.train(docs, vocab_size=500)
            ctx=pack_context([texts[i] for i,_,_ in hits], scores=[s for _,s,_ in hits], max_tokens=80, tokenizer=tok)
    html=tracer.export_html(); open('wee_trace.html','w').write(html); print('Trace exported to wee_trace.html')
if __name__=='__main__': main()
