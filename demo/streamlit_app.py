"""
weeRAG Interactive Demo (Streamlit)
====================================
Compare dense, hybrid, and reranked retrieval side by side.

Deploy: Streamlit Community Cloud (free), pointed at this file.
Local:  streamlit run demo/streamlit_app.py
"""

from __future__ import annotations

import textwrap
import time

import numpy as np
import streamlit as st

from wee import BM25, Reranker, Retriever, Tracer, VectorStore, pack_context

# ── Corpus ────────────────────────────────────────────────────────────────────

CORPUS: list[str] = [
    "Retrieval-Augmented Generation (RAG) combines a retriever with a generator. "
    "The retriever fetches relevant documents from a corpus, and the generator "
    "produces an answer conditioned on both the query and the retrieved context.",

    "BM25 is a bag-of-words ranking function used in information retrieval. "
    "It scores documents based on term frequency, inverse document frequency, "
    "and document length normalization. It remains a strong sparse baseline.",

    "Dense retrieval encodes queries and documents into fixed-dimensional vectors "
    "using neural encoders, then ranks by cosine similarity or dot product. "
    "Models like DPR and Contriever are popular dense retrievers.",

    "Transformer models use self-attention to capture long-range dependencies. "
    "Each layer consists of multi-head attention followed by a feed-forward "
    "network, with residual connections and layer normalization.",

    "Text chunking splits long documents into smaller segments for indexing. "
    "Common strategies include fixed-size windows with overlap, sentence-based "
    "splitting, and semantic chunking that respects topic boundaries.",

    "Reranking is a second-stage retrieval step where a cross-encoder scores "
    "each (query, document) pair jointly. Cross-encoders are more accurate than "
    "bi-encoders but too slow to run over the full corpus.",

    "Reciprocal Rank Fusion (RRF) merges ranked lists from multiple retrievers. "
    "Each document receives a score of 1/(K + rank) from each list, and the "
    "fused score is the sum. RRF is simple, effective, and parameter-free.",

    "Embedding quantization reduces memory by storing vectors as int8 or binary "
    "codes instead of float32. Binary quantization followed by rescoring can "
    "maintain over 95% of full-precision retrieval quality.",

    "HNSW (Hierarchical Navigable Small World) is a graph-based approximate "
    "nearest neighbor algorithm. It builds a multi-layer proximity graph that "
    "enables sub-linear search time with high recall.",

    "Contextual retrieval prepends a short document summary to each chunk before "
    "embedding. This helps the retriever understand what the chunk is about even "
    "when the chunk itself lacks sufficient context on its own.",
]

CORPUS_IDS: list[str] = [f"doc_{i}" for i in range(len(CORPUS))]

# ── Hash-based embedding (no model download needed) ──────────────────────────

EMBED_DIM = 64


def hash_embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    np.random.seed(hash(text) % 2**31)
    v = np.random.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def word_overlap_score(query: str, doc: str) -> float:
    q_tokens = set(query.lower().split())
    d_tokens = set(doc.lower().split())
    if not q_tokens or not d_tokens:
        return 0.0
    return len(q_tokens & d_tokens) / len(q_tokens | d_tokens)


# ── Build indexes (cached across reruns) ─────────────────────────────────────

@st.cache_resource
def build_indexes():
    vs = VectorStore(dim=EMBED_DIM)
    vs.add_texts(
        CORPUS, embed_fn=hash_embed, ids=CORPUS_IDS,
        metadata=[{"text": t} for t in CORPUS],
    )
    bm25 = BM25()
    bm25.add(CORPUS)

    dense_ret = Retriever(method="topk", k=5)
    dense_ret.attach_vectorstore(vs, embed_fn=hash_embed)

    hybrid_ret = Retriever(method="rrf", k=5)
    hybrid_ret.attach_vectorstore(vs, embed_fn=hash_embed)
    hybrid_ret.attach_bm25(bm25)

    reranker = Reranker(mode="cross_encoder")
    reranker.attach_cross_encoder(word_overlap_score)

    texts_by_id = dict(zip(CORPUS_IDS, CORPUS))
    return vs, bm25, dense_ret, hybrid_ret, reranker, texts_by_id


vs, bm25, dense_ret, hybrid_ret, reranker, TEXTS_BY_ID = build_indexes()


# ── Search functions ─────────────────────────────────────────────────────────

def run_dense(query: str):
    t0 = time.perf_counter()
    results = dense_ret.search(query)
    elapsed = time.perf_counter() - t0
    return results, elapsed


def run_hybrid(query: str):
    t0 = time.perf_counter()
    results = hybrid_ret.search(query)
    elapsed = time.perf_counter() - t0
    return results, elapsed


def run_reranked(query: str):
    t0 = time.perf_counter()
    candidates = dense_ret.search(query)
    results = reranker.rerank(query, candidates, texts_by_id=TEXTS_BY_ID, k=5)
    results_with_meta = []
    for doc_id, score, meta in results:
        if "text" not in meta:
            meta = {**meta, "text": TEXTS_BY_ID.get(doc_id, "")}
        results_with_meta.append((doc_id, score, meta))
    elapsed = time.perf_counter() - t0
    return results_with_meta, elapsed


def format_results(results, elapsed, label):
    lines = [f"**{label}** — {elapsed * 1000:.2f} ms", ""]
    for rank, (doc_id, score, meta) in enumerate(results, 1):
        text = meta.get("text", doc_id)
        snippet = textwrap.shorten(text, width=160, placeholder=" ...")
        lines.append(f"**{rank}.** `{doc_id}` — score {score:.4f}")
        lines.append(f"> {snippet}")
        lines.append("")
    if not results:
        lines.append("_No results._")
    return "\n".join(lines)


# ── Streamlit UI ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="weeRAG Demo", page_icon="🔍", layout="wide")

st.title("weeRAG Interactive Demo")
st.markdown(
    "Type a question and compare **dense**, **hybrid (RRF)**, and "
    "**cross-encoder reranked** retrieval side by side. "
    "Built from scratch with [weeRAG](https://github.com/ameya690/weeRAG). "
    "No model downloads or API keys — uses hash-based embeddings."
)

query = st.text_input(
    "Query",
    placeholder="e.g. How does BM25 rank documents?",
)

EXAMPLES = [
    "How does BM25 rank documents?",
    "What is retrieval-augmented generation?",
    "Explain reranking with cross-encoders",
    "How does HNSW approximate nearest neighbor search work?",
    "What is embedding quantization?",
]

st.markdown("**Examples:** " + " · ".join(f"`{e}`" for e in EXAMPLES))

if query and query.strip():
    dense_results, dense_time = run_dense(query)
    hybrid_results, hybrid_time = run_hybrid(query)
    reranked_results, reranked_time = run_reranked(query)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(format_results(dense_results, dense_time, "Dense (top-k)"))
    with col2:
        st.markdown(format_results(hybrid_results, hybrid_time, "Hybrid (RRF)"))
    with col3:
        st.markdown(format_results(reranked_results, reranked_time, "Reranked (cross-encoder)"))

    with st.expander("Pipeline Trace"):
        tracer = Tracer()
        with tracer.span("full_pipeline", query=query):
            with tracer.span("dense_retrieval", method="topk"):
                _ = dense_ret.search(query)
            with tracer.span("hybrid_retrieval", method="rrf"):
                _ = hybrid_ret.search(query)
            with tracer.span("reranked_retrieval", method="cross_encoder"):
                candidates = dense_ret.search(query)
                _ = reranker.rerank(query, candidates, texts_by_id=TEXTS_BY_ID, k=5)
            with tracer.span("pack_context"):
                chunks = [m.get("text", "") for _, _, m in dense_results]
                scores = [s for _, s, _ in dense_results]
                _ = pack_context(chunks, scores=scores, max_tokens=512)

        trace_lines = []
        def walk_span(span, depth=0):
            indent = "  " * depth
            dur = span.duration * 1000
            attrs = ""
            if span.attrs:
                parts = [f"{k}={v}" for k, v in span.attrs.items()]
                attrs = f"  [{', '.join(parts)}]"
            trace_lines.append(f"{indent}{span.name}  {dur:.2f} ms{attrs}")
            for child in span.children:
                walk_span(child, depth + 1)

        for root in tracer.root_spans:
            walk_span(root)

        st.code("\n".join(trace_lines))
