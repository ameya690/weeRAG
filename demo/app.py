"""
weeRAG Interactive Demo
=======================
A self-contained Gradio app that demonstrates dense, hybrid, and reranked
retrieval side by side with timing and trace visualization.

Run:
    pip install -e ".[demo]"
    python demo/app.py
"""

from __future__ import annotations

import time
import textwrap
import numpy as np

try:
    import gradio as gr
except ImportError:
    raise SystemExit(
        "Gradio is required for the demo. Install it with:\n"
        "    pip install -e '.[demo]'"
    )

from wee import VectorStore, Retriever, BM25, Reranker, Tracer, pack_context

# ---------------------------------------------------------------------------
# Corpus -- small set of RAG-related passages for demonstration
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Hash-based embedding -- deterministic, no model download required
# ---------------------------------------------------------------------------

EMBED_DIM = 64


def hash_embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Deterministic hash-based pseudo-embedding (demo only)."""
    np.random.seed(hash(text) % 2**31)
    v = np.random.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


# ---------------------------------------------------------------------------
# Word-overlap cross-encoder (dummy stand-in)
# ---------------------------------------------------------------------------


def word_overlap_score(query: str, doc: str) -> float:
    """Jaccard-ish overlap score used as a dummy cross-encoder."""
    q_tokens = set(query.lower().split())
    d_tokens = set(doc.lower().split())
    if not q_tokens or not d_tokens:
        return 0.0
    intersection = q_tokens & d_tokens
    return len(intersection) / len(q_tokens | d_tokens)


# ---------------------------------------------------------------------------
# Index setup (runs once at import time)
# ---------------------------------------------------------------------------

# Vector store
vs = VectorStore(dim=EMBED_DIM)
vs.add_texts(
    CORPUS,
    embed_fn=hash_embed,
    ids=CORPUS_IDS,
    metadata=[{"text": t} for t in CORPUS],
)

# BM25 index
bm25 = BM25()
bm25.add(CORPUS)

# Retrievers
dense_retriever = Retriever(method="topk", k=5)
dense_retriever.attach_vectorstore(vs, embed_fn=hash_embed)

hybrid_retriever = Retriever(method="rrf", k=5)
hybrid_retriever.attach_vectorstore(vs, embed_fn=hash_embed)
hybrid_retriever.attach_bm25(bm25)

# Reranker (cross-encoder mode with word-overlap dummy)
reranker = Reranker(mode="cross_encoder")
reranker.attach_cross_encoder(word_overlap_score)

# Text lookup for reranker
TEXTS_BY_ID: dict[str, str] = dict(zip(CORPUS_IDS, CORPUS))

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_results(
    results: list[tuple[str, float, dict]],
    elapsed: float,
    label: str,
) -> str:
    """Format retrieval results as Markdown."""
    lines = [f"### {label}", f"**{elapsed * 1000:.2f} ms**", ""]
    for rank, (doc_id, score, meta) in enumerate(results, 1):
        text = meta.get("text", doc_id)
        # Truncate for display
        snippet = textwrap.shorten(text, width=180, placeholder=" ...")
        lines.append(f"**{rank}.** `{doc_id}` &mdash; score {score:.4f}")
        lines.append(f"> {snippet}")
        lines.append("")
    if not results:
        lines.append("_No results._")
    return "\n".join(lines)


def _format_trace(tracer: Tracer) -> str:
    """Render the tracer spans as a readable Markdown trace."""
    lines = ["### Pipeline Trace", "```"]

    def _walk(span, depth: int = 0):
        indent = "  " * depth
        dur = span.duration * 1000
        attrs = ""
        if span.attrs:
            parts = [f"{k}={v}" for k, v in span.attrs.items()]
            attrs = f"  [{', '.join(parts)}]"
        lines.append(f"{indent}{span.name}  {dur:.2f} ms{attrs}")
        for child in span.children:
            _walk(child, depth + 1)

    for root in tracer.root_spans:
        _walk(root)

    lines.append("```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Search functions
# ---------------------------------------------------------------------------


def run_dense(query: str) -> str:
    tracer = Tracer()
    with tracer.span("dense_retrieval", method="topk", k=5):
        t0 = time.perf_counter()
        with tracer.span("embed_query"):
            _q = hash_embed(query)
        with tracer.span("vector_search"):
            results = dense_retriever.search(query)
        elapsed = time.perf_counter() - t0
    return _format_results(results, elapsed, "Dense Retrieval (top-k)")


def run_hybrid(query: str) -> str:
    tracer = Tracer()
    with tracer.span("hybrid_retrieval", method="rrf", k=5):
        t0 = time.perf_counter()
        with tracer.span("embed_query"):
            _q = hash_embed(query)
        with tracer.span("bm25_search"):
            _bm_hits = bm25.search(query, k=5)
        with tracer.span("rrf_fusion"):
            results = hybrid_retriever.search(query)
        elapsed = time.perf_counter() - t0
    return _format_results(results, elapsed, "Hybrid Retrieval (RRF)")


def run_reranked(query: str) -> str:
    tracer = Tracer()
    with tracer.span("reranked_retrieval", method="cross_encoder", k=5):
        t0 = time.perf_counter()
        with tracer.span("dense_first_stage"):
            candidates = dense_retriever.search(query)
        with tracer.span("cross_encoder_rerank"):
            results = reranker.rerank(
                query, candidates, texts_by_id=TEXTS_BY_ID, k=5
            )
        elapsed = time.perf_counter() - t0
    # Attach text metadata so formatting works
    results_with_meta = []
    for doc_id, score, meta in results:
        if "text" not in meta:
            meta = {**meta, "text": TEXTS_BY_ID.get(doc_id, "")}
        results_with_meta.append((doc_id, score, meta))
    return _format_results(results_with_meta, elapsed, "Reranked (cross-encoder)")


def run_trace(query: str) -> str:
    """Run all three pipelines under a single tracer and return the trace."""
    tracer = Tracer()

    with tracer.span("full_pipeline", query=query):
        with tracer.span("dense_retrieval", method="topk"):
            with tracer.span("embed_query"):
                _q = hash_embed(query)
            with tracer.span("vector_search"):
                dense_results = dense_retriever.search(query)

        with tracer.span("hybrid_retrieval", method="rrf"):
            with tracer.span("embed_query"):
                _q = hash_embed(query)
            with tracer.span("bm25_search"):
                _bm_hits = bm25.search(query, k=5)
            with tracer.span("rrf_fusion"):
                hybrid_results = hybrid_retriever.search(query)

        with tracer.span("reranked_retrieval", method="cross_encoder"):
            with tracer.span("dense_first_stage"):
                candidates = dense_retriever.search(query)
            with tracer.span("cross_encoder_rerank"):
                reranked_results = reranker.rerank(
                    query, candidates, texts_by_id=TEXTS_BY_ID, k=5
                )

        with tracer.span("pack_context"):
            chunks = [m.get("text", "") for _, _, m in dense_results]
            scores = [s for _, s, _ in dense_results]
            _packed = pack_context(chunks, scores=scores, max_tokens=512)

    return _format_trace(tracer)


def search_all(query: str) -> tuple[str, str, str, str]:
    """Run all three retrieval modes and return results + trace."""
    if not query or not query.strip():
        empty = "_Enter a query above._"
        return empty, empty, empty, ""
    dense_md = run_dense(query)
    hybrid_md = run_hybrid(query)
    reranked_md = run_reranked(query)
    trace_md = run_trace(query)
    return dense_md, hybrid_md, reranked_md, trace_md


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

EXAMPLES = [
    ["How does BM25 rank documents?"],
    ["What is retrieval-augmented generation?"],
    ["Explain reranking with cross-encoders"],
    ["How does HNSW approximate nearest neighbor search work?"],
    ["What is embedding quantization?"],
]

CSS = """
.result-col {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 12px;
    background: #fafafa;
}
"""

with gr.Blocks(
    title="weeRAG Interactive Demo",
    css=CSS,
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# weeRAG Interactive Demo\n"
        "Type a question and compare **dense**, **hybrid (RRF)**, and "
        "**reranked** retrieval side by side. Each column shows the top-5 "
        "results with scores and latency. The trace below visualizes the "
        "full pipeline span hierarchy.\n\n"
        "*This demo uses a small built-in corpus of 10 passages about RAG "
        "concepts. No model downloads or API keys required.*"
    )

    with gr.Row():
        query_box = gr.Textbox(
            label="Query",
            placeholder="e.g. How does BM25 rank documents?",
            scale=4,
        )
        submit_btn = gr.Button("Search", variant="primary", scale=1)

    with gr.Row(equal_height=True):
        dense_out = gr.Markdown(
            label="Dense",
            elem_classes=["result-col"],
        )
        hybrid_out = gr.Markdown(
            label="Hybrid (RRF)",
            elem_classes=["result-col"],
        )
        reranked_out = gr.Markdown(
            label="Reranked",
            elem_classes=["result-col"],
        )

    trace_out = gr.Markdown(label="Pipeline Trace")

    submit_btn.click(
        fn=search_all,
        inputs=[query_box],
        outputs=[dense_out, hybrid_out, reranked_out, trace_out],
    )
    query_box.submit(
        fn=search_all,
        inputs=[query_box],
        outputs=[dense_out, hybrid_out, reranked_out, trace_out],
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[query_box],
        outputs=[dense_out, hybrid_out, reranked_out, trace_out],
        fn=search_all,
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch()
