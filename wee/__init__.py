# wee/__init__.py
#
# Torch-dependent modules are guarded so the pure-numpy core
# (vectorstore, retriever, bm25, rerank, eval, etc.) works without torch.

from .bm25 import BM25
from .chunk import (
    chunk_by_semantic,
    chunk_by_sentences,
    chunk_by_tokens,
    chunk_by_words,
    chunk_with_parents,
)
from .context import pack_context
from .contextual import ContextualChunker
from .rerank import Reranker
from .retriever import Retriever
from .vectorstore import VectorStore

__all__ = [
    "BM25",
    "chunk_by_words",
    "chunk_by_sentences",
    "chunk_by_tokens",
    "chunk_by_semantic",
    "chunk_with_parents",
    "ContextualChunker",
    "VectorStore",
    "Retriever",
    "pack_context",
    "Reranker",
]

# ── Torch-dependent modules ──────────────────────────────────────────────────

try:
    from .attention import (
        GroupedQueryAttention,
        MultiHeadAttention,
        scaled_dot_product_attention,
    )
    from .tokenizer import Tokenizer
    from .transformer import GPT, GPTConfig

    __all__ += [
        "Tokenizer",
        "scaled_dot_product_attention",
        "GroupedQueryAttention",
        "MultiHeadAttention",
        "GPTConfig",
        "GPT",
    ]
except ImportError:
    pass

# ── Eval & ops ───────────────────────────────────────────────────────────────

from .cache import Cache, SemanticCache, cached
from .eval import (
    citation_support,
    context_precision_recall,
    evaluate_qa,
    exact_match,
    faithfulness,
    groundedness_score,
    max_f1,
)
from .judge import HeuristicJudge, Judge
from .trace import Tracer

__all__ += [
    "evaluate_qa", "exact_match", "max_f1", "faithfulness", "context_precision_recall",
    "groundedness_score", "citation_support",
    "Judge", "HeuristicJudge",
    "Cache", "cached", "SemanticCache",
    "Tracer",
]

# ── Performance & scaling (torch-dependent) ──────────────────────────────────

try:
    from .quant import QuantLinear, eval_perplexity, quantize_model, size_report

    __all__ += ["QuantLinear", "quantize_model", "size_report", "eval_perplexity"]
except ImportError:
    pass

try:
    from .router import Router
    from .stream import app as stream_app

    __all__ += ["Router", "stream_app"]
except ImportError:
    pass

# ── Knowledge & synthesis ────────────────────────────────────────────────────

from .graph import Graph
from .guard import Guard
from .synth import synth_qa

__all__ += ["Graph", "synth_qa", "Guard"]

# ── Agentic ──────────────────────────────────────────────────────────────────

from .agent import AgentLoop, SearchTool, Tool

__all__ += ["Tool", "SearchTool", "AgentLoop"]

# ── Retrieval depth ──────────────────────────────────────────────────────────

from .colbert import ColBERTIndex
from .embed import (
    Embedder,
    binary_search,
    dequantize_embeddings,
    hamming_distance,
    quantize_embeddings,
)
from .hnsw import HNSWIndex
from .parent_retriever import ParentRetriever
from .rewrite import QueryExpander, decompose_query, rewrite_query

__all__ += [
    "ColBERTIndex",
    "rewrite_query", "decompose_query", "QueryExpander",
    "ParentRetriever",
    "Embedder", "quantize_embeddings", "dequantize_embeddings",
    "hamming_distance", "binary_search",
    "HNSWIndex",
]
