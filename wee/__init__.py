# wee/__init__.py

from .tokenizer import Tokenizer
from .attention import scaled_dot_product_attention, GroupedQueryAttention, MultiHeadAttention
from .transformer import GPTConfig, GPT
from .bm25 import BM25
from .chunk import (
    chunk_by_words,
    chunk_by_sentences,
    chunk_by_tokens,
    chunk_by_semantic,
    chunk_with_parents,
)
from .contextual import ContextualChunker

# RAG kit
from .vectorstore import VectorStore
from .retriever import Retriever
from .context import pack_context
from .rerank import Reranker

__all__ = [
    "Tokenizer",
    "scaled_dot_product_attention",
    "GroupedQueryAttention",
    "MultiHeadAttention",
    "GPTConfig",
    "GPT",
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

from .eval import (
    evaluate_qa, exact_match, max_f1, faithfulness, context_precision_recall,
    groundedness_score, citation_support,
)
from .judge import Judge, HeuristicJudge
from .cache import Cache, cached, SemanticCache
from .trace import Tracer

__all__ += [
    "evaluate_qa", "exact_match", "max_f1", "faithfulness", "context_precision_recall",
    "groundedness_score", "citation_support",
    "Judge", "HeuristicJudge",
    "Cache", "cached", "SemanticCache",
    "Tracer",
]

# Wave 4
from .quant import QuantLinear, quantize_model, size_report, eval_perplexity
from .router import Router
from .stream import app as stream_app

__all__ += [
    "QuantLinear", "quantize_model", "size_report", "eval_perplexity",
    "Router", "stream_app",
]

# Wave 5
from .graph import Graph
from .synth import synth_qa
from .guard import Guard

__all__ += ["Graph", "synth_qa", "Guard"]

# Agentic
from .agent import Tool, SearchTool, AgentLoop

__all__ += ["Tool", "SearchTool", "AgentLoop"]

# Retrieval depth
from .colbert import ColBERTIndex
from .rewrite import rewrite_query, decompose_query, QueryExpander
from .parent_retriever import ParentRetriever
from .embed import (
    Embedder, quantize_embeddings, dequantize_embeddings,
    hamming_distance, binary_search,
)
from .hnsw import HNSWIndex

__all__ += [
    "ColBERTIndex",
    "rewrite_query", "decompose_query", "QueryExpander",
    "ParentRetriever",
    "Embedder", "quantize_embeddings", "dequantize_embeddings",
    "hamming_distance", "binary_search",
    "HNSWIndex",
]
