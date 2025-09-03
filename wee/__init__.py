# wee/__init__.py

from .tokenizer import Tokenizer
from .attention import scaled_dot_product_attention, MultiHeadAttention
from .transformer import GPTConfig, GPT
from .bm25 import BM25
from .chunk import (
    chunk_by_words,
    chunk_by_sentences,
    chunk_by_tokens,
)

# RAG kit
from .vectorstore import VectorStore
from .retriever import Retriever
from .context import pack_context
from .rerank import Reranker

__all__ = [
    "Tokenizer",
    "scaled_dot_product_attention",
    "MultiHeadAttention",
    "GPTConfig",
    "GPT",
    "BM25",
    "chunk_by_words",
    "chunk_by_sentences",
    "chunk_by_tokens",
    "VectorStore",
    "Retriever",
    "pack_context",
    "Reranker",
]

from .weeeval import evaluate_qa, exact_match, max_f1, faithfulness, context_precision_recall
from .weejudge import Judge, HeuristicJudge
from .weecache import Cache, cached
from .weetrace import Tracer

__all__ += [
    "evaluate_qa", "exact_match", "max_f1", "faithfulness", "context_precision_recall",
    "Judge", "HeuristicJudge",
    "Cache", "cached",
    "Tracer",
]

# Wave 4
from .quant import QuantLinear, quantize_model, size_report, eval_perplexity
from .kv import upgrade_gpt_for_kv, generate_with_cache, save_kv, load_kv
from .router import Router
from .stream import app as stream_app

__all__ += [
    "QuantLinear","quantize_model","size_report","eval_perplexity",
    "upgrade_gpt_for_kv","generate_with_cache","save_kv","load_kv",
    "Router","stream_app",
]

# Wave 5
from .graph import Graph
from .synth import synth_qa
from .guard import Guard

__all__ += ["Graph", "synth_qa", "Guard"]
