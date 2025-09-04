<p align="center">
  <img width="350" alt="weeRAG logo" src="https://github.com/user-attachments/assets/3c808464-ef6b-497f-a46c-d7b90edf9a7c" />
</p>

# weeRAG — tiny, teachable LLM & RAG foundations
 
**weeRAG** is a collection of small, readable implementations of building blocks behind  
**LLMs (Large Language Models)** and **RAG (Retrieval-Augmented Generation)** pipelines.  

The design goal: *clarity over performance*. Each component is implemented in a single file, with minimal dependencies, so you can read, learn, and hack.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [1. Foundations](#1-foundations)
- [2. RAG Toolkit](#2-rag-toolkit)
- [3. Evaluation & Ops](#3-evaluation--ops)
- [4. Performance & Scaling](#4-performance--scaling)
- [5. Extras](#5-extras)
- [License](#license)

---

## Overview

Why break things into categories?

- **Foundations** → the *core primitives* that any LLM or RAG pipeline builds on: tokenization, transformers, attention, lexical retrieval, chunking.  
- **RAG Toolkit** → higher-level components specifically for retrieval pipelines: vector stores, retrievers, context packing, rerankers.  
- **Evaluation & Ops** → how you measure, monitor, and operate RAG systems: metrics, judges, caches, tracing.  
- **Performance & Scaling** → the tricks that make systems practical: quantization, KV caches, routers, streaming.  
- **Extras** → goodies that extend RAG beyond basics: knowledge graphs, synthetic eval data, guardrails.

---

## Installation

```bash
git clone <your-fork-or-repo> weeRAG
cd weeRAG
pip install -e .
```

FastAPI + Uvicorn for streaming demos

---
## 1. Foundations

Core LLM building blocks. Minimal, readable versions of what production frameworks do at scale.

Tokenizer (wee.tokenizer.Tokenizer)

Char-BPE with ▁ word-boundary markers.

```python
from wee import Tokenizer
tok = Tokenizer()
tok.train(["hello world", "hello there"], vocab_size=200)
ids = tok.encode("hello world", add_special=True)
print(ids)             # -> [2, 29, 12, ..., 25, 3]
print(tok.decode(ids)) # -> "hello world"
```
```yaml
Encoded IDs: [2, 29, 12, ..., 25, 3]
Decoded: attention retrieves chunks
```

### Attention (wee.attention.MultiHeadAttention)

PyTorch multi-head attention with causal mask.

### Transformer (wee.transformer.GPT)

Minimal GPT-style decoder with .generate().

```yaml
Model params: 675840
```
### BM25 (wee.bm25.BM25)

Lexical retriever baseline.
```python
BM25 search: [(1, 1.105...), (0, 0.980...)]
```

### Chunking (wee.chunk)
Word, sentence, or token-based chunkers for long documents
```python
Sentence chunks: ['This is a sentence.', 'ence. Another sentence!', 'ence! And one more? Yep.']
```
## RAG Toolkit
Pieces to assemble an end-to-end retrieval pipeline.

### VectorStore (wee.vectorstore.VectorStore)

In-memory dense vector index with cosine search.

### Retriever (wee.retriever.Retriever)

Strategies:

topk — standard dense retrieval

mmr — Maximal Marginal Relevance (balances relevance & diversity)

rrf — Reciprocal Rank Fusion (combine dense + lexical)

```python
Dense hits: [('2-0', 0.04, 'BM25 ...'), ('1-0', ...)]
MMR hits: [('2-0', ...), ('1-0', ...)]
RRF hits: [('2-0', ...), ('0-0', 'RAG retrieves ...')]
```

### Reranker (wee.rerank.Reranker)
Re-score retrieved candidates with dense, lexical, or hybrid.

```python
Reranked: [('Context packing ...'), ('BM25 ...'), ('RAG retrieves ...')]
``` 

### Context (wee.context.pack_context)
Budgeted packing of chunks into the LLM’s context window.
```yaml
-- Packed Context ---
Context packing fits the best chunks into a token budget.
```

## Evaluation & Ops
Once you can retrieve & generate, you need to evaluate outputs and operate the system reliably.

### Eval Metrics (wee.weeeval)

Exact Match (EM)

F1 overlap

Faithfulness (are answers supported by context?)

Context precision/recall

Output:
```yaml
"metrics": {"em": 0.5, "f1": 0.611, "faithfulness": 0.5,
            "context_precision": 1.0, "context_recall": 1.0}
```

### Judge (wee.weejudge.Judge)
Heuristic (embedding sim) or LLM-as-judge.
```rust
What baseline...? -> score ≈ 0.51
```
### Cache (wee.weecache.Cache)
SQLite-backed caching + decorator.
```yaml
Cache hit: True
```
### Tracer (wee.weetrace.Tracer)
Hierarchical spans with JSON or HTML export.
```yaml
wee trace
pipeline (0.032s)
  index (0.030s)
  retrieve (0.000s)
  rerank (0.000s)
  pack (0.001s)
```
## Performance & Scaling
Toy versions of tricks production LLMs use for speed & efficiency.

### Quantization (wee.quant)
Post-training int8/int4 quantization of linear layers.
```yaml
Float32 size: {'parameters': 684032, 'bytes': 2736128}
PPL (fp32): 34.409
Quantized size: {'parameters': 24320, 'bytes': 97280}
PPL (int8): 34.432
```
### KV Cache (wee.kv)
Cache keys/values to accelerate generation.
```yaml
Generated: RAG retrieves context. Asouhn...
Time with KV cache: 0.021 s
```
### Router (wee.router.Router)
Route queries to models by quality, cost, or latency.
```yaml
Decision (quality): endpoint=mini, est_cost=0.022
Decision (cost): endpoint=mini, est_cost=0.044
```
### Streaming (wee.stream)
Minimal FastAPI SSE server to stream tokens.
```python
uvicorn wee.stream:app --reload
GET /stream?text=hello
```
## Extras
Beyond the basics: tools that make RAG practical and safer.
### Knowledge Graph (wee.graph.Graph)
Extract triples from text; explore & export DOT/JSON.
```yaml
Triples:
  (Transformers) -[->]-> (Tokens)
```
### Synthetic QA (wee.synth.synth_qa)
Generate tiny cloze/WH questions from context to build eval sets.
```yaml
CLOZE Q: Vector stores ____ similarity search.  A: ['enable']
CLOZE Q: ____ is a lexical baseline.            A: ['BM25']
WH Q:    What does RAG retrieve?                A: ['relevant context']
```
### Guard (wee.guard.Guard)
Check for PII, profanity, prompt-injection, or untrusted links.
```rust
Report: {'pii': ['email','phone'],
         'injection': ['ignore ...','act as administrator'],
         'unallowed_links': ['https://evil.com/model.bin'], 'risk': 0.9}
Sanitized:
 Contact me at [email] or [phone].
```
