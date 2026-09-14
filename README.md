

[![CI](https://github.com/ameya690/weeRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/ameya690/weeRAG/actions/workflows/ci.yml)
[![Bench](https://github.com/ameya690/weeRAG/actions/workflows/bench.yml/badge.svg)](https://github.com/ameya690/weeRAG/actions/workflows/bench.yml)

**weeRAG** — small, readable implementations of the building blocks behind LLMs and RAG pipelines.

Every component lives in a single file with minimal dependencies. The goal isn't a framework — it's a codebase you can read end-to-end in an afternoon and actually understand what each piece does, why it exists, and where the tradeoffs are.

---

## Installation

```bash
git clone https://github.com/ameya690/weeRAG.git
cd weeRAG
pip install -e .
```

```bash
pip install -e ".[torch]"       # transformer, attention, quantization modules
pip install -e ".[retrieval]"   # cross-encoder reranking, embeddings
pip install -e ".[demo]"        # Gradio + Streamlit demo apps
pip install -e ".[dev]"         # ruff, pytest, torch
```

---

## The Retrieval Stack

The modules build on each other. Each step in the progression is a separate file you can read independently.

| Stage | Module | What it does | When to use it |
|-------|--------|-------------|----------------|
| **Lexical baseline** | `wee.bm25` | Bag-of-words ranking (TF-IDF variant) | Always — it's your sanity-check baseline |
| **Dense retrieval** | `wee.vectorstore` + `wee.retriever` | Cosine search over embeddings, top-k / MMR / RRF strategies | When semantic similarity matters more than keyword overlap |
| **Hybrid (RRF)** | `wee.retriever` | Reciprocal Rank Fusion of dense + BM25 | Usually better than either alone, nearly free to add |
| **Cross-encoder rerank** | `wee.rerank` | Score (query, doc) pairs jointly | The single biggest quality lever — 2nd-stage over a candidate set |
| **Contextual retrieval** | `wee.contextual` | Prepend LLM-generated context blurb to chunks before embedding | Dramatic improvement when chunks lack self-contained context |
| **Query expansion** | `wee.rewrite` | Rewrite, decompose, or multi-rewrite queries before search | Complex or ambiguous queries; multi-hop questions |
| **Agentic retrieval** | `wee.agent` | ReAct loop — the model decides what to search and when to stop | Multi-hop questions where a single retrieve-then-generate pass isn't enough |
| **ColBERT (late interaction)** | `wee.colbert` | Per-token MaxSim scoring instead of single-vector | When token-level matching matters and you can afford the storage |

Every module accepts plain callables (`embed_fn`, `generate_fn`, `cross_encoder_fn`) — no LLM library lock-in.

---

## Foundations

| Module | Description |
|--------|-------------|
| `wee.tokenizer` | Char-BPE tokenizer with word-boundary markers, train/encode/decode |
| `wee.attention` | Multi-head attention and grouped-query attention (GQA) with `F.scaled_dot_product_attention` |
| `wee.transformer` | GPT-style decoder: RMSNorm, RoPE, SwiGLU, GQA, native KV cache, `.generate()` |
| `wee.chunk` | Word, sentence, token, and **semantic** chunking (split on embedding similarity breakpoints) |
| `wee.chunk` | **Parent-document** chunking — two-level hierarchy for retrieve-on-child, return-parent |

---

## Retrieval & Search

| Module | Description |
|--------|-------------|
| `wee.vectorstore` | In-memory dense index — add, search, delete, metadata filtering, save/load |
| `wee.retriever` | Retrieval strategies: top-k, MMR (diversity), RRF (hybrid fusion) |
| `wee.rerank` | Reranking modes: dense, lexical, hybrid, **cross-encoder**, cross-encoder+hybrid |
| `wee.contextual` | `ContextualChunker` — LLM-generated blurbs before embedding |
| `wee.rewrite` | `QueryExpander` — rewrite, decompose, multi-rewrite, search-with-expansion |
| `wee.agent` | `AgentLoop` with `SearchTool` — ReAct-style agentic retrieval |
| `wee.colbert` | `ColBERTIndex` — late-interaction retrieval with per-token MaxSim |
| `wee.parent_retriever` | `ParentRetriever` — retrieve on fine-grained children, return coarse parents |
| `wee.embed` | `Embedder` adapter with Matryoshka truncation + int8/binary quantization |
| `wee.hnsw` | Toy HNSW index — approximate nearest neighbors with recall-vs-latency tradeoff |
| `wee.context` | Budget-aware context packing into the LLM's window |

---

## Evaluation & Ops

| Module | Description |
|--------|-------------|
| `wee.eval` | EM, F1, faithfulness, **groundedness** (per-sentence detail), **citation support**, context precision/recall |
| `wee.judge` | Heuristic or LLM-as-judge scoring |
| `wee.cache` | SQLite-backed exact cache + **`SemanticCache`** (embedding similarity matching) |
| `wee.trace` | Hierarchical spans — JSON, HTML, and **OpenTelemetry GenAI** export with W3C trace context |
| `wee.guard` | PII detection, profanity, prompt-injection (regex baseline with [documented limits](wee/guard.py)), link allowlist |

---

## Performance & Scaling

| Module | Description |
|--------|-------------|
| `wee.quant` | Post-training int8/int4 quantization of linear layers with perplexity measurement |
| `wee.router` | Route queries to models by quality, cost, or latency |
| `wee.stream` | FastAPI SSE server for token streaming |
| `wee.graph` | Knowledge graph: extract triples, explore, export DOT/JSON |
| `wee.synth` | Synthetic QA generation (cloze + WH questions) for eval sets |

---

## Quick Example

```python
from wee import VectorStore, Retriever, BM25, Reranker, pack_context

# 1. Index
docs = ["RAG retrieves relevant context.", "BM25 is a lexical baseline.", "Cross-encoders score pairs jointly."]
embed_fn = lambda t: __import__('numpy').random.RandomState(hash(t) % 2**31).randn(64).astype('float32')

vs = VectorStore(dim=64)
vs.add_texts(docs, embed_fn, ids=["d0", "d1", "d2"])

bm25 = BM25()
bm25.add(docs)

# 2. Hybrid retrieve
ret = Retriever(method="rrf", k=3)
ret.attach_vectorstore(vs, embed_fn).attach_bm25(bm25)
hits = ret.search("How does lexical retrieval work?")

# 3. Rerank with cross-encoder
reranker = Reranker(mode="cross_encoder")
reranker.attach_cross_encoder(lambda q, d: len(set(q.lower().split()) & set(d.lower().split())) / len(set(q.lower().split()) | set(d.lower().split())))
texts = {id: doc for id, doc in zip(["d0", "d1", "d2"], docs)}
reranked = reranker.rerank("How does lexical retrieval work?", hits, texts, k=3)

# 4. Pack context
packed = pack_context([texts[id] for id, _, _ in reranked], max_tokens=100)
```

---

## Interactive Demo

[**Try it live on Streamlit Cloud**](https://wee-rag.streamlit.app)

Compare dense, hybrid, and reranked retrieval side by side with latency and pipeline traces.

```bash
pip install -e ".[demo]"
python demo/app.py              # Gradio (local)
streamlit run demo/streamlit_app.py  # Streamlit (local)
```

No model downloads or API keys needed — uses a built-in corpus with hash-based embeddings.

---

## Benchmarks

Retrieval quality on [HotpotQA](https://hotpotqa.github.io/) (500 questions, seed=42). Embedding: `all-MiniLM-L6-v2`. Cross-encoder: `ms-marco-MiniLM-L-6-v2`. LLM: Claude Haiku 4.5.

| | Stage | nDCG@10 | Recall@20 | MRR@10 | p50 (ms) | $/query |
|---|-------|---------|-----------|--------|----------|---------|
| **Baselines** | BM25 | 0.759 | 0.929 | 0.850 | 16.8 | — |
| | Dense (cosine) | 0.756 | 0.871 | 0.876 | 39.7 | — |
| | Hybrid (RRF) | 0.783 | 0.941 | 0.873 | 60.3 | — |
| **Reranking** | Cross-encoder rerank | **0.860** | **0.941** | **0.943** | 794.5 | — |
| | Query expansion + hybrid | 0.794 | 0.939 | 0.870 | 890.1 | $0.0001 |
| **LLM-augmented** | Contextual retrieval | 0.796 | 0.936 | 0.896 | 58.5 | $0.0042 |
| | Contextual + rerank | 0.859 | 0.936 | 0.942 | 777.4 | $0.0042 |

Cross-encoder reranking is the single biggest quality lever: **+10 nDCG** over hybrid for ~13x latency and zero LLM cost. Contextual retrieval improves MRR but only matches reranking when combined with it — at $0.0042/query in LLM calls. Query expansion adds +1 nDCG over vanilla hybrid: marginal for most use cases.

```bash
pip install -e ".[bench]"
make bench                  # reproduce with cached LLM outputs
make bench ARGS=--no-cache  # re-run LLM stages live (needs ANTHROPIC_API_KEY)
```

See [`bench/run.py`](bench/run.py) for methodology and [`bench/cache/`](bench/cache/) for cached outputs.

---

## Tests

```bash
pip install -e ".[dev]"
pytest -v
```

12 test files covering vectorstore, BM25, chunking, eval metrics, reranker, cache, ColBERT, embedder, HNSW, query rewriting, and guard (with injection TP/FP rate documentation).

---

## License

[MIT License](LICENSE)

## Contributing

Pull requests are welcome.
