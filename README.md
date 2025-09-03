# 🪶 weeRAG — tiny, teachable LLM & RAG foundations

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
