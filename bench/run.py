#!/usr/bin/env python3
"""
weeRAG Retrieval Benchmark
==========================
Evaluate 8 retrieval pipeline configurations on HotpotQA.

Usage:
    python bench/run.py                          # uses cached LLM outputs
    python bench/run.py --no-cache               # live LLM calls (needs ANTHROPIC_API_KEY)
    python bench/run.py --subset 100 --seed 0    # smaller run
    python bench/run.py --stages bm25 dense      # run specific stages only

Requirements:
    pip install -e ".[bench]"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).parent / "cache"
RESULTS_FILE = Path(__file__).parent / "results.json"

# Haiku 4.5 pricing (USD per 1M tokens)
HAIKU_INPUT_PRICE = 0.80
HAIKU_OUTPUT_PRICE = 4.00

# ── Metrics ──────────────────────────────────────────────────────────────────

def ndcg_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    retrieved = retrieved_ids[:k]
    dcg = sum(
        1.0 / np.log2(i + 2) for i, rid in enumerate(retrieved) if rid in gold_ids
    )
    ideal = sorted([1] * min(len(gold_ids), k), reverse=True)
    idcg = sum(1.0 / np.log2(i + 2) for i, _ in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 1.0
    retrieved = set(retrieved_ids[:k])
    return len(retrieved & gold_ids) / len(gold_ids)


def mrr_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in gold_ids:
            return 1.0 / (i + 1)
    return 0.0


# ── LLM Cache ────────────────────────────────────────────────────────────────

def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class LLMCache:
    def __init__(self, cache_dir: Path, stage: str, enabled: bool = True):
        self.path = cache_dir / f"{stage}.json"
        self.enabled = enabled
        self.data: dict[str, str] = {}
        self.hits = 0
        self.misses = 0
        if enabled and self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self.data = json.load(f)

    def get(self, key: str) -> str | None:
        h = _hash_key(key)
        if h in self.data:
            self.hits += 1
            return self.data[h]
        self.misses += 1
        return None

    def set(self, key: str, value: str) -> None:
        h = _hash_key(key)
        self.data[h] = value

    def save(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)


# ── Token / cost tracking ────────────────────────────────────────────────────

class CostTracker:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.embedding_calls = 0

    def add_llm(self, input_toks: int, output_toks: int):
        self.input_tokens += input_toks
        self.output_tokens += output_toks

    def cost_usd(self) -> float:
        return (
            self.input_tokens * HAIKU_INPUT_PRICE / 1_000_000
            + self.output_tokens * HAIKU_OUTPUT_PRICE / 1_000_000
        )

    def cost_per_query(self, n_queries: int) -> float:
        return self.cost_usd() / max(n_queries, 1)


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_hotpotqa(subset: int, seed: int) -> tuple[dict[str, str], list[dict]]:
    """Load HotpotQA distractor setting, return (corpus, queries).

    corpus: {passage_id: text}
    queries: [{question, gold_ids: set[str], answer}]
    """
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Install datasets: pip install -e '.[bench]'")

    ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    ds = ds.shuffle(seed=seed).select(range(min(subset, len(ds))))

    corpus: dict[str, str] = {}
    queries: list[dict] = []

    for ex in ds:
        titles = ex["context"]["title"]
        sentences_lists = ex["context"]["sentences"]
        gold_titles = set(ex["supporting_facts"]["title"])

        gold_ids: set[str] = set()
        for title, sents in zip(titles, sentences_lists):
            pid = _passage_id(title)
            text = title + ". " + " ".join(sents)
            corpus[pid] = text
            if title in gold_titles:
                gold_ids.add(pid)

        queries.append({
            "question": ex["question"],
            "answer": ex["answer"],
            "gold_ids": gold_ids,
        })

    return corpus, queries


def _passage_id(title: str) -> str:
    return title.strip().lower().replace(" ", "_")[:80]


# ── Embedding ─────────────────────────────────────────────────────────────────

def make_embed_fn() -> tuple:
    """Return (embed_fn, dim) using all-MiniLM-L6-v2."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("Install sentence-transformers: pip install -e '.[bench]'")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    dim = model.get_sentence_embedding_dimension()

    def embed(text: str) -> np.ndarray:
        return model.encode(text, normalize_embeddings=True).astype(np.float32)

    return embed, dim


def make_cross_encoder_fn():
    """Return a cross-encoder scoring function using ms-marco-MiniLM-L-6-v2."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        sys.exit("Install sentence-transformers: pip install -e '.[bench]'")

    ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def score(query: str, doc: str) -> float:
        return float(ce_model.predict([(query, doc)])[0])

    return score


# ── LLM ───────────────────────────────────────────────────────────────────────

def make_llm_fn(cache: LLMCache, cost: CostTracker):
    """Return a generate_fn that calls Claude Haiku with caching."""
    client = None

    def generate(prompt: str) -> str:
        nonlocal client

        cached = cache.get(prompt)
        if cached is not None:
            return cached

        if client is None:
            try:
                import anthropic
            except ImportError:
                sys.exit(
                    "Live LLM calls need the Anthropic SDK: pip install anthropic\n"
                    "Or run with cached outputs (the default)."
                )
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                sys.exit(
                    "Set ANTHROPIC_API_KEY for live LLM calls, "
                    "or run with cached outputs (the default)."
                )
            client = anthropic.Anthropic(api_key=api_key)

        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text
        cost.add_llm(resp.usage.input_tokens, resp.usage.output_tokens)
        cache.set(prompt, text)
        return text

    return generate


# ── Pipeline stages ───────────────────────────────────────────────────────────

def _build_index(corpus, embed_fn, dim):
    """Build VectorStore + BM25 index over corpus."""
    from wee import BM25, VectorStore

    ids = list(corpus.keys())
    texts = [corpus[pid] for pid in ids]

    vs = VectorStore(dim=dim)
    vs.add_texts(texts, embed_fn, ids=ids)

    bm25 = BM25()
    bm25.add(texts)

    return vs, bm25, ids, texts


def _extract_ids(results: list[tuple]) -> list[str]:
    return [r[0] for r in results]


def run_stage(
    stage_name: str,
    corpus: dict[str, str],
    queries: list[dict],
    embed_fn,
    dim: int,
    vs,
    bm25,
    texts_by_id: dict[str, str],
    cross_encoder_fn=None,
    llm_fn=None,
    llm_cache=None,
    cost_tracker=None,
) -> dict:
    """Run a single pipeline stage and return metrics."""
    from wee import AgentLoop, QueryExpander, Reranker, Retriever, SearchTool

    K_RETRIEVE = 20
    latencies: list[float] = []
    ndcgs: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []

    if stage_name == "bm25":
        for q in queries:
            t0 = time.perf_counter()
            hits = bm25.search(q["question"], k=K_RETRIEVE)
            ids_list = list(corpus.keys())
            retrieved = [ids_list[doc_idx] for doc_idx, _ in hits]
            latencies.append(time.perf_counter() - t0)
            ndcgs.append(ndcg_at_k(retrieved, q["gold_ids"], 10))
            recalls.append(recall_at_k(retrieved, q["gold_ids"], 20))
            mrrs.append(mrr_at_k(retrieved, q["gold_ids"], 10))

    elif stage_name == "dense":
        ret = Retriever(method="topk", k=K_RETRIEVE)
        ret.attach_vectorstore(vs, embed_fn)
        for q in queries:
            t0 = time.perf_counter()
            hits = ret.search(q["question"])
            retrieved = _extract_ids(hits)
            latencies.append(time.perf_counter() - t0)
            ndcgs.append(ndcg_at_k(retrieved, q["gold_ids"], 10))
            recalls.append(recall_at_k(retrieved, q["gold_ids"], 20))
            mrrs.append(mrr_at_k(retrieved, q["gold_ids"], 10))

    elif stage_name == "hybrid":
        ret = Retriever(method="rrf", k=K_RETRIEVE)
        ret.attach_vectorstore(vs, embed_fn).attach_bm25(bm25)
        for q in queries:
            t0 = time.perf_counter()
            hits = ret.search(q["question"])
            retrieved = _extract_ids(hits)
            latencies.append(time.perf_counter() - t0)
            ndcgs.append(ndcg_at_k(retrieved, q["gold_ids"], 10))
            recalls.append(recall_at_k(retrieved, q["gold_ids"], 20))
            mrrs.append(mrr_at_k(retrieved, q["gold_ids"], 10))

    elif stage_name == "rerank":
        ret = Retriever(method="rrf", k=K_RETRIEVE)
        ret.attach_vectorstore(vs, embed_fn).attach_bm25(bm25)
        reranker = Reranker(mode="cross_encoder")
        reranker.attach_cross_encoder(cross_encoder_fn)
        for q in queries:
            t0 = time.perf_counter()
            candidates = ret.search(q["question"])
            reranked = reranker.rerank(q["question"], candidates, texts_by_id, k=K_RETRIEVE)
            retrieved = _extract_ids(reranked)
            latencies.append(time.perf_counter() - t0)
            ndcgs.append(ndcg_at_k(retrieved, q["gold_ids"], 10))
            recalls.append(recall_at_k(retrieved, q["gold_ids"], 20))
            mrrs.append(mrr_at_k(retrieved, q["gold_ids"], 10))

    elif stage_name == "query_expansion":
        ret = Retriever(method="rrf", k=K_RETRIEVE)
        ret.attach_vectorstore(vs, embed_fn).attach_bm25(bm25)
        expander = QueryExpander(rewrite_fn=llm_fn)
        for q in queries:
            t0 = time.perf_counter()
            hits = expander.search_with_expansion(
                q["question"], ret.search, mode="rewrite"
            )[:K_RETRIEVE]
            retrieved = _extract_ids(hits)
            latencies.append(time.perf_counter() - t0)
            ndcgs.append(ndcg_at_k(retrieved, q["gold_ids"], 10))
            recalls.append(recall_at_k(retrieved, q["gold_ids"], 20))
            mrrs.append(mrr_at_k(retrieved, q["gold_ids"], 10))

    elif stage_name == "contextual":
        ctx_vs, ctx_bm25 = _build_contextual_index(
            corpus, embed_fn, dim, llm_fn, llm_cache, cost_tracker
        )
        ret = Retriever(method="rrf", k=K_RETRIEVE)
        ret.attach_vectorstore(ctx_vs, embed_fn).attach_bm25(ctx_bm25)
        for q in queries:
            t0 = time.perf_counter()
            hits = ret.search(q["question"])
            retrieved = _extract_ids(hits)
            latencies.append(time.perf_counter() - t0)
            ndcgs.append(ndcg_at_k(retrieved, q["gold_ids"], 10))
            recalls.append(recall_at_k(retrieved, q["gold_ids"], 20))
            mrrs.append(mrr_at_k(retrieved, q["gold_ids"], 10))

    elif stage_name == "contextual_rerank":
        ctx_vs, ctx_bm25 = _build_contextual_index(
            corpus, embed_fn, dim, llm_fn, llm_cache, cost_tracker
        )
        ret = Retriever(method="rrf", k=K_RETRIEVE)
        ret.attach_vectorstore(ctx_vs, embed_fn).attach_bm25(ctx_bm25)
        reranker = Reranker(mode="cross_encoder")
        reranker.attach_cross_encoder(cross_encoder_fn)
        for q in queries:
            t0 = time.perf_counter()
            candidates = ret.search(q["question"])
            reranked = reranker.rerank(q["question"], candidates, texts_by_id, k=K_RETRIEVE)
            retrieved = _extract_ids(reranked)
            latencies.append(time.perf_counter() - t0)
            ndcgs.append(ndcg_at_k(retrieved, q["gold_ids"], 10))
            recalls.append(recall_at_k(retrieved, q["gold_ids"], 20))
            mrrs.append(mrr_at_k(retrieved, q["gold_ids"], 10))

    elif stage_name == "agentic":
        ret = Retriever(method="rrf", k=10)
        ret.attach_vectorstore(vs, embed_fn).attach_bm25(bm25)
        search_tool = SearchTool(ret, texts_by_id)
        agent = AgentLoop(generate_fn=llm_fn, tools=[search_tool], max_steps=3)
        for q in queries:
            t0 = time.perf_counter()
            result = agent.run(q["question"])
            all_retrieved: list[str] = []
            for step in result.get("steps", []):
                obs = step.get("observation", "")
                for pid in corpus:
                    if pid in all_retrieved:
                        continue
                    if corpus[pid][:60] in obs:
                        all_retrieved.append(pid)
            latencies.append(time.perf_counter() - t0)
            ndcgs.append(ndcg_at_k(all_retrieved, q["gold_ids"], 10))
            recalls.append(recall_at_k(all_retrieved, q["gold_ids"], 20))
            mrrs.append(mrr_at_k(all_retrieved, q["gold_ids"], 10))

    else:
        raise ValueError(f"Unknown stage: {stage_name}")

    n = len(queries)
    sorted_lat = sorted(latencies)
    return {
        "stage": stage_name,
        "ndcg_10": np.mean(ndcgs),
        "recall_20": np.mean(recalls),
        "mrr_10": np.mean(mrrs),
        "p50_ms": sorted_lat[n // 2] * 1000 if sorted_lat else 0,
        "p95_ms": sorted_lat[int(n * 0.95)] * 1000 if sorted_lat else 0,
        "n_queries": n,
    }


_CONTEXTUAL_CACHE: dict[str, tuple] = {}


def _build_contextual_index(corpus, embed_fn, dim, llm_fn, llm_cache, cost_tracker):
    """Build a contextual index (cached across stages that share it)."""
    cache_key = "contextual_index"
    if cache_key in _CONTEXTUAL_CACHE:
        return _CONTEXTUAL_CACHE[cache_key]

    from wee import VectorStore
    from wee.bm25 import BM25 as WeeBM25

    def context_fn(document: str, chunk: str) -> str:
        prompt = (
            f"Situate this chunk within the document in 1-2 sentences.\n\n"
            f"Document: {document[:500]}\n\nChunk: {chunk[:300]}\n\nContext:"
        )
        return llm_fn(prompt)

    ctx_vs = VectorStore(dim=dim)
    ctx_bm25 = WeeBM25()
    ctx_texts: list[str] = []
    ctx_ids: list[str] = []

    for pid, text in corpus.items():
        blurb = context_fn(text, text)
        contextualized = f"{blurb}\n\n{text}"
        ctx_texts.append(contextualized)
        ctx_ids.append(pid)

    ctx_vs.add_texts(ctx_texts, embed_fn, ids=ctx_ids)
    ctx_bm25.add(ctx_texts)

    _CONTEXTUAL_CACHE[cache_key] = (ctx_vs, ctx_bm25)
    return ctx_vs, ctx_bm25


# ── Output ────────────────────────────────────────────────────────────────────

STAGE_LABELS = {
    "bm25": "BM25",
    "dense": "Dense",
    "hybrid": "Hybrid (RRF)",
    "rerank": "Cross-encoder rerank",
    "query_expansion": "Query expansion + hybrid",
    "contextual": "Contextual retrieval",
    "contextual_rerank": "Contextual + rerank",
    "agentic": "Agentic (multi-hop)",
}

TIERS = {
    "Baselines (no LLM)": ["bm25", "dense", "hybrid"],
    "Reranking": ["rerank", "query_expansion"],
    "LLM-augmented": ["contextual", "contextual_rerank", "agentic"],
}

LLM_STAGES = {"query_expansion", "contextual", "contextual_rerank", "agentic"}


def format_table(results: list[dict], costs: dict[str, float]) -> str:
    lines = [
        "| Stage | nDCG@10 | Recall@20 | MRR@10 | p50 (ms) | p95 (ms) | $/query |",
        "|-------|---------|-----------|--------|----------|----------|---------|",
    ]

    results_by_stage = {r["stage"]: r for r in results}

    for tier_name, stages in TIERS.items():
        tier_results = [results_by_stage[s] for s in stages if s in results_by_stage]
        if not tier_results:
            continue
        lines.append(f"| **{tier_name}** | | | | | | |")
        for r in tier_results:
            cost_str = f"${costs.get(r['stage'], 0):.4f}" if r["stage"] in LLM_STAGES else "—"
            lines.append(
                f"| {STAGE_LABELS[r['stage']]} "
                f"| {r['ndcg_10']:.3f} "
                f"| {r['recall_20']:.3f} "
                f"| {r['mrr_10']:.3f} "
                f"| {r['p50_ms']:.1f} "
                f"| {r['p95_ms']:.1f} "
                f"| {cost_str} |"
            )

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_STAGES = ["bm25", "dense", "hybrid", "rerank", "query_expansion",
              "contextual", "contextual_rerank", "agentic"]


def main():
    parser = argparse.ArgumentParser(description="weeRAG retrieval benchmark")
    parser.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa"])
    parser.add_argument("--subset", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stages", nargs="+", default=ALL_STAGES, choices=ALL_STAGES)
    parser.add_argument("--no-cache", action="store_true", help="disable LLM output cache")
    parser.add_argument("--k", type=int, default=20, help="retrieval depth")
    args = parser.parse_args()

    use_cache = not args.no_cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} (subset={args.subset}, seed={args.seed})...")
    corpus, queries = load_hotpotqa(args.subset, args.seed)
    print(f"  {len(corpus)} passages, {len(queries)} queries")

    needs_embeddings = any(s != "bm25" for s in args.stages)
    embed_fn, dim, vs, bm25_index = None, None, None, None
    texts_by_id: dict[str, str] = corpus

    if needs_embeddings:
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        embed_fn, dim = make_embed_fn()
        print("Building index...")
        vs, bm25_index, _, _ = _build_index(corpus, embed_fn, dim)
    else:
        from wee import BM25 as WeeBM25
        bm25_index = WeeBM25()
        bm25_index.add(list(corpus.values()))

    needs_ce = any(s in {"rerank", "contextual_rerank"} for s in args.stages)
    cross_encoder_fn = None
    if needs_ce:
        print("Loading cross-encoder (ms-marco-MiniLM-L-6-v2)...")
        cross_encoder_fn = make_cross_encoder_fn()

    needs_llm = any(s in LLM_STAGES for s in args.stages)
    llm_fn = None
    cost_tracker = CostTracker()
    llm_caches: dict[str, LLMCache] = {}
    if needs_llm:
        llm_cache = LLMCache(CACHE_DIR, "llm_responses", enabled=use_cache)
        llm_fn = make_llm_fn(llm_cache, cost_tracker)
        llm_caches["llm_responses"] = llm_cache

    results: list[dict] = []
    for stage in args.stages:
        print(f"Running {STAGE_LABELS.get(stage, stage)}...")
        cost_before = cost_tracker.cost_usd()

        r = run_stage(
            stage, corpus, queries, embed_fn, dim, vs, bm25_index,
            texts_by_id, cross_encoder_fn, llm_fn,
            llm_caches.get("llm_responses"), cost_tracker,
        )
        r["cost_usd"] = cost_tracker.cost_usd() - cost_before
        results.append(r)

        print(
            f"  nDCG@10={r['ndcg_10']:.3f}  "
            f"Recall@20={r['recall_20']:.3f}  "
            f"MRR@10={r['mrr_10']:.3f}  "
            f"p50={r['p50_ms']:.1f}ms"
        )

    for cache in llm_caches.values():
        cache.save()

    costs = {r["stage"]: r.get("cost_usd", 0) / max(r["n_queries"], 1) for r in results}

    print("\n" + "=" * 70)
    print(format_table(results, costs))
    print("=" * 70)

    all_results = {
        "dataset": args.dataset,
        "subset": args.subset,
        "seed": args.seed,
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "claude-haiku-4-5",
        "stages": results,
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Regression check: output exit code 1 if thresholds file exists and any metric regressed
    thresholds_path = Path(__file__).parent / "thresholds.json"
    if thresholds_path.exists():
        with open(thresholds_path, encoding="utf-8") as f:
            thresholds = json.load(f)
        failed = False
        for r in results:
            stage_thresholds = thresholds.get(r["stage"], {})
            for metric, min_val in stage_thresholds.items():
                actual = r.get(metric, 0)
                if actual < min_val:
                    print(f"REGRESSION: {r['stage']}.{metric} = {actual:.3f} < {min_val:.3f}")
                    failed = True
        if failed:
            sys.exit(1)
        print("All regression thresholds passed.")


if __name__ == "__main__":
    main()
