from wee.rewrite import rewrite_query, decompose_query, QueryExpander


# ── Dummy LLM functions (simulate model output) ──────────────────

def dummy_rewrite(prompt: str) -> str:
    """Pretend LLM that prefixes 'definition of' to the query."""
    # Extract the original query from the prompt
    if "Query:" in prompt:
        q = prompt.split("Query:")[-1].split("\n")[0].strip()
    else:
        q = prompt.strip()
    return f"definition of {q}"


def dummy_decompose(prompt: str) -> str:
    """Pretend LLM that returns canned sub-questions."""
    return (
        "1. What is retrieval-augmented generation?\n"
        "2. How does vector search work in RAG pipelines?\n"
        "3. What reranking strategies improve RAG accuracy?"
    )


# ── Standalone functions ──────────────────────────────────────────

def demo_standalone():
    print("=== rewrite_query ===")
    rewritten = rewrite_query("RAG accuracy", dummy_rewrite)
    print(f"  original : RAG accuracy")
    print(f"  rewritten: {rewritten}")

    print("\n=== decompose_query ===")
    subs = decompose_query("How does RAG improve LLM accuracy?", dummy_decompose)
    for i, s in enumerate(subs, 1):
        print(f"  {i}. {s}")


# ── QueryExpander class ──────────────────────────────────────────

def demo_expander():
    exp = QueryExpander(rewrite_fn=dummy_rewrite, decompose_fn=dummy_decompose)

    print("\n=== expand mode='rewrite' ===")
    for v in exp.expand("RAG accuracy", mode="rewrite"):
        print(f"  {v}")

    print("\n=== expand mode='decompose' ===")
    for v in exp.expand("How does RAG work?", mode="decompose"):
        print(f"  {v}")

    print("\n=== expand mode='both' ===")
    for v in exp.expand("RAG accuracy", mode="both"):
        print(f"  {v}")

    print("\n=== expand mode='multi_rewrite' ===")
    for v in exp.expand("RAG accuracy", mode="multi_rewrite"):
        print(f"  {v}")


# ── search_with_expansion ────────────────────────────────────────

def demo_search():
    # Fake document store
    docs = {
        "doc1": "Retrieval-augmented generation combines search with LLMs.",
        "doc2": "Vector databases store embeddings for similarity search.",
        "doc3": "BM25 is a classic lexical retrieval algorithm.",
        "doc4": "Reranking improves precision after initial retrieval.",
    }

    def fake_search(query: str):
        """Return docs whose text contains any word from the query."""
        words = set(query.lower().split())
        hits = []
        for doc_id, text in docs.items():
            if words & set(text.lower().split()):
                hits.append((doc_id, 1.0, text[:50]))
        return hits

    exp = QueryExpander(rewrite_fn=dummy_rewrite, decompose_fn=dummy_decompose)

    print("\n=== search_with_expansion (mode='decompose', deduplicate=True) ===")
    results = exp.search_with_expansion("How does RAG work?", fake_search, mode="decompose")
    for doc_id, score, snippet in results:
        print(f"  {doc_id} ({score:.1f}): {snippet}...")


if __name__ == "__main__":
    demo_standalone()
    demo_expander()
    demo_search()
