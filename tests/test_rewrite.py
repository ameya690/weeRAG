"""Tests for wee.rewrite (rewrite_query, decompose_query, QueryExpander)."""
from wee.rewrite import rewrite_query, decompose_query, QueryExpander


# ------------------------------------------------------------------
# Dummy LLM functions
# ------------------------------------------------------------------


def _dummy_rewrite_fn(prompt: str) -> str:
    """Dummy rewrite: uppercase the original query extracted from the prompt."""
    # The prompt contains "Query: <original>\n\nRewritten query:"
    # We just return a transformed version
    if "Query:" in prompt:
        query = prompt.split("Query:")[-1].split("\n")[0].strip()
        return f"improved {query}"
    return prompt.upper()


def _dummy_decompose_fn(prompt: str) -> str:
    """Dummy decompose: return a numbered list of sub-questions."""
    return (
        "1. What is the first aspect?\n"
        "2. What is the second aspect?\n"
        "3. What is the third aspect?"
    )


# ------------------------------------------------------------------
# test_rewrite_query
# ------------------------------------------------------------------


def test_rewrite_query():
    """rewrite_query passes the prompt to the rewrite_fn and returns result."""
    result = rewrite_query("machine learning basics", _dummy_rewrite_fn)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "machine learning basics" in result.lower()


# ------------------------------------------------------------------
# test_decompose_query
# ------------------------------------------------------------------


def test_decompose_query():
    """decompose_query parses a numbered list from the decompose_fn."""
    result = decompose_query("complex topic question", _dummy_decompose_fn)
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(q, str) for q in result)
    assert "first aspect" in result[0].lower()


# ------------------------------------------------------------------
# test_query_expander_modes
# ------------------------------------------------------------------


def test_query_expander_rewrite_mode():
    """QueryExpander mode='rewrite' returns a single rewritten query."""
    qe = QueryExpander(rewrite_fn=_dummy_rewrite_fn)
    variants = qe.expand("what is RAG", mode="rewrite")
    assert isinstance(variants, list)
    assert len(variants) == 1
    assert "what is rag" in variants[0].lower()


def test_query_expander_decompose_mode():
    """QueryExpander mode='decompose' returns sub-questions."""
    qe = QueryExpander(decompose_fn=_dummy_decompose_fn)
    variants = qe.expand("what is RAG", mode="decompose")
    assert isinstance(variants, list)
    assert len(variants) == 3


def test_query_expander_both_mode():
    """QueryExpander mode='both' rewrites first, then decomposes."""
    qe = QueryExpander(rewrite_fn=_dummy_rewrite_fn, decompose_fn=_dummy_decompose_fn)
    variants = qe.expand("what is RAG", mode="both")
    assert isinstance(variants, list)
    # decompose always returns 3 items in our dummy
    assert len(variants) == 3


def test_query_expander_multi_rewrite_mode():
    """QueryExpander mode='multi_rewrite' returns multiple style variants."""
    qe = QueryExpander(rewrite_fn=_dummy_rewrite_fn)
    variants = qe.expand("what is RAG", mode="multi_rewrite")
    assert isinstance(variants, list)
    # There are 3 style prompts in _MULTI_REWRITE_STYLES
    assert len(variants) == 3
    # Each should be a non-empty string
    assert all(len(v) > 0 for v in variants)
