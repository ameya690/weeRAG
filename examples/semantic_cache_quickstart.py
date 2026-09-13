"""Semantic cache quickstart — shows exact hits, semantic hits, and misses."""

import numpy as np

from wee.cache import SemanticCache

# ---------------------------------------------------------------------------
# Dummy embedding function
# Maps a few known phrases to hand-crafted vectors so we can demonstrate
# exact match, semantic match (similar wording), and miss behaviour
# without pulling in a real embedding model.
# ---------------------------------------------------------------------------

_PHRASE_VECTORS = {
    "What is the capital of France?": np.array([1.0, 0.0, 0.0, 0.0]),
    "Tell me France's capital city": np.array([0.96, 0.1, 0.0, 0.0]),  # similar
    "How tall is the Eiffel Tower?": np.array([0.0, 1.0, 0.0, 0.0]),  # different topic
    "What is the capital of Germany?": np.array([0.5, 0.0, 0.5, 0.0]),  # related but different
}

# Fallback: hash the string into a stable random vector
_rng = np.random.RandomState(42)
_fallback_cache: dict[str, np.ndarray] = {}


def dummy_embed(text: str) -> np.ndarray:
    if text in _PHRASE_VECTORS:
        return _PHRASE_VECTORS[text]
    if text not in _fallback_cache:
        _fallback_cache[text] = _rng.randn(4)
        _fallback_cache[text] /= np.linalg.norm(_fallback_cache[text])
    return _fallback_cache[text]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    cache = SemanticCache(embed_fn=dummy_embed, threshold=0.85, max_entries=100)

    # Populate the cache with one entry
    query = "What is the capital of France?"
    answer = "Paris"
    cache.set(query, answer)
    print(f"Cached: {query!r} -> {answer!r}\n")

    # 1. Exact match hit — same string
    result = cache.get("What is the capital of France?")
    print(f"Exact match query : 'What is the capital of France?'")
    print(f"  Result           : {result!r}  (expected 'Paris')\n")

    # 2. Semantic match hit — different wording, same meaning
    result = cache.get("Tell me France's capital city")
    print(f"Semantic match query: 'Tell me France's capital city'")
    print(f"  Result             : {result!r}  (expected 'Paris')\n")

    # 3. Miss — different topic entirely
    result = cache.get("How tall is the Eiffel Tower?")
    print(f"Miss query          : 'How tall is the Eiffel Tower?'")
    print(f"  Result             : {result!r}  (expected None)\n")

    # 4. Miss — related but not similar enough
    result = cache.get("What is the capital of Germany?")
    print(f"Partial overlap     : 'What is the capital of Germany?'")
    print(f"  Result             : {result!r}  (expected None)\n")

    # Stats
    print("Cache stats:", cache.stats())


if __name__ == "__main__":
    main()
