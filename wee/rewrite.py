from __future__ import annotations
from typing import List, Callable, Tuple, Any
import re


_REWRITE_PROMPT = (
    "Rewrite the following user query to be more specific and search-friendly. "
    "Return only the rewritten query, nothing else.\n\n"
    "Query: {query}\n\nRewritten query:"
)

_DECOMPOSE_PROMPT = (
    "Break the following complex question into 2-4 simpler, self-contained "
    "sub-questions that together cover the original intent. "
    "Return a numbered list (1. ... 2. ...) and nothing else.\n\n"
    "Question: {query}\n\nSub-questions:"
)

_MULTI_REWRITE_STYLES = [
    "Rewrite the following query for keyword search. "
    "Return only the rewritten query.\n\nQuery: {query}\n\nRewritten query:",
    "Rewrite the following query as a question. "
    "Return only the rewritten query.\n\nQuery: {query}\n\nRewritten query:",
    "Rewrite the following query using synonyms. "
    "Return only the rewritten query.\n\nQuery: {query}\n\nRewritten query:",
]

_NUMBERED_ITEM = re.compile(r"^\s*\d+[.)]\s*(.+)", re.MULTILINE)


def rewrite_query(query: str, rewrite_fn: Callable[[str], str]) -> str:
    """Rewrite *query* via an LLM callable to improve retrieval quality."""
    prompt = _REWRITE_PROMPT.format(query=query)
    return rewrite_fn(prompt).strip()


def decompose_query(query: str, decompose_fn: Callable[[str], str]) -> List[str]:
    """Break *query* into 2-4 simpler sub-questions via an LLM callable."""
    prompt = _DECOMPOSE_PROMPT.format(query=query)
    raw = decompose_fn(prompt)
    items = _NUMBERED_ITEM.findall(raw)
    return [item.strip() for item in items if item.strip()]


class QueryExpander:
    """
    Expand a user query before retrieval.

    Supports four modes:
      rewrite       - single LLM rewrite
      decompose     - split into sub-questions
      both          - rewrite then decompose
      multi_rewrite - several style-varied rewrites
    """

    MODES = ("rewrite", "decompose", "both", "multi_rewrite")

    def __init__(
        self,
        rewrite_fn: Callable[[str], str] | None = None,
        decompose_fn: Callable[[str], str] | None = None,
    ):
        self.rewrite_fn = rewrite_fn
        self.decompose_fn = decompose_fn

    # ------------------------------------------------------------------
    def expand(self, query: str, mode: str = "rewrite") -> List[str]:
        """Return a list of query variants according to *mode*."""
        assert mode in self.MODES, f"mode must be one of {self.MODES}"

        if mode == "rewrite":
            assert self.rewrite_fn is not None, "rewrite_fn required for mode='rewrite'"
            return [rewrite_query(query, self.rewrite_fn)]

        if mode == "decompose":
            assert self.decompose_fn is not None, "decompose_fn required for mode='decompose'"
            return decompose_query(query, self.decompose_fn)

        if mode == "both":
            assert self.rewrite_fn is not None, "rewrite_fn required for mode='both'"
            assert self.decompose_fn is not None, "decompose_fn required for mode='both'"
            rewritten = rewrite_query(query, self.rewrite_fn)
            return decompose_query(rewritten, self.decompose_fn)

        if mode == "multi_rewrite":
            assert self.rewrite_fn is not None, "rewrite_fn required for mode='multi_rewrite'"
            variants: List[str] = []
            for style_prompt in _MULTI_REWRITE_STYLES:
                result = self.rewrite_fn(style_prompt.format(query=query)).strip()
                if result:
                    variants.append(result)
            return variants

        raise ValueError(f"Unknown mode: {mode}")

    # ------------------------------------------------------------------
    def search_with_expansion(
        self,
        query: str,
        search_fn: Callable[[str], List[Tuple[Any, ...]]],
        mode: str = "rewrite",
        deduplicate: bool = True,
    ) -> List[Tuple[Any, ...]]:
        """Expand *query*, search each variant, merge results.

        *search_fn* should accept a query string and return a list of tuples
        whose first element is a document ID (used for deduplication).
        """
        variants = self.expand(query, mode=mode)
        all_results: List[Tuple[Any, ...]] = []
        seen_ids: set = set()

        for variant in variants:
            hits = search_fn(variant)
            for hit in hits:
                doc_id = hit[0]
                if deduplicate and doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)
                all_results.append(hit)

        return all_results
