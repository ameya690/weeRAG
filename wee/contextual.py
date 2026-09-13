from __future__ import annotations

from collections.abc import Callable

from .chunk import chunk_by_sentences


class ContextualChunker:
    """Contextual retrieval: prepend an LLM-generated blurb to each chunk
    so the embedding captures where the chunk fits within the full document.

    Parameters
    ----------
    generate_fn : callable (document, chunk) -> str
        Returns a short (1-2 sentence) contextual blurb situating *chunk*
        inside *document*.  Model-agnostic — wrap any LLM call you like.
    chunk_fn : callable (text) -> List[str], optional
        Splits a document into chunks.  Defaults to ``chunk_by_sentences``.
    """

    def __init__(
        self,
        generate_fn: Callable[[str, str], str],
        chunk_fn: Callable[[str], list[str]] | None = None,
    ) -> None:
        self.generate_fn = generate_fn
        self.chunk_fn = chunk_fn or chunk_by_sentences

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def contextualize(self, document: str) -> list[dict[str, str]]:
        """Chunk *document* and generate a contextual blurb for each chunk.

        Returns a list of dicts, one per chunk::

            {
                "original":        <raw chunk text>,
                "context":         <LLM-generated blurb>,
                "contextualized":  "<blurb>\\n\\n<chunk text>",
            }
        """
        chunks = self.chunk_fn(document)
        results: list[dict[str, str]] = []
        for chunk in chunks:
            blurb = self.generate_fn(document, chunk)
            results.append(
                {
                    "original": chunk,
                    "context": blurb,
                    "contextualized": f"{blurb}\n\n{chunk}",
                }
            )
        return results

    def contextualize_batch(
        self, documents: list[str]
    ) -> list[list[dict[str, str]]]:
        """Apply :meth:`contextualize` to every document in *documents*.

        Returns a nested list — one inner list per document.
        """
        return [self.contextualize(doc) for doc in documents]
