"""ColBERT-style late-interaction retrieval with MaxSim scoring.

Standard dense retrieval compresses an entire document into one vector, losing
fine-grained token information.  ColBERT keeps *per-token* embeddings for both
queries and documents and scores them with **MaxSim**: for every query-token
vector, take its maximum cosine similarity with any document-token vector, then
sum those maxima.  This preserves token-level matching while still allowing
pre-computation of document embeddings.

Reference: Khattab & Zaharia, "ColBERT: Efficient and Effective Passage Search
via Contextualized Late Interaction over BERT" (SIGIR 2020).
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np


class ColBERTIndex:
    """Late-interaction retrieval with per-token embeddings and MaxSim scoring."""

    def __init__(self, dim: int):
        self.dim = dim
        # Each entry: (doc_id, token_embeddings) where token_embeddings is (T, dim)
        self._docs: List[Tuple[str, np.ndarray]] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add(self, doc_id: str, token_embeddings: np.ndarray) -> None:
        """Add a document's per-token embeddings to the index.

        Parameters
        ----------
        doc_id : str
            Unique identifier for the document.
        token_embeddings : np.ndarray
            Shape ``(num_tokens, dim)`` — one vector per token.
            Vectors are L2-normalised before storage so that dot products
            equal cosine similarities.
        """
        token_embeddings = np.asarray(token_embeddings, dtype=np.float32)
        if token_embeddings.ndim != 2 or token_embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Expected shape (num_tokens, {self.dim}), "
                f"got {token_embeddings.shape}"
            )
        # L2-normalise each token vector
        norms = np.linalg.norm(token_embeddings, axis=1, keepdims=True) + 1e-9
        token_embeddings = token_embeddings / norms
        self._docs.append((doc_id, token_embeddings))

    def add_texts(
        self,
        doc_ids: List[str],
        texts: List[str],
        encode_fn: Callable[[str], np.ndarray],
    ) -> None:
        """Encode texts with *encode_fn* and add them to the index.

        Parameters
        ----------
        encode_fn : Callable[[str], np.ndarray]
            Maps a string to a ``(num_tokens, dim)`` matrix of token embeddings.
        """
        for doc_id, text in zip(doc_ids, texts):
            self.add(doc_id, encode_fn(text))

    # ------------------------------------------------------------------
    # MaxSim scoring
    # ------------------------------------------------------------------

    @staticmethod
    def maxsim(query_embeddings: np.ndarray, doc_embeddings: np.ndarray) -> float:
        """Compute the MaxSim score between query and document token embeddings.

        For each query token, find its maximum cosine similarity with any
        document token, then return the sum of those maxima.

        Parameters
        ----------
        query_embeddings : np.ndarray
            Shape ``(Q, dim)`` — per-token query vectors (should be normalised).
        doc_embeddings : np.ndarray
            Shape ``(D, dim)`` — per-token document vectors (should be normalised).

        Returns
        -------
        float
            The MaxSim score (higher is better).
        """
        # (Q, dim) @ (dim, D) -> (Q, D): cosine similarities (pre-normalised)
        sim_matrix = query_embeddings @ doc_embeddings.T
        # For each query token, take the best-matching doc token, then sum
        return float(sim_matrix.max(axis=1).sum())

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query_embeddings: np.ndarray, k: int = 5
    ) -> List[Tuple[str, float]]:
        """Score the query against every indexed document and return the top *k*.

        Parameters
        ----------
        query_embeddings : np.ndarray
            Shape ``(Q, dim)`` — per-token query vectors.
        k : int
            Number of results to return.

        Returns
        -------
        List[Tuple[str, float]]
            ``(doc_id, score)`` pairs sorted by descending MaxSim score.
        """
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
        # L2-normalise query tokens
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-9
        query_embeddings = query_embeddings / norms

        scored: List[Tuple[str, float]] = []
        for doc_id, doc_emb in self._docs:
            score = self.maxsim(query_embeddings, doc_emb)
            scored.append((doc_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def search_text(
        self,
        query: str,
        encode_fn: Callable[[str], np.ndarray],
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Convenience wrapper: encode *query* then search.

        Parameters
        ----------
        encode_fn : Callable[[str], np.ndarray]
            Same token-level encoder used at index time.
        """
        return self.search(encode_fn(query), k=k)
