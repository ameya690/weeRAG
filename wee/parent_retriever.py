"""Parent-document retriever: retrieve on child chunks, return parent documents."""

from __future__ import annotations
from typing import List, Dict, Optional, Callable
import numpy as np

from .vectorstore import VectorStore
from .chunk import chunk_with_parents


class ParentRetriever:
    """Retrieve on child chunks, return parent documents.

    The idea: child chunks are small and specific, so they match queries
    well.  But the LLM needs *surrounding context* to generate a good
    answer, so we hand back the larger parent chunk that the child
    belongs to.
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        embed_fn: Callable[[str], np.ndarray],
        parent_map: Dict[str, str],
    ):
        """
        Parameters
        ----------
        vectorstore : VectorStore
            Index of child-chunk embeddings.
        embed_fn : callable
            Function that maps a string to a 1-D embedding vector.
        parent_map : dict[str, str]
            Maps ``child_doc_id`` -> ``parent_text``.
        """
        self.vectorstore = vectorstore
        self.embed_fn = embed_fn
        self.parent_map = parent_map

    @classmethod
    def from_documents(
        cls,
        documents: List[str],
        embed_fn: Callable[[str], np.ndarray],
        parent_chunk_fn: Optional[Callable[[str], List[str]]] = None,
        child_chunk_fn: Optional[Callable[[str], List[str]]] = None,
    ) -> "ParentRetriever":
        """Build a :class:`ParentRetriever` from raw document strings.

        Parameters
        ----------
        documents : list[str]
            Raw documents to index.
        embed_fn : callable
            Embedding function (string -> 1-D numpy array).
        parent_chunk_fn, child_chunk_fn : callable, optional
            Forwarded to :func:`chunk_with_parents`.
        """
        vs = VectorStore()
        parent_map: Dict[str, str] = {}

        child_texts: List[str] = []
        child_ids: List[str] = []
        child_metadata: List[Dict] = []

        for doc_idx, doc in enumerate(documents):
            hierarchy = chunk_with_parents(
                doc,
                parent_chunk_fn=parent_chunk_fn,
                child_chunk_fn=child_chunk_fn,
            )
            for entry in hierarchy:
                parent_text = entry["parent"]
                parent_id = entry["parent_id"]
                for child in entry["children"]:
                    cid = f"doc{doc_idx}_p{parent_id}_c{child['child_id']}"
                    child_texts.append(child["text"])
                    child_ids.append(cid)
                    child_metadata.append({
                        "doc_idx": doc_idx,
                        "parent_id": parent_id,
                        "child_id": child["child_id"],
                        "child_text": child["text"],
                    })
                    parent_map[cid] = parent_text

        if child_texts:
            vs.add_texts(child_texts, embed_fn, ids=child_ids, metadata=child_metadata)
            vs._flush()  # materialise pending vectors so search() works immediately

        return cls(vectorstore=vs, embed_fn=embed_fn, parent_map=parent_map)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for child chunks, return deduplicated parent documents.

        Parameters
        ----------
        query : str
            The user query.
        k : int
            Number of child chunks to retrieve before mapping to parents.

        Returns
        -------
        list[dict]
            Each element has the shape::

                {"parent": str, "child": str, "score": float}

            Parents are deduplicated — if multiple children from the same
            parent match, only the highest-scoring child is kept.
        """
        q_vec = self.embed_fn(query)
        results = self.vectorstore.search(q_vec, k=k)

        # Deduplicate by parent text, keeping the highest score per parent
        seen_parents: Dict[str, Dict] = {}
        for doc_id, score, meta in results:
            parent_text = self.parent_map.get(doc_id, "")
            child_text = meta.get("child_text", "")
            if parent_text not in seen_parents or score > seen_parents[parent_text]["score"]:
                seen_parents[parent_text] = {
                    "parent": parent_text,
                    "child": child_text,
                    "score": score,
                }

        # Return sorted by score descending
        return sorted(seen_parents.values(), key=lambda x: x["score"], reverse=True)
