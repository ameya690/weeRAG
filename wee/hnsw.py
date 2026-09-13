"""Toy HNSW (Hierarchical Navigable Small World) index.

Educational implementation -- demonstrates the core algorithm:
  - Multi-layer graph where higher layers have fewer nodes
  - Greedy search through layers for fast approximate nearest neighbors
  - Insert new nodes by connecting to nearest neighbors at each layer

Not production-grade: no deletions, no serialisation, single-threaded.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np


class HNSWIndex:
    """Approximate nearest-neighbour search via HNSW graphs."""

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        max_level: int = 4,
    ):
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.max_level = max_level

        # Storage
        self.vectors: List[np.ndarray] = []
        self.doc_ids: List[Optional[str]] = []

        # Graph: graph[layer][node_id] -> set of neighbor node_ids
        self.graph: List[Dict[int, set]] = [dict() for _ in range(max_level)]

        # Entry point: node with the highest insertion level
        self.entry_point: Optional[int] = None
        self.entry_level: int = -1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _random_level(self) -> int:
        """Sample an insertion level from a geometric distribution.

        P(level >= l) = (1/M)^l  -- most nodes land at level 0.
        """
        level = 0
        while random.random() < (1.0 / self.M) and level < self.max_level - 1:
            level += 1
        return level

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        dot = float(np.dot(a, b))
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-12:
            return 0.0
        return dot / denom

    # ------------------------------------------------------------------
    # Search within a single layer
    # ------------------------------------------------------------------

    def _search_layer(
        self,
        query: np.ndarray,
        entry_point: int,
        ef: int,
        layer: int,
    ) -> List[Tuple[float, int]]:
        """Greedy BFS on *layer*, returning up to *ef* nearest neighbours.

        Returns a list of ``(similarity, node_id)`` pairs sorted descending
        by similarity.
        """
        visited: set = {entry_point}
        ep_sim = self._cosine_sim(query, self.vectors[entry_point])
        candidates: List[Tuple[float, int]] = [(-ep_sim, entry_point)]  # min-heap by neg sim
        results: List[Tuple[float, int]] = [(ep_sim, entry_point)]

        import heapq

        heapq.heapify(candidates)

        while candidates:
            neg_sim_c, c = heapq.heappop(candidates)
            # Furthest in results
            worst_sim = min(r[0] for r in results)
            if -neg_sim_c < worst_sim and len(results) >= ef:
                break

            for neighbor in self.graph[layer].get(c, set()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                n_sim = self._cosine_sim(query, self.vectors[neighbor])
                worst_sim = min(r[0] for r in results)
                if n_sim > worst_sim or len(results) < ef:
                    heapq.heappush(candidates, (-n_sim, neighbor))
                    results.append((n_sim, neighbor))
                    if len(results) > ef:
                        # Drop the worst
                        results.sort(key=lambda x: x[0])
                        results.pop(0)

        results.sort(key=lambda x: -x[0])
        return results

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def add(self, vector: np.ndarray, doc_id: Optional[str] = None):
        """Insert a single vector into the index."""
        vector = np.asarray(vector, dtype=np.float32).ravel()
        assert vector.shape[0] == self.dim, f"Expected dim {self.dim}, got {vector.shape[0]}"

        node_id = len(self.vectors)
        self.vectors.append(vector)
        self.doc_ids.append(doc_id)

        level = self._random_level()

        # Ensure the node exists in all layers up to its level
        for lyr in range(level + 1):
            self.graph[lyr][node_id] = set()

        # First node -- just set entry point
        if self.entry_point is None:
            self.entry_point = node_id
            self.entry_level = level
            return

        ep = self.entry_point

        # Phase 1: greedily descend through layers above insertion level
        for lyr in range(self.entry_level, level, -1):
            results = self._search_layer(vector, ep, ef=1, layer=lyr)
            ep = results[0][1]  # closest node becomes new entry point

        # Phase 2: at each layer from insertion level down to 0,
        #   find neighbours and connect
        for lyr in range(min(level, self.entry_level), -1, -1):
            results = self._search_layer(vector, ep, ef=self.ef_construction, layer=lyr)
            neighbors = [nid for (_sim, nid) in results[: self.M]]

            # Connect new node -> neighbours (bidirectional)
            for nbr in neighbors:
                self.graph[lyr][node_id].add(nbr)
                if nbr not in self.graph[lyr]:
                    self.graph[lyr][nbr] = set()
                self.graph[lyr][nbr].add(node_id)

                # Prune if neighbour exceeds M connections
                if len(self.graph[lyr][nbr]) > self.M:
                    # Keep the M closest
                    scored = [
                        (self._cosine_sim(self.vectors[nbr], self.vectors[n]), n)
                        for n in self.graph[lyr][nbr]
                    ]
                    scored.sort(key=lambda x: -x[0])
                    self.graph[lyr][nbr] = {n for _, n in scored[: self.M]}

            if results:
                ep = results[0][1]

        # Update entry point if new node has a higher level
        if level > self.entry_level:
            self.entry_point = node_id
            self.entry_level = level

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        ef: int = 50,
    ) -> List[Tuple[Optional[str], float]]:
        """Return the *k* approximate nearest neighbours.

        Returns ``(doc_id, similarity)`` pairs sorted by descending
        similarity.
        """
        if self.entry_point is None:
            return []

        query = np.asarray(query, dtype=np.float32).ravel()
        ep = self.entry_point

        # Descend through upper layers with ef=1
        for lyr in range(self.entry_level, 0, -1):
            results = self._search_layer(query, ep, ef=1, layer=lyr)
            ep = results[0][1]

        # Layer-0 search with the requested ef
        results = self._search_layer(query, ep, ef=max(ef, k), layer=0)

        top_k = results[:k]
        return [(self.doc_ids[nid], sim) for sim, nid in top_k]

    # ------------------------------------------------------------------
    # Evaluation helper
    # ------------------------------------------------------------------

    def recall_at_k(
        self,
        query: np.ndarray,
        k: int,
        exact_results: List,
    ) -> float:
        """Compare HNSW results against brute-force exact results.

        *exact_results* should be a sequence of ``(doc_id, similarity)``
        or similar tuples whose first element is the doc id.

        Returns ``recall = |intersection| / k``.
        """
        approx = self.search(query, k=k, ef=max(k * 10, 50))
        approx_ids = {r[0] for r in approx}
        exact_ids = {r[0] for r in exact_results[:k]}
        if k == 0:
            return 1.0
        return len(approx_ids & exact_ids) / k
