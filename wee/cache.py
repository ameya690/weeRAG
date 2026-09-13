from typing import Optional, Any, Callable, Dict, List, Tuple
import time
import json
import sqlite3
import hashlib
import threading

import numpy as np


def _now():
    return time.time()


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class Cache:
    def __init__(self, path: str = ".wee_cache.sqlite"):
        self.path = path
        self._lock = threading.Lock()
        self._con = sqlite3.connect(self.path, check_same_thread=False)
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL, "
            "created REAL NOT NULL, ttl REAL)"
        )
        self._con.commit()

    def set(self, key_signature: str, value: Any, ttl_seconds: Optional[float] = None):
        k = _sha256(key_signature)
        v = json.dumps(value, ensure_ascii=False)
        created = _now()
        ttl = float(ttl_seconds) if ttl_seconds is not None else None
        with self._lock:
            self._con.execute(
                "REPLACE INTO cache (key, value, created, ttl) VALUES (?, ?, ?, ?)",
                (k, v, created, ttl),
            )
            self._con.commit()

    def get(self, key_signature: str):
        k = _sha256(key_signature)
        with self._lock:
            row = self._con.execute(
                "SELECT value, created, ttl FROM cache WHERE key = ?", (k,)
            ).fetchone()
            if not row:
                return None
            value, created, ttl = row
            if ttl is not None and (_now() > created + ttl):
                self._con.execute("DELETE FROM cache WHERE key = ?", (k,))
                self._con.commit()
                return None
        return json.loads(value)

    def close(self):
        with self._lock:
            self._con.close()


def cached(cache: Cache, namespace: str, key_fn: Callable[..., str]):
    def _wrap(fn):
        def wrapper(*args, **kwargs):
            sig = f"{namespace}|{key_fn(*args, **kwargs)}"
            hit = cache.get(sig)
            if hit is not None:
                return hit
            out = fn(*args, **kwargs)
            cache.set(sig, out)
            return out
        return wrapper
    return _wrap


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class SemanticCache:
    """Cache that matches queries by embedding similarity, not exact string match.

    If a new query is semantically similar enough to a cached query
    (cosine similarity >= threshold), the cached result is returned.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], np.ndarray],
        threshold: float = 0.85,
        max_entries: int = 1000,
    ):
        self._embed_fn = embed_fn
        self._threshold = threshold
        self._max_entries = max_entries
        self._lock = threading.Lock()
        # Each entry: (embedding, query_text, value, timestamp)
        self._entries: List[Tuple[np.ndarray, str, Any, float]] = []
        self._hits = 0
        self._misses = 0

    def get(self, query: str) -> Optional[Any]:
        """Return cached value if a semantically similar query exists, else None."""
        query_emb = self._embed_fn(query)
        with self._lock:
            if not self._entries:
                self._misses += 1
                return None

            best_sim = -1.0
            best_value = None
            for emb, _text, value, _ts in self._entries:
                sim = _cosine_similarity(query_emb, emb)
                if sim > best_sim:
                    best_sim = sim
                    best_value = value

            if best_sim >= self._threshold:
                self._hits += 1
                return best_value

            self._misses += 1
            return None

    def set(self, query: str, value: Any):
        """Cache a query-value pair. Evicts the oldest entry if over capacity."""
        query_emb = self._embed_fn(query)
        ts = _now()
        with self._lock:
            self._entries.append((query_emb, query, value, ts))
            if len(self._entries) > self._max_entries:
                # Evict oldest (first) entry
                self._entries.pop(0)

    def clear(self):
        """Remove all cached entries and reset stats."""
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._entries),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }
