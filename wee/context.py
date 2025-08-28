from __future__ import annotations
from typing import List, Optional, Dict, Sequence, Tuple
from .tokenizer import Tokenizer

def _normalize(text: str) -> str:
    return " ".join(text.strip().split()).lower()

def pack_context(
    chunks: Sequence[str],
    scores: Optional[Sequence[float]] = None,
    max_tokens: int = 1024,
    tokenizer: Optional[Tokenizer] = None,
    strategy: str = "priority",
    dedup: bool = True,
    sep: str = "\n---\n",
) -> str:
    assert strategy in ("priority", "round_robin")
    if not chunks:
        return ""

    items = list(chunks)
    if scores is None:
        scores = [0.0] * len(items)
    scored = list(zip(items, scores))

    if strategy == "priority":
        scored.sort(key=lambda x: x[1], reverse=True)
    else:
        scored.sort(key=lambda x: x[1], reverse=True)
        third = max(1, len(scored) // 3)
        buckets = [scored[0:third], scored[third:2*third], scored[2*third:]]
        scored = []
        while any(buckets):
            for b in buckets:
                if b:
                    scored.append(b.pop(0))

    seen = set()
    ordered = []
    for text, s in scored:
        key = _normalize(text)
        if not dedup or key not in seen:
            ordered.append((text, s))
            seen.add(key)

    def count_tokens(s: str) -> int:
        if tokenizer is not None:
            return len(tokenizer.encode(s))
        return max(1, len(s.split()))

    budget = max_tokens
    out_chunks: List[str] = []
    for text, _ in ordered:
        need = count_tokens(text + (sep if out_chunks else ""))
        if need <= budget:
            out_chunks.append(text)
            budget -= need
        else:
            if tokenizer is None:
                words = text.split()
                sep_cost = count_tokens(sep) if out_chunks else 0
                allow = max(0, budget - sep_cost)
                if allow > 0 and len(words) > 0:
                    out_chunks.append(" ".join(words[:allow]))
                    budget = 0
            break

    return sep.join(out_chunks)
