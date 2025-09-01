from __future__ import annotations
from typing import List, Dict, Any, Optional
import math

class Router:
    """
    Minimal model router. You provide a list of endpoints with metadata:
    {
      "name": "gpt-mini",
      "price_in": 0.1,      # $ per 1k input tokens
      "price_out": 0.4,     # $ per 1k output tokens
      "latency": 0.5,       # seconds for typical prompt
      "quality": 0.65,      # [0,1] relative quality score
      "max_tokens": 8192
      "tags": ["fast","cheap"]
    }
    Policy selects endpoint by weighted objective given a request spec.
    """
    def __init__(self, endpoints: List[Dict[str, Any]]):
        self.endpoints = endpoints

    def route(self, prompt_len: int, max_output_tokens: int, budget_usd: Optional[float] = None, latency_target_s: Optional[float] = None, priority: str = "balance", required_tags: Optional[List[str]] = None) -> Dict[str, Any]:
        assert priority in ("balance", "quality", "speed", "cost")
        required_tags = required_tags or []
        # Filter by context length and tags
        cand = [e for e in self.endpoints if e["max_tokens"] >= (prompt_len + max_output_tokens) and all(t in e.get("tags",[]) for t in required_tags)]
        if not cand:
            raise ValueError("No endpoint satisfies constraints")
        # Cost estimate
        def est_cost(e):
            return (prompt_len/1000.0)*e["price_in"] + (max_output_tokens/1000.0)*e["price_out"]
        # Score
        best, best_score = None, -1e9
        for e in cand:
            cost = est_cost(e)
            if budget_usd is not None and cost > budget_usd:
                continue
            lat = e.get("latency", 1.0)
            qual = e.get("quality", 0.5)
            if priority == "quality":
                score = qual - 0.01*cost - 0.1*lat
            elif priority == "speed":
                score = -lat + 0.5*qual - 0.01*cost
            elif priority == "cost":
                score = -cost + 0.3*qual - 0.1*lat
            else:
                score = 0.5*qual - 0.2*lat - 0.3*cost
            if score > best_score:
                best, best_score = e, score
        if best is None:
            raise ValueError("No endpoint satisfies budget/latency")
        return {
            "endpoint": best,
            "est_cost": est_cost(best),
            "reason": priority,
        }
