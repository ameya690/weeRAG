# wee/graph.py
from __future__ import annotations
from typing import List, Tuple, Dict, Iterable, Optional, Set
import json
import re
from collections import defaultdict

Triple = Tuple[str, str, str]

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
_WORD = re.compile(r"[A-Za-z0-9_]+")

def _sentences(text: str) -> List[str]:
    text = text.strip()
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]

def _simple_triples(sent: str) -> List[Triple]:
    """
    Rule-based triple extractor (very naive but useful for demos).
    Patterns:
      - "<NOUN> is a|an <NOUN PHRASE>" -> (NOUN, "is_a", NOUN PHRASE)
      - "<X> uses <Y>"                 -> (X, "uses", Y)
      - "<X> retrieves <Y>"            -> (X, "retrieves", Y)
      - "<X> enables <Y>"              -> (X, "enables", Y)
      - "<X> fits <Y>"                 -> (X, "fits", Y)
    """
    s = sent.strip()
    triples: List[Triple] = []

    # is-a
    m = re.search(r"^([A-Z][\w\s-]{0,80}?)\s+is\s+(?:an?|the)\s+([\w\s-]{1,120})$", s, re.IGNORECASE)
    if m:
        h = m.group(1).strip()
        t = m.group(2).strip().rstrip(".")
        triples.append((h, "is_a", t))

    verbs = ["uses", "retrieves", "enables", "fits", "packs", "provides", "builds"]
    for v in verbs:
        m = re.search(rf"^([\w\s-]{{1,80}}?)\s+{v}\s+([\w\s-]{{1,120}})$", s, re.IGNORECASE)
        if m:
            h = m.group(1).strip()
            t = m.group(2).strip().rstrip(".")
            triples.append((h, v.lower(), t))

    # fallback: "X -> Y" arrows
    m = re.search(r"^([\w\s-]{1,80})\s*->\s*([\w\s-]{1,120})$", s)
    if m:
        triples.append((m.group(1).strip(), "->", m.group(2).strip()))
    return triples

class Graph:
    def __init__(self):
        self.triples: List[Triple] = []
        self.nodes: Set[str] = set()
        self.adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # head -> [(rel, tail)]

    def add(self, triples: Iterable[Triple]):
        for h, r, t in triples:
            h, r, t = h.strip(), r.strip(), t.strip()
            self.triples.append((h, r, t))
            self.nodes.add(h); self.nodes.add(t)
            self.adj[h].append((r, t))

    def build_from_texts(self, texts: Iterable[str]):
        for txt in texts:
            for s in _sentences(txt):
                self.add(_simple_triples(s))

    def neighbors(self, node: str) -> List[Tuple[str,str]]:
        return list(self.adj.get(node, []))

    def find(self, pattern: str) -> List[Triple]:
        rx = re.compile(pattern, re.IGNORECASE)
        return [(h,r,t) for (h,r,t) in self.triples if rx.search(h) or rx.search(r) or rx.search(t)]

    def to_json(self) -> str:
        return json.dumps({"nodes": sorted(self.nodes), "triples": self.triples}, ensure_ascii=False, indent=2)

    def to_dot(self) -> str:
        lines = ["digraph wee {", "  rankdir=LR; node [shape=box, style=rounded];"]
        for h,r,t in self.triples:
            lines.append(f'  "{h}" -> "{t}" [label="{r}"];')
        lines.append("}")
        return "\n".join(lines)
