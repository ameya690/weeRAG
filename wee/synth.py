# wee/synth.py
from __future__ import annotations
from typing import List, Dict, Optional
import re
import random

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def _sents(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]

def _titlecase(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s

def _cloze(sent: str) -> Optional[Dict]:
    words = [w for w in re.findall(r"[A-Za-z0-9'-]+", sent) if len(w) > 3]
    if not words:
        return None
    ans = random.choice(words)
    q = sent.replace(ans, "____", 1)
    return {"question": q, "answers": [ans], "type": "cloze"}

def _wh(sent: str) -> Optional[Dict]:
    # Naive WH: “X uses Y” -> What does X use?
    m = re.search(r"^([\w\s-]{2,80})\s+uses\s+([\w\s-]{2,80})", sent, re.IGNORECASE)
    if m:
        x, y = m.group(1).strip(), m.group(2).strip().rstrip(".")
        return {"question": f"What does {_titlecase(x)} use?", "answers": [y], "type": "wh"}
    m = re.search(r"^([\w\s-]{2,80})\s+retrieves\s+([\w\s-]{2,80})", sent, re.IGNORECASE)
    if m:
        x, y = m.group(1).strip(), m.group(2).strip().rstrip(".")
        return {"question": f"What does {_titlecase(x)} retrieve?", "answers": [y], "type": "wh"}
    return None

def synth_qa(texts: List[str], n_per_text: int = 3, seed: int = 0) -> List[Dict]:
    """
    Make tiny synthetic QA pairs from raw texts.
    Returns dicts: {"question", "answers", "context"}
    """
    random.seed(seed)
    out: List[Dict] = []
    for txt in texts:
        sents = _sents(txt)
        cand = []
        for s in sents:
            s = s.strip()
            if not s: continue
            q = _wh(s) or _cloze(s)
            if q:
                q["context"] = s
                cand.append(q)
        random.shuffle(cand)
        out.extend(cand[:n_per_text])
    return out
