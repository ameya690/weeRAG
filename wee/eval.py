from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

_WORD = re.compile(r"\w+")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _tokens(s: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(s)]


def exact_match(pred: str, golds: Sequence[str]) -> int:
    p = _norm(pred)
    return int(any(p == _norm(g) for g in golds))


def f1_score(pred: str, gold: str) -> float:
    p = _tokens(pred)
    g = _tokens(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    cp = Counter(p)
    g_tokens = _tokens(gold)
    overlap = sum(min(cp[t], g_tokens.count(t)) for t in set(p))
    if overlap == 0:
        return 0.0
    prec = overlap / len(p)
    rec = overlap / len(g)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def max_f1(pred: str, golds: Sequence[str]) -> float:
    return max((f1_score(pred, g) for g in golds), default=0.0)


def jaccard(a: str, b: str) -> float:
    sa = set(_tokens(a))
    sb = set(_tokens(b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def sentence_split(s: str) -> List[str]:
    return [seg.strip() for seg in re.split(r"(?<=[\.\!?])\s+", s.strip()) if seg.strip()]


def faithfulness(answer: str, contexts: Sequence[str], thr: float = 0.5) -> float:
    sents = sentence_split(answer)
    if not sents:
        return 1.0
    supported = sum(1 for s in sents if any(jaccard(s, c) >= thr for c in contexts))
    return supported / len(sents)


def context_precision_recall(
    selected: Sequence[str], gold: Sequence[str], thr: float = 0.5
) -> Tuple[float, float]:
    if not selected and not gold:
        return (1.0, 1.0)
    if not selected:
        return (0.0, 0.0)
    if not gold:
        return (0.0, 1.0)
    matched = 0
    used: set = set()
    for s in selected:
        for i, g in enumerate(gold):
            if i in used:
                continue
            if jaccard(s, g) >= thr:
                matched += 1
                used.add(i)
                break
    prec = matched / max(len(selected), 1)
    rec = matched / max(len(gold), 1)
    return (prec, rec)


def groundedness_score(
    answer: str, contexts: Sequence[str], thr: float = 0.3
) -> Dict[str, Any]:
    """Check which answer sentences are grounded in the retrieved contexts.

    Like :func:`faithfulness` but returns per-sentence detail so users can
    debug which claims lack supporting evidence.
    """
    sents = sentence_split(answer)
    if not sents:
        return {
            "score": 1.0,
            "grounded_sentences": 0,
            "total_sentences": 0,
            "ungrounded": [],
        }
    grounded = 0
    ungrounded: List[str] = []
    for s in sents:
        if any(jaccard(s, c) >= thr for c in contexts):
            grounded += 1
        else:
            ungrounded.append(s)
    return {
        "score": grounded / len(sents),
        "grounded_sentences": grounded,
        "total_sentences": len(sents),
        "ungrounded": ungrounded,
    }


def citation_support(
    answer: str,
    citations: Sequence[str],
    contexts: Sequence[str],
    thr: float = 0.5,
) -> Dict[str, Any]:
    """Check whether each cited passage actually appears in the retrieved context.

    *citations* are the passage strings the model claims to be citing.
    A citation is "supported" when it has ``jaccard >= thr`` with at least one
    context passage.
    """
    if not citations:
        return {"score": 1.0, "supported": 0, "total": 0, "unsupported": []}
    supported = 0
    unsupported: List[str] = []
    for cite in citations:
        if any(jaccard(cite, c) >= thr for c in contexts):
            supported += 1
        else:
            unsupported.append(cite)
    return {
        "score": supported / len(citations),
        "supported": supported,
        "total": len(citations),
        "unsupported": unsupported,
    }


def evaluate_qa(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"samples": []}
    em_total = 0
    f1_total = 0.0
    faith_total = 0.0
    ground_total = 0.0
    cp_total = 0.0
    cr_total = 0.0
    n_context = 0

    for ex in samples:
        pred = ex.get("pred", " ")
        golds = ex.get("answers", [])
        ctx = ex.get("contexts", [])
        gold_citations = ex.get("gold_citations", None)

        em = exact_match(pred, golds)
        f1 = max_f1(pred, golds)
        faith = faithfulness(pred, ctx)
        ground = groundedness_score(pred, ctx)

        rec: Dict[str, Any] = {
            "em": em,
            "f1": f1,
            "faithfulness": faith,
            "groundedness": ground["score"],
        }

        if gold_citations is not None:
            prec, recall = context_precision_recall(ctx, gold_citations)
            rec["context_precision"] = prec
            rec["context_recall"] = recall
            cp_total += prec
            cr_total += recall
            n_context += 1

        out["samples"].append(rec)
        em_total += em
        f1_total += f1
        faith_total += faith
        ground_total += ground["score"]

    n = max(len(samples), 1)
    out["metrics"] = {
        "em": em_total / n,
        "f1": f1_total / n,
        "faithfulness": faith_total / n,
        "groundedness": ground_total / n,
    }
    if n_context > 0:
        out["metrics"]["context_precision"] = cp_total / n_context
        out["metrics"]["context_recall"] = cr_total / n_context

    return out
