from typing import List, Dict, Any, Sequence, Tuple, Optional
import re
_WORD = re.compile(r"\w+")
def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip().lower()); return s
def _tokens(s: str): return [t.lower() for t in _WORD.findall(s)]
def exact_match(pred: str, golds: Sequence[str]) -> int:
    p=_norm(pred); return int(any(p==_norm(g) for g in golds))
def f1_score(pred: str, gold: str) -> float:
    p=_tokens(pred); g=_tokens(gold)
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    from collections import Counter
    cp=Counter(p); overlap=sum(min(cp[t], _tokens(gold).count(t)) for t in set(p))
    if overlap==0: return 0.0
    prec=overlap/len(p); rec=overlap/len(g); return 2*prec*rec/max(prec+rec,1e-9)
def max_f1(pred: str, golds: Sequence[str]) -> float:
    return max((f1_score(pred,g) for g in golds), default=0.0)
def jaccard(a: str, b: str) -> float:
    A=set(_tokens(a)); B=set(_tokens(b)); 
    return len(A&B)/max(len(A|B),1) if A or B else 1.0
def sentence_split(s: str): 
    return [seg.strip() for seg in re.split(r'(?<=[\.\!?])\s+', s.strip()) if seg.strip()]
def faithfulness(answer: str, contexts: Sequence[str], thr: float=0.5) -> float:
    sents=sentence_split(answer); 
    if not sents: return 1.0
    sup=sum(1 for s in sents if any(jaccard(s,c)>=thr for c in contexts)); 
    return sup/len(sents)
def context_precision_recall(selected: Sequence[str], gold: Sequence[str], thr: float=0.5):
    if not selected and not gold: return (1.0,1.0)
    if not selected: return (0.0,0.0)
    if not gold: return (0.0,1.0)
    matched=0; used=set()
    for s in selected:
        for i,g in enumerate(gold):
            if i in used: continue
            if jaccard(s,g)>=thr:
                matched+=1; used.add(i); break
    prec=matched/max(len(selected),1); rec=matched/max(len(gold),1); return (prec,rec)
def evaluate_qa(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out={"samples": []}; em_t=0; f1_t=0.0; faith_t=0.0; cp_t=0.0; cr_t=0.0; ncp=0
    for ex in samples:
        pred=ex.get("pred"," "); golds=ex.get("answers",[]); ctx=ex.get("contexts",[]); gc=ex.get("gold_citations",None)
        em=exact_match(pred,golds); f1=max_f1(pred,golds); faith=faithfulness(pred,ctx)
        rec={"em":em,"f1":f1,"faithfulness":faith}
        if gc is not None:
            prec,recall=context_precision_recall(ctx,gc); rec["context_precision"]=prec; rec["context_recall"]=recall; cp_t+=prec; cr_t+=recall; ncp+=1
        out["samples"].append(rec); em_t+=em; f1_t+=f1; faith_t+=faith
    n=max(len(samples),1)
    out["metrics"]={"em":em_t/n,"f1":f1_t/n,"faithfulness":faith_t/n}
    if ncp>0: out["metrics"].update({"context_precision":cp_t/ncp,"context_recall":cr_t/ncp})
    return out
