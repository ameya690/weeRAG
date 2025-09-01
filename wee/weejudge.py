from typing import List, Dict, Any, Callable, Optional
import numpy as np
class HeuristicJudge:
    def __init__(self, embed_fn: Callable[[str], np.ndarray]): self.embed=embed_fn
    def score(self, question: str, answer: str, references: List[str]) -> float:
        if not references:
            qv=self.embed(question); av=self.embed(answer); sim=float(qv@av); return max(0.0,min(1.0,(sim+1)/2))
        av=self.embed(answer); refs=np.vstack([self.embed(r) for r in references]).astype(np.float32); sims=refs@av.reshape(-1,1); sims=(sims+1.0)*0.5; return float(np.max(sims))
class Judge:
    def __init__(self, llm_fn: Optional[Callable[[str], float]]=None, embed_fn: Optional[Callable[[str], np.ndarray]]=None, aggregation: str="mean", rubric: Optional[str]=None):
        assert aggregation in ("mean","median"); self.llm_fn=llm_fn; self.rubric=rubric or "Return a score [0,1]."; self.aggregation=aggregation; self.heuristic=None
        if llm_fn is None: assert embed_fn is not None; self.heuristic=HeuristicJudge(embed_fn)
    def _agg(self, xs: List[float]) -> float:
        if not xs: return 0.0
        return float(np.mean(xs) if self.aggregation=="mean" else np.median(xs))
    def judge(self, question: str, answer: str, references: List[str], n: int=3) -> Dict[str,Any]:
        scores=[]
        if self.llm_fn is not None:
            prompt=f"{self.rubric}\nQuestion:{question}\nAnswer:{answer}\nReferences:\n"+"\n".join("- "+r for r in references)+"\nScore:"
            for _ in range(n): scores.append(max(0.0,min(1.0,float(self.llm_fn(prompt)))))
        else:
            for _ in range(n): scores.append(self.heuristic.score(question, answer, references))
        return {"scores": scores, "score": self._agg(scores)}
