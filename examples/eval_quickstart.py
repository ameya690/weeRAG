import numpy as np
from wee import evaluate_qa, Judge
def hash_embed(dim=128):
    def _emb(text: str):
        rng=np.random.default_rng(abs(hash(text))%(2**32)); v=rng.standard_normal(dim).astype(np.float32); v/= (np.linalg.norm(v)+1e-9); return v
    return _emb
def main():
    samples=[
        {"question":"What baseline is commonly used for lexical retrieval?","pred":"BM25 is a strong lexical baseline for retrieval.","answers":["BM25"],"contexts":["BM25 provides a strong lexical baseline for retrieval."],"gold_citations":["BM25 baseline for retrieval"]},
        {"question":"What technique fits the best chunks under a token budget?","pred":"Context packing","answers":["Context packing"],"contexts":["Context packing fits the best chunks into a token budget."],"gold_citations":["Context packing fits the best chunks"]},
    ]
    j=Judge(embed_fn=hash_embed(128))
    for ex in samples: print(ex["question"], "->", j.judge(ex["question"], ex["pred"], ex["contexts"], n=3))
    print(evaluate_qa(samples))
if __name__=='__main__': main()
