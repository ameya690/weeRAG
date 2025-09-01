import numpy as np
from wee import Cache, cached
cache=Cache('.wee_cache.sqlite')
@cached(cache,'embed',lambda text: text)
def hash_embed(text: str, dim: int=128):
    rng=np.random.default_rng(abs(hash(text))%(2**32)); v=rng.standard_normal(dim).astype(np.float32).tolist(); return v
def main():
    q='retrieval augmented generation'; v1=hash_embed(q); v2=hash_embed(q); print('Cache hit:', v1==v2)
if __name__=='__main__': main()
