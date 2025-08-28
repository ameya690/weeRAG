# wee — tiny, teachable LLM/RAG foundations (Wave 1)

Wave 1 modules included:

- `wee.tokenizer.Tokenizer` — SentencePiece-like char-BPE with a word-boundary marker.
- `wee.attention.MultiHeadAttention` — PyTorch multi-head attention + scaled-dot-product core.
- `wee.transformer.GPT` — minimal GPT (decoder-only) with greedy sampling.
- `wee.bm25.BM25` — dead-simple BM25 index/search.
- `wee.chunk` — word/sentence/token chunkers.

## Install (editable)

```bash
pip install -e .
# requires torch for attention/transformer modules
```

## Quickstart

```python
from wee import Tokenizer, BM25, GPTConfig, GPT
from wee import chunk_by_words, chunk_by_sentences, chunk_by_tokens

# 1) Tokenizer
tok = Tokenizer()
tok.train(["hello world", "hello there"], vocab_size=200)
ids = tok.encode("hello world")
print(ids, tok.decode(ids))

# 2) BM25
bm = BM25()
bm.add(["LLMs are fun", "RAG retrieves chunks", "Transformers attend to tokens"])
print(bm.search("RAG tokens", k=2))

# 3) Chunking
text = "This is a sentence. Another sentence! And one more? Yep."
print(chunk_by_sentences(text, max_chars=30, overlap=5))

# 4) Transformer (toy)
cfg = GPTConfig(vocab_size=tok.vocab_size, d_model=128, n_heads=4, n_layers=2, max_seq_len=64)
model = GPT(cfg)
```

> Note: This is educational code with tiny, readable implementations—not optimized for production.
