# Train wee GPT on a tiny corpus, then compare perplexity and generations
# before/after int8 quantization.
# Run: python examples/gpt_train_quant_demo.py
import math, time, torch
from wee import Tokenizer, GPTConfig, GPT
from wee import eval_perplexity, quantize_model, size_report
from wee import upgrade_gpt_for_kv, generate_with_cache

# 1) Tiny corpus
corpus = (
    'retrieval augments generation by fetching relevant context.\n'
    'transformers use attention to model token dependencies.\n'
    'bm25 is a lexical baseline for retrieval.\n'
    'context packing fits the best chunks into a token budget.\n'
    'vector stores enable fast similarity search over embeddings.\n'
    'rag pipelines retrieve, rerank, and pack context into prompts.\n'
    'good retrieval improves answers, and concise prompts help models.\n'
)

# 2) Tokenize
tok = Tokenizer(); tok.train([corpus], vocab_size=1000)
ids = torch.tensor(tok.encode(corpus), dtype=torch.long)    # shape [N]

# 3) Build model
cfg = GPTConfig(vocab_size=tok.vocab_size, d_model=192, n_heads=4, n_layers=3, max_seq_len=256, dropout=0.1)
model = GPT(cfg)
upgrade_gpt_for_kv(model)   # swap in cacheable attention (weights preserved)

# Helper to compute ppl on the corpus
def ppl(m):
    return eval_perplexity(m, ids, tok.vocab_size, chunk_len=64)

print("== Model size (float32) ==")
print(size_report(model))

print("\n== Perplexity before training ==")
print("PPL (init fp32):", round(ppl(model), 3))

# 4) Train a few hundred steps
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
T = 128
steps = 600   # bump to 1500+ for better quality
for step in range(steps):
    i = torch.randint(0, max(1, len(ids)-T-1), (1,)).item()
    x = ids[i:i+T].unsqueeze(0)
    y = ids[i+1:i+1+T].unsqueeze(0)
    logits, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 50 == 0:
        print(f"step {step:4d} | loss {loss.item():.3f}")

model.eval()
print("\n== Perplexity after training ==")
ppl_trained = ppl(model)
print("PPL (trained fp32):", round(ppl_trained, 3))

# 5) Generate text (greedy or simple top-k sampling if available)
def sample(model, prompt, max_new_tokens=80, temperature=0.8, top_k=40):
    import torch
    ids_in = torch.tensor([tok.encode(prompt, add_special=True)], dtype=torch.long)
    # Use our KV-cache generator (works for fp32 and after quantization)
    out = generate_with_cache(model, ids_in, max_new_tokens=max_new_tokens,
                              temperature=temperature, top_k=top_k)
    return tok.decode(out[0].tolist())

prompt = "How do I get relevant passages for a question?"
print("\n== Generation after training (fp32) ==")
print(sample(model, prompt))

# 6) Quantize to int8 and re-evaluate
quantize_model(model, bits=8, per_channel=True)
print("\n== Model size after int8 quantization ==")
print(size_report(model))

print("\n== Perplexity after int8 quantization ==")
ppl_q = ppl(model)
print("PPL (trained int8):", round(ppl_q, 3))

print("\n== Generation after int8 quantization ==")
print(sample(model, prompt))
