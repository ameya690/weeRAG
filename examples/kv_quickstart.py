import torch, time
from wee import Tokenizer, GPTConfig, GPT, upgrade_gpt_for_kv, generate_with_cache

def main():
    tok = Tokenizer()
    tok.train(["RAG retrieves context. KV cache speeds up generation."], vocab_size=300)
    ids = torch.tensor([tok.encode("RAG retrieves context.", add_special=True)], dtype=torch.long)

    cfg = GPTConfig(vocab_size=tok.vocab_size, d_model=128, n_heads=4, n_layers=2, max_seq_len=256, dropout=0.0)
    model = GPT(cfg)

    # Upgrade attention modules to be cacheable
    upgrade_gpt_for_kv(model)

    t0 = time.perf_counter()
    out = generate_with_cache(model, ids, max_new_tokens=50, temperature=1.0, top_k=20)
    t1 = time.perf_counter()
    print("Generated:", tok.decode(out[0].tolist()))
    print("Time with KV cache (includes first full pass):", round(t1 - t0, 4), "s")

if __name__ == "__main__":
    main()
