from wee import Tokenizer, BM25, GPTConfig, GPT, chunk_by_sentences

def main():
    # Tokenizer demo
    tok = Tokenizer()
    tok.train([
        "Large language models (LLMs) use attention.",
        "Retrieval augmented generation (RAG) retrieves relevant chunks."
    ], vocab_size=300)
    ids = tok.encode("attention retrieves chunks", add_special=True)
    print("Encoded IDs:", ids)
    print("Decoded:", tok.decode(ids))

    # BM25 demo
    bm = BM25()
    docs = [
        "Transformers attend to tokens",
        "RAG retrieves chunks",
        "BM25 is a lexical baseline",
    ]
    bm.add(docs)
    print("BM25 search:", bm.search("RAG tokens", k=2))

    # Chunking demo
    text = "This is a sentence. Another sentence! And one more? Yep."
    print("Sentence chunks:", chunk_by_sentences(text, max_chars=30, overlap=5))

    # Transformer config (toy)
    cfg = GPTConfig(vocab_size=tok.vocab_size, d_model=128, n_heads=4, n_layers=2, max_seq_len=64)
    model = GPT(cfg)
    print("Model params:", sum(p.numel() for p in model.parameters()))

if __name__ == "__main__":
    main()
