import torch, time
from wee import Tokenizer, GPTConfig, GPT, quantize_model, size_report, eval_perplexity

def make_data(tok):
    text = "Transformers use attention. BM25 is a lexical baseline. Context packing fits best chunks."
    tok.train([text], vocab_size=400)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    return ids

def main():
    tok = Tokenizer()
    data_ids = make_data(tok)

    cfg = GPTConfig(vocab_size=tok.vocab_size, d_model=128, n_heads=4, n_layers=2, max_seq_len=128, dropout=0.0)
    model = GPT(cfg)

    print("Float32 size:", size_report(model))
    ppl_fp = eval_perplexity(model, data_ids, tok.vocab_size, chunk_len=32)
    print("PPL (fp32):", round(ppl_fp, 3))

    quantize_model(model, bits=8, per_channel=True)
    print("Quantized size:", size_report(model))
    ppl_q = eval_perplexity(model, data_ids, tok.vocab_size, chunk_len=32)
    print("PPL (int8 weights dequantized on the fly):", round(ppl_q, 3))

if __name__ == "__main__":
    main()
