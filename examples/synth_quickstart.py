from wee import synth_qa

text = """RAG retrieves relevant context. Transformers use attention. 
BM25 is a lexical baseline. Vector stores enable similarity search."""
pairs = synth_qa([text], n_per_text=5, seed=42)
for p in pairs:
    print(p["type"].upper(), "Q:", p["question"])
    print("  A:", p["answers"])
    print("  ctx:", p["context"])
