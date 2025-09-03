from wee import Graph

docs = [
    "Transformers use attention to model token dependencies.",
    "RAG retrieves relevant context from a knowledge base.",
    "BM25 is a lexical baseline for retrieval.",
    "Vector stores enable similarity search over embeddings.",
    "Context packing fits the best chunks into a token budget.",
    "Transformers -> Tokens",
]

g = Graph()
g.build_from_texts(docs)

print("Triples:")
for h,r,t in g.triples:
    print(f"  ({h}) -[{r}]-> ({t})")

print("\nNeighbors of 'Transformers':", g.neighbors("Transformers"))
print("\nFind 'retrieves':", g.find("retrieves"))

print("\nDOT:")
print(g.to_dot())
