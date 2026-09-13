"""Agentic retrieval quickstart.

Shows the AgentLoop with a SearchTool backed by a Retriever.
Uses a dummy generate_fn that simulates an LLM deciding to search once
and then answering, so no API key is needed.
"""

import numpy as np
from wee import (
    VectorStore,
    Retriever,
    chunk_by_sentences,
)
from wee.agent import SearchTool, AgentLoop


# -- tiny embedding function (deterministic hash) ---------------------------

def hash_embed(dim=128):
    def _emb(text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(dim).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        return v
    return _emb


# -- dummy generate_fn that simulates a ReAct agent -------------------------

def make_dummy_generate():
    """Return a callable that acts like an LLM doing one search then answering."""
    call_count = {"n": 0}

    def generate(prompt: str) -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call: the "LLM" decides to search
            return (
                "Thought: I need to find information about retrieval in "
                "a knowledge base. Let me search.\n"
                "Action: search\n"
                "Action Input: retrieval knowledge base"
            )
        else:
            # Second call: the "LLM" has seen the search results and answers
            return (
                "Thought: I now have enough information to answer.\n"
                "Final Answer: RAG retrieves relevant context from a "
                "knowledge base and uses it to generate grounded answers."
            )

    return generate


def main():
    # -- build a small knowledge base ----------------------------------------
    docs = [
        "RAG retrieves relevant context from a knowledge base.",
        "Transformers use attention to model token dependencies.",
        "BM25 provides a strong lexical baseline for retrieval.",
        "Vector stores enable fast similarity search over embeddings.",
        "Context packing fits the best chunks into a token budget.",
    ]

    chunks, meta = [], []
    for i, d in enumerate(docs):
        for j, c in enumerate(chunk_by_sentences(d, max_chars=200)):
            chunks.append(c)
            meta.append({"doc_id": i, "chunk_id": j, "text": c})

    emb = hash_embed(128)
    vs = VectorStore(dim=128, normalize=True)
    vs.add_texts(
        [m["text"] for m in meta],
        embed_fn=emb,
        ids=[f"{m['doc_id']}-{m['chunk_id']}" for m in meta],
        metadata=meta,
    )

    texts_by_id = {f"{m['doc_id']}-{m['chunk_id']}": m["text"] for m in meta}

    # -- set up the agent ----------------------------------------------------
    retriever = Retriever(method="topk", k=3).attach_vectorstore(vs, emb)
    search_tool = SearchTool(retriever, texts_by_id)

    agent = AgentLoop(
        generate_fn=make_dummy_generate(),
        tools=[search_tool],
        max_steps=5,
    )

    # -- run -----------------------------------------------------------------
    question = "How does RAG retrieve relevant passages?"
    result = agent.run(question)

    print("Question:", question)
    print()
    for i, step in enumerate(result["steps"], 1):
        print(f"--- Step {i} ---")
        print(f"Thought:      {step['thought']}")
        print(f"Action:       {step['action']}")
        print(f"Action Input: {step['action_input']}")
        print(f"Observation:  {step['observation'][:200]}...")
        print()
    print("Final Answer:", result["answer"])
    print(f"Tool calls:   {result['tool_calls']}")


if __name__ == "__main__":
    main()
