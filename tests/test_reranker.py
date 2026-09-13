"""Tests for wee.rerank.Reranker."""
import numpy as np

from wee.rerank import Reranker


def dummy_embed(text, dim=32):
    np.random.seed(hash(text) % 2**31)
    v = np.random.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _make_candidates_and_texts():
    """Build a small candidate set for reranking tests."""
    texts_by_id = {
        "d0": "the cat sat on the mat",
        "d1": "quantum computing research paper",
        "d2": "the cat chased a mouse",
        "d3": "deep learning neural networks",
    }
    # Candidates: (id, initial_score, metadata)
    candidates = [
        ("d0", 0.9, {"text": texts_by_id["d0"]}),
        ("d1", 0.8, {"text": texts_by_id["d1"]}),
        ("d2", 0.7, {"text": texts_by_id["d2"]}),
        ("d3", 0.6, {"text": texts_by_id["d3"]}),
    ]
    return candidates, texts_by_id


# ------------------------------------------------------------------
# test_dense_rerank
# ------------------------------------------------------------------


def test_dense_rerank():
    """mode='dense' reranks by cosine similarity with dummy embeddings."""
    candidates, texts_by_id = _make_candidates_and_texts()

    rr = Reranker(mode="dense")
    rr.attach_dense(dummy_embed)

    results = rr.rerank("the cat sat on the mat", candidates, texts_by_id, k=4)

    assert len(results) == 4
    # Top result should be d0 (exact text match -> highest cosine sim)
    assert results[0][0] == "d0"
    # Scores should be descending
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)


# ------------------------------------------------------------------
# test_lexical_rerank
# ------------------------------------------------------------------


def test_lexical_rerank():
    """mode='lexical' reranks using BM25 over candidate texts."""
    candidates, texts_by_id = _make_candidates_and_texts()

    rr = Reranker(mode="lexical")

    results = rr.rerank("cat sat mat", candidates, texts_by_id, k=4)

    assert len(results) == 4
    # d0 contains all three query terms; should rank first
    assert results[0][0] == "d0"


# ------------------------------------------------------------------
# test_cross_encoder_rerank
# ------------------------------------------------------------------


def test_cross_encoder_rerank():
    """mode='cross_encoder' uses a dummy scoring function."""
    candidates, texts_by_id = _make_candidates_and_texts()

    # Dummy cross-encoder: score based on word overlap
    def dummy_cross_encoder(query: str, doc: str) -> float:
        q_words = set(query.lower().split())
        d_words = set(doc.lower().split())
        overlap = len(q_words & d_words)
        return float(overlap) / max(len(q_words), 1)

    rr = Reranker(mode="cross_encoder")
    rr.attach_cross_encoder(dummy_cross_encoder)

    results = rr.rerank("the cat sat on the mat", candidates, texts_by_id, k=4)

    assert len(results) == 4
    # d0 is the exact match, should score highest
    assert results[0][0] == "d0"
    # Scores should be in descending order
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)


# ------------------------------------------------------------------
# test_hybrid_rerank
# ------------------------------------------------------------------


def test_hybrid_rerank():
    """mode='hybrid' fuses dense and lexical ranks."""
    candidates, texts_by_id = _make_candidates_and_texts()

    rr = Reranker(mode="hybrid", alpha=0.5)
    rr.attach_dense(dummy_embed)

    results = rr.rerank("the cat sat on the mat", candidates, texts_by_id, k=4)

    assert len(results) == 4
    # d0 should still rank at or near the top with hybrid fusion
    result_ids = [r[0] for r in results]
    assert "d0" in result_ids[:2]

    # Should return all 4 candidates
    assert set(result_ids) == {"d0", "d1", "d2", "d3"}
