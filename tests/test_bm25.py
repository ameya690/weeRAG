"""Tests for wee.bm25.BM25."""
from wee.bm25 import BM25


# ------------------------------------------------------------------
# test_basic_search
# ------------------------------------------------------------------


def test_basic_search():
    """Add docs, search for a term, verify ranking makes sense."""
    bm = BM25()
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "a fox is a cunning animal",
        "the dog barked at the mailman",
        "quick quick quick fox fox fox",  # high TF for "quick" and "fox"
    ]
    bm.add(corpus)

    results = bm.search("quick fox", k=4)

    # Should return results
    assert len(results) > 0

    # Scores should be in descending order
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)

    # Doc 3 (index 3) has the highest TF for both "quick" and "fox"
    # It should rank highly
    top_ids = [doc_id for doc_id, _ in results]
    assert 3 in top_ids[:2], f"Expected doc 3 in top 2, got {top_ids}"

    # Doc 2 mentions neither "quick" nor "fox" -- shouldn't be top
    if 2 in top_ids:
        assert top_ids.index(2) > top_ids.index(3)


# ------------------------------------------------------------------
# test_empty
# ------------------------------------------------------------------


def test_empty():
    """Search on an empty corpus returns no results."""
    bm = BM25()
    results = bm.search("hello world", k=5)
    assert results == []


# ------------------------------------------------------------------
# test_single_doc
# ------------------------------------------------------------------


def test_single_doc():
    """Single document retrieval should return that document."""
    bm = BM25()
    bm.add(["machine learning is fascinating"])

    results = bm.search("machine learning", k=5)
    assert len(results) == 1
    assert results[0][0] == 0  # doc_id
    assert results[0][1] > 0   # positive score
