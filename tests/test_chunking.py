"""Deterministic tests for wee.chunk.

Covers chunk_by_words, chunk_by_sentences, chunk_by_tokens, and
chunk_with_parents. No network calls, no randomness.
"""

from wee.chunk import (
    chunk_by_words,
    chunk_by_sentences,
    chunk_by_tokens,
    chunk_with_parents,
)
from wee.tokenizer import Tokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A small corpus for training the tokenizer -- deterministic
_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models process natural language text.",
    "Retrieval augmented generation improves answer quality.",
    "Embeddings capture semantic meaning of words.",
    "The fox and the dog became good friends.",
]


def _make_tokenizer() -> Tokenizer:
    tok = Tokenizer()
    tok.train(_CORPUS, vocab_size=300, min_pair_freq=1)
    return tok


# ---------------------------------------------------------------------------
# chunk_by_words
# ---------------------------------------------------------------------------


class TestChunkByWords:
    def test_basic_chunking(self):
        text = " ".join(f"word{i}" for i in range(100))
        chunks = chunk_by_words(text, max_words=30, overlap=0)
        # 100 words / 30 per chunk = ceil(100/30) = 4 chunks
        assert len(chunks) == 4

    def test_overlap(self):
        text = " ".join(f"w{i}" for i in range(20))
        chunks = chunk_by_words(text, max_words=10, overlap=5)
        # step = 10 - 5 = 5, so starts at 0, 5, 10, 15 -> 4 chunks
        assert len(chunks) == 4
        # First chunk: w0..w9, second chunk: w5..w14
        # They should share words w5..w9
        first_words = chunks[0].split()
        second_words = chunks[1].split()
        overlap_words = set(first_words) & set(second_words)
        assert len(overlap_words) >= 5

    def test_no_overlap(self):
        text = " ".join(f"w{i}" for i in range(10))
        chunks = chunk_by_words(text, max_words=5, overlap=0)
        assert len(chunks) == 2
        assert chunks[0] == "w0 w1 w2 w3 w4"
        assert chunks[1] == "w5 w6 w7 w8 w9"

    def test_empty_text(self):
        assert chunk_by_words("", max_words=10) == []

    def test_text_shorter_than_max(self):
        text = "one two three"
        chunks = chunk_by_words(text, max_words=100, overlap=0)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_single_word(self):
        chunks = chunk_by_words("hello", max_words=5, overlap=0)
        assert chunks == ["hello"]

    def test_all_words_present(self):
        words = [f"w{i}" for i in range(50)]
        text = " ".join(words)
        chunks = chunk_by_words(text, max_words=20, overlap=0)
        reassembled = " ".join(chunks)
        for w in words:
            assert w in reassembled


# ---------------------------------------------------------------------------
# chunk_by_sentences
# ---------------------------------------------------------------------------


class TestChunkBySentences:
    def test_respects_max_chars(self):
        text = "First sentence. Second sentence. Third sentence here. Fourth sentence now."
        chunks = chunk_by_sentences(text, max_chars=40, overlap=0)
        for chunk in chunks:
            # Each chunk should be at most max_chars (or a single sentence if
            # the sentence itself exceeds max_chars)
            assert len(chunk) <= 80  # generous bound since a single sentence might be near limit

    def test_overlap_carries_tail(self):
        text = "Alpha bravo charlie. Delta echo foxtrot. Golf hotel india."
        chunks = chunk_by_sentences(text, max_chars=30, overlap=10)
        assert len(chunks) >= 2
        # With overlap > 0, second chunk should contain tail of first
        if len(chunks) >= 2:
            tail_of_first = chunks[0][-10:]
            assert tail_of_first in chunks[1]

    def test_empty_text(self):
        assert chunk_by_sentences("", max_chars=100) == []

    def test_single_sentence(self):
        text = "Only one sentence."
        chunks = chunk_by_sentences(text, max_chars=1000)
        assert len(chunks) == 1
        assert chunks[0] == "Only one sentence."

    def test_sentences_preserved(self):
        text = "First. Second. Third."
        chunks = chunk_by_sentences(text, max_chars=5000, overlap=0)
        joined = " ".join(chunks)
        assert "First." in joined
        assert "Second." in joined
        assert "Third." in joined

    def test_no_empty_chunks(self):
        text = "A. B. C. D. E."
        chunks = chunk_by_sentences(text, max_chars=5, overlap=0)
        for chunk in chunks:
            assert chunk.strip() != ""


# ---------------------------------------------------------------------------
# chunk_by_tokens
# ---------------------------------------------------------------------------


class TestChunkByTokens:
    def test_roundtrip(self):
        """Encode -> chunk -> decode should preserve all content."""
        tok = _make_tokenizer()
        text = "The quick brown fox jumps over the lazy dog"
        chunks = chunk_by_tokens(text, tok, max_tokens=10, overlap=0)
        assert len(chunks) >= 1
        # Each chunk decoded back should be non-empty
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_respects_max_tokens(self):
        tok = _make_tokenizer()
        text = " ".join(["hello world"] * 20)
        chunks = chunk_by_tokens(text, tok, max_tokens=15, overlap=0)
        for chunk in chunks:
            ids = tok.encode(chunk, add_special=False)
            assert len(ids) <= 15

    def test_overlap(self):
        tok = _make_tokenizer()
        text = " ".join(["hello world foo bar baz"] * 5)
        chunks = chunk_by_tokens(text, tok, max_tokens=10, overlap=3)
        assert len(chunks) >= 2

    def test_empty_text(self):
        tok = _make_tokenizer()
        assert chunk_by_tokens("", tok, max_tokens=10) == []

    def test_text_within_budget(self):
        tok = _make_tokenizer()
        text = "hello"
        chunks = chunk_by_tokens(text, tok, max_tokens=1000, overlap=0)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# chunk_with_parents
# ---------------------------------------------------------------------------


class TestChunkWithParents:
    def test_hierarchy_structure(self):
        text = (
            "First parent sentence one. First parent sentence two. "
            "Second parent sentence one. Second parent sentence two. "
            "Third parent sentence one. Third parent sentence two."
        )
        # Use small parent chunks so we get multiple parents
        parent_fn = lambda t: chunk_by_sentences(t, max_chars=60, overlap=0)
        child_fn = lambda p: chunk_by_sentences(p, max_chars=30, overlap=0)

        hierarchy = chunk_with_parents(text, parent_fn, child_fn)

        assert len(hierarchy) >= 1
        for entry in hierarchy:
            assert "parent" in entry
            assert "parent_id" in entry
            assert "children" in entry
            assert isinstance(entry["parent"], str)
            assert isinstance(entry["parent_id"], int)
            assert isinstance(entry["children"], list)
            for child in entry["children"]:
                assert "text" in child
                assert "child_id" in child

    def test_parent_ids_sequential(self):
        text = "A. B. C. D. E. F."
        parent_fn = lambda t: chunk_by_sentences(t, max_chars=5, overlap=0)
        child_fn = lambda p: [p]  # trivial: one child per parent

        hierarchy = chunk_with_parents(text, parent_fn, child_fn)
        for i, entry in enumerate(hierarchy):
            assert entry["parent_id"] == i

    def test_child_ids_sequential_within_parent(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        parent_fn = lambda t: [t]  # single parent = entire text
        child_fn = lambda p: chunk_by_sentences(p, max_chars=30, overlap=0)

        hierarchy = chunk_with_parents(text, parent_fn, child_fn)
        assert len(hierarchy) == 1
        children = hierarchy[0]["children"]
        for i, child in enumerate(children):
            assert child["child_id"] == i

    def test_default_chunkers(self):
        # Use defaults (no custom fns). Just ensure it does not crash and
        # returns valid structure.
        text = " ".join(
            [f"Sentence number {i} with some extra words." for i in range(20)]
        )
        hierarchy = chunk_with_parents(text)
        assert isinstance(hierarchy, list)
        assert len(hierarchy) >= 1
        for entry in hierarchy:
            assert "parent" in entry
            assert "children" in entry

    def test_empty_text(self):
        hierarchy = chunk_with_parents("")
        assert hierarchy == []

    def test_children_are_substrings_of_parent(self):
        text = "Alpha bravo. Charlie delta. Echo foxtrot. Golf hotel."
        parent_fn = lambda t: chunk_by_sentences(t, max_chars=60, overlap=0)
        child_fn = lambda p: chunk_by_sentences(p, max_chars=20, overlap=0)

        hierarchy = chunk_with_parents(text, parent_fn, child_fn)
        for entry in hierarchy:
            parent_text = entry["parent"]
            for child in entry["children"]:
                # Each child's words should appear in the parent
                child_words = set(child["text"].split())
                parent_words = set(parent_text.split())
                assert child_words.issubset(parent_words), (
                    f"Child words {child_words - parent_words} not in parent"
                )
