from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np

from .tokenizer import Tokenizer

_SENT_SPLIT = re.compile(r'(?<=[\.!?])\s+')

def chunk_by_words(text: str, max_words: int = 200, overlap: int = 20) -> list[str]:
    """
    Simple whitespace word chunking with optional overlap (by words).
    Returns list of text chunks.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    step = max(1, max_words - overlap)
    while i < len(words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
        i += step
    return chunks

def chunk_by_sentences(text: str, max_chars: int = 1000, overlap: int = 100) -> list[str]:
    """
    Sentence-aware chunking: pack sentences into chunks up to max_chars with overlap.
    """
    sents = _SENT_SPLIT.split(text.strip())
    if not sents or sents == [""]:
        return []
    chunks = []
    cur = ""
    for s in sents:
        if not cur:
            cur = s.strip()
            continue
        if len(cur) + 1 + len(s) <= max_chars:
            cur = cur + " " + s.strip()
        else:
            chunks.append(cur)
            if overlap > 0:
                # keep a tail of the previous chunk
                cur_tail = cur[-overlap:]
                cur = (cur_tail + " " + s.strip()).strip()
            else:
                cur = s.strip()
    if cur:
        chunks.append(cur)
    return chunks

def chunk_by_tokens(text: str, tokenizer: Tokenizer, max_tokens: int = 256, overlap: int = 32) -> list[str]:
    """
    Token-budgeted chunking using wee.Tokenizer. Chunks are decoded back to text.
    """
    ids = tokenizer.encode(text, add_special=False)
    if not ids:
        return []
    chunks = []
    step = max(1, max_tokens - overlap)
    for i in range(0, len(ids), step):
        window = ids[i:i+max_tokens]
        chunks.append(tokenizer.decode(window))
        if i + max_tokens >= len(ids):
            break
    return chunks


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    a = a.ravel()
    b = b.ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def chunk_by_semantic(
    text: str,
    embed_fn: Callable[[str], np.ndarray],
    threshold: float = 0.5,
    min_chunk_size: int = 50,
) -> list[str]:
    """Semantic chunking: split where consecutive-sentence similarity drops below *threshold*.

    Parameters
    ----------
    text : str
        The document to chunk.
    embed_fn : Callable[[str], np.ndarray]
        A function that maps a string to a 1-D embedding vector.
    threshold : float
        Cosine-similarity cutoff.  A split is placed between sentence *i*
        and *i+1* whenever their similarity is below this value.
    min_chunk_size : int
        Minimum character count per chunk.  Chunks smaller than this are
        merged into the following chunk.
    """
    sentences = _SENT_SPLIT.split(text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [sentences[0]]

    # Embed every sentence
    embeddings = [np.asarray(embed_fn(s), dtype=np.float32) for s in sentences]

    # Find breakpoints where similarity drops below threshold
    breakpoints: list[int] = []
    for i in range(len(sentences) - 1):
        sim = _cosine_sim(embeddings[i], embeddings[i + 1])
        if sim < threshold:
            breakpoints.append(i + 1)  # split *before* sentence i+1

    # Build raw chunks from breakpoints
    raw_chunks: list[str] = []
    prev = 0
    for bp in breakpoints:
        raw_chunks.append(" ".join(sentences[prev:bp]))
        prev = bp
    raw_chunks.append(" ".join(sentences[prev:]))

    # Enforce min_chunk_size by merging tiny chunks forward
    merged: list[str] = []
    buf = ""
    for chunk in raw_chunks:
        if buf:
            buf = buf + " " + chunk
        else:
            buf = chunk
        if len(buf) >= min_chunk_size:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] = merged[-1] + " " + buf
        else:
            merged.append(buf)

    return merged


def chunk_with_parents(
    text: str,
    parent_chunk_fn: Callable[[str], list[str]] | None = None,
    child_chunk_fn: Callable[[str], list[str]] | None = None,
) -> list[dict]:
    """Create a two-level chunk hierarchy (parent + child).

    Parameters
    ----------
    text : str
        The document to chunk.
    parent_chunk_fn : callable, optional
        Splits *text* into coarse parent chunks.
        Defaults to ``chunk_by_sentences(text, max_chars=2000)``.
    child_chunk_fn : callable, optional
        Splits a single *parent* chunk into finer child chunks.
        Defaults to ``chunk_by_sentences(parent, max_chars=500)``.

    Returns
    -------
    list[dict]
        Each element has the shape::

            {
                "parent": str,
                "parent_id": int,
                "children": [{"text": str, "child_id": int}, ...]
            }

    The intended workflow: embed and retrieve on the *child* chunks (which
    are more specific), but return the *parent* chunk (which carries more
    context) to the LLM.
    """
    if parent_chunk_fn is None:
        def parent_chunk_fn(t):
            return chunk_by_sentences(t, max_chars=2000)
    if child_chunk_fn is None:
        def child_chunk_fn(p):
            return chunk_by_sentences(p, max_chars=500)

    parents = parent_chunk_fn(text)
    hierarchy: list[dict] = []
    for pid, parent in enumerate(parents):
        children_texts = child_chunk_fn(parent)
        children = [
            {"text": ct, "child_id": cid}
            for cid, ct in enumerate(children_texts)
        ]
        hierarchy.append({
            "parent": parent,
            "parent_id": pid,
            "children": children,
        })
    return hierarchy
