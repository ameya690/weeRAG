import re
from typing import List, Dict, Tuple, Iterable, Optional
from .tokenizer import Tokenizer

_SENT_SPLIT = re.compile(r'(?<=[\.!?])\s+')

def chunk_by_words(text: str, max_words: int = 200, overlap: int = 20) -> List[str]:
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

def chunk_by_sentences(text: str, max_chars: int = 1000, overlap: int = 100) -> List[str]:
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

def chunk_by_tokens(text: str, tokenizer: Tokenizer, max_tokens: int = 256, overlap: int = 32) -> List[str]:
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
