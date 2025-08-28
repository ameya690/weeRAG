\
import json
import re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Iterable, Optional

BOUNDARY = "▁"  # word boundary marker (SentencePiece style)

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

def _preprocess(text: str) -> List[List[str]]:
    """
    Convert text into a list of words, each represented as a list of characters
    with a leading BOUNDARY symbol to mark a space.
    """
    words = []
    for w in re.findall(r"\S+", text):
        # Keep punctuation as part of the word; BPE will learn merges
        words.append([BOUNDARY] + list(w))
    return words

def _get_stats(words: List[List[str]], counts: Counter) -> Counter:
    """
    Count frequency of adjacent symbol pairs across the corpus.
    Each word contributes its corpus frequency (counts key).
    """
    pairs = Counter()
    for word, c in zip(words, counts.values()):  # counts aligned with words order
        for i in range(len(word) - 1):
            pairs[(word[i], word[i+1])] += c
    return pairs

def _merge_pair(pair: Tuple[str, str], words: List[List[str]]) -> List[List[str]]:
    """
    Merge all occurrences of `pair` in the words (greedy, left-to-right).
    """
    a, b = pair
    merged_symbol = a + b
    new_words = []
    for w in words:
        i = 0
        new_w = []
        while i < len(w):
            if i < len(w) - 1 and w[i] == a and w[i+1] == b:
                new_w.append(merged_symbol)
                i += 2
            else:
                new_w.append(w[i])
                i += 1
        new_words.append(new_w)
    return new_words

class Tokenizer:
    """
    Tiny SentencePiece-like char-BPE tokenizer.
    - Train from raw text (list[str]) to a vocab of size N.
    - Encode/decode deterministically.
    - Save/load to JSON.
    Notes:
      * Uses '▁' to mark spaces. Decoding replaces it with ' '.
      * Robust to arbitrary Unicode (operates on Python characters).
    """
    def __init__(self):
        self.merges: List[Tuple[str, str]] = []
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.vocab_size: int = 0

    def train(self, corpus: Iterable[str], vocab_size: int = 2000, min_pair_freq: int = 2):
        # Build initial word list and counts (word types + frequency)
        counter = Counter()
        word_types = []
        for line in corpus:
            words = _preprocess(line)
            for w in words:
                word_types.append(tuple(w))
                counter[tuple(w)] += 1
        # unique word forms
        unique_words = [list(w) for w in counter.keys()]
        counts = Counter(counter.values())  # aligned by iteration order below

        # initial symbols are all characters that appear
        symbols = Counter()
        for w, c in counter.items():
            for ch in w:
                symbols[ch] += c
        vocab = set(symbols.keys())

        merges: List[Tuple[str, str]] = []

        # keep merging most frequent pairs
        while True:
            # compute pair stats weighted by word frequencies; align counts order
            # turn counts into list matching unique_words order
            aligned_counts = Counter()
            for w, c in counter.items():
                aligned_counts[w] = c
            pairs = Counter()
            for w in unique_words:
                # use corpus count for this word
                c = aligned_counts[tuple(w)]
                for i in range(len(w)-1):
                    pairs[(w[i], w[i+1])] += c

            if not pairs:
                break
            (a, b), freq = pairs.most_common(1)[0]
            if freq < min_pair_freq:
                break
            if len(vocab) + 1 >= vocab_size:
                break
            # merge
            unique_words = _merge_pair((a, b), unique_words)
            merges.append((a, b))
            vocab.add(a + b)

        # finalize vocab with special tokens first
        all_tokens = list(SPECIAL_TOKENS) + sorted(vocab, key=lambda s: (len(s), s))
        self.token2id = {tok: i for i, tok in enumerate(all_tokens)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}
        self.merges = merges
        self.vocab_size = len(all_tokens)

    def _apply_merges(self, symbols: List[str]) -> List[str]:
        # Apply merges in learned order
        for a, b in self.merges:
            i = 0
            out = []
            while i < len(symbols):
                if i < len(symbols)-1 and symbols[i] == a and symbols[i+1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            symbols = out
        return symbols

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        words = _preprocess(text)  # list of [▁, c1, c2, ...]
        tokens: List[str] = []
        for w in words:
            sym = self._apply_merges(w[:])
            tokens.extend(sym)
        ids = []
        if add_special:
            ids.append(self.token2id.get("<bos>", 0))
        for t in tokens:
            ids.append(self.token2id.get(t, self.token2id.get("<unk>", 1)))
        if add_special:
            ids.append(self.token2id.get("<eos>", 3))
        return ids

    def decode(self, ids: List[int]) -> str:
        toks = [self.id2token.get(i, "<unk>") for i in ids]
        # strip special tokens
        toks = [t for t in toks if t not in SPECIAL_TOKENS]
        s = "".join(toks)
        # convert boundary back to space
        s = s.replace(BOUNDARY, " ")
        # normalize extra spaces
        return re.sub(r"\s+", " ", s).strip()

    def save(self, path: str):
        data = {
            "merges": self.merges,
            "token2id": self.token2id,
            "vocab_size": self.vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.merges = [tuple(x) for x in data["merges"]]
        tok.token2id = {k: int(v) for k, v in data["token2id"].items()}
        tok.id2token = {i: t for t, i in tok.token2id.items()}
        tok.vocab_size = int(data["vocab_size"])
        return tok
