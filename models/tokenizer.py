# models/tokenizer.py
# WordPiece tokenizer built completely from scratch (no HuggingFace or external libraries)
# Uses only Python standard library: collections, re, os, json

import re
import json
import os
from collections import defaultdict


# ── Special tokens ────────────────────────────────────────────────────────────
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
PAD_ID  = 0
UNK_ID  = 1
CLS_ID  = 2
SEP_ID  = 3
MASK_ID = 4


class Encoding:
    """
    Thin wrapper so tokenizer.encode(text).ids works
    the same way as HuggingFace tokenizers — used throughout trainer/evaluate.
    """
    def __init__(self, ids: list):
        self.ids = ids


class WordPieceTokenizer:
    """
    WordPiece tokenizer built from scratch.

    Training:
      1. Count word frequencies in corpus.
      2. Initialize vocabulary with all unique characters (prefixed with '##'
         for non-initial positions) plus special tokens.
      3. Repeatedly find the pair of adjacent subwords whose merge most
         increases the likelihood of the corpus (approximated by frequency),
         merge them, and add to vocab — until vocab_size is reached.

    Tokenization (inference):
      For each word, use greedy longest-match from the vocabulary:
        - Try to match the longest prefix from vocab as the first token.
        - For the remainder, try longest match prefixed with '##'.
        - If no match found at any position, return [UNK].

    NOTE:
      - encode() returns an Encoding object with a .ids attribute (list of ints),
        NOT including [CLS] / [SEP]. Callers add those manually.
      - tokenize() returns a full token list WITH [CLS] / [SEP] for human inspection.
    """

    def __init__(self):
        self.vocab: dict = {}
        self.inv_vocab: dict = {}
        self.vocab_size: int = 0

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, corpus: list, vocab_size: int = 1000):
        print("  [Tokenizer] Counting word frequencies...")
        word_freqs = self._count_word_frequencies(corpus)

        print("  [Tokenizer] Initializing character vocabulary...")
        splits = {
            word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
            for word in word_freqs
        }

        vocab = list(SPECIAL_TOKENS)
        for word in splits:
            for token in splits[word]:
                if token not in vocab:
                    vocab.append(token)

        print(f"  [Tokenizer] Initial vocab size: {len(vocab)} | Target: {vocab_size}")

        while len(vocab) < vocab_size:
            pair_freqs = self._compute_pair_frequencies(splits, word_freqs)
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)
            a, b = best_pair
            new_token = a + (b[2:] if b.startswith("##") else b)
            vocab.append(new_token)
            splits = self._merge_pair(splits, a, b, new_token)

        self.vocab     = {token: idx for idx, token in enumerate(vocab)}
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        print(f"  [Tokenizer] Training complete. Final vocab size: {self.vocab_size}")

    def _count_word_frequencies(self, corpus: list) -> dict:
        freqs = defaultdict(int)
        for sentence in corpus:
            for word in re.findall(r"[a-zA-Z]+", sentence.lower()):
                freqs[word] += 1
        return dict(freqs)

    def _compute_pair_frequencies(self, splits: dict, word_freqs: dict) -> dict:
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            tokens = splits[word]
            for i in range(len(tokens) - 1):
                pair_freqs[(tokens[i], tokens[i + 1])] += freq
        return dict(pair_freqs)

    def _merge_pair(self, splits: dict, a: str, b: str, new_token: str) -> dict:
        new_splits = {}
        for word, tokens in splits.items():
            merged = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    merged.append(new_token)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            new_splits[word] = merged
        return new_splits

    # ── Tokenization (inference) ───────────────────────────────────────────────

    def tokenize(self, text: str) -> list:
        """Returns full token list WITH [CLS] and [SEP] — for human inspection."""
        tokens = ["[CLS]"]
        for word in re.findall(r"[a-zA-Z]+", text.lower()):
            tokens.extend(self._tokenize_word(word))
        tokens.append("[SEP]")
        return tokens

    def _tokenize_word(self, word: str) -> list:
        tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            found = None
            while start < end:
                substr = word[start:end]
                candidate = substr if start == 0 else f"##{substr}"
                if candidate in self.vocab:
                    found = candidate
                    break
                end -= 1
            if found is None:
                return ["[UNK]"]
            tokens.append(found)
            start = end
        return tokens

    def encode(self, text: str) -> "Encoding":
        """
        Encode text to token IDs — WITHOUT [CLS] / [SEP].
        Returns an Encoding object so callers can use .ids (HuggingFace-style).
        Callers (trainer, evaluate) prepend/append CLS_ID and SEP_ID themselves.
        """
        words = re.findall(r"[a-zA-Z]+", text.lower())
        tokens = []
        for word in words:
            tokens.extend(self._tokenize_word(word))
        ids = [self.vocab.get(t, UNK_ID) for t in tokens]
        return Encoding(ids)

    def decode(self, ids: list) -> str:
        tokens = [self.inv_vocab.get(i, "[UNK]") for i in ids]
        result = []
        for token in tokens:
            if token in SPECIAL_TOKENS:
                continue
            if token.startswith("##"):
                if result:
                    result[-1] += token[2:]
                else:
                    result.append(token[2:])
            else:
                result.append(token)
        return " ".join(result)

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        print(f"  [Tokenizer] Saved to {path}")

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.inv_vocab  = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        print(f"  [Tokenizer] Loaded. Vocab size: {self.vocab_size}")


def train_tokenizer(corpus: list, vocab_size: int = 1000, save_path: str = "tokenizer_output"):
    tok = WordPieceTokenizer()
    tok.train(corpus, vocab_size=vocab_size)
    os.makedirs(save_path, exist_ok=True)
    tok.save(os.path.join(save_path, "tokenizer.json"))
    return tok


def load_tokenizer(save_path: str = "tokenizer_output"):
    tok = WordPieceTokenizer()
    tok.load(os.path.join(save_path, "tokenizer.json"))
    return tok


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.corpus import load_corpus, preprocess
    corpus = preprocess(load_corpus())
    tok = WordPieceTokenizer()
    tok.train(corpus, vocab_size=500)
    test = "Diabetes is treated by Insulin"
    print(f"Tokens:  {tok.tokenize(test)}")
    enc = tok.encode(test)
    print(f"IDs:     {enc.ids}")
    print(f"Decoded: {tok.decode(enc.ids)}")
