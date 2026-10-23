"""Prompt handling for Grounding DINO: normalise, tokenize, phrase masks, padding.

Grounding DINO reads a caption such as "person . red car . dog ." The text
encoder sees every phrase on its own: tokens attend only within their phrase
(block-diagonal mask) and position ids restart at each phrase. This module
reproduces `generate_masks_with_special_tokens_and_transfer_map` from
transformers exactly, and pads the result to a fixed token count T for the MLA.

Padding is exact for the real tokens: a pad token is not a special token, so it
gets an identity mask row (it only sees itself), no real token attends to it,
and every later use of the text (fusion, decoder text attention, contrastive
head) masks it with the token mask.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SPECIAL_TOKENS = (101, 102, 1012, 1029)  # [CLS], [SEP], ".", "?"
CLS, SEP, DOT, PAD = 101, 102, 1012, 0
MASK_NEG = -1e4  # additive mask; exp(-1e4) is exactly 0 in bf16/fp32


def normalize(text: str) -> str:
    """Lower-case, split on '.', ',', ';' or newlines, rejoin as 'a . b .'"""
    parts = [p.strip() for p in re.split(r"[.,;\n]+", text.lower())]
    parts = [re.sub(r"\s+", " ", p) for p in parts if p]
    return " . ".join(parts) + " ." if parts else ""


@dataclass
class Prompt:
    caption: str                     # normalised caption fed to the tokenizer
    phrases: list[str]               # phrases in order
    input_ids: np.ndarray            # (T,) int64, padded with 0
    n_tokens: int                    # real tokens incl. [CLS]/[SEP]
    token_mask: np.ndarray           # (T,) bool, True for real tokens
    self_mask: np.ndarray            # (T, T) bool, True = may attend
    position_ids: np.ndarray         # (T,) int64
    phrase_of_token: np.ndarray      # (T,) int, phrase index or -1
    truncated: list[str] = field(default_factory=list)  # phrases dropped to fit T

    def additive_self_mask(self) -> np.ndarray:
        return np.where(self.self_mask, 0.0, MASK_NEG).astype(np.float32)

    def additive_token_mask(self) -> np.ndarray:
        return np.where(self.token_mask, 0.0, MASK_NEG).astype(np.float32)


def special_masks(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """numpy port of transformers' generate_masks_with_special_tokens_and_transfer_map (batch 1)."""
    n = len(ids)
    special = np.isin(ids, SPECIAL_TOKENS)
    mask = np.eye(n, dtype=bool)
    pos = np.zeros(n, np.int64)
    prev = 0
    for col in np.nonzero(special)[0]:
        if col == 0 or col == n - 1:
            mask[col, col] = True
            pos[col] = 0
        else:
            mask[prev + 1: col + 1, prev + 1: col + 1] = True
            pos[prev + 1: col + 1] = np.arange(0, col - prev)
        prev = col
    return mask, pos


class WordPiece:
    """Pure-Python BERT (uncased) tokenizer: basic tokenization + greedy WordPiece.

    Used where the `tokenizers` package is missing (the DevKit). Matches
    BertTokenizer(do_lower_case=True, strip_accents=None, tokenize_chinese_chars=True).
    """

    def __init__(self, vocab_path: str | Path):
        self.vocab = {}
        with open(vocab_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                self.vocab[line.rstrip("\n")] = i
        self.inv = {i: t for t, i in self.vocab.items()}
        self.unk = self.vocab["[UNK]"]

    @staticmethod
    def _is_punct(ch):
        import unicodedata
        cp = ord(ch)
        if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
            return True
        return unicodedata.category(ch).startswith("P")

    @staticmethod
    def _is_cjk(cp):
        return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF
                or 0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF
                or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)

    def _basic(self, text):
        import unicodedata
        out = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or unicodedata.category(ch) in ("Cc", "Cf") and ch not in "\t\n\r":
                continue
            out.append(f" {ch} " if self._is_cjk(cp) else (" " if ch.isspace() else ch))
        words = []
        for w in "".join(out).lower().split():
            w = "".join(c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) != "Mn")
            cur = ""
            for ch in w:
                if self._is_punct(ch):
                    if cur:
                        words.append(cur)
                        cur = ""
                    words.append(ch)
                else:
                    cur += ch
            if cur:
                words.append(cur)
        return words

    def _wordpiece(self, word):
        if len(word) > 100:
            return [self.unk]
        ids, start = [], 0
        while start < len(word):
            end, cur = len(word), None
            while start < end:
                sub = word[start:end] if start == 0 else "##" + word[start:end]
                if sub in self.vocab:
                    cur = self.vocab[sub]
                    break
                end -= 1
            if cur is None:
                return [self.unk]
            ids.append(cur)
            start = end
        return ids

    def encode(self, text: str) -> list[int]:
        ids = [CLS]
        for w in self._basic(text):
            ids += self._wordpiece(w)
        return ids + [SEP]

    def decode(self, ids) -> str:
        toks = [self.inv.get(int(i), "[UNK]") for i in ids if int(i) not in (CLS, SEP, PAD)]
        out = ""
        for t in toks:
            out += t[2:] if t.startswith("##") else (" " + t if out else t)
        return out


class Tokenizer:
    """BERT WordPiece: the `tokenizers` package when installed, else the pure-Python WordPiece.

    `path` is tokenizer.json or a directory holding tokenizer.json / vocab.txt.
    """

    def __init__(self, path: str | Path | None = None, pure_python: bool = False):
        if path is None:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download("IDEA-Research/grounding-dino-base", "tokenizer.json")).parent
        path = Path(path)
        folder = path if path.is_dir() else path.parent
        self.tok = None
        if not pure_python:
            try:
                from tokenizers import Tokenizer as HFTok
                self.tok = HFTok.from_file(str(folder / "tokenizer.json"))
                self.tok.no_padding()
                self.tok.no_truncation()
            except ImportError:
                self.tok = None
        self.wp = None if self.tok is not None else WordPiece(folder / "vocab.txt")

    def encode(self, text: str) -> list[int]:
        if self.tok is not None:
            return self.tok.encode(text).ids          # adds [CLS] ... [SEP]
        return self.wp.encode(text)

    def decode(self, ids) -> str:
        if self.tok is not None:
            return self.tok.decode(list(ids))
        return self.wp.decode(ids)


def build(text: str, tok: Tokenizer, T: int = 64, normalise: bool = True) -> Prompt:
    """Tokenize a caption into a fixed-length prompt, dropping whole phrases that don't fit."""
    caption = normalize(text) if normalise else text
    phrases = [p.strip() for p in caption.split(" .") if p.strip()]
    kept, dropped = [], []
    ids = [CLS]
    for ph in phrases:
        piece = tok.encode(ph)[1:-1] + [DOT]
        if len(ids) + len(piece) + 1 <= T and not dropped:
            ids += piece
            kept.append(ph)
        else:
            dropped.append(ph)
    ids.append(SEP)
    # Reference tokenization of the kept caption must match piecewise tokenization.
    kept_caption = " . ".join(kept) + " ." if kept else ""
    n = len(ids)
    full = np.zeros(T, np.int64)
    full[:n] = ids
    m, pos = special_masks(full[:n])
    self_mask = np.eye(T, dtype=bool)
    self_mask[:n, :n] = m
    position_ids = np.zeros(T, np.int64)
    position_ids[:n] = pos
    token_mask = np.zeros(T, bool)
    token_mask[:n] = True
    phrase_of = -np.ones(T, int)
    p, i = 0, 1
    for ph in kept:
        k = len(tok.encode(ph)) - 2
        phrase_of[i:i + k] = p
        i += k + 1
        p += 1
    return Prompt(kept_caption, kept, full, n, token_mask, self_mask, position_ids, phrase_of, dropped)


def sine_embed_1d(pos: np.ndarray, num_feats: int = 256, temperature: float = 10000.0) -> np.ndarray:
    """get_sine_pos_embed for a scalar position (text position ids): (T,) -> (T, num_feats)."""
    scale = 2 * np.pi
    dim_t = np.arange(num_feats, dtype=np.float64)
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)
    x = pos.astype(np.float64)[:, None] * scale / dim_t
    out = np.empty_like(x)
    out[:, 0::2] = np.sin(x[:, 0::2])
    out[:, 1::2] = np.cos(x[:, 1::2])
    return out.astype(np.float32)
