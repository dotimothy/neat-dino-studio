"""CLIP's byte-level BPE tokenizer in plain Python (no `tokenizers` / `regex` packages needed).

Follows OpenAI's simple_tokenizer (used by SAM 3's text encoder): lowercase, collapse
whitespace, split into words / digits / punctuation, BPE each word with </w> marking its end,
then <|startoftext|> ... <|endoftext|>.
"""
from __future__ import annotations

import html
import json
import re
from functools import lru_cache
from pathlib import Path


@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


# words (letters), single digits, contractions, other non-space runs; specials first
_PAT = re.compile(r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|[^\s\w]+|_+", re.IGNORECASE)


class ClipTokenizer:
    def __init__(self, vocab_json: str | Path, merges_txt: str | Path):
        self.encoder = json.loads(Path(vocab_json).read_text(encoding="utf-8"))
        merges = Path(merges_txt).read_text(encoding="utf-8").split("\n")
        merges = [tuple(m.split()) for m in merges if m and not m.startswith("#version")]
        self.ranks = {m: i for i, m in enumerate(merges) if len(m) == 2}
        self.byte_encoder = bytes_to_unicode()
        self.bos, self.eos = self.encoder["<|startoftext|>"], self.encoder["<|endoftext|>"]
        self.cache = {}

    def _bpe(self, token: str) -> list[str]:
        if token in self.cache:
            return self.cache[token]
        word = list(token[:-1]) + [token[-1] + "</w>"]
        while len(word) > 1:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            a, b = best
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = out
        self.cache[token] = word
        return word

    def encode(self, text: str, max_tokens: int = 32) -> list[int]:
        text = html.unescape(html.unescape(text))
        text = re.sub(r"\s+", " ", text).strip().lower()
        ids = [self.bos]
        for tok in _PAT.findall(text):
            tok = "".join(self.byte_encoder[b] for b in tok.encode("utf-8"))
            ids.extend(self.encoder[t] for t in self._bpe(tok))
        ids = ids[: max_tokens - 1] + [self.eos]
        return ids
