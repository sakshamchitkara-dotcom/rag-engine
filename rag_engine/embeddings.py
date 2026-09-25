"""Dense vectors for semantic-ish retrieval.

Default is a feature-hashing vectorizer (word unigrams, word bigrams and in-word
character trigrams, signed hashing, log1p tf, L2-normalised). It needs no model
download, is deterministic and works offline; the character trigrams give it fuzzy
matching that BM25 lacks. If `sentence-transformers` is installed you can opt into a
real embedding model with `--embedder st:<model-name>`.
"""

from __future__ import annotations

import hashlib
import math
from array import array
from collections import Counter

from .text import tokenize

Vector = array  # array('f'), unit length


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
    return h % dim, (1.0 if (h >> 63) & 1 else -1.0)


class HashingEmbedder:
    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.name = f"hash:{dim}"

    def features(self, text: str) -> Counter:
        words = tokenize(text)
        feats: Counter = Counter()
        for w in words:
            feats["w:" + w] += 1.0
            padded = f"<{w}>"
            for i in range(len(padded) - 2):
                feats["c:" + padded[i : i + 3]] += 0.3
        for a, b in zip(words, words[1:]):
            feats[f"b:{a} {b}"] += 0.7
        return feats

    def embed(self, texts: list[str]) -> list[Vector]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for feat, weight in self.features(text).items():
                idx, sign = _bucket(feat, self.dim)
                vec[idx] += sign * math.log1p(weight)  # sublinear tf
            out.append(_normalise(vec))
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = SentenceTransformer(model_name)
        self.name = f"st:{model_name}"

    def embed(self, texts: list[str]) -> list[Vector]:
        return [_normalise(list(map(float, v))) for v in self.model.encode(texts)]


def _normalise(vec: list[float]) -> Vector:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return array("f", (x / norm for x in vec))


def cosine(a: Vector, b: Vector) -> float:
    """Dot product; inputs are already unit length."""
    return sum(x * y for x, y in zip(a, b))


def get_embedder(name: str = "hash:1024"):
    kind, _, arg = name.partition(":")
    if kind == "hash":
        return HashingEmbedder(int(arg or 1024))
    if kind == "st":
        try:
            return SentenceTransformerEmbedder(arg or "all-MiniLM-L6-v2")
        except ImportError as exc:
            raise SystemExit("sentence-transformers is not installed; use the default 'hash' embedder") from exc
    raise ValueError(f"unknown embedder {name!r} (expected 'hash[:dim]' or 'st:<model>')")
