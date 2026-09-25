"""Okapi BM25 over an in-memory inverted index. No dependencies."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from .text import tokenize


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)  # term -> [(doc, tf)]

    def add(self, text: str) -> int:
        doc_id = len(self.doc_len)
        tokens = tokenize(text)
        self.doc_len.append(len(tokens))
        for term, tf in Counter(tokens).items():
            self.postings[term].append((doc_id, tf))
        return doc_id

    def __len__(self) -> int:
        return len(self.doc_len)

    def idf(self, term: str) -> float:
        # BM25+ style smoothing keeps idf positive even for terms in most documents.
        n = len(self.doc_len)
        df = len(self.postings.get(term, ()))
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Return [(doc_id, score)] best first. Duplicate query terms count once."""
        if not self.doc_len:
            return []
        avgdl = sum(self.doc_len) / len(self.doc_len) or 1.0
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for doc_id, tf in postings:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / avgdl)
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + norm)
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
