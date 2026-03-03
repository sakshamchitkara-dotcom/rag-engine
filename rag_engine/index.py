"""Persistent chunk store (SQLite) with BM25, dense and hybrid (RRF) retrieval."""

from __future__ import annotations

import sqlite3
import threading
from array import array
from dataclasses import dataclass, field
from pathlib import Path

from .bm25 import BM25
from .chunking import Chunk, chunk_document
from .embeddings import cosine, get_embedder
from .loaders import Document
from .rerank import POOL, RERANKERS, mmr, proximity_score
from .text import tokenize

DEFAULT_EMBEDDER = "hash:1024"
RRF_K = 60
MODES = ("hybrid", "bm25", "dense")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    heading TEXT NOT NULL,
    text TEXT NOT NULL,
    position INTEGER NOT NULL,
    vector BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source);
"""


@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # retriever -> 1-based rank


def rrf(rankings: dict[str, list[int]], k: int = RRF_K) -> list[tuple[int, float, dict[str, int]]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank(d))."""
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, ranking in rankings.items():
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(doc, {})[name] = rank
    ordered = sorted(scores, key=lambda d: (-scores[d], d))
    return [(d, scores[d], ranks[d]) for d in ordered]


class Index:
    def __init__(self, path: str | Path, embedder: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript(_SCHEMA)
        stored = self._meta("embedder")
        if stored and embedder and stored != embedder:
            self.db.close()
            raise ValueError(
                f"index at {self.path} was built with embedder {stored!r}; "
                f"re-ingest with --reset to switch to {embedder!r}"
            )
        self.embedder_name = stored or embedder or DEFAULT_EMBEDDER
        self._embedder = None
        self._chunks: list[Chunk] | None = None
        self._vectors: list[array] = []
        self._bm25: BM25 | None = None
        self._lock = threading.Lock()  # lazy load may race under the threaded server

    # -- persistence ---------------------------------------------------------

    def _meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder(self.embedder_name)
        return self._embedder

    def reset(self) -> None:
        with self.db:
            self.db.execute("DELETE FROM chunks")
            self.db.execute("DELETE FROM meta")
        self._invalidate()

    def _invalidate(self) -> None:
        self._chunks, self._vectors, self._bm25 = None, [], None

    def add_documents(self, docs: list[Document], *, max_chars: int = 800, overlap: int = 150) -> int:
        """Chunk, embed and store documents; re-ingesting a source replaces its old chunks."""
        chunks = [
            c
            for doc in docs
            for c in chunk_document(doc.text, source=doc.source, title=doc.title,
                                    max_chars=max_chars, overlap=overlap)
        ]
        vectors = self.embedder.embed([c.indexed_text for c in chunks]) if chunks else []
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('embedder', ?)", (self.embedder_name,)
            )
            self.db.executemany("DELETE FROM chunks WHERE source = ?", [(d.source,) for d in docs])
            self.db.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (c.id, c.source, c.title, c.heading, c.text, c.position, v.tobytes())
                    for c, v in zip(chunks, vectors)
                ],
            )
        self._invalidate()
        return len(chunks)

    def _load(self) -> None:
        with self._lock:
            if self._chunks is None:
                self._load_locked()

    def _load_locked(self) -> None:
        rows = self.db.execute(
            "SELECT id, source, title, heading, text, position, vector FROM chunks ORDER BY source, position"
        ).fetchall()
        chunks, vectors, bm25 = [], [], BM25()
        # ponytail: BM25 is rebuilt in memory on load and dense search is brute force;
        # fine to ~100k chunks, move to a persisted inverted index / ANN beyond that.
        for cid, source, title, heading, text, position, blob in rows:
            chunk = Chunk(cid, source, title, heading, text, position)
            chunks.append(chunk)
            vec = array("f")
            vec.frombytes(blob)
            vectors.append(vec)
            bm25.add(chunk.indexed_text)
        self._vectors, self._bm25 = vectors, bm25
        self._chunks = chunks  # assigned last: it is the "loaded" flag

    @property
    def chunks(self) -> list[Chunk]:
        self._load()
        return self._chunks

    def sources(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT DISTINCT source FROM chunks ORDER BY source")]

    # -- retrieval -----------------------------------------------------------

    def _bm25_ranking(self, query: str, n: int) -> list[tuple[int, float]]:
        self._load()
        return self._bm25.search(query, n)

    def _dense_ranking(self, query: str, n: int) -> list[tuple[int, float]]:
        self._load()
        if not self._vectors:
            return []
        q = self.embedder.embed([query])[0]
        scored = [(i, cosine(q, v)) for i, v in enumerate(self._vectors)]
        scored = [s for s in scored if s[1] > 0]
        return sorted(scored, key=lambda s: (-s[1], s[0]))[:n]

    def _first_stage(self, query: str, n: int, mode: str) -> list[tuple[int, float, dict[str, int]]]:
        if mode == "bm25":
            return [(i, s, {"bm25": r}) for r, (i, s) in enumerate(self._bm25_ranking(query, n), 1)]
        if mode == "dense":
            return [(i, s, {"dense": r}) for r, (i, s) in enumerate(self._dense_ranking(query, n), 1)]
        return rrf({
            "bm25": [i for i, _ in self._bm25_ranking(query, max(n, 50))],
            "dense": [i for i, _ in self._dense_ranking(query, max(n, 50))],
        })[:n]

    def _rerank(self, query: str, pool: list[tuple[int, float, dict[str, int]]], k: int,
                rerank: str) -> list[tuple[int, float, dict[str, int]]]:
        if rerank == "proximity":
            q_terms = set(tokenize(query))
            prox = {i: proximity_score(q_terms, tokenize(self._chunks[i].indexed_text), self._bm25.idf)
                    for i, _, _ in pool}
            by_prox = sorted(prox, key=lambda i: (-prox[i], i))
            first = {i: ranks for i, _, ranks in pool}
            fused = rrf({"first": [i for i, _, _ in pool], "proximity": by_prox})
            return [(i, s, {**first[i], "proximity": r["proximity"]}) for i, s, r in fused[:k]]
        top = pool[0][1] or 1.0
        order = mmr([s / top for _, s, _ in pool],
                    lambda a, b: cosine(self._vectors[pool[a][0]], self._vectors[pool[b][0]]), k)
        return [(pool[p][0], pool[p][1], {**pool[p][2], "mmr": r}) for r, p in enumerate(order, 1)]

    def search(self, query: str, k: int = 5, mode: str = "hybrid", rerank: str | None = None) -> list[Hit]:
        """Top-k chunks for `query`; `rerank` reorders the first-stage top POOL candidates."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if rerank not in (None, "none", *RERANKERS):
            raise ValueError(f"rerank must be one of {('none', *RERANKERS)}")
        if not query.strip():
            return []
        self._load()
        if rerank in (None, "none"):
            results = self._first_stage(query, k, mode)
        else:
            pool = self._first_stage(query, max(k, POOL), mode)
            results = self._rerank(query, pool, k, rerank) if pool else []
        return [Hit(self._chunks[i], score, ranks) for i, score, ranks in results]

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "Index":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
