"""Multi-query retrieval: Claude rewrites a question into a few search queries, each
is searched, and the rankings are fused with RRF. The original question is always
the first query.

There is no offline rewriter: splitting compound questions into their parts and
pseudo-relevance feedback (adding top terms from the first results) both lowered
MRR on the sample eval sets, so without Claude this is a plain search.
"""

from __future__ import annotations

import re

from .generate import MODEL, _fallback_warning, claude_available, claude_complete
from .index import Hit, Index, rrf
from .rerank import POOL

MAX_QUERIES = 4

REWRITE_PROMPT = """You turn a user's question into search queries for a keyword + vector search over technical documentation.

Write up to 3 short search queries, one per line, with no numbering or commentary. Split a question that asks several things into one query per part, and use the words the documentation is likely to use. Do not answer the question."""


def rewrite_queries(question: str, *, use_llm: bool = True, model: str = MODEL) -> tuple[list[str], str | None]:
    """(queries, warning): the question plus Claude's rewrites, or just the question
    (with a warning when Claude was asked for but failed)."""
    if use_llm and claude_available():
        try:
            lines = claude_complete(REWRITE_PROMPT, question, model).splitlines()
            extra = [re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", ln).strip() for ln in lines]
            queries = [question.strip()] + [q for q in extra if q and q.lower() != question.strip().lower()]
            return queries[:MAX_QUERIES], None
        except Exception as exc:
            warning = _fallback_warning(exc)
            if warning is None:
                raise
            return [question], warning.replace("used extractive fallback", "searched the question as written")
    return [question], None


def multi_search(index: Index, queries: list[str], k: int = 5, mode: str = "hybrid", rerank: str | None = None,
                 *, sources: list[str] | None = None, tags: list[str] | None = None) -> list[Hit]:
    """Search each query and fuse the rankings with RRF; one query is a plain search."""
    if len(queries) == 1:
        return index.search(queries[0], k=k, mode=mode, rerank=rerank, sources=sources, tags=tags)
    depth = max(k, POOL)
    by_id: dict[str, Hit] = {}
    rankings: dict[str, list[str]] = {}
    for n, q in enumerate(queries, 1):
        hits = index.search(q, k=depth, mode=mode, rerank=rerank, sources=sources, tags=tags)
        rankings[f"q{n}"] = [h.chunk.id for h in hits]
        for h in hits:
            by_id.setdefault(h.chunk.id, h)
    return [Hit(by_id[cid].chunk, score, ranks) for cid, score, ranks in rrf(rankings)[:k]]
