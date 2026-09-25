"""Second-stage rerankers applied to the top candidates of a first-stage retriever.

- proximity: rewards chunks where the query terms appear close together, which
  first-stage BM25 and bag-of-words vectors ignore. Fused with the first-stage
  order by RRF so it reorders near-ties rather than overriding retrieval.
- mmr: maximal marginal relevance; trades a little relevance for diversity so the
  top-k does not fill up with near-duplicate chunks (e.g. overlapping neighbours).
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Sequence

RERANKERS = ("proximity", "mmr")
POOL = 20  # first-stage candidates the reranker sees
MMR_LAMBDA = 0.7  # 1.0 = pure relevance, 0.0 = pure diversity


def min_span(positions: dict[str, list[int]]) -> int:
    """Length (in tokens) of the shortest window containing every term in `positions` (non-empty)."""
    events = sorted((p, t) for t, ps in positions.items() for p in ps)
    need, have, counts = len(positions), 0, Counter()
    best, left = events[-1][0] - events[0][0] + 1, 0
    for pos, term in events:
        counts[term] += 1
        have += counts[term] == 1
        while have == need:
            best = min(best, pos - events[left][0] + 1)
            lterm = events[left][1]
            counts[lterm] -= 1
            have -= counts[lterm] == 0
            left += 1
    return best


def proximity_score(query_terms: set[str], tokens: list[str], idf: Callable[[str], float]) -> float:
    """IDF-weighted query coverage, scaled up when the matched terms sit close together."""
    total = sum(idf(t) for t in query_terms)
    positions: dict[str, list[int]] = {}
    for i, tok in enumerate(tokens):
        if tok in query_terms:
            positions.setdefault(tok, []).append(i)
    if not positions or not total:
        return 0.0
    coverage = sum(idf(t) for t in positions) / total
    tightness = len(positions) / min_span(positions)  # 1.0 when the terms are adjacent
    return coverage * (0.5 + 0.5 * tightness)


def mmr(relevance: Sequence[float], similarity: Callable[[int, int], float], k: int,
        lam: float = MMR_LAMBDA) -> list[int]:
    """Greedy MMR over candidates 0..n-1; `relevance` should be scaled to [0, 1]."""
    remaining = list(range(len(relevance)))
    picked: list[int] = []
    while remaining and len(picked) < k:
        def gain(i: int) -> float:
            redundancy = max((similarity(i, j) for j in picked), default=0.0)
            return lam * relevance[i] - (1 - lam) * redundancy

        best = max(remaining, key=lambda i: (gain(i), -i))
        picked.append(best)
        remaining.remove(best)
    return picked

