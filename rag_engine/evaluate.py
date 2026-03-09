"""Retrieval and answer evaluation over a labelled question set.

Retrieval: recall@k, MRR and top-k diversity per retriever, optionally reranked
('hybrid+proximity'). Answers: how often the offline extractive
answer contains the labelled phrase, and how much of it is padding.

Question file format (JSON list):
    [{"question": "...", "source": "pricing.md", "contains": "20% discount"}, ...]

A retrieved chunk is relevant when it comes from `source` and its text contains the
`contains` phrase (case-insensitive). Labelling by phrase rather than chunk id keeps
the labels valid when chunking parameters change.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .generate import NO_ANSWER, extractive_answer
from .index import MODES, Index
from .rerank import RERANKERS

DEFAULT_MODES = (*MODES, *(f"hybrid+{r}" for r in RERANKERS))


@dataclass
class QuestionResult:
    question: str
    rank: int | None  # 1-based rank of the first relevant chunk, None if not retrieved
    sources: int = 0  # distinct documents in the retrieved top-k (diversity)


@dataclass
class ModeReport:
    mode: str
    ks: tuple[int, ...]
    results: list[QuestionResult]

    def recall_at(self, k: int) -> float:
        return sum(r.rank is not None and r.rank <= k for r in self.results) / len(self.results)

    @property
    def mrr(self) -> float:
        return sum(1 / r.rank for r in self.results if r.rank) / len(self.results)

    @property
    def avg_sources(self) -> float:
        return sum(r.sources for r in self.results) / len(self.results)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            **{f"recall@{k}": round(self.recall_at(k), 4) for k in self.ks},
            "mrr": round(self.mrr, 4),
            "sources": round(self.avg_sources, 2),
            "misses": [r.question for r in self.results if r.rank is None],
        }


def load_questions(path: str | Path) -> list[dict]:
    questions = json.loads(Path(path).read_text())
    for q in questions:
        missing = {"question", "source", "contains"} - q.keys()
        if missing:
            raise ValueError(f"question {q!r} is missing {sorted(missing)}")
    if not questions:
        raise ValueError(f"{path} contains no questions")
    return questions


def is_relevant(chunk, label: dict) -> bool:
    return chunk.source == label["source"] and label["contains"].lower() in chunk.text.lower()


def parse_mode(spec: str) -> tuple[str, str | None]:
    """'hybrid' -> ('hybrid', None); 'hybrid+mmr' -> ('hybrid', 'mmr')."""
    mode, _, rerank = spec.partition("+")
    if mode not in MODES or (rerank and rerank not in RERANKERS):
        raise ValueError(f"unknown mode {spec!r}; use one of {MODES}, optionally +{'/+'.join(RERANKERS)}")
    return mode, rerank or None


def evaluate(index: Index, questions: list[dict], ks=(1, 3, 5), modes=DEFAULT_MODES) -> list[ModeReport]:
    """One report per mode spec, e.g. 'bm25' or 'hybrid+proximity' (retriever + reranker)."""
    depth = max(ks)
    parsed = [(spec, *parse_mode(spec)) for spec in modes]  # validate before doing any work
    reports = []
    for spec, mode, rerank in parsed:
        results = []
        for q in questions:
            hits = index.search(q["question"], k=depth, mode=mode, rerank=rerank)
            rank = next((i for i, h in enumerate(hits, 1) if is_relevant(h.chunk, q)), None)
            results.append(QuestionResult(q["question"], rank, len({h.chunk.source for h in hits})))
        reports.append(ModeReport(spec, tuple(ks), results))
    return reports


_CITED_SENTENCE = re.compile(r"(.+?)\s\[(\d+)\](?:\s+|$)")


@dataclass
class AnswerResult:
    question: str
    found: bool  # the labelled phrase appears in the answer
    sentences: int
    on_target: int  # sentences cited from a chunk that holds the labelled answer


@dataclass
class AnswerReport:
    mode: str
    results: list[AnswerResult]

    @property
    def found_rate(self) -> float:
        return sum(r.found for r in self.results) / len(self.results)

    @property
    def precision(self) -> float:
        """Share of all answer sentences that come from an answer-bearing chunk."""
        total = sum(r.sentences for r in self.results)
        return sum(r.on_target for r in self.results) / total if total else 0.0

    @property
    def avg_sentences(self) -> float:
        return sum(r.sentences for r in self.results) / len(self.results)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "found": round(self.found_rate, 4),
            "precision": round(self.precision, 4),
            "avg_sentences": round(self.avg_sentences, 2),
            "misses": [r.question for r in self.results if not r.found],
        }


def evaluate_answers(index: Index, questions: list[dict], k: int = 5, mode: str = "hybrid") -> AnswerReport:
    """Score the extractive answerer (deterministic and offline, unlike Claude) on top-k hits."""
    retriever, rerank = parse_mode(mode)
    results = []
    for q in questions:
        hits = index.search(q["question"], k=k, mode=retriever, rerank=rerank)
        text = extractive_answer(q["question"], hits)
        cited = [] if text == NO_ANSWER else [int(n) for _, n in _CITED_SENTENCE.findall(text)]
        on_target = sum(1 <= n <= len(hits) and is_relevant(hits[n - 1].chunk, q) for n in cited)
        results.append(AnswerResult(q["question"], q["contains"].lower() in text.lower(), len(cited), on_target))
    return AnswerReport(mode, results)


def format_table(reports: list[ModeReport]) -> str:
    ks = reports[0].ks
    header = ["mode"] + [f"recall@{k}" for k in ks] + ["MRR", f"docs@{max(ks)}"]
    rows = [[r.mode] + [f"{r.recall_at(k):.3f}" for k in ks] + [f"{r.mrr:.3f}", f"{r.avg_sources:.2f}"]
            for r in reports]
    widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()  # noqa: E731
    return "\n".join([line(header), line(["-" * w for w in widths])] + [line(r) for r in rows])


def format_answer_table(reports: list[AnswerReport]) -> str:
    header = ["mode", "found", "precision", "sentences"]
    rows = [[r.mode, f"{r.found_rate:.3f}", f"{r.precision:.3f}", f"{r.avg_sentences:.2f}"] for r in reports]
    widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()  # noqa: E731
    return "\n".join([line(header), line(["-" * w for w in widths])] + [line(r) for r in rows])
