"""Retrieval evaluation: recall@k and MRR over a labelled question set.

Question file format (JSON list):
    [{"question": "...", "source": "pricing.md", "contains": "20% discount"}, ...]

A retrieved chunk is relevant when it comes from `source` and its text contains the
`contains` phrase (case-insensitive). Labelling by phrase rather than chunk id keeps
the labels valid when chunking parameters change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .index import MODES, Index


@dataclass
class QuestionResult:
    question: str
    rank: int | None  # 1-based rank of the first relevant chunk, None if not retrieved


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

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            **{f"recall@{k}": round(self.recall_at(k), 4) for k in self.ks},
            "mrr": round(self.mrr, 4),
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


def evaluate(index: Index, questions: list[dict], ks=(1, 3, 5), modes=MODES) -> list[ModeReport]:
    depth = max(ks)
    reports = []
    for mode in modes:
        results = []
        for q in questions:
            hits = index.search(q["question"], k=depth, mode=mode)
            rank = next((i for i, h in enumerate(hits, 1) if is_relevant(h.chunk, q)), None)
            results.append(QuestionResult(q["question"], rank))
        reports.append(ModeReport(mode, tuple(ks), results))
    return reports


def format_table(reports: list[ModeReport]) -> str:
    ks = reports[0].ks
    header = ["mode"] + [f"recall@{k}" for k in ks] + ["MRR"]
    rows = [[r.mode] + [f"{r.recall_at(k):.3f}" for k in ks] + [f"{r.mrr:.3f}"] for r in reports]
    widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths))  # noqa: E731
    return "\n".join([line(header), line(["-" * w for w in widths])] + [line(r) for r in rows])
