"""Retrieval and answer evaluation over a labelled question set.

Judged answers (`rag eval --judge`): the answers `rag ask` would give, graded for
correctness and grounding by Claude, or by a word-overlap heuristic offline.

Retrieval: recall@k, MRR and top-k diversity per retriever, optionally reranked
('hybrid+proximity') and/or with Claude multi-query retrieval ('hybrid+multi'). Answers: how often the offline extractive
answer contains the labelled phrase, and how much of it is padding.

Question file format (JSON list):
    [{"question": "...", "source": "pricing.md", "contains": "20% discount"}, ...]

A question may carry "history", the earlier questions of a conversation (oldest
first); it is then condensed into a standalone question before it is searched, as
`rag chat` does. Retrieval and extractive answers use the offline heuristic so the
numbers stay deterministic; `--judge` uses Claude when it answers with Claude.

A retrieved chunk is relevant when it comes from `source` and its text contains the
`contains` phrase (case-insensitive). Labelling by phrase rather than chunk id keeps
the labels valid when chunking parameters change.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .generate import NO_ANSWER, _fallback_warning, answer, claude_available, claude_complete, extractive_answer
from .conversation import condense
from .index import MODES, Index
from .rerank import RERANKERS
from .rewrite import multi_search, rewrite_queries
from .text import tokenize

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
        history = q.get("history", [])
        if not isinstance(history, list) or not all(isinstance(h, str) for h in history):
            raise ValueError(f"question {q['question']!r}: 'history' must be a list of strings")
    if not questions:
        raise ValueError(f"{path} contains no questions")
    return questions


def standalone(q: dict, use_llm: bool = False) -> str:
    """The question to search and answer: condensed with its conversation history, if any."""
    return condense(q["question"], q.get("history", []), use_llm=use_llm)[0]


def is_relevant(chunk, label: dict) -> bool:
    return chunk.source == label["source"] and label["contains"].lower() in chunk.text.lower()


def parse_mode(spec: str) -> tuple[str, str | None, bool]:
    """'hybrid' -> ('hybrid', None, False); 'hybrid+mmr+multi' -> ('hybrid', 'mmr', True).

    '+multi' fuses searches for Claude's rewrites of the question (needs Claude).
    """
    mode, *extras = spec.split("+")
    multi = "multi" in extras
    extras = [e for e in extras if e != "multi"]
    if mode not in MODES or len(extras) > 1 or (extras and extras[0] not in RERANKERS):
        raise ValueError(f"unknown mode {spec!r}; use one of {MODES}, optionally +{'/+'.join(RERANKERS)} "
                         "and/or +multi")
    if multi and not claude_available():
        raise ValueError(f"{spec!r} needs Claude to rewrite queries: set ANTHROPIC_API_KEY "
                         "and install the claude extra")
    return mode, extras[0] if extras else None, multi


def _retrieve(index: Index, question: str, k: int, spec: tuple[str, str | None, bool]):
    mode, rerank, multi = spec
    queries = rewrite_queries(question)[0] if multi else [question]
    return multi_search(index, queries, k=k, mode=mode, rerank=rerank)


def evaluate(index: Index, questions: list[dict], ks=(1, 3, 5), modes=DEFAULT_MODES) -> list[ModeReport]:
    """One report per mode spec, e.g. 'bm25' or 'hybrid+proximity' (retriever + reranker)."""
    depth = max(ks)
    parsed = [(spec, parse_mode(spec)) for spec in modes]  # validate before doing any work
    reports = []
    for spec, parts in parsed:
        results = []
        for q in questions:
            hits = _retrieve(index, standalone(q), depth, parts)
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
    parts = parse_mode(mode)
    results = []
    for q in questions:
        query = standalone(q)
        hits = _retrieve(index, query, k, parts)
        text = extractive_answer(query, hits)
        cited = [] if text == NO_ANSWER else [int(n) for _, n in _CITED_SENTENCE.findall(text)]
        on_target = sum(1 <= n <= len(hits) and is_relevant(hits[n - 1].chunk, q) for n in cited)
        results.append(AnswerResult(q["question"], q["contains"].lower() in text.lower(), len(cited), on_target))
    return AnswerReport(mode, results)


JUDGE_PROMPT = """You grade answers produced by a retrieval-augmented QA system.

You get a question, the reference fact a correct answer must convey, the numbered sources the system retrieved, and its answer with [n] citations. Decide:
- correct: the answer conveys the reference fact (wording may differ) and does not contradict it.
- grounded: every claim in the answer is supported by the source it cites.

Reply with only a JSON object: {"correct": true|false, "grounded": true|false, "reason": "<one short sentence>"}"""


@dataclass
class JudgedAnswer:
    question: str
    answer: str
    correct: bool
    grounded: bool
    judge: str  # "claude" or "heuristic"
    reason: str = ""


@dataclass
class JudgeReport:
    mode: str
    answered_by: str  # "claude", "extractive" or "mixed"
    results: list[JudgedAnswer]

    @property
    def correct(self) -> float:
        return sum(r.correct for r in self.results) / len(self.results)

    @property
    def grounded(self) -> float:
        return sum(r.grounded for r in self.results) / len(self.results)

    def to_dict(self) -> dict:
        return {"mode": self.mode, "answered_by": self.answered_by,
                "judges": sorted({r.judge for r in self.results}),
                "correct": round(self.correct, 4), "grounded": round(self.grounded, 4),
                "failures": [{"question": r.question, "answer": r.answer, "correct": r.correct,
                              "grounded": r.grounded, "reason": r.reason}
                             for r in self.results if not (r.correct and r.grounded)]}


def heuristic_judgement(label: dict, answer_text: str, sources: list[dict]) -> tuple[bool, bool, str]:
    """(correct, grounded, reason) without an LLM.

    correct: every content word of the labelled phrase appears in the answer.
    grounded: every claim (the text before a run of [n] citations, or uncited text at
    the end) shares at least half of its content words with the sources it cites.
    """
    words = set(tokenize(answer_text))
    missing = set(tokenize(label["contains"])) - words
    if answer_text == NO_ANSWER:
        return False, True, "no answer"
    unsupported, start = [], 0
    # Each claim is the text before a run of citations: "Claim one [1]. Claim two [2][3]."
    for m in list(re.finditer(r"(?:\[\d+\])+", answer_text)) + [None]:
        claim = answer_text[start:m.start() if m else None]
        cited = [int(n) for n in re.findall(r"\d+", m.group(0))] if m else []
        start = m.end() if m else start
        own = set(tokenize(claim))
        cited_words = set().union(*(set(tokenize(sources[n - 1]["text"])) for n in cited if 1 <= n <= len(sources)))
        if own and len(own & cited_words) < len(own) / 2:
            unsupported.append(claim.strip())
    reason = "; ".join(filter(None, [f"missing {sorted(missing)}" if missing else "",
                                     f"not supported by its citation: {unsupported[0][:80]!r}"
                                     if unsupported else ""]))
    return not missing, not unsupported, reason


def claude_judgement(question: str, label: dict, answer_text: str, sources: list[dict]) -> tuple[bool, bool, str]:
    user = (f"Question: {question}\n\nReference fact: {label['contains']}\n\n<sources>\n"
            + "\n".join(f'<source id="{s["n"]}">{s["text"]}</source>' for s in sources)
            + f"\n</sources>\n\nAnswer:\n{answer_text}")
    reply = claude_complete(JUDGE_PROMPT, user)
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    verdict = json.loads(match.group(0)) if match else None
    if not isinstance(verdict, dict) or not {"correct", "grounded"} <= verdict.keys():
        raise RuntimeError(f"judge reply was not the expected JSON: {reply[:80]!r}")
    return bool(verdict["correct"]), bool(verdict["grounded"]), str(verdict.get("reason", ""))


def judge_answers(index: Index, questions: list[dict], k: int = 5, mode: str = "hybrid",
                  use_llm: bool = True) -> JudgeReport:
    """Answer each question the way `rag ask` does (Claude when available, else the
    extractive fallback) and grade the answer: Claude as judge when available, else
    heuristic_judgement(). A failed judge call falls back to the heuristic for that
    question."""
    parts = parse_mode(mode)
    llm = use_llm and claude_available()
    results, modes = [], set()
    for q in questions:
        query = standalone(q, use_llm=llm)
        hits = _retrieve(index, query, k, parts)
        result = answer(query, hits, use_llm=llm)
        modes.add(result.mode)
        verdict, judge = None, "heuristic"
        if llm:
            try:
                verdict, judge = claude_judgement(query, q, result.text, result.sources), "claude"
            except Exception as exc:
                if _fallback_warning(exc) is None and not isinstance(exc, ValueError):
                    raise
        verdict = verdict or heuristic_judgement(q, result.text, result.sources)
        results.append(JudgedAnswer(q["question"], result.text, *verdict[:2], judge, verdict[2]))
    return JudgeReport(mode, modes.pop() if len(modes) == 1 else "mixed", results)


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
