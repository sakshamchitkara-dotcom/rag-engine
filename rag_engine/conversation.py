"""Conversation mode: turn a follow-up ("does the old one keep working?") into a
standalone question that can be searched and answered on its own.

With Claude, the earlier questions and the follow-up are sent to Claude, which
rewrites the follow-up. Offline (or if Claude fails), a heuristic decides whether the
follow-up depends on the earlier turns (pronouns such as "it" or "they", openers such
as "and" or "what about", or at most two content words) and if so appends the
content words of the previous standalone question that the follow-up lacks.
"""

from __future__ import annotations

import re

from .generate import MODEL, _fallback_warning, claude_available, claude_complete
from .text import STOPWORDS, tokenize

MAX_TURNS = 10  # earlier questions considered; older ones are dropped
MAX_CARRIED = 8  # context words appended by the heuristic

_REFERRING = frozenset("it its they them their theirs this that these those there one ones same".split())
_OPENERS = ("and ", "also ", "but ", "so ", "then ", "what about", "how about", "what if")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*")

CONDENSE_PROMPT = """You rewrite the latest question in a conversation about technical documentation so that it can be understood without the conversation.

Resolve pronouns and references ("it", "the old one", "what about X") using the earlier questions. Keep the user's wording where you can. If the question already stands on its own, return it unchanged. Reply with only the rewritten question on one line; do not answer it."""


def is_follow_up(question: str) -> bool:
    """Whether a question probably leans on the conversation before it."""
    words = [w.lower() for w in _WORD_RE.findall(question)]
    if not words:
        return False
    if _REFERRING & set(words) or question.strip().lower().startswith(_OPENERS):
        return True
    return len(tokenize(question)) <= 2


def carry_over(question: str, previous: str) -> str:
    """The follow-up plus the content words of `previous` it does not already contain."""
    if not is_follow_up(question):
        return question
    have = set(tokenize(question))
    carried: list[str] = []
    for word in _WORD_RE.findall(previous):
        stem = tokenize(word)
        if word.lower() in STOPWORDS or word.lower() in _REFERRING or not stem or stem[0] in have:
            continue
        have.add(stem[0])
        carried.append(word)
    # ponytail: a bag of words from one earlier turn; Claude does the real coreference.
    carried = carried[:MAX_CARRIED]
    return f"{question.strip()} ({' '.join(carried)})" if carried else question


def heuristic_condense(question: str, history: list[str]) -> str:
    standalone = ""
    for turn in [*history[-MAX_TURNS:], question]:
        standalone = carry_over(turn, standalone) if standalone else turn
    return standalone


def condense(question: str, history: list[str], *, use_llm: bool = True,
             model: str = MODEL) -> tuple[str, str | None]:
    """(standalone question, warning). `history` holds the earlier questions, oldest first."""
    history = [h for h in history if h.strip()]
    if not history:
        return question, None
    if use_llm and claude_available():
        turns = "\n".join(f"{n}. {h.strip()}" for n, h in enumerate(history[-MAX_TURNS:], 1))
        try:
            # claude_complete strips and rejects empty output, so the first line is the rewrite.
            return claude_complete(CONDENSE_PROMPT, f"Earlier questions:\n{turns}\n\nLatest question: {question}",
                                   model).splitlines()[0].strip(), None
        except Exception as exc:
            warning = _fallback_warning(exc)
            if warning is None:
                raise
        return heuristic_condense(question, history), warning.replace("used extractive fallback",
                                                                      "used the offline follow-up heuristic")
    return heuristic_condense(question, history), None
