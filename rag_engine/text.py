"""Tokenization and sentence splitting shared by the indexers and the generator."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9`])")

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers herself him himself his how
    i if in into is it its itself just me more most my myself no nor not now of off on
    once only or other our ours ourselves out over own same she should so some such than
    that the their theirs them themselves then there these they this those through to too
    under until up very was we were what when where which while who whom why will with
    would you your yours yourself yourselves""".split()
)


def _stem(token: str) -> str:
    """Tiny suffix stripper so 'limits'/'limit' and 'configured'/'configure' collide.

    ponytail: crude rule-based stemming; swap for a Porter stemmer if recall on
    morphology-heavy corpora matters.
    """
    if len(token) <= 3 or not token.isalpha():
        return token
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("s", "")):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            if suffix == "s" and token.endswith(("ss", "us", "is")):
                break
            token = token[: -len(suffix)] + repl
            break
    if len(token) > 4 and token.endswith("e"):
        token = token[:-1]
    return token


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Lowercase, split on non-alphanumerics (keeping 'v2.1', 'x-api-key'), drop stopwords, stem."""
    tokens = _TOKEN_RE.findall(text.lower())
    if not keep_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [_stem(t) for t in tokens]


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences. Good enough for English docs; not a full segmenter."""
    text = " ".join(text.split())
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
