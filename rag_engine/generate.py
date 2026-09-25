"""Answer generation with numbered citations.

Uses Claude through the official `anthropic` SDK when it is installed and
ANTHROPIC_API_KEY is set. Otherwise (or if the call fails) it falls back to an
extractive answer built from the best-matching retrieved sentences, so the
pipeline always produces a cited answer.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

from .index import Hit
from .text import split_sentences, tokenize

MODEL = "claude-opus-5-5"
NO_ANSWER = "I couldn't find an answer to that in the indexed documents."

SYSTEM_PROMPT = """You answer questions using only the numbered sources provided by the user.

Rules:
- Cite every factual claim with the number of the source that supports it, in square brackets, e.g. [1] or [2][3]. Put the citation right after the claim.
- Use only information in the sources. If they do not contain the answer, say so plainly and do not guess.
- Be concise: a few sentences or a short list. Do not repeat the question or describe the sources."""


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)
    mode: str = "extractive"  # "claude" or "extractive"
    warning: str | None = None

    def to_dict(self) -> dict:
        return {"answer": self.text, "sources": self.sources, "mode": self.mode, "warning": self.warning}


def _source_list(hits: list[Hit]) -> list[dict]:
    return [
        {
            "n": n,
            "id": h.chunk.id,
            "source": h.chunk.source,
            "title": h.chunk.title,
            "heading": h.chunk.heading,
            "score": round(h.score, 4),
            "text": h.chunk.text,
        }
        for n, h in enumerate(hits, start=1)
    ]


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = []
    for n, h in enumerate(hits, start=1):
        c = h.chunk
        blocks.append(
            f'<source id="{n}" document="{c.source}" section="{c.heading}">\n{c.text}\n</source>'
        )
    return "<sources>\n" + "\n".join(blocks) + f"\n</sources>\n\nQuestion: {question}"


def extractive_answer(question: str, hits: list[Hit], max_sentences: int = 3) -> str:
    """Pick the retrieved sentences that best cover the query terms, each with its citation."""
    q_terms = set(tokenize(question))
    if not q_terms or not hits:
        return NO_ANSWER
    candidates: list[tuple[str, int, set[str]]] = []
    for n, h in enumerate(hits, start=1):
        for para in h.chunk.text.split("\n\n"):
            if para.lstrip().startswith(("```", "|")):
                continue  # code blocks and tables make poor answer sentences
            for sentence in split_sentences(para):
                candidates.append((sentence, n, set(tokenize(sentence))))
    # IDF over the candidate sentences so rare query terms dominate common ones.
    df = {t: sum(t in toks for _, _, toks in candidates) for t in q_terms}
    total = len(candidates)
    scored = []
    for i, (sentence, n, toks) in enumerate(candidates):
        overlap = q_terms & toks
        if not overlap:
            continue
        weight = sum(math.log(1 + total / df[t]) for t in overlap)
        score = weight / math.sqrt(len(toks) + 1) + 0.15 / n  # slight preference for top-ranked chunks
        scored.append((score, i, sentence, n))
    if not scored:
        return NO_ANSWER
    picked, seen = [], set()
    for _, i, sentence, n in sorted(scored, key=lambda s: (-s[0], s[1])):
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        picked.append((i, sentence, n))
        if len(picked) == max_sentences:
            break
    picked.sort()  # keep document order so the answer reads naturally
    return " ".join(f"{s} [{n}]" for _, s, n in picked)


def claude_available() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def claude_answer(question: str, hits: list[Hit], model: str = MODEL) -> str:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    response = client.messages.create(
        model=model,
        max_tokens=16000,
        # Claude Opus 5.5 always thinks; effort is the only dial. Grounded Q&A over a
        # handful of passages does not need deep deliberation.
        output_config={"effort": "low"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to answer this request")
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise RuntimeError(f"Claude returned no text (stop_reason={response.stop_reason})")
    return text


def answer(question: str, hits: list[Hit], *, use_llm: bool = True, model: str = MODEL) -> Answer:
    sources = _source_list(hits)
    if not hits:
        return Answer(NO_ANSWER, sources)
    warning = None
    if use_llm and claude_available():
        import anthropic

        try:
            return Answer(claude_answer(question, hits, model), sources, mode="claude")
        except anthropic.AuthenticationError:
            warning = "ANTHROPIC_API_KEY was rejected; used extractive fallback"
        except anthropic.RateLimitError:
            warning = "Claude rate limit hit; used extractive fallback"
        except anthropic.APIStatusError as exc:
            warning = f"Claude API error {exc.status_code}; used extractive fallback"
        except anthropic.APIConnectionError:
            warning = "could not reach the Claude API; used extractive fallback"
        except RuntimeError as exc:
            warning = f"{exc}; used extractive fallback"
    return Answer(extractive_answer(question, hits), sources, mode="extractive", warning=warning)


def cited_numbers(text: str) -> list[int]:
    """Citation numbers used in an answer, in order of first appearance."""
    seen: list[int] = []
    for m in re.finditer(r"\[(\d+)\]", text):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen
