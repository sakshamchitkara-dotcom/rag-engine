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
MIN_SENTENCE_TERMS = 3
MIN_COVERAGE = 1 / 3  # share of query IDF weight a sentence from another chunk must match
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


def _merge_fragments(sentences: list[str]) -> list[str]:
    """Glue fragments such as "Defaults to 8420." or "Required." onto the sentence before.

    Alone they win on length normalisation while saying nothing ("Required. [3]");
    attached, "`BEACON_PORT` - HTTP port ... Defaults to 8420." reads as an answer.
    """
    out: list[str] = []
    for sentence in sentences:
        if out and len(tokenize(sentence)) < MIN_SENTENCE_TERMS:
            out[-1] = f"{out[-1]} {sentence}"
        else:
            out.append(sentence)
    return out


def extractive_answer(question: str, hits: list[Hit], max_sentences: int = 3) -> str:
    """Pick the retrieved sentences that best cover the query terms, each with its citation."""
    q_terms = set(tokenize(question))
    if not q_terms or not hits:
        return NO_ANSWER
    candidates: list[tuple[str, int, set[str], set[str]]] = []
    for n, h in enumerate(hits, start=1):
        # The section heading is context for every sentence under it ("Rate limits" ->
        # "300 requests per minute"), so heading terms count at a discount.
        heading = set(tokenize(h.chunk.heading.rsplit(" > ", 1)[-1]))
        for para in h.chunk.text.split("\n\n"):
            if para.lstrip().startswith(("```", "|")):
                continue  # code blocks and tables make poor answer sentences
            lines = para.splitlines()
            units = ([ln.lstrip("-* ").strip() for ln in lines]
                     if all(ln.lstrip().startswith(("- ", "* ")) for ln in lines) else [para])
            for unit in units:
                for sentence in _merge_fragments(split_sentences(re.sub(r"\*\*|__", "", unit))):
                    candidates.append((sentence, n, set(tokenize(sentence)), heading))
    # IDF over the candidate sentences so rare query terms dominate common ones.
    total = len(candidates)
    idf = {t: math.log(1 + total / max(1, sum(t in (a | b) for _, _, a, b in candidates))) for t in q_terms}
    scored = []
    for i, (sentence, n, toks, heading) in enumerate(candidates):
        direct = q_terms & toks
        if not direct and not q_terms & heading:
            continue
        weight = sum(idf[t] for t in direct) + 0.5 * sum(idf[t] for t in (q_terms & heading) - toks)
        score = weight / math.sqrt(len(toks) + 1) + 0.15 / n  # slight preference for top-ranked chunks
        scored.append((score, i, sentence, n))
    if not scored:
        return NO_ANSWER
    scored.sort(key=lambda s: (-s[0], s[1]))
    query_weight = sum(idf.values())
    floor = scored[0][0] * 0.5  # drop sentences that only graze the query
    picked, seen, covered = [], set(), set()
    for score, i, sentence, n in scored:
        if score < floor:
            break
        key = sentence.lower()
        toks = candidates[i][2]
        if key in seen:
            continue
        # Padding guards: a follow-up sentence must come from the same chunk as the
        # best one (it continues that passage), or match a third of the query's IDF
        # weight and add a query term nothing picked so far covers. Stops "Beacon is
        # free." riding along on the word "free" in a question about the free trial.
        if picked and n != picked[0][2]:
            coverage = sum(idf[t] for t in q_terms & toks) / query_weight
            if coverage < MIN_COVERAGE or not (q_terms & toks) - covered:
                continue
        seen.add(key)
        covered |= q_terms & toks
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
