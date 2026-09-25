"""Structure-aware chunking.

Documents arrive as Markdown-flavoured text (HTML and PDF loaders convert to it).
Chunks never cross a heading boundary, prefer to break between paragraphs, fall back
to sentence boundaries for long paragraphs (line boundaries for tables, which
repeat their header, and fenced code, which stays fenced), and carry a sentence-aligned overlap
from the previous chunk of the same section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .text import split_sentences

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_TABLE_RULE_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    title: str
    heading: str
    text: str
    position: int

    @property
    def indexed_text(self) -> str:
        """Text used for retrieval: heading context helps both lexical and dense matching."""
        return f"{self.title}\n{self.heading}\n{self.text}"


def split_sections(text: str) -> list[tuple[str, list[str]]]:
    """Return [(heading_path, paragraphs)] where heading_path is 'H1 > H2 > H3'."""
    sections: list[tuple[str, list[str]]] = []
    stack: list[tuple[int, str]] = []
    paragraphs: list[str] = []
    buf: list[str] = []
    in_fence = False

    def flush_para() -> None:
        if buf:
            para = "\n".join(buf).strip()
            if para:
                paragraphs.append(para)
            buf.clear()

    def flush_section() -> None:
        flush_para()
        if paragraphs:
            sections.append((" > ".join(h for _, h in stack), paragraphs.copy()))
            paragraphs.clear()

    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            buf.append(line)
            continue
        if in_fence:
            buf.append(line)
            continue
        m = _HEADING_RE.match(line)
        if m:
            flush_section()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2).strip()))
        elif not line.strip():
            flush_para()
        else:
            buf.append(line)
    flush_section()
    return sections


def is_table(paragraph: str) -> bool:
    lines = paragraph.splitlines()
    return len(lines) >= 2 and all(line.lstrip().startswith("|") for line in lines)


def _is_code(paragraph: str) -> bool:
    return bool(_FENCE_RE.match(paragraph))


def _line_pieces(head: list[str], body: list[str], tail: list[str], max_chars: int) -> list[str]:
    """Pack whole lines into pieces of at most max_chars, each wrapped in head and tail lines."""
    out: list[str] = []
    lines: list[str] = []
    for line in body:
        if lines and len("\n".join(head + lines + [line] + tail)) > max_chars:
            out.append("\n".join(head + lines + tail))
            lines = []
        lines.append(line)  # a single line longer than max_chars stays whole
    out.append("\n".join(head + lines + tail))
    return out


def _pieces(paragraph: str, max_chars: int) -> list[str]:
    """Split a paragraph into pieces no longer than max_chars, on sentence then word boundaries."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    if is_table(paragraph):  # split between rows; every piece repeats the header
        lines = paragraph.splitlines()
        head = lines[:2] if _TABLE_RULE_RE.match(lines[1]) else []
        return _line_pieces(head, lines[len(head):], [], max_chars)
    if _is_code(paragraph):  # split between lines; every piece is a fenced block
        lines = paragraph.splitlines()
        closed = len(lines) > 1 and _FENCE_RE.match(lines[-1])
        return _line_pieces(lines[:1], lines[1:-1] if closed else lines[1:], lines[-1:] if closed else [], max_chars)
    out: list[str] = []
    for sentence in split_sentences(paragraph):
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            out.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            out.append(sentence)
    return out


def _tail(text: str, overlap: int) -> str:
    """Trailing whole sentences of text totalling at most `overlap` characters."""
    if overlap <= 0:
        return ""
    picked: list[str] = []
    size = 0
    for sentence in reversed(split_sentences(text)):
        if size + len(sentence) > overlap:
            break
        picked.insert(0, sentence)
        size += len(sentence) + 1
    return " ".join(picked)


def chunk_document(
    text: str, *, source: str, title: str, max_chars: int = 800, overlap: int = 150
) -> list[Chunk]:
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    chunks: list[Chunk] = []

    def emit(heading: str, body: str) -> None:
        chunks.append(
            Chunk(
                id=f"{source}#{len(chunks)}",
                source=source,
                title=title,
                heading=heading,
                text=body,
                position=len(chunks),
            )
        )

    for heading, paragraphs in split_sections(text):
        pieces = [p for para in paragraphs for p in _pieces(para, max_chars)]
        current: list[str] = []
        size = 0
        fresh = 0  # pieces added since the last emit (overlap alone is not worth emitting)
        for piece in pieces:
            if current and size + len(piece) + 2 > max_chars:
                body = "\n\n".join(current)
                emit(heading, body)
                seed = "" if is_table(current[-1]) or _is_code(current[-1]) else _tail(body, overlap)
                current, size, fresh = ([seed], len(seed), 0) if seed else ([], 0, 0)
                if current and size + len(piece) + 2 > max_chars:
                    current, size = [], 0
            current.append(piece)
            size += len(piece) + 2
            fresh += 1
        if current and fresh:
            emit(heading, "\n\n".join(current))
    return chunks
