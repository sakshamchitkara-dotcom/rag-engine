"""Load .md / .txt / .html / .pdf from files, folders or URLs into Markdown-ish text."""

from __future__ import annotations

import io
import logging
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf"}
MAX_URL_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class Document:
    source: str
    title: str
    text: str


class _HTMLToMarkdown(HTMLParser):
    """Keep headings (as '#' lines) and block structure; drop scripts, styles and nav chrome."""

    _SKIP = {"script", "style", "noscript", "template", "svg", "nav", "footer", "head"}
    _BLOCK = {"p", "div", "section", "article", "li", "br", "pre", "blockquote",
              "ul", "ol", "dd", "dt", "main", "header"}
    _TABLE = {"table", "thead", "tbody", "tfoot", "tr", "td", "th", "caption", "colgroup", "col"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False
        self._in_pre = False
        # ponytail: one table level; a table nested in a cell is flattened into that cell.
        self._rows: list[list[str]] | None = None  # rows of the open <table>
        self._cell: list[str] | None = None  # text of the open <td>/<th>

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in self._SKIP:
            self._skip_depth += 1
        elif self._skip_depth:
            return
        elif self._rows is not None:
            if tag == "tr":
                self._rows.append([])
            elif tag in {"td", "th", "caption"}:
                if not self._rows and tag != "caption":
                    self._rows.append([])
                self._cell = []
            elif self._cell is not None:
                self._cell.append(" ")  # <br>, <p> etc. inside a cell
        elif tag == "table":
            self._rows = []
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in self._BLOCK:
            self.parts.append("\n\n")
            if tag == "li":
                self.parts.append("- ")
            if tag == "pre":
                self._in_pre = True
                self.parts.append("```\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif self._skip_depth:
            return
        elif self._rows is not None:
            if tag == "caption" and self._cell is not None:
                self.parts.append("\n\n" + " ".join("".join(self._cell).split()) + "\n\n")
                self._cell = None
            elif tag in {"td", "th"} and self._cell is not None:
                self._rows[-1].append(" ".join("".join(self._cell).split()).replace("|", "\\|"))
                self._cell = None
            elif tag == "table":
                self.parts.append("\n\n" + _markdown_table(self._rows) + "\n\n")
                self._rows = self._cell = None
            elif self._cell is not None:
                self._cell.append(" ")
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"} or tag in self._BLOCK:
            self.parts.append("\n\n")
            if tag == "pre":
                self._in_pre = False
                self.parts.insert(len(self.parts) - 1, "\n```")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
            return
        if self._skip_depth:
            return
        if self._rows is not None:
            if self._cell is not None:
                self._cell.append(data)
            return  # whitespace between cells
        self.parts.append(data if self._in_pre else re.sub(r"\s+", " ", data))

    def markdown(self) -> str:
        out: list[str] = []
        in_fence = False
        for line in "".join(self.parts).splitlines():
            if line.strip().startswith("```"):
                in_fence = not in_fence
                line = line.strip()
            line = line.rstrip() if in_fence else line.strip()
            if line or (out and out[-1]):
                out.append(line)
        return "\n".join(out).strip()


def _markdown_table(rows: list[list[str]]) -> str:
    """Render rows as a Markdown pipe table; the first row is the header."""
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    lines = ["| " + " | ".join(r + [""] * (width - len(r))) + " |" for r in rows]
    lines.insert(1, "|" + "---|" * width)
    return "\n".join(lines)


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, markdown-ish body)."""
    parser = _HTMLToMarkdown()
    parser.feed(html)
    parser.close()
    return " ".join(parser.title.split()), parser.markdown()


def pdf_to_text(data: bytes) -> str | None:
    """Extract text page by page with pypdf; None if pypdf is not installed."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    logging.getLogger("pypdf").setLevel(logging.ERROR)  # a broken PDF is reported once, by load_path
    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(p for p in pages if p)


def text_to_markdown(text: str) -> str:
    """Promote plain-text headings (short standalone lines without end punctuation) to Markdown.

    Leaves text alone if it already has Markdown headings.
    """
    if re.search(r"^#{1,6}\s", text, re.MULTILINE):
        return text
    out, first = [], True
    for para in re.split(r"\n\s*\n", text.strip()):
        para = para.strip()
        if "\n" not in para and 0 < len(para) <= 70 and para[-1] not in ".,;:!?)":
            para = ("# " if first else "## ") + para
        first = False
        out.append(para)
    return "\n\n".join(out)


def _title_from_markdown(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _decode(data: bytes, source: str, kind: str) -> Document | None:
    stem = Path(urllib.parse.urlparse(source).path).stem or source
    if kind == "pdf":
        text = pdf_to_text(data)
        if text is None:
            print(f"skip {source}: install 'pypdf' to ingest PDFs", file=sys.stderr)
            return None
        return Document(source, stem, text)
    raw = data.decode("utf-8", errors="replace")
    if kind == "html":
        title, text = html_to_text(raw)
        return Document(source, title or _title_from_markdown(text, stem), text)
    text = text_to_markdown(raw) if kind == "text" and not source.lower().endswith((".md", ".markdown")) else raw
    return Document(source, _title_from_markdown(text, stem), text)


def _kind_for_suffix(suffix: str) -> str:
    suffix = suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".html", ".htm"}:
        return "html"
    return "text"


def load_file(path: Path, source: str | None = None) -> Document | None:
    return _decode(path.read_bytes(), source or str(path), _kind_for_suffix(path.suffix))


def load_url(url: str, timeout: float = 20.0) -> Document | None:
    req = urllib.request.Request(url, headers={"User-Agent": "rag-engine/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(MAX_URL_BYTES + 1)
        ctype = resp.headers.get_content_type()
    if len(data) > MAX_URL_BYTES:
        raise ValueError(f"{url} is larger than {MAX_URL_BYTES} bytes")
    if ctype == "application/pdf":
        kind = "pdf"
    elif ctype in {"text/html", "application/xhtml+xml"}:
        kind = "html"
    else:
        kind = _kind_for_suffix(Path(urllib.parse.urlparse(url).path).suffix)
    return _decode(data, url, kind)


def load_path(target: str) -> list[Document]:
    """Load a URL, a single file, or every supported file under a folder (recursively)."""
    if target.startswith(("http://", "https://")):
        doc = load_url(target)
        return [doc] if doc else []
    path = Path(target)
    if not path.exists():
        raise FileNotFoundError(target)
    if path.is_file():
        files, base = [path], path.parent
    else:
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
        base = path
    docs = []
    for f in files:
        try:
            doc = load_file(f, source=f.relative_to(base).as_posix())
        except Exception as exc:  # a corrupt PDF or unreadable file must not abort the whole ingest
            print(f"skip {f}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if doc and doc.text.strip():
            docs.append(doc)
    return docs
