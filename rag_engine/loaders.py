"""Load .md / .txt / .html / .pdf from files, folders or URLs into Markdown-ish text."""

from __future__ import annotations

import io
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
    _BLOCK = {"p", "div", "section", "article", "li", "tr", "br", "pre", "blockquote",
              "table", "ul", "ol", "dd", "dt", "main", "header"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False
        self._in_pre = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in self._SKIP:
            self._skip_depth += 1
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
    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(p for p in pages if p)


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
    return Document(source, _title_from_markdown(raw, stem), raw)


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
        doc = load_file(f, source=f.relative_to(base).as_posix())
        if doc and doc.text.strip():
            docs.append(doc)
    return docs
