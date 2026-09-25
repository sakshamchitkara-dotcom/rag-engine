import builtins
import io
import importlib.util
import tempfile
import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from rag_engine.loaders import html_to_text, load_path, pdf_to_text, text_to_markdown

CORPUS = Path(__file__).resolve().parent.parent / "examples" / "corpus"


class HTMLTest(unittest.TestCase):
    def test_headings_lists_and_skipped_chrome(self):
        title, text = html_to_text(
            "<html><head><title>API  Ref</title><style>x{}</style></head><body>"
            "<nav>menu</nav><h1>API</h1><p>Hello <b>bold</b> world &amp; more.</p>"
            "<ul><li>one</li><li>two</li></ul><h2>Auth</h2><pre>curl -H x\n  y</pre>"
            "<script>bad()</script><footer>foot</footer></body></html>"
        )
        self.assertEqual(title, "API Ref")
        self.assertIn("# API", text)
        self.assertIn("Hello bold world & more.", text)
        self.assertIn("- one", text)
        self.assertIn("## Auth", text)
        self.assertIn("```\ncurl -H x\n  y\n```", text)
        for junk in ("menu", "bad()", "foot", "x{}"):
            self.assertNotIn(junk, text)


    def test_tables_become_markdown_tables(self):
        _, text = html_to_text(
            "<h2>Plans</h2><table><caption>Hosted plans</caption>"
            "<thead><tr><th>Plan</th><th>Price</th></tr></thead>"
            "<tbody><tr><td>Team<br>plan</td><td>$25 | seat</td></tr><tr><td>Free</td></tr></tbody>"
            "</table><p>After the table.</p>"
        )
        self.assertIn(
            "Hosted plans\n\n| Plan | Price |\n|---|---|\n| Team plan | $25 \\| seat |\n| Free |  |\n\nAfter the table.",
            text,
        )


class TextTest(unittest.TestCase):
    def test_promotes_standalone_short_lines(self):
        md = text_to_markdown("Guide\n\nIntro text here.\n\nSection two\n\nBody. More body.")
        self.assertEqual(md, "# Guide\n\nIntro text here.\n\n## Section two\n\nBody. More body.")

    def test_leaves_markdown_alone(self):
        self.assertEqual(text_to_markdown("# Real\n\nShort line"), "# Real\n\nShort line")


class LoadPathTest(unittest.TestCase):
    def test_loads_sample_corpus(self):
        docs = {d.source: d for d in load_path(str(CORPUS))}
        self.assertIn("api-reference.html", docs)
        self.assertIn("troubleshooting.txt", docs)
        self.assertEqual(docs["api-reference.html"].title, "Beacon REST API Reference")
        self.assertEqual(docs["pricing.md"].title, "Plans and Pricing")
        self.assertEqual(docs["troubleshooting.txt"].title, "Troubleshooting Beacon")

    def test_ignores_unsupported_and_empty_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.md").write_text("# A\n\nhello")
            Path(tmp, "b.json").write_text("{}")
            Path(tmp, "c.txt").write_text("   ")
            self.assertEqual([d.source for d in load_path(tmp)], ["a.md"])

    def test_missing_path(self):
        with self.assertRaises(FileNotFoundError):
            load_path("/no/such/path")

    def test_pdf_skipped_without_pypdf(self):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pypdf":
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", fake_import):
            self.assertIsNone(pdf_to_text(b"%PDF-1.4"))
            with tempfile.TemporaryDirectory() as tmp:
                Path(tmp, "doc.pdf").write_bytes(b"%PDF-1.4")
                Path(tmp, "a.md").write_text("# A\n\nhello")
                with mock.patch("sys.stderr"):
                    self.assertEqual([d.source for d in load_path(tmp)], ["a.md"])

    def test_unreadable_file_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "broken.pdf").write_bytes(b"garbage")
            Path(tmp, "a.md").write_text("# A\n\nhello")
            err = io.StringIO()
            with mock.patch("rag_engine.loaders.pdf_to_text", side_effect=ValueError("stream ended")), \
                    mock.patch("sys.stderr", err):
                self.assertEqual([d.source for d in load_path(tmp)], ["a.md"])
        self.assertIn("broken.pdf: ValueError: stream ended", err.getvalue())

    def test_url_kind_comes_from_content_type_and_size_is_capped(self):
        class Handler(_Quiet):
            def do_GET(self):
                body = b"Plain notes\n\nServed as text." if self.path == "/notes" else b"x" * 64
                self.send_response(200)
                self.send_header("Content-Type", "text/plain" if self.path == "/notes" else "application/pdf")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            [doc] = load_path(base + "/notes")  # no suffix: text/plain decides
            self.assertEqual((doc.title, doc.text), ("Plain notes", "# Plain notes\n\nServed as text."))
            with mock.patch("rag_engine.loaders.pdf_to_text", return_value="pdf text") as pdf:
                [doc] = load_path(base + "/report")  # no suffix: application/pdf decides
            pdf.assert_called_once()
            with mock.patch("rag_engine.loaders.MAX_URL_BYTES", 10), self.assertRaises(ValueError):
                load_path(base + "/report")
        finally:
            server.shutdown()
            server.server_close()

    def test_loads_html_from_url(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_Quiet, directory=str(CORPUS)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/api-reference.html"
            [doc] = load_path(url)
            self.assertEqual(doc.source, url)
            self.assertEqual(doc.title, "Beacon REST API Reference")
            self.assertIn("## Rate limits", doc.text)
        finally:
            server.shutdown()
            server.server_close()


def minimal_pdf(lines: list[str]) -> bytes:
    """Build a one-page PDF with the given text lines (Helvetica), with a valid xref table."""
    ops = "BT /F1 12 Tf 72 720 Td 14 TL " + " ".join(f"({ln}) Tj T*" for ln in lines) + " ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(ops)} >>\nstream\n{ops}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


@unittest.skipUnless(importlib.util.find_spec("pypdf"), "pypdf not installed")
class PDFTest(unittest.TestCase):
    def test_extracts_pdf_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "manual.pdf").write_bytes(minimal_pdf(["Beacon manual", "Relays cache flag rules."]))
            [doc] = load_path(tmp)
        self.assertEqual(doc.source, "manual.pdf")
        self.assertIn("Relays cache flag rules.", doc.text)


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


if __name__ == "__main__":
    unittest.main()
