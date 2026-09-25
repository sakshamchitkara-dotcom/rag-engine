import builtins
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


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


if __name__ == "__main__":
    unittest.main()
