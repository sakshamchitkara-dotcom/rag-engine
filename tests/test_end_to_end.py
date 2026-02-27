"""End-to-end: CLI ingest/ask/eval on the bundled corpus, plus the HTTP API."""

import contextlib
import io
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from rag_engine.cli import main
from rag_engine.evaluate import evaluate, evaluate_answers, load_questions
from rag_engine.index import Index
from rag_engine.server import make_handler

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "examples" / "corpus"
QUESTIONS = ROOT / "examples" / "eval_questions.json"


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        code = main(list(argv))
    return code, out.getvalue()


class EndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.index_path = str(Path(cls.tmp.name) / "index.sqlite")
        cls.env = mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""})  # force offline answers
        cls.env.start()
        code, out = run("--index", cls.index_path, "ingest", str(CORPUS), "--reset")
        assert code == 0, out

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        cls.tmp.cleanup()

    def test_ingest_reports_counts(self):
        _, out = run("--index", self.index_path, "ingest", str(CORPUS / "pricing.md"))
        self.assertIn("ingested 1 documents", out)
        self.assertIn("from 9 documents", out)  # re-ingest replaced, not duplicated

    def test_ask_json(self):
        code, out = run("--index", self.index_path, "ask", "What is the REST API rate limit?", "--json")
        self.assertEqual(code, 0)
        result = json.loads(out)
        self.assertEqual(result["mode"], "extractive")
        self.assertIn("300 requests per minute", result["answer"])
        self.assertEqual(result["sources"][0]["source"], "api-reference.html")

    def test_ask_text_marks_cited_sources(self):
        _, out = run("--index", self.index_path, "ask", "How long is the free trial?")
        self.assertIn("14-day free trial", out)
        self.assertIn("*[1] pricing.md", out)

    def test_ask_on_empty_index(self):
        code, _ = run("--index", str(Path(self.tmp.name) / "empty.sqlite"), "ask", "anything")
        self.assertEqual(code, 1)

    def test_eval_quality_floor(self):
        questions = load_questions(QUESTIONS)
        index = Index(self.index_path)
        reports = {r.mode: r for r in evaluate(index, questions)}
        index.close()
        self.assertGreaterEqual(reports["hybrid"].recall_at(5), 0.9)
        self.assertGreaterEqual(reports["hybrid"].mrr, 0.7)

    def test_answer_quality_floor(self):
        index = Index(self.index_path)
        report = evaluate_answers(index, load_questions(QUESTIONS))
        index.close()
        self.assertGreaterEqual(report.found_rate, 0.8)
        self.assertGreaterEqual(report.precision, 0.5)

    def test_eval_json_has_answer_metrics(self):
        code, out = run("--index", self.index_path, "eval", "--questions", str(QUESTIONS), "--json")
        self.assertEqual(code, 0)
        result = json.loads(out)
        self.assertEqual([r["mode"] for r in result["retrieval"]], ["hybrid", "bm25", "dense"])
        self.assertEqual(set(result["answers"]), {"found", "precision", "avg_sentences", "misses"})

    def test_eval_cli(self):
        code, out = run("--index", self.index_path, "eval", "--questions", str(QUESTIONS))
        self.assertEqual(code, 0)
        self.assertIn("recall@5", out)
        self.assertIn("hybrid", out)
        self.assertIn("Answer quality", out)


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.index = Index(Path(cls.tmp.name) / "index.sqlite")
        from rag_engine.loaders import load_path

        cls.index.add_documents(load_path(str(CORPUS)))
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.index))
        cls.server.RequestHandlerClass.log_message = lambda *a: None
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.index.close()
        cls.tmp.cleanup()

    def post(self, payload):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        req = urllib.request.Request(self.base + "/api/ask", data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as err:
            with err:
                return err.code, json.loads(err.read())

    def test_page_and_health(self):
        with urllib.request.urlopen(self.base + "/") as resp:
            self.assertIn(b"<title>rag-engine</title>", resp.read())
        with urllib.request.urlopen(self.base + "/api/health") as resp:
            health = json.loads(resp.read())
        self.assertEqual(health["sources"], 9)
        self.assertGreater(health["chunks"], 9)

    def test_ask(self):
        status, body = self.post({"question": "How often are backups taken?", "k": 3, "llm": False})
        self.assertEqual(status, 200)
        self.assertIn("every 6 hours", body["answer"])
        self.assertEqual(len(body["sources"]), 3)

    def test_validation(self):
        self.assertEqual(self.post({"question": ""})[0], 400)
        self.assertEqual(self.post({"question": "x", "k": 0})[0], 400)
        self.assertEqual(self.post({"question": "x", "k": True})[0], 400)
        self.assertEqual(self.post({"question": "x", "mode": "magic"})[0], 400)
        self.assertEqual(self.post(b"not json")[0], 400)
        self.assertEqual(self.post(["list"])[0], 400)
        self.assertEqual(self.post({"question": "x" * 3000})[0], 400)

    def test_unknown_route(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self.base + "/nope")
        with ctx.exception:
            self.assertEqual(ctx.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
