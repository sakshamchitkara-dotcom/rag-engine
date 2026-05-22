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
        self.assertIn("0 added, 0 updated, 1 unchanged, 0 removed -> 0 chunks written", out)
        self.assertIn("from 9 documents", out)  # re-ingest replaced, not duplicated

    def test_reingesting_a_folder_picks_up_edits_and_deletions(self):
        index = str(Path(self.tmp.name) / "sync.sqlite")
        folder = Path(self.tmp.name) / "docs"
        folder.mkdir()
        (folder / "a.md").write_text("# A\n\nAlpha rotates keys weekly.")
        (folder / "b.md").write_text("# B\n\nBravo backs up hourly.")
        run("--index", index, "ingest", str(folder))
        (folder / "a.md").write_text("# A\n\nAlpha rotates keys daily.")
        (folder / "b.md").unlink()
        (folder / "c.md").write_text("# C\n\nCharlie caches responses.")
        code, out = run("--index", index, "ingest", str(folder))
        self.assertEqual(code, 0)
        self.assertIn("1 added, 1 updated, 0 unchanged, 1 removed", out)
        self.assertIn("removed b.md", out)
        with Index(index) as ix:
            self.assertEqual(ix.sources(), ["a.md", "c.md"])
            self.assertIn("daily", ix.search("alpha keys", k=1)[0].chunk.text)

    def test_remove(self):
        index = str(Path(self.tmp.name) / "remove.sqlite")
        run("--index", index, "ingest", str(CORPUS))
        code, out = run("--index", index, "remove", "pricing.md", str(CORPUS / "security.md"))
        self.assertEqual(code, 0)
        self.assertIn("removed pricing.md", out)
        self.assertIn("removed security.md", out)
        self.assertIn("from 7 documents", out)
        self.assertEqual(run("--index", index, "remove", "pricing.md")[0], 1)  # already gone
        code, out = run("--index", index, "remove", str(CORPUS))  # whole ingested folder
        self.assertEqual(code, 0)
        self.assertIn("0 chunks from 0 documents", out)

    def test_stats(self):
        code, out = run("--index", self.index_path, "stats", "--json")
        self.assertEqual(code, 0)
        st = json.loads(out)
        self.assertEqual(st["documents"], 9)
        self.assertEqual(st["embedder"], "hash:1024")
        pricing = next(d for d in st["sources"] if d["source"] == "pricing.md")
        self.assertEqual(pricing["origin"], str(CORPUS))
        self.assertGreater(pricing["chunks"], 0)
        self.assertEqual(sum(d["chunks"] for d in st["sources"]), st["chunks"])
        _, out = run("--index", self.index_path, "stats")
        self.assertIn("contents  %d chunks from 9 documents" % st["chunks"], out)

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

    def test_ask_with_reranker(self):
        code, out = run("--index", self.index_path, "ask", "What is the REST API rate limit?",
                        "--rerank", "proximity", "--json")
        self.assertEqual(code, 0)
        self.assertIn("300 requests per minute", json.loads(out)["answer"])

    def test_ask_with_source_filter(self):
        code, out = run("--index", self.index_path, "ask", "How long is the free trial?",
                        "--source", "troubleshooting.*", "--json")
        self.assertEqual(code, 0)
        self.assertEqual({s["source"] for s in json.loads(out)["sources"]}, {"troubleshooting.txt"})

    def test_ask_on_empty_index(self):
        code, _ = run("--index", str(Path(self.tmp.name) / "empty.sqlite"), "ask", "anything")
        self.assertEqual(code, 1)

    def test_cli_errors_are_one_line_not_tracebacks(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(main(["--index", str(ROOT / "README.md"), "stats"]), 2)
            self.assertEqual(main(["--index", self.index_path, "ingest", "http://127.0.0.1:9/x.html"]), 2)
        lines = err.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("is not a usable rag index", lines[0])
        self.assertIn("could not fetch URL", lines[1])

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
        modes = ["hybrid", "bm25", "dense", "hybrid+proximity", "hybrid+mmr"]
        self.assertEqual([r["mode"] for r in result["retrieval"]], modes)
        self.assertEqual([a["mode"] for a in result["answers"]], modes)
        self.assertEqual(set(result["answers"][0]), {"mode", "found", "precision", "avg_sentences", "misses"})

    def test_reranked_eval_quality_floor(self):
        index = Index(self.index_path)
        (report,) = evaluate(index, load_questions(QUESTIONS), modes=["hybrid+proximity"])
        index.close()
        self.assertGreaterEqual(report.recall_at(5), 0.9)
        self.assertGreaterEqual(report.mrr, 0.85)

    def test_eval_rejects_unknown_mode(self):
        code, _ = run("--index", self.index_path, "eval", "--questions", str(QUESTIONS), "--modes", "hybrid+magic")
        self.assertEqual(code, 2)

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
            page = resp.read()
        self.assertIn(b"<title>rag-engine</title>", page)
        self.assertIn(b"/api/ask/stream", page)
        with urllib.request.urlopen(self.base + "/api/health") as resp:
            health = json.loads(resp.read())
        self.assertEqual(health["sources"], 9)
        self.assertGreater(health["chunks"], 9)

    def test_ask(self):
        status, body = self.post({"question": "How often are backups taken?", "k": 3, "llm": False})
        self.assertEqual(status, 200)
        self.assertIn("every 6 hours", body["answer"])
        self.assertEqual(len(body["sources"]), 3)

    def test_ask_reranked(self):
        status, body = self.post({"question": "How often are backups taken?", "rerank": "mmr", "llm": False})
        self.assertEqual(status, 200)
        self.assertIn("every 6 hours", body["answer"])

    def test_ask_filtered(self):
        status, body = self.post({"question": "How often are backups taken?", "llm": False,
                                  "source": ["security.md", "pricing.*"]})
        self.assertEqual(status, 200)
        self.assertTrue(body["sources"])
        self.assertLessEqual({s["source"] for s in body["sources"]}, {"security.md", "pricing.md"})
        status, body = self.post({"question": "backups", "llm": False, "tag": "no-such-tag"})
        self.assertEqual((status, body["sources"]), (200, []))

    def stream(self, payload):
        req = urllib.request.Request(self.base + "/api/ask/stream", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            self.assertTrue(resp.headers["Content-Type"].startswith("text/event-stream"))
            raw = resp.read().decode()
        events = []
        for block in raw.strip().split("\n\n"):
            fields = dict(line.split(": ", 1) for line in block.splitlines())
            events.append((fields["event"], json.loads(fields["data"])))
        return events

    def test_stream_offline_fallback(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            events = self.stream({"question": "How often are backups taken?", "k": 3})
        self.assertEqual([e for e, _ in events], ["sources", "delta", "done"])
        self.assertEqual(len(events[0][1]), 3)
        self.assertIn("every 6 hours", events[1][1])
        self.assertEqual(events[2][1]["mode"], "extractive")

    def test_stream_claude_deltas(self):
        from tests.test_generate import FakeAnthropic, response

        fake = FakeAnthropic(response(""), deltas=["Backups run ", "every 6 hours [1]."])
        with mock.patch.dict("sys.modules", {"anthropic": fake}), \
                mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            events = self.stream({"question": "How often are backups taken?"})
        self.assertEqual([d for e, d in events if e == "delta"], ["Backups run ", "every 6 hours [1]."])
        self.assertEqual(events[-1], ("done", {"mode": "claude", "warning": None}))

    def test_stream_validation(self):
        req = urllib.request.Request(self.base + "/api/ask/stream", data=b'{"question": ""}')
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        with ctx.exception:
            self.assertEqual(ctx.exception.code, 400)

    def test_validation(self):
        self.assertEqual(self.post({"question": ""})[0], 400)
        self.assertEqual(self.post({"question": "x", "k": 0})[0], 400)
        self.assertEqual(self.post({"question": "x", "k": True})[0], 400)
        self.assertEqual(self.post({"question": "x", "mode": "magic"})[0], 400)
        self.assertEqual(self.post({"question": "x", "rerank": "magic"})[0], 400)
        self.assertEqual(self.post({"question": "x", "source": 3})[0], 400)
        self.assertEqual(self.post({"question": "x", "tag": [""]})[0], 400)
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
