import tempfile
import unittest
from pathlib import Path

from rag_engine.index import Index
from rag_engine.loaders import Document
from rag_engine.rerank import min_span, mmr, proximity_score

ONE = lambda term: 1.0  # noqa: E731  uniform idf


class ProximityTest(unittest.TestCase):
    def test_min_span(self):
        self.assertEqual(min_span({"a": [0, 9], "b": [5]}), 5)  # positions 5..9
        self.assertEqual(min_span({"a": [3]}), 1)
        self.assertEqual(min_span({"a": [0, 4], "b": [1], "c": [2]}), 3)

    def test_adjacent_terms_beat_scattered_terms(self):
        q = {"rate", "limit"}
        near = proximity_score(q, "the rate limit is 300".split(), ONE)
        far = proximity_score(q, "rate a b c d e f limit".split(), ONE)
        partial = proximity_score(q, "rate only".split(), ONE)
        self.assertEqual(near, 1.0)
        self.assertGreater(near, far)
        self.assertGreater(far, partial)
        self.assertEqual(proximity_score(q, "nothing here".split(), ONE), 0.0)


class MMRTest(unittest.TestCase):
    def test_skips_near_duplicate(self):
        sim = lambda i, j: 1.0 if {i, j} == {0, 1} else 0.0  # noqa: E731
        self.assertEqual(mmr([1.0, 0.8, 0.5], sim, 2), [0, 2])

    def test_pure_relevance_when_lambda_is_one(self):
        self.assertEqual(mmr([0.2, 0.9, 0.5], lambda i, j: 1.0, 3, lam=1.0), [1, 2, 0])


class IndexRerankTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ix = Index(Path(self.tmp.name) / "index.sqlite")
        self.ix.add_documents([
            Document("a.md", "A", "# A\n\nThe limit on retries is five. Rate changes are logged daily."),
            Document("b.md", "B", "# B\n\nThe rate limit is 300 requests per minute."),
            Document("c.md", "C", "# C\n\nThe rate limit is 300 requests per minute, as noted."),
        ])

    def tearDown(self):
        self.ix.close()
        self.tmp.cleanup()

    def test_proximity_records_rank_and_keeps_k(self):
        hits = self.ix.search("rate limit", k=2, rerank="proximity")
        self.assertEqual(len(hits), 2)
        self.assertNotEqual(hits[0].chunk.source, "a.md")  # terms far apart
        self.assertIn("proximity", hits[0].ranks)

    def test_mmr_demotes_duplicate(self):
        plain = [h.chunk.source for h in self.ix.search("rate limit requests", k=3)]
        diverse = [h.chunk.source for h in self.ix.search("rate limit requests", k=3, rerank="mmr")]
        self.assertEqual(plain[:2], ["b.md", "c.md"])
        self.assertEqual(diverse[:2], ["b.md", "a.md"])

    def test_unknown_reranker(self):
        with self.assertRaises(ValueError):
            self.ix.search("x", rerank="magic")
        self.assertEqual(len(self.ix.search("rate", k=1, rerank="none")), 1)


if __name__ == "__main__":
    unittest.main()
