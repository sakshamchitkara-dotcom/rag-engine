import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rag_engine.evaluate import parse_mode
from rag_engine.index import Index
from rag_engine.loaders import Document
from rag_engine.rewrite import multi_search, rewrite_queries
from tests.test_generate import FakeAnthropic, response


def with_claude(fake):
    return mock.patch.dict(sys.modules, {"anthropic": fake}), mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "k"})


class RewriteTest(unittest.TestCase):
    def test_claude_rewrites_are_cleaned_and_capped(self):
        fake = FakeAnthropic(response("1. rotate sdk key\n- sdk key grace period\n\nHow do I rotate?\n* a\n* b"))
        mods, env = with_claude(fake)
        with mods, env:
            queries, warning = rewrite_queries("How do I rotate?")
        self.assertEqual(queries, ["How do I rotate?", "rotate sdk key", "sdk key grace period", "a"])
        self.assertIsNone(warning)
        self.assertEqual(fake.calls[0]["messages"][0]["content"], "How do I rotate?")

    def test_failure_or_no_claude_searches_the_question_as_written(self):
        fake = FakeAnthropic(error=FakeAnthropic.RateLimitError("slow down"))
        mods, env = with_claude(fake)
        with mods, env:
            queries, warning = rewrite_queries("q?")
        self.assertEqual(queries, ["q?"])
        self.assertEqual(warning, "Claude rate limit hit; searched the question as written")
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            self.assertEqual(rewrite_queries("q?"), (["q?"], None))

    def test_eval_multi_mode_needs_claude(self):
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            with self.assertRaises(ValueError):
                parse_mode("hybrid+multi")
        mods, env = with_claude(FakeAnthropic(response("x")))
        with mods, env:
            self.assertEqual(parse_mode("hybrid+proximity+multi"), ("hybrid", "proximity", True))


class MultiSearchTest(unittest.TestCase):
    def test_fuses_one_ranking_per_query(self):
        with tempfile.TemporaryDirectory() as tmp, Index(Path(tmp) / "i.sqlite") as ix:
            ix.add_documents([
                Document("limits.md", "Limits", "# Limits\n\nThe REST API allows 300 requests per minute."),
                Document("pets.md", "Pets", "# Pets\n\nCats and dogs are popular pets."),
                Document("keys.md", "Keys", "# Keys\n\nRotate keys from the settings page."),
            ])
            single = multi_search(ix, ["api requests per minute"], k=2)
            self.assertEqual([h.chunk.source for h in single], [h.chunk.source for h in ix.search("api requests per minute", k=2)])
            fused = multi_search(ix, ["api requests per minute", "cats and dogs"], k=2)
            self.assertEqual({h.chunk.source for h in fused}, {"limits.md", "pets.md"})
            self.assertLessEqual(set(fused[0].ranks), {"q1", "q2"})


if __name__ == "__main__":
    unittest.main()
