import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rag_engine.evaluate import heuristic_judgement, judge_answers
from rag_engine.index import Index
from rag_engine.loaders import load_path
from tests.test_generate import FakeAnthropic, response

CORPUS = Path(__file__).resolve().parent.parent / "examples" / "corpus"
SOURCES = [{"n": 1, "text": "Backups are taken every 6 hours and kept for 30 days."},
           {"n": 2, "text": "The REST API allows 300 requests per minute."}]
LABEL = {"question": "How often are backups taken?", "source": "security.md", "contains": "every 6 hours"}


class HeuristicJudgeTest(unittest.TestCase):
    def test_correct_and_grounded(self):
        self.assertEqual(heuristic_judgement(LABEL, "Backups are taken every 6 hours [1].", SOURCES),
                         (True, True, ""))

    def test_claim_citing_the_wrong_source_is_ungrounded(self):
        correct, grounded, reason = heuristic_judgement(
            LABEL, "Backups are taken every 6 hours [1]. Backups are encrypted with AES-256 [2].", SOURCES)
        self.assertTrue(correct)
        self.assertFalse(grounded)
        self.assertIn("encrypted", reason)

    def test_uncited_trailing_claim_is_ungrounded(self):
        self.assertFalse(heuristic_judgement(LABEL, "Every 6 hours [1]. Restores take minutes.", SOURCES)[1])

    def test_wrong_answer(self):
        correct, _, reason = heuristic_judgement(LABEL, "Backups are kept for 30 days [1].", SOURCES)
        self.assertFalse(correct)
        self.assertIn("missing", reason)


class JudgeAnswersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.index = Index(Path(cls.tmp.name) / "i.sqlite")
        cls.index.add_documents(load_path(str(CORPUS)))

    @classmethod
    def tearDownClass(cls):
        cls.index.close()
        cls.tmp.cleanup()

    def test_offline_uses_extractive_answers_and_heuristic_judge(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            report = judge_answers(self.index, [LABEL])
        self.assertEqual((report.answered_by, report.correct, report.grounded), ("extractive", 1.0, 1.0))
        self.assertEqual(report.to_dict()["judges"], ["heuristic"])

    def test_claude_judges_when_available(self):
        verdict = '{"correct": false, "grounded": true, "reason": "gives the retention, not the frequency"}'
        fake = FakeAnthropic(response(verdict))
        with mock.patch.dict(sys.modules, {"anthropic": fake}), \
                mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            report = judge_answers(self.index, [LABEL])
        self.assertEqual(len(fake.calls), 2)  # answer, then judge
        self.assertIn("Reference fact: every 6 hours", fake.calls[1]["messages"][0]["content"])
        self.assertEqual((report.answered_by, report.results[0].judge), ("claude", "claude"))
        self.assertEqual(report.to_dict()["failures"][0]["reason"], "gives the retention, not the frequency")

    def test_unparseable_verdict_falls_back_to_heuristic(self):
        fake = FakeAnthropic(response("Backups are taken every 6 hours [1]."))  # not JSON
        with mock.patch.dict(sys.modules, {"anthropic": fake}), \
                mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            report = judge_answers(self.index, [LABEL])
        self.assertEqual(report.results[0].judge, "heuristic")
        self.assertTrue(report.results[0].correct)


if __name__ == "__main__":
    unittest.main()
