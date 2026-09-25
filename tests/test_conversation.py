import unittest
from unittest import mock

from rag_engine.conversation import condense, is_follow_up
from tests.test_generate import FakeAnthropic, response
from tests.test_rewrite import with_claude


class HeuristicTest(unittest.TestCase):
    def test_detects_follow_ups(self):
        for q in ("Does the old one keep working?", "And the Team plan?", "What about Admins?", "How long?"):
            self.assertTrue(is_follow_up(q), q)
        for q in ("What port does beacon-server listen on by default?", "How do I rotate an SDK key?", ""):
            self.assertFalse(is_follow_up(q), q)

    def test_carries_context_words_from_the_previous_question(self):
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            self.assertEqual(condense("Does the old one keep working?", ["How do I rotate an SDK key?"]),
                             ("Does the old one keep working? (rotate SDK key)", None))
            # Chained: the second follow-up inherits the first one's carried context.
            self.assertEqual(condense("How long should I wait?",
                                      ["What is the REST API rate limit?", "What happens when I go over it?"])[0],
                             "How long should I wait? (go REST API rate limit)")

    def test_standalone_questions_and_empty_history_pass_through(self):
        q = "What port does beacon-server listen on by default?"
        self.assertEqual(condense(q, ["Is there a free trial?"], use_llm=False), (q, None))
        self.assertEqual(condense("Does it?", ["", "  "], use_llm=False), ("Does it?", None))


class ClaudeTest(unittest.TestCase):
    def test_claude_rewrites_the_follow_up(self):
        fake = FakeAnthropic(response("Does the old SDK key keep working after rotation?\n"))
        mods, env = with_claude(fake)
        with mods, env:
            self.assertEqual(condense("Does the old one keep working?", ["How do I rotate an SDK key?"]),
                             ("Does the old SDK key keep working after rotation?", None))
        prompt = fake.calls[0]["messages"][0]["content"]
        self.assertIn("1. How do I rotate an SDK key?", prompt)
        self.assertIn("Latest question: Does the old one keep working?", prompt)

    def test_claude_failure_falls_back_to_the_heuristic(self):
        fake = FakeAnthropic(error=FakeAnthropic.RateLimitError("slow down"))
        mods, env = with_claude(fake)
        with mods, env:
            self.assertEqual(condense("How long is it?", ["Is there a free trial?"]),
                             ("How long is it? (free trial)",
                              "Claude rate limit hit; used the offline follow-up heuristic"))

    def test_no_claude_call_without_history(self):
        fake = FakeAnthropic(response("unused"))
        mods, env = with_claude(fake)
        with mods, env:
            self.assertEqual(condense("How long is it?", []), ("How long is it?", None))
        self.assertEqual(fake.calls, [])


if __name__ == "__main__":
    unittest.main()
