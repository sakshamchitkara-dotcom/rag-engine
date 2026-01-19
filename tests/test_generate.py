import sys
import types
import unittest
from unittest import mock

from rag_engine import generate
from rag_engine.chunking import Chunk
from rag_engine.index import Hit


def hit(text, heading="Section", source="doc.md", score=1.0):
    return Hit(Chunk(f"{source}#0", source, "Doc", heading, text, 0), score)


HITS = [
    hit("The REST API allows 300 requests per minute per token. Tokens are created in settings.", "API > Rate limits"),
    hit("Every new workspace starts with a 14-day free trial. No credit card is required.", "Pricing > Trial"),
]


class ExtractiveTest(unittest.TestCase):
    def test_cites_the_supporting_source(self):
        text = generate.extractive_answer("How many requests per minute does the API allow?", HITS)
        self.assertIn("300 requests per minute", text)
        self.assertIn("[1]", text)
        self.assertNotIn("[2]", text)

    def test_second_source_numbering(self):
        text = generate.extractive_answer("How long is the free trial?", HITS)
        self.assertTrue(text.startswith("Every new workspace starts with a 14-day free trial. [2]"))

    def test_no_overlap_means_no_answer(self):
        self.assertEqual(generate.extractive_answer("capital of France", HITS), generate.NO_ANSWER)
        self.assertEqual(generate.extractive_answer("anything", []), generate.NO_ANSWER)

    def test_skips_code_blocks(self):
        text = generate.extractive_answer("docker run port", [hit("```\ndocker run -p 8420\n```\n\nUse docker run to start it.")])
        self.assertEqual(text, "Use docker run to start it. [1]")

    def test_cited_numbers(self):
        self.assertEqual(generate.cited_numbers("a [2] b [1][2] c [3]"), [2, 1, 3])


class FakeAnthropic(types.ModuleType):
    """Stand-in for the anthropic SDK: records the request, returns a canned response."""

    class APIStatusError(Exception):
        status_code = 500

    class AuthenticationError(APIStatusError):
        status_code = 401

    class RateLimitError(APIStatusError):
        status_code = 429

    class APIConnectionError(Exception):
        pass

    def __init__(self, response=None, error=None):
        super().__init__("anthropic")
        self.calls = []
        fake = self

        class Messages:
            def create(self, **kwargs):
                fake.calls.append(kwargs)
                if error:
                    raise error
                return response

        class Anthropic:
            def __init__(self):
                self.messages = Messages()

        self.Anthropic = Anthropic


def response(text, stop_reason="end_turn"):
    blocks = [types.SimpleNamespace(type="thinking", thinking=""), types.SimpleNamespace(type="text", text=text)]
    return types.SimpleNamespace(content=blocks, stop_reason=stop_reason)


class ClaudePathTest(unittest.TestCase):
    def run_with(self, fake, key="sk-test"):
        env = {"ANTHROPIC_API_KEY": key} if key else {}
        with mock.patch.dict(sys.modules, {"anthropic": fake}), mock.patch.dict("os.environ", env, clear=True):
            return generate.answer("How many requests per minute?", HITS)

    def test_uses_claude_with_numbered_sources(self):
        fake = FakeAnthropic(response("The API allows 300 requests per minute [1]."))
        result = self.run_with(fake)
        self.assertEqual(result.mode, "claude")
        self.assertEqual(result.text, "The API allows 300 requests per minute [1].")
        call = fake.calls[0]
        self.assertEqual(call["model"], "claude-opus-5-5")
        self.assertNotIn("thinking", call)  # Opus 5.5 rejects disabling thinking; effort is the dial
        self.assertIn("effort", call["output_config"])
        prompt = call["messages"][0]["content"]
        self.assertIn('<source id="1" document="doc.md" section="API > Rate limits">', prompt)
        self.assertIn('<source id="2"', prompt)
        self.assertIn("Question: How many requests per minute?", prompt)
        self.assertEqual([s["n"] for s in result.sources], [1, 2])

    def test_no_key_uses_extractive_without_calling_api(self):
        fake = FakeAnthropic(response("unused"))
        result = self.run_with(fake, key=None)
        self.assertEqual(result.mode, "extractive")
        self.assertEqual(fake.calls, [])
        self.assertIsNone(result.warning)

    def test_api_error_falls_back_with_warning(self):
        result = self.run_with(FakeAnthropic(error=FakeAnthropic.RateLimitError()))
        self.assertEqual(result.mode, "extractive")
        self.assertIn("rate limit", result.warning)
        self.assertIn("[1]", result.text)

    def test_refusal_falls_back(self):
        result = self.run_with(FakeAnthropic(response("", stop_reason="refusal")))
        self.assertEqual(result.mode, "extractive")
        self.assertIn("declined", result.warning)

    def test_no_llm_flag(self):
        fake = FakeAnthropic(response("unused"))
        with mock.patch.dict(sys.modules, {"anthropic": fake}), mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "k"}):
            result = generate.answer("requests per minute", HITS, use_llm=False)
        self.assertEqual(result.mode, "extractive")
        self.assertEqual(fake.calls, [])


if __name__ == "__main__":
    unittest.main()
