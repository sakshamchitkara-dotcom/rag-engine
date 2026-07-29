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

    def test_drops_loosely_related_sentences_from_other_chunks(self):
        hits = [
            hit("Every new workspace starts with a 14-day free trial. No credit card is required.", "Pricing > Free trial"),
            hit("The self-hosted edition is free forever.", "Pricing"),
        ]
        text = generate.extractive_answer("How long is the free trial?", hits)
        self.assertIn("14-day free trial", text)
        self.assertNotIn("free forever", text)  # only shares the word "free"

    def test_keeps_other_chunk_that_answers_another_part_of_the_question(self):
        hits = [
            hit("Keys are rotated from the settings page under environments.", "SDK keys > Rotation"),
            hit("After a rotation the old key keeps working for a 24-hour grace period.", "SDK keys > Grace period"),
        ]
        text = generate.extractive_answer("How do I rotate a key and does the old key keep working?", hits)
        self.assertIn("[1]", text)
        self.assertIn("24-hour grace period. [2]", text)

    def test_answers_from_the_matching_table_row(self):
        table = ("| Plan | Price | Seats |\n|---|---|---|\n| Community | Free | Unlimited |\n"
                 "| Team | $25 per seat | Up to 50 |\n| Enterprise | Custom | Unlimited |")
        text = generate.extractive_answer("How many seats does the Team plan allow?",
                                          [hit("Plans are listed below.\n\n" + table, "Pricing > Plans")])
        self.assertEqual(text, "Plan: Team; Price: $25 per seat; Seats: Up to 50 [1]")

    def test_fragments_attach_to_the_previous_sentence(self):
        hits = [hit("- `PORT` - HTTP port for the admin UI. Defaults to 8420.\n- `LOG_LEVEL` - log verbosity. Required.")]
        text = generate.extractive_answer("What port does the server use by default?", hits)
        self.assertEqual(text, "`PORT` - HTTP port for the admin UI. Defaults to 8420. [1]")

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

    def __init__(self, response=None, error=None, deltas=(), stream_error=None):
        super().__init__("anthropic")
        self.calls = []
        fake = self

        class Stream:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            @property
            def text_stream(self):
                for d in deltas:
                    yield d
                if stream_error:
                    raise stream_error

            def get_final_message(self):
                return response

        class Messages:
            def create(self, **kwargs):
                fake.calls.append(kwargs)
                if error:
                    raise error
                return response

            def stream(self, **kwargs):
                fake.calls.append(kwargs)
                if error:
                    raise error
                return Stream()

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

    def test_each_api_failure_has_its_own_warning(self):
        cases = [
            (FakeAnthropic.AuthenticationError(), "ANTHROPIC_API_KEY was rejected"),
            (FakeAnthropic.APIStatusError(), "Claude API error 500"),
            (FakeAnthropic.APIConnectionError(), "could not reach the Claude API"),
        ]
        for error, warning in cases:
            with self.subTest(error=type(error).__name__):
                result = self.run_with(FakeAnthropic(error=error))
                self.assertEqual(result.mode, "extractive")
                self.assertEqual(result.warning, f"{warning}; used extractive fallback")

    def test_empty_reply_falls_back(self):
        result = self.run_with(FakeAnthropic(response("  ")))
        self.assertIn("returned no text", result.warning)

    def test_programming_errors_are_not_swallowed(self):
        with self.assertRaises(KeyError):
            self.run_with(FakeAnthropic(error=KeyError("bug")))

    def test_sdk_not_installed_means_no_claude(self):
        with mock.patch.dict(sys.modules, {"anthropic": None}), \
                mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
            self.assertFalse(generate.claude_available())
            self.assertEqual(generate.answer("How many requests per minute?", HITS).mode, "extractive")

    def test_no_llm_flag(self):
        fake = FakeAnthropic(response("unused"))
        with mock.patch.dict(sys.modules, {"anthropic": fake}), mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "k"}):
            result = generate.answer("requests per minute", HITS, use_llm=False)
        self.assertEqual(result.mode, "extractive")
        self.assertEqual(fake.calls, [])


class StreamAnswerTest(unittest.TestCase):
    def events(self, fake, key="sk-test", hits=HITS):
        env = {"ANTHROPIC_API_KEY": key} if key else {}
        with mock.patch.dict(sys.modules, {"anthropic": fake}), mock.patch.dict("os.environ", env, clear=True):
            return list(generate.stream_answer("How many requests per minute?", hits))

    def test_streams_claude_deltas(self):
        fake = FakeAnthropic(response("", stop_reason="end_turn"), deltas=["The API allows ", "300 [1]."])
        events = self.events(fake)
        self.assertEqual(events[0][0], "sources")
        self.assertEqual([d for e, d in events if e == "delta"], ["The API allows ", "300 [1]."])
        self.assertEqual(events[-1], ("done", {"mode": "claude", "warning": None}))
        self.assertIn("effort", fake.calls[0]["output_config"])

    def test_error_before_any_text_falls_back_to_extractive_delta(self):
        events = self.events(FakeAnthropic(error=FakeAnthropic.APIConnectionError()))
        kinds = [e for e, _ in events]
        self.assertEqual(kinds, ["sources", "delta", "done"])
        self.assertIn("300 requests per minute", events[1][1])
        self.assertEqual(events[-1][1]["mode"], "extractive")
        self.assertIn("could not reach", events[-1][1]["warning"])

    def test_error_mid_stream_replaces_partial_text(self):
        fake = FakeAnthropic(response(""), deltas=["The API "], stream_error=FakeAnthropic.RateLimitError())
        kinds = [e for e, _ in self.events(fake)]
        self.assertEqual(kinds, ["sources", "delta", "replace", "done"])

    def test_refusal_after_stream_replaces(self):
        fake = FakeAnthropic(response("", stop_reason="refusal"), deltas=["I can't"])
        events = self.events(fake)
        self.assertEqual(events[-2][0], "replace")
        self.assertIn("declined", events[-1][1]["warning"])

    def test_offline_and_empty(self):
        events = self.events(FakeAnthropic(response("unused")), key=None)
        self.assertEqual([e for e, _ in events], ["sources", "delta", "done"])
        self.assertIsNone(events[-1][1]["warning"])
        self.assertEqual(self.events(FakeAnthropic(), key=None, hits=[])[1], ("delta", generate.NO_ANSWER))


if __name__ == "__main__":
    unittest.main()
