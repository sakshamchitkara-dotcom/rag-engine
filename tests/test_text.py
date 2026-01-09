import unittest

from rag_engine.text import split_sentences, tokenize


class TokenizeTest(unittest.TestCase):
    def test_drops_stopwords_and_lowercases(self):
        self.assertEqual(tokenize("The API and the Keys"), ["api", "key"])

    def test_stemming_collapses_inflections(self):
        self.assertEqual(len(set(tokenize("configure configured configures"))), 1)
        self.assertEqual(tokenize("limits queries"), ["limit", "query"])

    def test_keeps_versions_and_identifiers(self):
        self.assertEqual(tokenize("v2.1 x-api-key 3.4.0"), ["v2.1", "x-api-key", "3.4.0"])

    def test_short_or_ss_words_untouched(self):
        self.assertEqual(tokenize("bus class"), ["bus", "class"])


class SentenceTest(unittest.TestCase):
    def test_splits_on_terminal_punctuation(self):
        self.assertEqual(split_sentences("One. Two! Three? four"), ["One.", "Two!", "Three? four"])

    def test_does_not_split_version_numbers(self):
        self.assertEqual(split_sentences("Use v2.1 today. Then stop."), ["Use v2.1 today.", "Then stop."])

    def test_empty(self):
        self.assertEqual(split_sentences("   "), [])


if __name__ == "__main__":
    unittest.main()
