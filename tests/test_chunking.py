import unittest

from rag_engine.chunking import chunk_document, split_sections

DOC = """# Guide

Intro paragraph.

## Setup

First setup paragraph.

```
# a comment, not a heading
run --fast
```

## Limits
### API
The API allows 300 requests per minute.
"""


class SectionsTest(unittest.TestCase):
    def test_heading_paths_and_code_fences(self):
        sections = split_sections(DOC)
        self.assertEqual([h for h, _ in sections], ["Guide", "Guide > Setup", "Guide > Limits > API"])
        setup = sections[1][1]
        self.assertIn("# a comment, not a heading", setup[1])

    def test_sibling_heading_pops_stack(self):
        sections = split_sections("# A\n## B\nb\n## C\nc\n# D\nd")
        self.assertEqual([h for h, _ in sections], ["A > B", "A > C", "D"])


class ChunkTest(unittest.TestCase):
    def test_chunks_never_cross_sections(self):
        chunks = chunk_document(DOC, source="g.md", title="Guide")
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[2].heading, "Guide > Limits > API")
        self.assertEqual([c.id for c in chunks], ["g.md#0", "g.md#1", "g.md#2"])

    def test_long_sections_respect_max_and_overlap(self):
        body = " ".join(f"Sentence {i} describes step {i} in detail." for i in range(60))
        chunks = chunk_document("# T\n" + body, source="t.md", title="T", max_chars=300, overlap=100)
        self.assertGreater(len(chunks), 3)
        for c in chunks:
            self.assertLessEqual(len(c.text), 300)
        for prev, nxt in zip(chunks, chunks[1:]):
            last_sentence = prev.text.split("\n\n")[-1]
            seed = nxt.text.split("\n\n")[0]
            self.assertTrue(seed.endswith(last_sentence), "next chunk should open with the overlap")
            self.assertLessEqual(len(seed), 100)
        joined = " ".join(c.text for c in chunks)
        for i in range(60):
            self.assertIn(f"Sentence {i} ", joined)

    def test_giant_sentence_is_hard_split(self):
        chunks = chunk_document("word " * 500, source="w.txt", title="W", max_chars=200, overlap=0)
        self.assertTrue(all(len(c.text) <= 200 for c in chunks))

    def test_overlap_must_be_smaller_than_chunk(self):
        with self.assertRaises(ValueError):
            chunk_document("x", source="x", title="x", max_chars=100, overlap=100)

    def test_indexed_text_includes_heading(self):
        chunk = chunk_document(DOC, source="g.md", title="Guide")[2]
        self.assertIn("Limits > API", chunk.indexed_text)


if __name__ == "__main__":
    unittest.main()
