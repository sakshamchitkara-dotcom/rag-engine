import tempfile
import unittest
from pathlib import Path

from rag_engine.bm25 import BM25
from rag_engine.embeddings import cosine, get_embedder
from rag_engine.index import Index, rrf
from rag_engine.loaders import Document


class BM25Test(unittest.TestCase):
    def setUp(self):
        self.bm25 = BM25()
        for text in ["the cat sat on the mat", "dogs chase cats in the park",
                     "the API allows 300 requests per minute", "cats cats cats everywhere"]:
            self.bm25.add(text)

    def test_ranks_matching_doc_first(self):
        self.assertEqual(self.bm25.search("api request limit")[0][0], 2)

    def test_term_frequency_saturates_but_counts(self):
        ranked = [d for d, _ in self.bm25.search("cat")]
        self.assertEqual(ranked[0], 3)
        self.assertEqual(set(ranked), {0, 1, 3})

    def test_no_match_and_empty(self):
        self.assertEqual(self.bm25.search("zebra"), [])
        self.assertEqual(BM25().search("cat"), [])

    def test_rare_terms_weigh_more(self):
        self.assertGreater(self.bm25.idf("api"), self.bm25.idf("cat"))


class EmbeddingTest(unittest.TestCase):
    def test_deterministic_unit_vectors(self):
        e = get_embedder("hash:256")
        a, b = e.embed(["rate limits"]), e.embed(["rate limits"])
        self.assertEqual(list(a[0]), list(b[0]))
        self.assertAlmostEqual(cosine(a[0], a[0]), 1.0, places=5)

    def test_similar_text_scores_higher(self):
        e = get_embedder()
        q, near, far = e.embed(["API rate limiting", "the api rate limit is 300", "bananas are yellow"])
        self.assertGreater(cosine(q, near), cosine(q, far))

    def test_char_ngrams_give_fuzzy_match(self):
        e = get_embedder()
        q, typo = e.embed(["authentication", "authentcation"])
        self.assertGreater(cosine(q, typo), 0.5)

    def test_unknown_embedder(self):
        with self.assertRaises(ValueError):
            get_embedder("nope")


class RRFTest(unittest.TestCase):
    def test_fusion_rewards_agreement(self):
        fused = rrf({"a": [1, 2, 3], "b": [2, 1, 4]}, k=60)
        top_two = {fused[0][0], fused[1][0]}
        self.assertEqual(top_two, {1, 2})
        self.assertAlmostEqual(fused[0][1], 1 / 61 + 1 / 62)
        self.assertEqual(dict((d, r) for d, _, r in fused)[4], {"b": 3})


DOCS = [
    Document("limits.md", "Limits", "# Limits\n\n## API\n\nThe REST API allows 300 requests per minute."),
    Document("pets.md", "Pets", "# Pets\n\nCats and dogs are popular pets."),
]


class IndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "idx" / "index.sqlite"

    def tearDown(self):
        self.tmp.cleanup()

    def test_persists_and_searches_in_every_mode(self):
        ix = Index(self.path)
        self.assertEqual(ix.add_documents(DOCS), 2)
        ix.close()
        ix = Index(self.path)
        self.assertEqual(len(ix.chunks), 2)
        for mode in ("hybrid", "bm25", "dense"):
            hits = ix.search("how many requests per minute", k=1, mode=mode)
            self.assertEqual(hits[0].chunk.source, "limits.md", mode)
        hybrid = ix.search("requests per minute", k=2)[0]
        self.assertEqual(set(hybrid.ranks), {"bm25", "dense"})
        ix.close()

    def test_reingest_replaces_source(self):
        ix = Index(self.path)
        ix.add_documents(DOCS)
        ix.add_documents([Document("pets.md", "Pets", "# Pets\n\nParrots talk.\n\n## More\n\nFish swim.")])
        self.assertEqual(sorted(c.source for c in ix.chunks), ["limits.md", "pets.md", "pets.md"])
        self.assertEqual(ix.search("cats dogs", mode="bm25"), [])
        ix.close()

    def test_embedder_is_pinned(self):
        ix = Index(self.path, embedder="hash:128")
        ix.add_documents(DOCS)
        ix.close()
        self.assertEqual(Index(self.path).embedder_name, "hash:128")
        with self.assertRaises(ValueError):
            Index(self.path, embedder="hash:256")

    def test_bad_mode_and_blank_query(self):
        ix = Index(self.path)
        ix.add_documents(DOCS)
        with self.assertRaises(ValueError):
            ix.search("x", mode="magic")
        self.assertEqual(ix.search("   "), [])
        ix.close()


if __name__ == "__main__":
    unittest.main()
