# rag-engine

[![CI](https://github.com/sakshamchitkara-dotcom/rag-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/sakshamchitkara-dotcom/rag-engine/actions/workflows/ci.yml)

A small retrieval-augmented generation (RAG) engine with no required dependencies. It
ingests Markdown, text, HTML and PDF documents, indexes them with an in-repo BM25
implementation plus dense vectors, fuses the two rankings with reciprocal rank fusion,
and answers questions with numbered citations using Claude. Without an API key it falls
back to an extractive answerer, so every command works offline.

It includes a sample corpus about **Beacon**, a made-up feature-flag service, and a
30-question retrieval eval set, so you can try every command right after cloning.

## Quickstart

```bash
git clone https://github.com/sakshamchitkara-dotcom/rag-engine
cd rag-engine
python -m venv .venv && source .venv/bin/activate
pip install -e .              # core: standard library only
# pip install -e ".[all]"     # optional: anthropic SDK (Claude answers) + pypdf (PDF ingest)

rag ingest examples/corpus
rag ask "How do I rotate an SDK key, and does the old one stop working immediately?"
rag eval
rag serve                     # http://127.0.0.1:8000
```

To get answers from Claude, install the `claude` extra and set your key:

```bash
pip install -e ".[claude]"
export ANTHROPIC_API_KEY=sk-ant-...
rag ask "What does the Enterprise plan add over Team?"
```

### Example output (offline, extractive mode)

```text
$ rag ingest examples/corpus
ingested 9 documents -> 42 chunks (embedder hash:1024)
index .rag/index.sqlite: 42 chunks from 9 documents

$ rag ask "How do I rotate an SDK key, and does the old one stop working immediately?"
Keys are rotated from Settings > Environments > Rotate key. [1] After rotation, the old key keeps
working for a 24-hour grace period so that you can redeploy without downtime. [1] SDK keys cannot
be used with the REST API; they only work for flag evaluation. [4]

Sources:
 *[1] troubleshooting.txt - Troubleshooting Beacon > Error "401 invalid sdk key"  (score 0.0328)
  [2] sdk-guide.md - SDK Guide  (score 0.0313)
  [3] troubleshooting.txt - Troubleshooting Beacon > SDK always returns the default value  (score 0.0313)
 *[4] api-reference.html - REST API Reference > Authentication  (score 0.0308)
  [5] sdk-guide.md - SDK Guide > Initializing the Python SDK  (score 0.0306)

(answer mode: extractive; * = cited)

$ rag eval
30 questions, 42 chunks

mode    recall@1  recall@3  recall@5  MRR
------  --------  --------  --------  -----
hybrid  0.767     1.000     1.000     0.878
bm25    0.733     1.000     1.000     0.861
dense   0.700     1.000     1.000     0.839
```

## Commands

| Command | What it does |
|---|---|
| `rag ingest <path-or-url>... [--reset] [--chunk-size 800] [--overlap 150] [--embedder hash:1024]` | Load files, folders (recursively) or `http(s)` URLs, chunk them, embed them and store them. Ingesting a source again replaces its old chunks. |
| `rag ask "<question>" [-k 5] [--mode hybrid\|bm25\|dense] [--no-llm] [--json]` | Retrieve the top-k chunks and answer with `[n]` citations. `*` marks the sources the answer cites. |
| `rag eval [--questions FILE] [--k 1,3,5] [--modes hybrid,bm25,dense] [--json]` | Report recall@k and MRR for each retriever and list the questions each one missed. |
| `rag serve [--host 127.0.0.1] [--port 8000] [-k 5]` | Start a JSON API and a small chat page. |

Every command accepts `--index PATH` (default `.rag/index.sqlite`, or the `RAG_INDEX` environment variable).

### HTTP API

```bash
curl -s localhost:8000/api/health
# {"chunks": 42, "sources": 9, "llm": false}

curl -s localhost:8000/api/ask -H 'Content-Type: application/json' \
  -d '{"question": "How do I authenticate REST API calls?", "k": 3}'
# {"answer": "... [1] ...", "sources": [{"n": 1, "source": "api-reference.html", ...}], "mode": "extractive", "warning": null}
```

`POST /api/ask` accepts `question` (required, up to 2000 characters), `k` (1-20), `mode`, and `llm` (set it to false to skip Claude). Invalid input gets a 400 response with an `error` message.

## Architecture

```
            ingest                                   ask / serve
 files / folders / URLs                                 question
          │                                                │
     loaders.py  md · txt · html→md · pdf (pypdf)          ├──────────────┐
          │                                                ▼              ▼
    chunking.py  heading sections → paragraphs →        BM25 top-50   dense top-50
                 sentences, sentence-aligned overlap       └──── RRF ─────┘
          │                                                       │ top-k chunks
  embeddings.py  hashing vectorizer (or sentence-transformers)    ▼
          │                                                generate.py
      index.py   SQLite: chunks + float32 vectors + meta   Claude (claude-opus-5-5) with [n] citations
                                                           └─ fallback: extractive sentences with [n]
```

| Module | Responsibility |
|---|---|
| `rag_engine/loaders.py` | Reads `.md`, `.txt`, `.html` and `.pdf` from files, folders or URLs and turns each into Markdown-style text. HTML keeps its headings, lists and `<pre>` blocks, and drops `script`, `style`, `nav` and `footer`. In plain text, short standalone lines become headings. PDFs are read with `pypdf` if it is installed; otherwise they are skipped with a notice. |
| `rag_engine/chunking.py` | Splits text into sections at headings, so a chunk never crosses a section boundary. It packs whole paragraphs up to `max_chars`, splits long paragraphs at sentence boundaries (and word boundaries as a last resort), and starts each chunk with the last few sentences of the previous chunk in the same section. Each chunk keeps its heading path, such as `REST API Reference > Rate limits`, and that path is indexed with the text. |
| `rag_engine/text.py` | Tokenizer, stopword list, a light suffix stemmer and a sentence splitter, shared by every component. |
| `rag_engine/bm25.py` | Okapi BM25 (k1=1.5, b=0.75) over an inverted index. |
| `rag_engine/embeddings.py` | The default embedder is a feature-hashing vectorizer: word unigrams and bigrams plus in-word character trigrams, signed hashing into 1024 dimensions, log-scaled term frequencies and L2 normalization. It is deterministic, needs no download, and the character trigrams handle typos that BM25 misses. `--embedder st:<model>` uses `sentence-transformers` instead, if it is installed. |
| `rag_engine/index.py` | Stores chunks and their float32 vectors in SQLite and records which embedder built the index, so queries always use the same one. Search modes are `bm25`, `dense` (brute-force cosine) and `hybrid`, which takes the top 50 from each retriever and merges them with reciprocal rank fusion, `score = Σ 1/(60 + rank)`. |
| `rag_engine/generate.py` | Sends numbered `<source>` blocks to Claude with instructions to answer only from the sources and cite every claim. The fallback scores each retrieved sentence by IDF-weighted overlap with the query (counting its section heading at a discount) and returns the best sentences with their citations. |
| `rag_engine/evaluate.py` | Computes recall@k and MRR. A question counts as answered when a retrieved chunk comes from the expected file and contains the expected phrase, so the labels stay valid when you change chunking settings. |
| `rag_engine/server.py` | `http.server`-based JSON API and single-file chat page (no build step, no external assets). |

### Claude integration

`generate.py` calls `client.messages.create` in the official `anthropic` SDK with model
`claude-opus-5-5`. The key comes from `ANTHROPIC_API_KEY`. Thinking cannot be turned off
on this model, so the request sets `output_config={"effort": "low"}` to keep grounded
Q&A fast and cheap. The code reads `text` blocks by type, because the response also
contains `thinking` blocks. If Claude refuses (`stop_reason == "refusal"`), rejects the
key, rate-limits the request, returns an API error or can't be reached, the answer falls
back to extractive mode and includes a `warning`, so the command never fails because of
the LLM.

### Design choices and limits

- The BM25 postings are rebuilt in memory each time the index loads, and dense search is a brute-force scan. This takes milliseconds for thousands of chunks and should work up to about 100k. Beyond that, persist the inverted index and use an approximate-nearest-neighbour library.
- The stemmer is a small rule-based suffix stripper. Use a Porter or Snowball stemmer if your corpus needs better handling of word forms.
- The hashing embedder captures word overlap, not meaning. On the sample eval it is the weakest retriever alone, but it still helps the hybrid, which ranks the right chunk first more often than BM25 alone. For real semantic retrieval, use `--embedder st:all-MiniLM-L6-v2`.

## Using your own data

```bash
rag --index my.sqlite ingest ~/notes https://example.com/docs/page.html --reset
rag --index my.sqlite ask "..."
```

To evaluate retrieval on your data, write a JSON list of `{"question", "source", "contains"}`
objects (see `examples/eval_questions.json`) and run `rag eval --questions your.json`.

## Development

```bash
pip install -e ".[all]"
python -m unittest discover -s tests -v
```

The tests cover every module. They also run the CLI (ingest, ask, eval) on the sample
corpus, start the HTTP server and exercise the Claude code path against a fake SDK, so
they need no network access or API key. CI runs them on Python 3.10 through 3.13.

## License

MIT
