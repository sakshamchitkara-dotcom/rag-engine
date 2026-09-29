# rag-engine

[![CI](https://github.com/sakshamchitkara-dotcom/rag-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/sakshamchitkara-dotcom/rag-engine/actions/workflows/ci.yml)

A small retrieval-augmented generation (RAG) engine with no required dependencies. It
ingests Markdown, text, HTML and PDF documents, indexes them with an in-repo BM25
implementation plus dense vectors, fuses the two rankings with reciprocal rank fusion,
optionally reranks the candidates, and answers questions with numbered citations using
Claude (streamed over Server-Sent Events in the HTTP API). Without an API key it falls
back to an extractive answerer, so every command works offline. Re-ingesting only
re-embeds files whose content changed, and questions can be restricted to documents by
source pattern or tag. Tables survive ingest as Markdown tables (HTML tables included)
and the offline answerer can answer from a table row. With Claude, questions can be
rewritten into several search queries (`--multi-query`), and `rag eval --judge` grades
end-to-end answers for correctness and grounding. `rag chat` and the web chat page keep
a conversation: follow-ups such as "does the old one keep working?" are rewritten into
standalone questions (by Claude, or offline by a heuristic), and the chat page links
each `[n]` citation to its source with the query terms highlighted.

It includes a sample corpus about **Beacon**, a made-up feature-flag service, a
31-question eval set and a 10-question set of two-part questions, so you can try every
command right after cloning.

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

Or with Docker (the index lives in the `/data` volume; the sample corpus is in the image):

```bash
docker build -t rag-engine .
docker run --rm -v rag-data:/data rag-engine ingest /app/examples/corpus
docker run --rm -p 8000:8000 -v rag-data:/data rag-engine          # serves on :8000
docker run --rm -v rag-data:/data -e ANTHROPIC_API_KEY rag-engine ask "..."
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
examples/corpus: 9 added, 0 updated, 0 unchanged, 0 removed -> 42 chunks written (embedder hash:1024)
index .rag/index.sqlite: 42 chunks from 9 documents

$ rag ingest examples/corpus        # again: nothing changed, nothing re-embedded
examples/corpus: 0 added, 0 updated, 9 unchanged, 0 removed -> 0 chunks written (embedder hash:1024)
index .rag/index.sqlite: 42 chunks from 9 documents

$ rag ask "How do I rotate an SDK key, and does the old one stop working immediately?"
Keys are rotated from Settings > Environments > Rotate key. [1] After rotation, the old key keeps
working for a 24-hour grace period so that you can redeploy without downtime. [1]

Sources:
 *[1] troubleshooting.txt - Troubleshooting Beacon > Error "401 invalid sdk key"  (score 0.0328)
  [2] sdk-guide.md - SDK Guide  (score 0.0313)
  [3] troubleshooting.txt - Troubleshooting Beacon > SDK always returns the default value  (score 0.0313)
  [4] api-reference.html - REST API Reference > Authentication  (score 0.0308)
  [5] sdk-guide.md - SDK Guide > Initializing the Python SDK  (score 0.0306)

(answer mode: extractive; * = cited)

$ rag ask "How many seats does the Team plan allow?"
Plan: Team; Price: $25 per seat per month; Seats: Up to 50; Monthly active users: 100,000 [1]

Sources:
 *[1] pricing.md - Plans and Pricing > Plans  (score 0.0328)
  [2] pricing.md - Plans and Pricing > Free trial  (score 0.0323)
  ...

$ rag eval --judge
31 questions, 42 chunks

mode              recall@1  recall@3  recall@5  MRR    docs@5
----------------  --------  --------  --------  -----  ------
hybrid            0.774     1.000     1.000     0.882  3.16
bm25              0.742     1.000     1.000     0.866  3.13
dense             0.710     1.000     1.000     0.844  3.10
hybrid+proximity  0.839     1.000     1.000     0.914  3.52
hybrid+mmr        0.774     0.903     0.968     0.844  3.94

hybrid+mmr missed (not in top 5):
  - The SDK keeps returning the default value, what should I check?

Answer quality (extractive answerer over the top 5):
mode              found  precision  sentences
----------------  -----  ---------  ---------
hybrid            0.903  0.839      1.81
bm25              0.871  0.797      1.90
dense             0.871  0.839      1.81
hybrid+proximity  0.903  0.817      1.94
hybrid+mmr        0.871  0.839      1.81
found = answer contains the labelled phrase; precision = share of answer sentences
cited from a chunk that holds the answer; sentences = average per answer

Judged answers (hybrid, top 5; answered by extractive, judged by heuristic):
correct   0.935
grounded  1.000
  - How can my app keep working when the flag server is unreachable at startup?
    missing ['bootstrap']
  - How much does the Team plan cost?
    missing ['25', 'per', 'seat']

$ rag remove pricing.md security.md && rag vacuum
removed pricing.md
removed security.md
index .rag/index.sqlite: 34 chunks from 7 documents
index .rag/index.sqlite: 224 KiB -> 188 KiB (36 KiB reclaimed)
```

`docs@5` is the average number of distinct documents in the top 5 (diversity). The
proximity reranker ranks the right chunk first more often (MRR 0.882 -> 0.914); MMR
spreads the top 5 over more documents at a small recall cost. The judged answers show
the extractive answerer's limits: it copies sentences, so it is always grounded, but it
misses "How much does the Team plan cost?" because the table says "Price", not "cost".

The two-part questions in `examples/eval_compound.json` are harder to rank (each
question is listed once per labelled part):

```text
$ rag eval --questions examples/eval_compound.json --modes hybrid,bm25,dense,hybrid+proximity
20 questions, 42 chunks

mode              recall@1  recall@3  recall@5  MRR    docs@5
----------------  --------  --------  --------  -----  ------
hybrid            0.500     0.950     1.000     0.702  3.10
bm25              0.500     0.900     1.000     0.708  3.50
dense             0.550     0.850     0.950     0.706  3.10
hybrid+proximity  0.500     1.000     1.000     0.725  3.50
```

With `ANTHROPIC_API_KEY` set, add `hybrid+multi` to `--modes` to measure Claude query
rewriting on it.

## Commands

| Command | What it does |
|---|---|
| `rag ingest <path-or-url>... [--reset] [--tag NAME]... [--prefix NAME] [--chunk-size 800] [--overlap 150] [--embedder hash:1024]` | Load files, folders (recursively) or `http(s)` URLs, chunk them, embed them and store them. Only new or changed documents are re-embedded, documents deleted from an ingested folder are removed, and unreadable files are skipped with a warning. `--tag` labels the documents of this run (omit it to keep existing tags). `--prefix` stores them as `NAME/<path>`; a document whose source name already belongs to another file is skipped with a warning suggesting it. |
| `rag ask "<question>" [-k 5] [--mode hybrid\|bm25\|dense] [--rerank none\|proximity\|mmr] [--multi-query] [--source GLOB]... [--tag NAME]... [--no-llm] [--json]` | Retrieve the top-k chunks and answer with `[n]` citations. `*` marks the sources the answer cites. `--multi-query` has Claude rewrite the question into up to 3 extra search queries and fuses their rankings. |
| `rag chat [same options as ask, except --json]` | Read questions from stdin, one per line. A follow-up is condensed with the earlier questions into a standalone question (shown as `(as: ...)`) before searching. `/reset` starts a new conversation, `/quit` or Ctrl+D exits. |
| `rag remove <source-or-path>...` | Remove documents by source name (as `rag stats` shows it), folder prefix, or the path of an ingested file or folder. |
| `rag stats [--json]` | Show the index file, embedder, and each document's chunk count, ingest time and tags. |
| `rag vacuum` | Compact the index file; SQLite does not shrink it after `remove` or re-ingests on its own. |
| `rag eval [--questions FILE] [--k 1,3,5] [--modes hybrid,bm25,dense,hybrid+proximity,hybrid+mmr] [--judge] [--no-llm] [--json]` | Report recall@k, MRR and top-k diversity per retriever (optionally `+proximity`/`+mmr` reranked, or `+multi` with Claude), plus extractive answer quality, and list the questions each one missed. `--judge` also grades the answers `rag ask` would give. |
| `rag serve [--host 127.0.0.1] [--port 8000] [-k 5]` | Start a JSON API and a small chat page that streams answers. The server picks up documents that other `rag` commands add or remove while it runs. |

Every command accepts `--index PATH` (default `.rag/index.sqlite`, or the `RAG_INDEX` environment variable).

### HTTP API

```bash
curl -s localhost:8000/api/health
# {"chunks": 42, "sources": 9, "llm": false}

curl -s localhost:8000/api/ask -H 'Content-Type: application/json' \
  -d '{"question": "How do I authenticate REST API calls?", "k": 3}'
# {"answer": "... [1] ...", "sources": [{"n": 1, "source": "api-reference.html", ...}], "mode": "extractive", "warning": null}
```

`POST /api/ask` accepts `question` (required, up to 2000 characters), `k` (1-20), `mode`,
`rerank` (`none`, `proximity` or `mmr`), `source` and `tag` (a string or a list of strings),
`multi_query` (true to fuse searches for Claude's rewrites of the question), `history`
(the conversation's earlier questions, oldest first; the last 10 are used), and `llm`
(set it to false to skip Claude). Responses report the standalone `question` that was
answered, and each source has `highlights`, the `[start, end)` character spans of query
terms in its `text`. Invalid input gets a 400 response with an
`error` message.

`POST /api/ask/stream` takes the same body and answers with Server-Sent Events:

```text
$ curl -N localhost:8000/api/ask/stream -d '{"question": "What is the REST API rate limit?", "k": 2}'
event: sources
data: [{"n": 1, "id": "api-reference.html#2", "source": "api-reference.html", ...}, ...]

event: delta
data: "The REST API allows 300 requests per minute per access token. [1] ..."

event: done
data: {"mode": "extractive", "warning": null}
```

With Claude, `delta` events arrive as the answer is written. Without Claude (no key,
`"llm": false`, or an API error before any text) the extractive answer comes as a single
`delta`. If Claude fails after some text was sent, a `replace` event carries the full
extractive answer, which supersedes the partial text; `done` then has
`"mode": "extractive"` and a `warning`.

### Filters

`--source` takes shell-style globs matched against source names (`api-*`, `guides/*`);
`--tag` matches tags given at ingest time. Repeating a flag matches any of the values;
using both flags requires both to match.

```bash
rag ingest ~/handbook --tag handbook
rag ingest ~/runbooks --tag ops
rag ask "How do we rotate credentials?" --tag ops --source 'db-*'
```

## Architecture

```
            ingest                                   ask / serve
 files / folders / URLs                                 question
          │                                                │
     loaders.py  md · txt · html→md (tables              rewrite.py  --multi-query: Claude adds up
                 kept as Markdown) · pdf (pypdf)           │         to 3 queries, one search each,
          │                                                │         rankings fused by RRF
    chunking.py  heading sections → paragraphs →           ├──────────────┐
                 sentences (tables: rows, code: lines),    ▼              ▼
                 sentence-aligned overlap               --source / --tag filters
          │                                                │              │
  embeddings.py  hashing vectorizer (or sentence-       BM25 top-50   dense top-50
                 transformers)                             └──── RRF ─────┘
          │                                                       │ top 20
      index.py   SQLite: chunks + float32 vectors +        rerank.py  proximity | MMR (optional)
                 documents (hash, origin, tags) + meta            │ top-k chunks
                                                                  ▼
                                                           generate.py
                                                           Claude (claude-opus-5-5) with [n] citations,
                                                           blocking or streamed
                                                           └─ fallback: extractive sentences / table rows with [n]
```

| Module | Responsibility |
|---|---|
| `rag_engine/loaders.py` | Reads `.md`, `.txt`, `.html` and `.pdf` from files, folders or URLs and turns each into Markdown-style text. HTML keeps its headings, lists and `<pre>` blocks, turns tables into Markdown pipe tables (captions kept), and drops `script`, `style`, `nav` and `footer`. In plain text, short standalone lines become headings. PDFs are read with `pypdf` if it is installed; otherwise they are skipped with a notice. A file that fails to load (such as a corrupt PDF) is reported and skipped. |
| `rag_engine/chunking.py` | Splits text into sections at headings, so a chunk never crosses a section boundary. It packs whole paragraphs up to `max_chars`, splits long paragraphs at sentence boundaries (and word boundaries as a last resort), splits long tables between rows (each piece repeats the header row) and long fenced code between lines (each piece stays fenced), and starts each chunk with the last few sentences of the previous chunk in the same section. Each chunk keeps its heading path, such as `REST API Reference > Rate limits`, and that path is indexed with the text. |
| `rag_engine/text.py` | Tokenizer, stopword list, a light suffix stemmer and a sentence splitter, shared by every component. |
| `rag_engine/bm25.py` | Okapi BM25 (k1=1.5, b=0.75) over an inverted index. |
| `rag_engine/embeddings.py` | The default embedder is a feature-hashing vectorizer: word unigrams and bigrams plus in-word character trigrams, signed hashing into 1024 dimensions, log-scaled term frequencies and L2 normalization. It is deterministic, needs no download, and the character trigrams handle typos that BM25 misses. `--embedder st:<model>` uses `sentence-transformers` instead, if it is installed. |
| `rag_engine/index.py` | Stores chunks and their float32 vectors in SQLite and records which embedder built the index, so queries always use the same one. A `documents` table keeps each source's content hash (text plus chunking settings), origin and tags; ingest skips documents whose hash is unchanged and prunes ones deleted from an ingested folder. A source name that already belongs to a different file is refused and reported rather than overwritten. Before each search it checks SQLite's `data_version` and reloads if another process changed the index. Search modes are `bm25`, `dense` (brute-force cosine over the query's non-zero dimensions) and `hybrid`, which takes the top 50 from each retriever and merges them with reciprocal rank fusion, `score = Σ 1/(60 + rank)`. Source and tag filters are applied before the rankings are cut. |
| `rag_engine/rewrite.py` | Multi-query retrieval: Claude rewrites the question into up to three search queries (the original always comes first); each is searched and the rankings are fused with RRF. Without Claude it is a plain search. |
| `rag_engine/rerank.py` | Optional second stage over the top 20 candidates. `proximity` scores IDF-weighted query coverage, boosted when the matched terms sit in a short window, and fuses that order with the first-stage order by RRF. `mmr` (maximal marginal relevance, λ = 0.7) uses the dense vectors to push near-duplicate chunks down. |
| `rag_engine/generate.py` | Sends numbered `<source>` blocks to Claude with instructions to answer only from the sources and cite every claim. `stream_answer` does the same over `messages.stream` and yields `sources` / `delta` / `replace` / `done` events. The fallback scores each retrieved sentence by IDF-weighted overlap with the query (counting its section heading at a discount), glues fragments such as "Defaults to 8420." onto the sentence before them, reads each table row as a sentence ("Plan: Team; Price: $25 ...", column names counted like headings, and a row must match the question in its own cells), and returns up to three sentences. To avoid padding, a sentence from a chunk other than the best one must match at least a third of the query's IDF weight and cover a query term the answer does not cover yet. |
| `rag_engine/evaluate.py` | Computes recall@k, MRR and docs@k for each retriever or `retriever+reranker`. A question counts as answered when a retrieved chunk comes from the expected file and contains the expected phrase, so the labels stay valid when you change chunking settings. It also scores the extractive answerer: `found` (the answer contains the expected phrase), `precision` (share of answer sentences cited from a chunk that contains it) and sentences per answer. `--judge` answers each question as `rag ask` would and grades it for correctness (against the labelled phrase) and grounding (in the cited sources), with Claude as the judge or, offline, a word-overlap heuristic. |
| `rag_engine/server.py` | `http.server`-based JSON API with an SSE streaming endpoint, and a single-file chat page that renders streamed answers (no build step, no external assets). |

### Claude integration

`generate.py` calls `client.messages.create` in the official `anthropic` SDK with model
`claude-opus-5-5`. The key comes from `ANTHROPIC_API_KEY`. Thinking cannot be turned off
on this model, so the request sets `output_config={"effort": "low"}` to keep grounded
Q&A fast and cheap. The code reads `text` blocks by type, because the response also
contains `thinking` blocks. If Claude refuses (`stop_reason == "refusal"`), rejects the
key, rate-limits the request, returns an API error or can't be reached, the answer falls
back to extractive mode and includes a `warning`, so the command never fails because of
the LLM. The streaming path calls `client.messages.stream(...)` with the same parameters
and forwards `text_stream` deltas; it uses the same fallbacks, and a failure after text
was sent is resolved with a `replace` event.

### Design choices and limits

- Filtered searches score every chunk and then filter. That is fine at this scale; a
  large multi-tenant index would want per-tag postings.
- Documents are identified by source name, which for folders is the path relative to the
  folder. When two ingested folders both contain `README.md`, the second one is skipped
  with a warning; ingest it with `--prefix NAME` to keep both.
- The BM25 postings are rebuilt in memory each time the index loads (and again after
  another process changes the index), and dense search is a brute-force scan over the
  query's non-zero dimensions. On a 12,833-chunk index (Python 3.14, Apple silicon)
  loading takes under a second and a query about 5 ms (BM25), 24 ms (dense) or 29 ms
  (hybrid). Beyond about 100k chunks, persist the inverted index and use an
  approximate-nearest-neighbour library. With `--embedder st:...` the query vector is
  dense, so dense search costs a full scan again.
- Searches in `rag serve` hold one lock, so they run one at a time (tens of
  milliseconds each); the Claude call runs outside it.
- There is no offline query rewriter. Splitting compound questions into their parts
  lowered hybrid MRR on `eval_compound.json` from 0.702 to 0.583, and pseudo-relevance
  feedback lowered it on the main set, so `--multi-query` needs Claude. The Claude
  rewriting, streaming and judge paths are tested against a fake SDK in the test suite;
  their effect on answer quality has not been measured here.
- Conversation mode condenses a follow-up into a standalone question. Offline, a
  heuristic treats a question as a follow-up when it has a referring word ("it", "they",
  "that", "one"...), starts with "and"/"what about" and similar, or has at most two
  content words, and then appends the previous question's content words. On
  `examples/eval_followups.json` (12 follow-ups written for this corpus, so optimistic)
  hybrid recall@5 goes from 0.667 searched as written to 1.000 and MRR from 0.521 to
  0.958. It cannot tell a new topic that happens to say "it" from a follow-up; Claude's
  rewrite handles that but has not been measured here. Answers are generated for the
  standalone question; earlier answers are not sent to Claude.
- The heuristic judge only checks word overlap: an extractive answer is grounded by
  construction, and a paraphrased correct answer can be marked wrong.
- Table support covers Markdown tables and HTML tables. PDF tables are not
  reconstructed; they come out as whatever plain text `pypdf` extracts.
- The stemmer is a small rule-based suffix stripper. Use a Porter or Snowball stemmer if your corpus needs better handling of word forms.
- The hashing embedder captures word overlap, not meaning. On the sample eval it is the weakest retriever alone, but it still helps the hybrid, which ranks the right chunk first more often than BM25 alone. For real semantic retrieval, use `--embedder st:all-MiniLM-L6-v2`.

## Using your own data

```bash
rag --index my.sqlite ingest ~/notes https://example.com/docs/page.html --reset
rag --index my.sqlite ask "..."
```

To evaluate retrieval on your data, write a JSON list of `{"question", "source", "contains"}`
objects (see `examples/eval_questions.json`) and run `rag eval --questions your.json`. Add `"history": ["earlier question", ...]` to evaluate
follow-ups (see `examples/eval_followups.json`).

## Development

```bash
pip install -e ".[all]"
python -m unittest discover -s tests -v
```

The tests cover every module. They also run the CLI (ingest, ask, eval) on the sample
corpus, start the HTTP server and exercise the Claude code path against a fake SDK, so
they need no network access or API key. CI runs them on Python 3.10 through 3.13, runs a
CLI smoke test, and builds the Docker image and queries it.

## License

MIT
