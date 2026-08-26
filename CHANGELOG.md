# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/) (0.x: minor versions may break things).

## [0.3.0] - 2026-09-25

### Added

- Table-aware ingest: HTML tables become Markdown pipe tables, long tables are split
  between rows with the header repeated in every chunk, and the extractive answerer
  answers from a matching row ("Plan: Team; Price: $25 per seat per month; Seats: Up to
  50"). Hybrid answer found-rate on the eval set: 0.871 -> 0.903.
- `rag ask --multi-query` and `"multi_query"` in the API: Claude rewrites the question
  into up to three search queries whose rankings are fused with RRF. `rag eval` accepts
  `+multi` modes, and `examples/eval_compound.json` holds 10 two-part questions.
- `rag eval --judge [--no-llm]`: grades the answers `rag ask` would give for correctness
  and grounding, with Claude as the judge or a word-overlap heuristic offline.
- `rag vacuum` compacts the index file.
- `rag ingest --prefix NAME` stores a run's documents as `NAME/<path>`.
- Docker image (`Dockerfile`) and a CI job that builds and queries it.
- A table question in `examples/eval_questions.json` (now 31 questions).

### Fixed

- HTML table cells were glued together ("Team$25").
- Long tables and fenced code blocks were flattened by the sentence splitter; they are
  now split between lines and keep their header or fences.
- Ingesting two folders that both contain the same relative path silently replaced the
  first document; the second is now skipped with a warning suggesting `--prefix`.
- One unreadable file (such as a corrupt PDF) aborted the whole ingest; it is now
  skipped with a warning.
- `rag serve` did not see documents added or removed by other `rag` commands until it
  was restarted.
- Unreachable URLs, HTTP errors and a non-SQLite `--index` printed tracebacks; they are
  now one-line errors with exit status 2.
- Out-of-range numeric options (`-k -2`, `--port 70000`, `--chunk-size 10`, `--k 0`)
  were accepted or failed obscurely; they are now usage errors.
- URL ingest and the server reported version 0.1 in their HTTP headers.

### Changed

- Dense search only visits the query's non-zero dimensions: on a 12,833-chunk index,
  dense queries went from 466 to 24 ms and hybrid from 411 to 29 ms, with identical
  results.
- `rag ask --json` output includes the `queries` that were searched.
- `IngestResult` has a `conflicts` field; `evaluate.parse_mode` returns
  `(mode, rerank, multi)`.

## [0.2.0] - 2026-09-25

### Added

- Reranking stage: `rag ask --rerank proximity|mmr` and `"rerank"` in the API.
  `proximity` rewards query terms that appear close together; `mmr` diversifies the
  top-k. On the sample eval, `hybrid+proximity` raises MRR from 0.878 to 0.911.
- `rag eval` compares `retriever+reranker` modes (default
  `hybrid,bm25,dense,hybrid+proximity,hybrid+mmr`), reports `docs@k` diversity, and
  scores extractive answer quality (`found`, `precision`, sentences per answer).
- Incremental ingest: documents are tracked by a content hash, so unchanged documents
  are not re-embedded and documents deleted from an ingested folder are removed.
- `rag remove <source-or-path>` and `rag stats`.
- `rag ingest --tag`, and `--source` / `--tag` filters in `rag ask` and
  `"source"` / `"tag"` in `POST /api/ask`.
- `POST /api/ask/stream`: Server-Sent Events that stream Claude's answer, with the
  extractive fallback as a single `delta`, or as a `replace` event when Claude fails
  partway through. The chat page renders answers as they stream.

### Fixed

- Extractive answers no longer pad themselves with loosely related sentences, such as
  one that shares a single word with the question or a bare fragment like
  "Required.". Answer precision on the sample eval went from 0.671 to 0.833, with
  1.80 sentences per answer instead of 2.33 and no answers lost.

### Changed

- `rag eval --json` returns `{"retrieval": [...], "answers": [...]}` instead of a list.
- `Index.add_documents` returns an `IngestResult` instead of a chunk count.
- `rag ingest` prints added / updated / unchanged / removed counts for each target.
- Indexes built by 0.1.0 still open; on the next ingest their documents count as new
  and are re-embedded once.

## [0.1.0] - 2026-09-25

First release: md/txt/html/pdf loaders, heading-aware chunking, in-repo BM25, hashing
embedder, SQLite index with hybrid RRF retrieval, Claude answers with citations and an
extractive fallback, a recall@k/MRR eval, and a stdlib HTTP API with a chat page.
