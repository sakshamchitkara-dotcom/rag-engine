# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/) (0.x: minor versions may break things).

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
