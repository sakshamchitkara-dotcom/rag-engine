"""Command line interface: rag ingest | ask | eval | serve."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .evaluate import DEFAULT_MODES, evaluate, evaluate_answers, format_answer_table, format_table, load_questions
from .generate import answer, cited_numbers
from .index import MODES, Index
from .loaders import load_path
from .rerank import RERANKERS

DEFAULT_INDEX = os.environ.get("RAG_INDEX", ".rag/index.sqlite")
DEFAULT_QUESTIONS = "examples/eval_questions.json"


def _open(args, embedder: str | None = None) -> Index:
    return Index(args.index, embedder=embedder)


def cmd_ingest(args) -> int:
    docs = []
    for target in args.paths:
        loaded = load_path(target)
        if not loaded:
            print(f"warning: no supported documents found in {target}", file=sys.stderr)
        docs.extend(loaded)
    if args.reset:
        with _open(args) as index:
            index.reset()
    with _open(args, args.embedder) as index:
        n = index.add_documents(docs, max_chars=args.chunk_size, overlap=args.overlap)
        print(f"ingested {len(docs)} documents -> {n} chunks (embedder {index.embedder_name})")
        print(f"index {index.path}: {len(index.chunks)} chunks from {len(index.sources())} documents")
    return 0


def _require_chunks(index: Index) -> bool:
    if index.chunks:
        return True
    print(f"index {index.path} is empty; run `rag ingest <path>` first", file=sys.stderr)
    return False


def cmd_ask(args) -> int:
    with _open(args) as index:
        return _ask(args, index)


def _ask(args, index: Index) -> int:
    if not _require_chunks(index):
        return 1
    hits = index.search(args.question, k=args.k, mode=args.mode, rerank=args.rerank)
    result = answer(args.question, hits, use_llm=not args.no_llm)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    if result.warning:
        print(f"warning: {result.warning}", file=sys.stderr, flush=True)
    print(result.text)
    cited = set(cited_numbers(result.text))
    print("\nSources:")
    for s in result.sources:
        mark = "*" if s["n"] in cited else " "
        print(f" {mark}[{s['n']}] {s['source']} - {s['heading'] or s['title']}  (score {s['score']})")
    print(f"\n(answer mode: {result.mode}; * = cited)")
    return 0


def cmd_eval(args) -> int:
    with _open(args) as index:
        return _eval(args, index)


def _eval(args, index: Index) -> int:
    if not _require_chunks(index):
        return 1
    questions = load_questions(args.questions)
    ks = tuple(sorted({int(k) for k in args.k.split(",")}))
    modes = args.modes.split(",")
    reports = evaluate(index, questions, ks=ks, modes=modes)
    answers = [evaluate_answers(index, questions, k=5, mode=m) for m in modes]
    if args.json:
        print(json.dumps({"retrieval": [r.to_dict() for r in reports],
                          "answers": [a.to_dict() for a in answers]}, indent=2))
        return 0
    print(f"{len(questions)} questions, {len(index.chunks)} chunks\n")
    print(format_table(reports))
    for r in reports:
        misses = [q.question for q in r.results if q.rank is None]
        if misses:
            print(f"\n{r.mode} missed (not in top {max(ks)}):")
            for m in misses:
                print(f"  - {m}")
    print("\nAnswer quality (extractive answerer over the top 5):")
    print(format_answer_table(answers))
    print("found = answer contains the labelled phrase; precision = share of answer sentences\n"
          "cited from a chunk that holds the answer; sentences = average per answer")
    return 0


def cmd_serve(args) -> int:
    from .server import serve

    with _open(args) as index:
        if not _require_chunks(index):
            return 1
        serve(index, host=args.host, port=args.port, k=args.k)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag", description="Small hybrid-retrieval RAG engine.")
    p.add_argument("--version", action="version", version=f"rag-engine {__version__}")
    p.add_argument("--index", default=DEFAULT_INDEX, help=f"index file (default: {DEFAULT_INDEX}, env RAG_INDEX)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest", help="load and index files, folders or URLs")
    s.add_argument("paths", nargs="+", help="file, folder or http(s) URL")
    s.add_argument("--reset", action="store_true", help="clear the index first")
    s.add_argument("--embedder", default=None, help="'hash[:dim]' (default hash:1024) or 'st:<model>'")
    s.add_argument("--chunk-size", type=int, default=800, help="max characters per chunk (default 800)")
    s.add_argument("--overlap", type=int, default=150, help="overlap characters between chunks (default 150)")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("ask", help="answer a question with citations")
    s.add_argument("question")
    s.add_argument("-k", type=int, default=5, help="chunks to retrieve (default 5)")
    s.add_argument("--mode", choices=MODES, default="hybrid")
    s.add_argument("--rerank", choices=("none", *RERANKERS), default="none",
                   help="rerank the top 20 candidates: proximity (query terms close together) or mmr (diversity)")
    s.add_argument("--no-llm", action="store_true", help="skip Claude; use the extractive answerer")
    s.add_argument("--json", action="store_true", help="print the full result as JSON")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("eval", help="retrieval eval (recall@k, MRR) and extractive answer quality")
    s.add_argument("--questions", default=DEFAULT_QUESTIONS, help=f"question file (default {DEFAULT_QUESTIONS})")
    s.add_argument("--k", default="1,3,5", help="comma-separated cutoffs (default 1,3,5)")
    s.add_argument("--modes", default=",".join(DEFAULT_MODES),
                   help="comma-separated retrievers to compare; add +proximity or +mmr to rerank "
                        f"(default {','.join(DEFAULT_MODES)})")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("serve", help="HTTP JSON API + chat page")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("-k", type=int, default=5)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
