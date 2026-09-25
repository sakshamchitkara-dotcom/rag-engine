"""Command line interface: rag ingest | remove | stats | ask | eval | serve."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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


def _origin(target: str) -> str:
    return target if target.startswith(("http://", "https://")) else str(Path(target).resolve())


def cmd_ingest(args) -> int:
    batches = []
    for target in args.paths:
        loaded = load_path(target)
        if not loaded:
            print(f"warning: no supported documents found in {target}", file=sys.stderr)
        batches.append((target, loaded))
    if args.reset:
        with _open(args) as index:
            index.reset()
    with _open(args, args.embedder) as index:
        for target, docs in batches:
            r = index.add_documents(docs, max_chars=args.chunk_size, overlap=args.overlap,
                                    origin=_origin(target), prune=Path(target).is_dir(), tags=tuple(args.tag) if args.tag else None)
            print(f"{target}: {len(r.added)} added, {len(r.updated)} updated, {len(r.unchanged)} unchanged, "
                  f"{len(r.removed)} removed -> {r.chunks} chunks written (embedder {index.embedder_name})")
            for source in r.removed:
                print(f"  removed {source} (no longer in {target})")
        print(f"index {index.path}: {len(index.chunks)} chunks from {len(index.sources())} documents")
    return 0


def cmd_remove(args) -> int:
    with _open(args) as index:
        targets = {t: index.find_sources(t, _origin(t)) for t in args.paths}
        for target, found in targets.items():
            if not found:
                print(f"warning: nothing in the index matches {target}", file=sys.stderr)
        removed = index.remove(sorted({s for found in targets.values() for s in found}))
        for source in removed:
            print(f"removed {source}")
        print(f"index {index.path}: {len(index.chunks)} chunks from {len(index.sources())} documents")
    return 0 if removed else 1


def cmd_stats(args) -> int:
    with _open(args) as index:
        st = index.stats()
    if args.json:
        print(json.dumps(st, indent=2))
        return 0
    print(f"index     {st['path']} ({st['bytes'] / 1024:.0f} KiB)")
    print(f"embedder  {st['embedder']}")
    print(f"contents  {st['chunks']} chunks from {st['documents']} documents")
    if st["sources"]:
        width = max(len(d["source"]) for d in st["sources"])
        print(f"\n{'source'.ljust(width)}  chunks   chars  {'ingested (UTC)'.ljust(25)}  tags")
        for d in st["sources"]:
            print(f"{d['source'].ljust(width)}  {d['chunks']:>6}  {d['chars']:>6}  "
                  f"{(d['ingested_at'] or '-').ljust(25)}  {','.join(d['tags'])}".rstrip())
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
    hits = index.search(args.question, k=args.k, mode=args.mode, rerank=args.rerank,
                        sources=args.source, tags=args.tag)
    if not hits and (args.source or args.tag):
        print("warning: no indexed chunks match the --source/--tag filters", file=sys.stderr)
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
    s.add_argument("--tag", action="append", default=[],
                   help="tag the ingested documents (repeatable), replacing their tags; without --tag "
                        "existing tags are kept. Filter with `rag ask --tag`")
    s.add_argument("--embedder", default=None, help="'hash[:dim]' (default hash:1024) or 'st:<model>'")
    s.add_argument("--chunk-size", type=int, default=800, help="max characters per chunk (default 800)")
    s.add_argument("--overlap", type=int, default=150, help="overlap characters between chunks (default 150)")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("remove", help="remove documents from the index")
    s.add_argument("paths", nargs="+",
                   help="source name as shown by `rag stats`, a folder prefix, or an ingested file/folder path")
    s.set_defaults(func=cmd_remove)

    s = sub.add_parser("stats", help="show what the index contains")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("ask", help="answer a question with citations")
    s.add_argument("question")
    s.add_argument("-k", type=int, default=5, help="chunks to retrieve (default 5)")
    s.add_argument("--mode", choices=MODES, default="hybrid")
    s.add_argument("--rerank", choices=("none", *RERANKERS), default="none",
                   help="rerank the top 20 candidates: proximity (query terms close together) or mmr (diversity)")
    s.add_argument("--source", action="append", default=[],
                   help="only search documents whose source matches this glob, e.g. 'api-*' (repeatable: any)")
    s.add_argument("--tag", action="append", default=[],
                   help="only search documents with this ingest tag (repeatable: any)")
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
