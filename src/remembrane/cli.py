"""remembrane CLI: inspect and manage a memory database from the terminal."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from .store import MemoryStore


def _fmt_ts(ts) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "-"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="remembrane", description="Inspect an agent memory database.")
    parser.add_argument("--db", default="remembrane.db", help="Path to the SQLite database")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_store = sub.add_parser("store", help="Store a memory")
    p_store.add_argument("content", nargs="?", default=None)
    p_store.add_argument("--file", dest="from_file", default=None,
                         help="Read content from a file ('-' for stdin) — avoids OS argv length limits")
    p_store.add_argument("--namespace", default="default")
    p_store.add_argument("--importance", type=float, default=0.5)

    p_recall = sub.add_parser("recall", help="Recall memories for a query")
    p_recall.add_argument("query")
    p_recall.add_argument("--namespace", default="default")
    p_recall.add_argument("-k", type=int, default=5)
    p_recall.add_argument("--mode", choices=["hybrid", "vector", "keyword"], default="hybrid")
    p_recall.add_argument("--explain", action="store_true", help="Show ranking breakdown per result")

    p_list = sub.add_parser("list", help="List all memories")
    p_list.add_argument("--namespace", default=None)

    p_forget = sub.add_parser("forget", help="Forget a memory by id")
    p_forget.add_argument("memory_id")

    p_export = sub.add_parser("export", help="Export memories as JSON")
    p_export.add_argument("--namespace", default=None)

    sub.add_parser("stats", help="Show database stats")

    p_snap = sub.add_parser("snapshot", help="Record a named point in time")
    p_snap.add_argument("label")

    p_log = sub.add_parser("log", help="Show memory history (newest first)")
    p_log.add_argument("--namespace", default=None)
    p_log.add_argument("--limit", type=int, default=30)

    p_diff = sub.add_parser("diff", help="What changed between two snapshots (or snapshot..now)")
    p_diff.add_argument("a", help="snapshot label")
    p_diff.add_argument("b", nargs="?", default=None, help="snapshot label (default: now)")

    p_conf = sub.add_parser("conflicts", help="Surface memories in tension")
    p_conf.add_argument("query", nargs="?", default=None)
    p_conf.add_argument("--namespace", default="default")
    p_conf.add_argument("--min-confidence", choices=["possible", "likely"], default="possible")

    p_fb = sub.add_parser("feedback", help="Record task-outcome feedback for a memory")
    p_fb.add_argument("memory_id")
    g = p_fb.add_mutually_exclusive_group(required=True)
    g.add_argument("--useful", action="store_true")
    g.add_argument("--useless", action="store_true")

    p_pack = sub.add_parser("pack", help="Optimal memory set for a token budget")
    p_pack.add_argument("query")
    p_pack.add_argument("--budget", type=int, default=800)
    p_pack.add_argument("--namespace", default="default")

    p_merge = sub.add_parser("merge", help="Merge another memory db into this one")
    p_merge.add_argument("source", help="path to the other .db file")
    p_merge.add_argument("--dedupe-threshold", type=float, default=0.95)

    args = parser.parse_args(argv)
    try:
        store = MemoryStore(args.db)
    except Exception as exc:  # pragma: no cover - depends on host fs
        print(f"error: cannot open database {args.db!r}: {exc}", file=sys.stderr)
        return 1

    if args.cmd == "store":
        content = args.content
        if args.from_file:
            content = sys.stdin.read() if args.from_file == "-" else open(args.from_file, encoding="utf-8").read()
        if content is None:
            raise ValueError("provide CONTENT or --file")
        mem = store.store(content, namespace=args.namespace, importance=args.importance)
        print(f"stored {mem.id}")
    elif args.cmd == "recall":
        for r in store.recall(args.query, k=args.k, namespace=args.namespace,
                              touch=False, mode=args.mode):
            print(f"{r.score:.3f}  [{r.memory.id[:8]}]  {r.memory.content}")
            if args.explain:
                print(f"        {r.explain_text()}")
    elif args.cmd == "list":
        for m in store.all(args.namespace):
            print(f"[{m.id[:8]}] ({m.namespace}, imp={m.importance:.2f}) {m.content}")
    elif args.cmd == "forget":
        n = store.forget(args.memory_id)
        print(f"forgot {n} memory(ies)")
    elif args.cmd == "export":
        json.dump(store.export(args.namespace), sys.stdout, indent=2)
        print()
    elif args.cmd == "stats":
        print(f"memories: {store.count()}")
        for ns in store.namespaces():
            print(f"  {ns}: {store.count(ns)}")
    elif args.cmd == "snapshot":
        ts = store.snapshot(args.label)
        print(f"snapshot {args.label!r} at {_fmt_ts(ts)}")
    elif args.cmd == "log":
        for e in store.log(namespace=args.namespace, limit=args.limit):
            detail = e.payload.get("content", "") or e.payload.get("reason", "")
            print(f"{_fmt_ts(e.ts)}  {e.op:<11} [{e.memory_id[:8]}] {detail[:70]}")
    elif args.cmd == "diff":
        d = store.diff(args.a, args.b)
        for item in d["added"]:
            print(f"+ {item['content']}")
        for item in d["removed"]:
            print(f"- {item['content']}")
        for item in d["changed"]:
            print(f"~ {item['content']} (importance {item['importance_before']:.2f} -> {item['importance_after']:.2f})")
        if not any(d.values()):
            print("no changes")
    elif args.cmd == "conflicts":
        found = store.conflicts(args.query, namespace=args.namespace,
                                min_confidence=args.min_confidence)
        if not found:
            print("no conflicts detected")
        for c in found:
            print(f"--- {c.confidence} (ids {c.a.id[:8]} vs {c.b.id[:8]})")
            print(c.describe())
    elif args.cmd == "feedback":
        mem = store.feedback(args.memory_id, args.useful)
        print(f"usefulness now {mem.usefulness:+.1f}" if mem else "no such memory")
    elif args.cmd == "pack":
        chosen = store.pack(args.query, budget_tokens=args.budget, namespace=args.namespace,
                            touch=False)
        total = sum(r.tokens for r in chosen)
        for r in chosen:
            print(f"{r.score:.3f}  ({r.tokens:>4} tok)  {r.memory.content}")
        print(f"-- {len(chosen)} memories, {total}/{args.budget} tokens")
    elif args.cmd == "merge":
        result = store.merge_from(args.source, dedupe_threshold=args.dedupe_threshold)
        print(f"added {result['added']}, merged {result['merged']} duplicates")
    store.close()
    return 0


def _main_wrapper(argv=None) -> int:
    try:
        return main(argv)
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(_main_wrapper())
