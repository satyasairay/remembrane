"""remembrane CLI: inspect and manage a memory database from the terminal."""
from __future__ import annotations

import argparse
import json
import sys

from .store import MemoryStore


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="remembrane", description="Inspect an agent memory database.")
    parser.add_argument("--db", default="remembrane.db", help="Path to the SQLite database")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_store = sub.add_parser("store", help="Store a memory")
    p_store.add_argument("content")
    p_store.add_argument("--namespace", default="default")
    p_store.add_argument("--importance", type=float, default=0.5)

    p_recall = sub.add_parser("recall", help="Recall memories for a query")
    p_recall.add_argument("query")
    p_recall.add_argument("--namespace", default="default")
    p_recall.add_argument("-k", type=int, default=5)

    p_list = sub.add_parser("list", help="List all memories")
    p_list.add_argument("--namespace", default=None)

    p_forget = sub.add_parser("forget", help="Forget a memory by id")
    p_forget.add_argument("memory_id")

    p_export = sub.add_parser("export", help="Export memories as JSON")
    p_export.add_argument("--namespace", default=None)

    sub.add_parser("stats", help="Show database stats")

    args = parser.parse_args(argv)
    store = MemoryStore(args.db)

    if args.cmd == "store":
        mem = store.store(args.content, namespace=args.namespace, importance=args.importance)
        print(f"stored {mem.id}")
    elif args.cmd == "recall":
        for r in store.recall(args.query, k=args.k, namespace=args.namespace, touch=False):
            print(f"{r.score:.3f}  [{r.memory.id[:8]}]  {r.memory.content}")
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
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
