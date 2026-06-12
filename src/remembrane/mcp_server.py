"""MCP server exposing remembrane to any MCP-capable agent (Claude, etc.).

Requires the optional mcp extra::

    pip install remembrane[mcp]
    remembrane-mcp --db agent.db        # or: python -m remembrane.mcp_server

Tools exposed: memory_store, memory_recall, memory_forget, memory_reinforce,
memory_stats.
"""
from __future__ import annotations

import argparse
import os
import sys

from .store import MemoryStore


def build_server(db_path: str):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'mcp' package is required for the MCP server. "
            "Run: pip install remembrane[mcp]"
        ) from exc

    store = MemoryStore(db_path)
    server = FastMCP(
        "remembrane",
        instructions=(
            "Persistent local memory. Use memory_store to remember facts worth "
            "keeping across sessions, memory_recall to retrieve relevant ones."
        ),
    )

    max_content = int(os.environ.get("REMEMBRANE_MAX_CONTENT", "100000"))

    @server.tool()
    def memory_store(content: str, namespace: str = "default", importance: float = 0.5) -> str:
        """Store a memory for later recall. importance: 0.0 (trivial) to 1.0 (critical)."""
        if len(content) > max_content:
            return (f"Refused: content is {len(content)} chars; limit is {max_content} "
                    "(set REMEMBRANE_MAX_CONTENT to change).")
        mem = store.store(content, namespace=namespace, importance=importance)
        return f"Stored memory {mem.id}"

    @server.tool()
    def memory_recall(query: str, namespace: str = "default", k: int = 5) -> str:
        """Recall the memories most relevant to a query."""
        results = store.recall(query, k=k, namespace=namespace)
        if not results:
            return "No memories found."
        return "\n".join(f"[{r.score:.2f}] {r.memory.content}" for r in results)

    @server.tool()
    def memory_forget(memory_id: str) -> str:
        """Permanently delete a memory by id."""
        n = store.forget(memory_id)
        return f"Deleted {n} memory(ies)."

    @server.tool()
    def memory_reinforce(memory_id: str) -> str:
        """Strengthen a memory so it decays slower and ranks higher."""
        mem = store.reinforce(memory_id)
        return f"Reinforced {memory_id}" if mem else f"No memory {memory_id}"

    @server.tool()
    def memory_conflicts(query: str = "", namespace: str = "default") -> str:
        """Check for conflicting memories (optionally scoped to a query) before answering.
        Returns tensions for you to adjudicate or ask the user about."""
        found = store.conflicts(query or None, namespace=namespace)
        if not found:
            return "No conflicting memories detected."
        return "\n\n".join(
            f"[{c.a.id} vs {c.b.id}]\n{c.describe()}" for c in found[:5]
        )

    @server.tool()
    def memory_resolve(keep_id: str, drop_id: str, reason: str = "") -> str:
        """Settle a conflict: keep one memory, forget the other. Journaled and auditable."""
        n = store.resolve(keep_id, [drop_id], reason=reason)
        return f"Kept {keep_id}, dropped {n} memory(ies)."

    @server.tool()
    def memory_feedback(memory_id: str, useful: bool) -> str:
        """After completing a task, report whether a recalled memory actually helped.
        Useful memories rank higher in future; useless ones fade."""
        mem = store.feedback(memory_id, useful)
        if not mem:
            return f"No memory {memory_id}"
        return f"Recorded. {memory_id} usefulness is now {mem.usefulness:+.1f}."

    @server.tool()
    def memory_pack(query: str, budget_tokens: int = 800, namespace: str = "default") -> str:
        """Get the optimal set of memories that fits a token budget (exact, deduplicated)."""
        chosen = store.pack(query, budget_tokens=budget_tokens, namespace=namespace)
        if not chosen:
            return "No memories found."
        return "\n".join(f"[{r.memory.id}|{r.tokens}tok] {r.memory.content}" for r in chosen)

    @server.tool()
    def memory_stats() -> str:
        """Count memories per namespace."""
        lines = [f"total: {store.count()}"]
        lines += [f"{ns}: {store.count(ns)}" for ns in store.namespaces()]
        return "\n".join(lines)

    return server


def main(argv=None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(prog="remembrane-mcp")
    parser.add_argument("--db", default=os.environ.get("REMEMBRANE_DB", "remembrane.db"))
    args = parser.parse_args(argv)
    build_server(args.db).run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
