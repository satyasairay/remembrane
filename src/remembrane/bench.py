"""Measure remembrane on YOUR machine: python -m remembrane.bench

Published numbers never travel between machines — this prints the same table
the README shows, measured locally, with your platform stamped on it.
"""
from __future__ import annotations

import argparse
import platform
import random
import statistics
import sys
import time

from .store import MemoryStore
from . import store as _store_mod

WORDS = (
    "deploy user prefers dark mode aws tokyo pipeline token memory agent "
    "schedule meeting budget python sqlite laptop error region latency"
).split()


def _sentence(rng: random.Random) -> str:
    return " ".join(rng.choices(WORDS, k=rng.randint(5, 14)))


def run(sizes, runs: int = 5, seed: int = 7) -> None:
    rng = random.Random(seed)
    numpy_state = "numpy" if _store_mod._np is not None else "pure python"
    print(f"remembrane bench | {platform.system()} {platform.machine()} | "
          f"Python {platform.python_version()} | {numpy_state}")
    print(f"{'memories':>10} | {'store/1k':>9} | {'recall (median)':>16} | {'pack 800tok':>12}")
    print("-" * 58)
    for n in sizes:
        m = MemoryStore(":memory:")
        t0 = time.perf_counter()
        for start in range(0, n, 1000):
            m.store_many([_sentence(rng) for _ in range(min(1000, n - start))])
        store_ms = (time.perf_counter() - t0) / max(1, n / 1000) * 1000
        m.recall("deploy pipeline tokyo", touch=False)  # warm cache
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            m.recall("deploy pipeline tokyo", touch=False)
            times.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        m.pack("deploy pipeline tokyo", budget_tokens=800, touch=False)
        pack_ms = (time.perf_counter() - t0) * 1000
        print(f"{n:>10,} | {store_ms:>7.0f}ms | {statistics.median(times):>14.1f}ms | {pack_ms:>10.1f}ms")
        m.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m remembrane.bench")
    parser.add_argument("--sizes", default="1000,10000,50000",
                        help="comma-separated store sizes (default 1000,10000,50000)")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args(argv)
    run([int(x) for x in args.sizes.split(",")], runs=args.runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
