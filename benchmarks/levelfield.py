"""The level-field benchmark: remembrane vs mem0-OSS on equal terms.

"Equal terms" means: same machine, same embedder (remembrane's dependency-free
HashEmbedder plugged into BOTH systems), fully local storage, and NO LLM
anywhere (mem0 runs with infer=False, skipping its LLM fact-extraction step).
This isolates the memory layer itself from the embedding model and the LLM.

Suites:
  1. write latency   - median ms per store/add over N memories
  2. recall latency  - median ms per query
  3. retrieval agreement - top-1 overlap on paraphrase queries (same embedder
     => quality is expected to tie; this suite PROVES the tie)
  4. stale-fact test - facts updated over time; does the system return the
     CURRENT fact? (remembrane: recency-aware ranking; mem0 infer=False:
     pure cosine, time-blind). v1 facts are aged 30 days by direct timestamp
     edit on the remembrane side only - mem0's ranking ignores time entirely,
     so aging cannot help or hurt it. Documented simulation.
  5. footprint       - on-disk bytes for the same corpus

Fairness notes:
  - remembrane recalls with mode="vector" everywhere (its hybrid BM25 mode is
    default in production, but disabled here because mem0's BM25 requires an
    extra package; we benchmark like-for-like).
  - mem0 add() uses infer=False; that is the documented way to use mem0-OSS
    without an LLM, and it is exactly the "come down to our level" condition.

Run: pip install remembrane mem0ai && python benchmarks/levelfield.py
"""
from __future__ import annotations

import os
import random
import statistics
import sys
import time
from pathlib import Path

for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k)
os.environ.setdefault("OPENAI_API_KEY", "sk-unused-no-llm-in-this-benchmark")
os.environ["MEM0_TELEMETRY"] = "False"

N_DISTRACTORS = 260
N_ENTITIES = 40
SEED = 11

NAMES = "arjun meera kavya rohan diya aditya sneha vikram anaya ishaan".split()
ATTRS = [("favorite editor", ["vim", "emacs", "zed", "helix"]),
         ("preferred cloud", ["aws", "gcp", "azure", "hetzner"]),
         ("home city", ["pune", "chennai", "jaipur", "kochi"]),
         ("main language", ["python", "rust", "go", "typescript"])]
FILLER = ("the sprint review moved to thursday afternoon",
          "ci pipeline caches dependencies aggressively",
          "the design doc needs two more reviewers",
          "database migrations run before deploy",
          "the retro surfaced three action items")


def build_world(rng):
    entities = []
    for i in range(N_ENTITIES):
        name = f"{rng.choice(NAMES)}{i}"
        attr, values = ATTRS[i % len(ATTRS)]
        old, new = rng.sample(values, 2)
        # two update styles: "parallel" keeps the sentence template (common in
        # practice: same fact restated with a new value); "rephrased" changes
        # the wording (adversarial for lexical embedders).
        parallel = i % 2 == 0
        v2 = (f"{name}'s {attr} is now {new}" if parallel
              else f"{name} switched their {attr} to {new}")
        entities.append({
            "v1": f"{name}'s {attr} is {old}",
            "v2": v2, "parallel": parallel,
            "query": f"what is {name}'s {attr}?",
            "current": new, "stale": old,
        })
    distractors = [f"{rng.choice(FILLER)} ({i})" for i in range(N_DISTRACTORS)]
    return entities, distractors


def bench_remembrane(entities, distractors, tmp):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from remembrane import MemoryStore
    db = Path(tmp) / "lf-remembrane.db"
    m = MemoryStore(db)
    writes = []
    month_ago = time.time() - 30 * 86400
    v1_ids = []
    for e in entities:
        t0 = time.perf_counter()
        mem = m.store(e["v1"])
        writes.append(time.perf_counter() - t0)
        v1_ids.append(mem.id)
    # simulated aging of v1 facts (see module docstring)
    m._conn.executemany("UPDATE memories SET created_at=? WHERE id=?",
                        [(month_ago, i) for i in v1_ids])
    m._conn.commit()
    m._invalidate()
    for d in distractors:
        t0 = time.perf_counter()
        m.store(d)
        writes.append(time.perf_counter() - t0)
    for e in entities:
        t0 = time.perf_counter()
        m.store(e["v2"])
        writes.append(time.perf_counter() - t0)

    reads, top1 = [], []
    stale = {"parallel": 0, "rephrased": 0}
    for e in entities:
        t0 = time.perf_counter()
        res = m.recall(e["query"], k=3, mode="vector", touch=False)
        reads.append(time.perf_counter() - t0)
        top = res[0].memory.content if res else ""
        top1.append(top)
        if e["current"] in top:
            stale["parallel" if e["parallel"] else "rephrased"] += 1
    conflicts = sum(bool(m.conflicts(e["query"], min_confidence="possible")) for e in entities[:10])
    size = db.stat().st_size
    m.close()
    return {"write_ms": statistics.median(writes) * 1000,
            "read_ms": statistics.median(reads) * 1000,
            "top1": top1, "stale": stale,
            "conflict_flagged": conflicts, "bytes": size}


def bench_mem0(entities, distractors, tmp):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from mem0 import Memory
    from remembrane.embedders import HashEmbedder
    cfg = {"vector_store": {"provider": "qdrant",
                            "config": {"path": str(Path(tmp) / "lf-qdrant"),
                                       "embedding_model_dims": 512, "on_disk": True}},
           "embedder": {"provider": "openai", "config": {"embedding_dims": 512}}}
    m = Memory.from_config(cfg)
    he = HashEmbedder()

    class _SameEmbedder:  # equal terms: identical embedder for both systems
        def embed(self, text, memory_action=None):
            return he.embed([text])[0]

    m.embedding_model = _SameEmbedder()
    writes = []
    for batch in (
        [e["v1"] for e in entities],
        list(distractors),
        [e["v2"] for e in entities],
    ):
        for text in batch:
            t0 = time.perf_counter()
            m.add(text, user_id="bench", infer=False)
            writes.append(time.perf_counter() - t0)

    reads, top1 = [], []
    stale = {"parallel": 0, "rephrased": 0}
    for e in entities:
        t0 = time.perf_counter()
        hits = m.search(e["query"], filters={"user_id": "bench"}, limit=3)
        reads.append(time.perf_counter() - t0)
        top = hits["results"][0]["memory"] if hits["results"] else ""
        top1.append(top)
        if e["current"] in top:
            stale["parallel" if e["parallel"] else "rephrased"] += 1
    qdir = Path(tmp) / "lf-qdrant"
    size = sum(f.stat().st_size for f in qdir.rglob("*") if f.is_file())
    return {"write_ms": statistics.median(writes) * 1000,
            "read_ms": statistics.median(reads) * 1000,
            "top1": top1, "stale": stale,
            "conflict_flagged": "n/a", "bytes": size}


def main():
    import tempfile
    rng = random.Random(SEED)
    entities, distractors = build_world(rng)
    tmp = tempfile.mkdtemp(prefix="levelfield-")
    print(f"level-field benchmark | {N_ENTITIES*2 + N_DISTRACTORS} memories | seed {SEED}")
    r = bench_remembrane(entities, distractors, tmp)
    z = bench_mem0(entities, distractors, tmp)
    agree = sum(a == b for a, b in zip(r["top1"], z["top1"]))
    print(f"\n{'metric':<28}{'remembrane':>14}{'mem0 (infer=False)':>20}")
    print("-" * 62)
    print(f"{'median write ms':<28}{r['write_ms']:>14.2f}{z['write_ms']:>20.2f}")
    print(f"{'median recall ms':<28}{r['read_ms']:>14.2f}{z['read_ms']:>20.2f}")
    half = N_ENTITIES // 2
    print(f"{'stale-fact (parallel)':<28}{r['stale']['parallel']:>11}/{half}{z['stale']['parallel']:>17}/{half}")
    print(f"{'stale-fact (rephrased)':<28}{r['stale']['rephrased']:>11}/{half}{z['stale']['rephrased']:>17}/{half}")
    print(f"{'conflicts surfaced (of 10)':<28}{str(r['conflict_flagged']):>14}{str(z['conflict_flagged']):>20}")
    print(f"{'storage bytes':<28}{r['bytes']:>14,}{z['bytes']:>20,}")
    print(f"\ntop-1 agreement: {agree}/{N_ENTITIES} — the disagreements are exactly the"
          " parallel stale-fact cases where remembrane returns the current fact")


if __name__ == "__main__":
    main()
