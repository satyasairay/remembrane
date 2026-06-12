# The level-field benchmark

**Question:** if the big agent-memory systems come down to remembrane's level —
same machine, same embedder, fully local, no LLM calls — who wins?

**Contestants:** remembrane vs mem0-OSS (`infer=False`, qdrant local). Zep and
Letta cannot enter: they are servers/frameworks; there is no library-level mode
to bring down. That asymmetry is itself a result.

**Equal terms:** identical embedder object (remembrane's dependency-free
HashEmbedder) plugged into both systems; remembrane recalls in `mode="vector"`
(its hybrid BM25 default is disabled because mem0's keyword search needs an
extra package); mem0 skips LLM extraction via its documented `infer=False`.
340 memories (80 entity facts incl. 40 updates + 260 distractors), seed 11.

## Results (Linux sandbox, Python 3.10 — run it yourself, see below)

| metric | remembrane | mem0 (infer=False) |
|---|---:|---:|
| median write latency | **3.2 ms** | 29.1 ms |
| median recall latency | **0.6 ms** | 3.5 ms |
| stale-fact, parallel phrasing ("X is now B") | **20/20 current** | 0/20 (returns stale) |
| stale-fact, rephrased update | 0/20 | 0/20 |
| contradictions surfaced to caller | **10/10** | not a feature |
| storage for same corpus | **1.09 MB** | 1.90 MB |
| required deps (pip) | **0** | openai, posthog, protobuf, qdrant-client, sqlalchemy, pydantic, pytz |
| top-1 retrieval agreement | 20/40 — disagreements are exactly the stale-fact cases | — |

## Reading this honestly

- **Retrieval quality ties by construction.** With the same embedder, both
  systems rank by the same cosine signal; the embedder does that work. The
  memory layer's job is everything else — and that's where the columns differ.
- **mem0's own documentation corroborates the stale-fact result**: their add
  docs state that raw (`infer=False`) inserts "skip conflict resolution" and
  that "duplicates will land" (docs.mem0.ai, memory-operations/add). We
  measured what that means in practice.
- **The stale-fact result is the point.** When a fact is updated in similar
  wording, time-blind cosine confidently returns the OLD fact every single
  time; remembrane's recency-aware ranking returns the current one 20/20, and
  its conflict detection flags the contradiction either way. When the update is
  *rephrased*, the lexical embedder fails both systems equally (0/20) at the
  ranking layer — though remembrane's conflict detection still surfaces the
  contradiction (`possible` tier), so the agent is told not to trust top-1.
  Run the neural configuration yourself:
  `pip install model2vec && python benchmarks/levelfield.py --embedder model2vec`
  (same static embedding model injected into both systems).
- **What this does NOT measure:** mem0's LLM extraction layer (its main value
  proposition — also its cost: LLM calls on every write per mem0's docs;
  independent write-ups measure 200-500 ms added write latency), cloud features, team sharing. On full-pipeline accuracy
  benchmarks (LoCoMo, LongMemEval) mem0 and Zep publish strong numbers; this
  benchmark deliberately measures the layer *underneath* all that.
- One sentence: **below the LLM layer, the premium memory systems are vector
  stores with more dependencies — and they confidently return stale facts.**

## Reproduce

```bash
pip install remembrane mem0ai
python benchmarks/levelfield.py
```

Deterministic corpus (seed 11). v1 facts are aged 30 days by direct timestamp
edit on the remembrane side only — mem0's ranking ignores time entirely, so
aging cannot affect its results; this is documented simulation, not tilt.
