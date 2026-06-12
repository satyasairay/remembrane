# remembrane

**Local-first persistent memory for AI agents.** One SQLite file, zero required dependencies. Exact hybrid recall (vector + BM25 — never approximate), explainable ranking, time-travel over memory history, conflict-aware recall that admits uncertainty, salience learned from task outcomes, budget-capped context packing (exactly optimal when numpy is present), and deterministic behavior you can unit-test in CI. Adapters for LangChain and CrewAI, plus a built-in MCP server.

```bash
pip install remembrane
```

## Why

Agents forget everything between sessions. Existing memory solutions are cloud APIs, require a vector database, or drag in a heavyweight framework. `remembrane` is the opposite:

- **One file.** Your agent's entire memory is a SQLite database you can copy, back up, diff, or delete.
- **Zero required dependencies.** The default embedder is pure stdlib. `pip install remembrane` pulls in nothing else.
- **Human-like recall.** Results are ranked by a weighted sum of similarity, recency decay (halves every week by default), importance, and outcome-earned usefulness. Recalled memories are *reinforced* — spaced repetition for agents.
- **Exact, not approximate.** Large systems use approximate nearest-neighbor search and accept missed results. At agent-memory scale, remembrane scores *every* memory — hybrid vector + BM25 keyword in one pass, guaranteed complete.
- **A memory you can debug.** Every store/forget/reinforce is journaled. Snapshot, diff, and reconstruct what your agent knew at any point in time. Every recall result explains exactly why it ranked where it did.
- **Testable in CI.** Deterministic embedder + frozen-time recall = reproducible memory behavior. `remembrane.testing` ships pytest-friendly assertions.
- **Framework-agnostic.** Use it bare, through the LangChain or CrewAI adapters, or expose it to any MCP-capable agent (like Claude) as an MCP server.

## Quick start

```python
from remembrane import MemoryStore

mem = MemoryStore("agent.db")            # or ":memory:" for ephemeral

mem.store("User prefers dark mode", importance=0.8)
mem.store("Deploy target is AWS us-east-1", namespace="ops")

results = mem.recall("what theme does the user like?")
print(results[0].memory.content)         # → "User prefers dark mode"
print(results[0].score)                  # weighted: similarity + recency + importance + usefulness
```

### Memory lifecycle

```python
mem.reinforce(memory_id)                  # strengthen: slower decay, higher rank
mem.forget(memory_id)                     # delete one
mem.forget(namespace="ops")               # delete a namespace
mem.forget(older_than_seconds=30*86400)   # prune stale memories
mem.consolidate()                         # merge near-duplicates
mem.export()                              # plain dicts, ready for json.dump
```

### Tuning recall

```python
from remembrane import MemoryStore, ScoringConfig

mem = MemoryStore(
    "agent.db",
    scoring=ScoringConfig(
        weight_similarity=0.65,
        weight_recency=0.15,
        weight_importance=0.10,
        weight_usefulness=0.10,            # earned from mark_useful()/mark_useless()
        half_life_seconds=7 * 24 * 3600,   # recency halves every week
    ),
)
```

## Embedders

The default `HashEmbedder` is deterministic, offline, and dependency-free — it hashes word and character n-grams. That makes similarity *lexical*, not semantic. It works well for typical agent memories (facts, preferences, short statements). For true semantic recall, plug in a real model:

```python
from remembrane import MemoryStore, SentenceTransformerEmbedder, OpenAIEmbedder

mem = MemoryStore("agent.db", embedder=SentenceTransformerEmbedder())   # local, pip install remembrane[sentence-transformers]
mem = MemoryStore("agent.db", embedder=OpenAIEmbedder())                # API,   pip install remembrane[openai]
```

Any object with `embed(texts) -> List[List[float]]` and a `dimension` attribute works.

Note: don't mix embedders in one database. Vectors from different embedders aren't comparable.

## LangChain

For current LangChain (verified against langchain-core 1.4):

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from remembrane import MemoryStore
from remembrane.adapters import RemembraneChatMessageHistory

store = MemoryStore("agent.db")
chain = RunnableWithMessageHistory(
    runnable,
    lambda session_id: RemembraneChatMessageHistory(store, session_id),
)
```

Needs `pip install langchain-core` (lazily imported — the rest of remembrane stays dependency-free). For legacy pre-1.x code, `RemembraneChatMemory` still provides the old `save_context` / `load_memory_variables` interface with semantic retrieval — no langchain install required at all.

## CrewAI

```python
from remembrane import MemoryStore
from remembrane.adapters import RemembraneStorage

storage = RemembraneStorage(MemoryStore("crew.db"))
storage.save("the deadline is next friday", metadata={"task": "planning"})
storage.search("when is the deadline?")          # also: delete / update / list_records / reset
```

A storage helper, duck-typed (save/search/delete/update/list_records/get_record/count/reset, kwargs-tolerant). **Known limitation:** it is not a registered `crewai.StorageBackend` subclass and returns dicts rather than `(MemoryRecord, score)` tuples, so plugging it directly into `crewai.Memory(...)` does not work as of crewai 1.14 — use it directly or behind a thin shim. Native StorageBackend integration is on the roadmap. Note CrewAI itself phones home (`telemetry.crewai.com`); set `CREWAI_DISABLE_TELEMETRY=true` if that matters to you — bare remembrane opens no sockets (verified by audit under Python audit hooks).

## MCP server

Give any MCP-capable agent (e.g. Claude Desktop, Claude Code) persistent memory:

```bash
pip install remembrane[mcp]
remembrane-mcp --db ~/agent-memory.db
```

```json
{
  "mcpServers": {
    "remembrane": {
      "command": "remembrane-mcp",
      "args": ["--db", "/path/to/agent-memory.db"]
    }
  }
}
```

Tools exposed: `memory_store`, `memory_recall`, `memory_forget`, `memory_reinforce`, `memory_conflicts`, `memory_resolve`, `memory_feedback`, `memory_pack`, `memory_stats`. Stored content is capped at 100k chars per memory (`REMEMBRANE_MAX_CONTENT` to change).

## CLI

```bash
remembrane --db agent.db store "the user prefers dark mode" --importance 0.8
remembrane --db agent.db recall "what theme?"
remembrane --db agent.db list
remembrane --db agent.db stats
remembrane --db agent.db export > backup.json
```



## Conflict-aware recall

Every other memory system silently resolves contradictions and returns one confident answer — which is how agents end up confidently wrong. remembrane surfaces the tension and lets the agent adjudicate (or ask the user):

```python
mem.store("the user lives in London")
mem.store("the user moved to Tokyo, no longer in London")

for c in mem.conflicts("where does the user live?"):
    print(c.describe())
# Conflicting memories (likely, change_markers=['longer', 'moved', 'no']):
#   older: 'the user lives in London' (recalled 4x)
#   newer: 'the user moved to Tokyo, no longer in London' (recalled 0x)

mem.resolve(keep_id=newer.id, drop_ids=[older.id], reason="user confirmed Tokyo")
```

Detection is deterministic and free (anchor-word overlap, negation markers, numeric mismatches, value substitution — honest heuristics, not hidden LLM judgments). Two confidence tiers: `likely` (strong negation, a numeric/weekday/month mismatch with corroboration, or a same-template value swap like "written in Python" → "written in Rust") and `possible` (topical tension worth a look). Independent audit on a 30-pair adversarial set measured the `likely` tier at 0.875 precision / 0.70 recall on v0.4 and 0.889 / 0.80 on v0.5.0 — better, not perfect. v0.5.1 adds the value-substitution signal, which catches the two false negatives that audit named (technology and region swaps); the full set has not been re-measured. Known limitations: rephrased (non-template) contradictions can stay in the `possible` tier, and "old X / new X" pairs describing two coexisting things can flag as `likely` (suppressing those would also suppress real old→new contradictions, so we don't). It remains a heuristic: treat conflicts as *candidates for the agent to adjudicate*, which is the design intent. Filter with `conflicts(min_confidence='likely')`. Resolutions are journaled, so every settled conflict stays auditable via `log()` and `as_of()`. Also exposed as the `memory_conflicts` / `memory_resolve` MCP tools and `remembrane conflicts` CLI.

## Salience earned from outcomes

Cloud systems decide what matters *at write time*, with an LLM call you pay for on every memory. remembrane inverts it: writes are free, and importance is **earned by helping**:

```python
results = mem.recall("how do I deploy this?")
# ... agent completes its task using results[0] ...
mem.mark_useful(results[0].memory.id)     # this memory rises
mem.mark_useless(results[2].memory.id)    # this one fades
```

Feedback accumulates into a usefulness signal (sigmoid-squashed into ranking, neutral at zero). Memories that keep helping outrank memories that merely match — learned per-deployment, from real outcomes, with zero LLM calls.

## Token-budget packing

Agents don't want "top 5 results"; they want the best use of the context window space they have left:

```python
context = mem.pack("user preferences", budget_tokens=800)
sum(r.tokens for r in context)   # <= 800, guaranteed
```

`pack()` scores every candidate exactly, suppresses near-duplicates so the budget is never spent saying the same thing twice, then solves the selection with a 0/1 knapsack. The budget is a hard guarantee in every configuration (verified over thousands of randomized trials). Optimality depends on the path: with numpy installed the solution is *exact* at 1-token granularity; the pure-python fallback uses coarsened weights plus a greedy refill and is documented as near-optimal, not optimal (worst observed loss 16% on adversarial random instances — real memory stores sit nowhere near that). Deterministic, no LLM. Pass `token_estimator=your_tokenizer` for exact counts.

## Time travel

Every mutation is journaled, so the past is queryable:

```python
mem.snapshot("before-research")
# ... agent runs, learns things, forgets things ...

mem.diff("before-research")
# {'added': [{'content': 'competitor launched a new pricing tier', ...}],
#  'removed': [...], 'changed': [...]}

mem.as_of("before-research")          # full memory state at that point
mem.log()                             # newest-first history of every operation
```

Or from the CLI: `remembrane snapshot v1`, `remembrane diff v1`, `remembrane log`.
"What did my agent believe last Tuesday, and what changed its mind?" is now an answerable question.

## Explainable recall

No black boxes — every result carries its full ranking breakdown:

```python
r = mem.recall("what theme does the user like?")[0]
r.explain()
# {'score': 0.6087, 'components': {'vector_similarity': 0.71, 'keyword_bm25': 1.0,
#   'combined_similarity': 0.81, 'recency': 0.98, 'importance': 0.8}, ...}
r.explain_text()
# 'score 0.609 = similarity 0.812 (vector 0.713, keyword 1.000) + recency 0.984 + importance 0.80 | recalled 3x'
```

## Testing your agent's memory

Deterministic recall means memory behavior is unit-testable — something no cloud memory API can offer:

```python
from remembrane.testing import assert_recalls, assert_recalls_first, assert_not_recalls

def test_agent_remembers_allergies():
    mem = build_agent_memory()
    assert_recalls_first(mem, "any food allergies?", "peanuts")
    assert_not_recalls(mem, "any food allergies?", "dark mode", k=1)
```

Pass `now=...` to `recall()` to freeze time and make recency scoring reproducible.

## Merging memories

Memory files are portable — merge two agents' brains, with near-duplicate absorption:

```python
mem.merge_from("other-agent.db")            # {'added': 12, 'merged': 3}
mem.merge_from("backup.db", namespaces=["prefs"], dedupe_threshold=0.95)
```

CLI: `remembrane --db a.db merge b.db`


## Performance

Performance numbers don't travel between machines, so measure your own first:

```bash
python -m remembrane.bench
```

Two reference points (hybrid recall, 512-dim default embedder, warm cache):

| memories | recall / pack (Linux sandbox, py3.10, numpy) | recall / pack (independent audit: Windows, py3.12, numpy) | recall (pure python, audit machine) |
|---|---|---|---|
| 1,000 | ~2 ms / ~17 ms | ~5 ms / ~32 ms | ~113 ms |
| 10,000 | ~30 ms / ~44 ms | ~51 ms / ~76 ms | ~1.2 s |
| 50,000 | ~205 ms / ~222 ms | ~1.0 s / ~430 ms | ~6.8 s |

The core stays dependency-free; if numpy is importable it is used automatically (`pip install remembrane[fast]`), and a broken numpy install is ignored rather than fatal. For sub-10ms recall beyond ~10k memories, or anything beyond ~50k, you've outgrown the design — that's vector-database territory, and remembrane won't pretend otherwise.

### Concurrency

Multiple connections, threads, and processes can share one memory file: file-backed stores default to SQLite WAL mode with a busy timeout and immediate write transactions, caches detect external writes via SQLite's `data_version`, and residual lock races are retried. Our test suite hammers 3 connections × 6 threads and 8 processes against a single file with zero errors. Two caveats: WAL keeps transient `-wal`/`-shm` sidecar files next to the db (pass `journal_mode="DELETE"` for strict single-file behavior), and SQLite on network filesystems (NFS/SMB) is unsafe regardless of mode — keep memory files on local disk.

## How ranking works

```
score      = 0.65·similarity + 0.15·recency + 0.10·importance + 0.10·usefulness
recency    = exp(−ln2 · age / half_life)
usefulness = sigmoid(outcome feedback)
```

Scoring is a weighted sum (weights normalize to 1), with one hard rule on top: similarity must be positive for a memory to be returned at all — recency and importance rank relevant memories, they never substitute for relevance.

`age` is measured from the memory's **last access**, not creation — every recall resets the decay clock. Frequently-used memories stay vivid; untouched ones fade. In the default hybrid mode, similarity is `0.65·cosine + 0.35·bm25`. All weights, the mode, and the half-life are configurable.

## Design choices

- **SQLite over a vector DB** — agent memory stores are small (thousands, not billions, of rows). Exact brute-force scoring at that scale is fast enough (see Performance for measured numbers), and you gain transactions, a single portable file, and zero infra.
- **No background daemon** — decay is computed at read time, so nothing runs when your agent doesn't.
- **Duck-typed adapters** — `remembrane` never imports langchain or crewai; the adapters match their interfaces structurally, so there are no version-pinning fights.

## Scope notes

- The CLI writes wherever `--db` points, with the invoking user's permissions — it is a local tool, not a sandbox. Wrap it if you expose it to untrusted input.
- OS argv limits apply to `remembrane store "<content>"`; use `--file path` or `--file -` (stdin) for large content.
- MCP argument validation follows pydantic's lax coercion (e.g. `useful="yes"` coerces to `True`).
- Recall `touch` updates (access stats) are statistics, not events — they are intentionally not journaled, and `as_of()` reconstructs content/importance state only.
- `export()`/`merge_from()` carry memories (content, importance, metadata, access stats, usefulness) but not the source's journal history; embeddings are regenerated by the destination's embedder.
- Journal entries with corrupt payloads are surfaced in `log()` (with a `_corrupt` key) and skipped by `as_of()` reconstruction.
- A missing or corrupt embedding blob is self-healed on first read (re-embedded from content) and a `RuntimeWarning` is emitted naming the affected ids — availability is preserved, and the repair is loud, not silent.
- WAL `-wal`/`-shm` sidecar files can grow during sustained concurrent writes; they shrink back on checkpoint and disappear when all handles close. Reported sizes in our tests are settled-state, not peak.
- A process killed during initial db creation can leave an empty file; reopening it repairs the schema automatically.

## Development

```bash
git clone https://github.com/satyasairay/remembrane
cd remembrane
pip install -e .[dev]
pytest
```

## License

MIT
