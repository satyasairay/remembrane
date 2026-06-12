# remembrane

**Local-first persistent memory for AI agents.** One SQLite file, zero required dependencies. Exact hybrid recall (vector + BM25 — never approximate), explainable ranking, time-travel over memory history, and deterministic behavior you can unit-test in CI. Adapters for LangChain and CrewAI, plus a built-in MCP server.

```bash
pip install remembrane
```

## Why

Agents forget everything between sessions. Existing memory solutions are cloud APIs, require a vector database, or drag in a heavyweight framework. `remembrane` is the opposite:

- **One file.** Your agent's entire memory is a SQLite database you can copy, back up, diff, or delete.
- **Zero required dependencies.** The default embedder is pure stdlib. `pip install remembrane` pulls in nothing else.
- **Human-like recall.** Results are ranked by a composite of similarity, recency decay (memories halve in weight every week by default), and importance. Recalled memories are *reinforced* — spaced repetition for agents.
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
print(results[0].score)                  # similarity × recency × importance
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
        weight_similarity=0.7,
        weight_recency=0.15,
        weight_importance=0.15,
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

```python
from remembrane import MemoryStore
from remembrane.adapters import RemembraneChatMemory

memory = RemembraneChatMemory(MemoryStore("agent.db"), session_id="user-42")

memory.save_context({"input": "my favorite color is teal"}, {"output": "Noted!"})
memory.load_memory_variables({"input": "what color do I like?"})
# {'history': 'human: my favorite color is teal\nai: Noted!'}
```

Unlike buffer memory, this retrieves the exchanges *relevant to the current input* — the context window stays small no matter how long the history grows.

## CrewAI

```python
from remembrane import MemoryStore
from remembrane.adapters import RemembraneStorage

storage = RemembraneStorage(MemoryStore("crew.db"))
storage.save("the deadline is next friday", metadata={"task": "planning"})
storage.search("when is the deadline?")
```

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

Tools exposed: `memory_store`, `memory_recall`, `memory_forget`, `memory_reinforce`, `memory_stats`.

## CLI

```bash
remembrane --db agent.db store "the user prefers dark mode" --importance 0.8
remembrane --db agent.db recall "what theme?"
remembrane --db agent.db list
remembrane --db agent.db stats
remembrane --db agent.db export > backup.json
```


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

## How ranking works

```
score = 0.7·similarity + 0.15·recency + 0.15·importance
recency = exp(−ln2 · age / half_life)
```

`age` is measured from the memory's **last access**, not creation — every recall resets the decay clock. Frequently-used memories stay vivid; untouched ones fade. In the default hybrid mode, similarity is `0.65·cosine + 0.35·bm25`. All weights, the mode, and the half-life are configurable.

## Design choices

- **SQLite over a vector DB** — agent memory stores are small (thousands, not billions, of rows). Brute-force cosine over a few thousand vectors is sub-millisecond, and you gain transactions, a single portable file, and zero infra.
- **No background daemon** — decay is computed at read time, so nothing runs when your agent doesn't.
- **Duck-typed adapters** — `remembrane` never imports langchain or crewai; the adapters match their interfaces structurally, so there are no version-pinning fights.

## Development

```bash
git clone https://github.com/satyasairay/remembrane
cd remembrane
pip install -e .[dev]
pytest
```

## License

MIT
