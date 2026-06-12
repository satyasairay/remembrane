# Changelog

## 0.3.1 — 2026-06-12

Fixes from an independent package audit. No new features.

### Fixed
- Zero-relevance memories are no longer returned: similarity must be positive
  to qualify; recency/importance rank relevant memories but never substitute
  for relevance. New `min_similarity` recall parameter.
- `recall(None)` now raises a clear `ValueError` instead of an AttributeError.
- `MemoryStore("missing/dir/db.sqlite")` creates parent directories instead of
  failing with a raw sqlite error.
- CLI prints one-line errors instead of Python tracebacks.
- `metadata` must now be a dict; previously lists/strings were accepted and
  failed later.
- README scoring documentation matched to the implemented formula and current
  default weights (it previously described multiplicative scoring and v0.2
  weights).

## 0.3.0 — 2026-06-12

### Added
- **Conflict-aware recall**: `conflicts()` surfaces memories in tension
  (deterministic heuristics: anchor overlap, change markers, numeric
  mismatches) instead of silently picking a winner; `resolve()` settles them
  with a journaled, auditable decision. New MCP tools `memory_conflicts` /
  `memory_resolve`, CLI `remembrane conflicts`.
- **Outcome feedback**: `mark_useful()` / `mark_useless()` / `feedback()` —
  salience is earned from task outcomes (sigmoid-squashed `usefulness` term in
  ranking) instead of guessed at write time. New MCP tool `memory_feedback`,
  CLI `remembrane feedback`.
- **Token-budget packing**: `pack(query, budget_tokens=800)` returns the
  provably optimal, deduplicated memory set within a token budget (exact 0/1
  knapsack). New MCP tool `memory_pack`, CLI `remembrane pack`.

### Changed
- Scoring now has four normalized weights (similarity/recency/importance/
  usefulness, default 0.65/0.15/0.10/0.10).
- Existing 0.2.x databases are migrated automatically (adds the `usefulness`
  column).

## 0.2.0 — 2026-06-12

### Added
- **Time travel**: every store/forget/reinforce/consolidate is journaled.
  New `snapshot(label)`, `diff(a, b)`, `as_of(point)`, `log()` on MemoryStore,
  and `snapshot` / `diff` / `log` CLI commands.
- **Exact hybrid search**: recall now combines exact cosine similarity with
  exact BM25 keyword scoring (pure stdlib) in one pass. New `mode` parameter
  ("hybrid" default, "vector", "keyword") and `--mode` / `--explain` CLI flags.
- **Explainable recall**: `RecallResult.explain()` and `.explain_text()` expose
  the full ranking breakdown (vector, keyword, recency, importance).
- **Testing kit**: `remembrane.testing` with `assert_recalls`,
  `assert_recalls_first`, `assert_not_recalls`, `recalled_contents`; recall
  accepts `now=` for fully deterministic, CI-friendly behavior.
- **Merge**: `MemoryStore.merge_from()` merges another memory db with
  near-duplicate absorption; `remembrane merge` CLI command.

### Changed
- Default recall mode is hybrid (was pure vector). Pass `mode="vector"` for
  the 0.1.x behavior.

## 0.1.0 — 2026-06-12

Initial release: SQLite-backed MemoryStore, similarity x recency x importance
recall, dependency-free hash embedder (optional OpenAI / sentence-transformers),
LangChain & CrewAI adapters, MCP server, CLI.
