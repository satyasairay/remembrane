# Changelog

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
