# Changelog

## 0.5.1 — 2026-06-12

Response to round 4 of independent auditing — the final verification audit,
whose verdict was: yes, 0.5.0 can hold production agent memory, including
multi-process writers on local disk. Findings were Medium and below; all
addressed same-day.

### Fixed
- **Conflict detection value-substitution false negatives (audit: Medium).**
  A new substitution signal upgrades same-template value swaps ("written in
  Python" -> "written in Rust", "us-east-1" -> "eu-west-1") to the `likely`
  tier. General by construction: equal-arity word diffs + identical token
  template, no whitelist of value classes. A leading-diff-word guard keeps
  subject swaps ("alice drinks coffee" / "bob drinks coffee") from upgrading.
  The audit's "old laptop 16GB / new laptop 32GB" benign false positive is
  retained and documented — suppressing it would also suppress real old->new
  contradictions.
- **Corrupt embedding self-heal is now loud (audit: Low).** Re-embedding a
  missing/corrupt blob emits a RuntimeWarning naming the affected ids.
  Availability behavior is unchanged.

### Documentation
- README conflict wording tightened per audit: cites v0.4 (0.875 P / 0.70 R)
  and v0.5.0 (0.889 P / 0.80 R) measured numbers, names the remaining miss
  classes, and no longer reads as "all v0.4 false negatives are fixed".
- WAL sidecar limitation now notes reported sizes are settled-state, not peak
  (audit: Low).
- Python 3.9 and 3.13, unverifiable on the audit host, are exercised by CI on
  Linux/Windows/macOS (all 15 jobs green at the audited commit).

### Tests
- tests/test_round4_fixes.py: auditor-named pairs as regressions, including
  the documented false positive asserted *as* a limitation — if behavior
  changes silently, the suite fails. 130 tests total.

## 0.5.0 — 2026-06-12

Response to round 3 of independent auditing ("mother of all audits").

### Fixed
- **Cross-connection cache coherence (audit: High, production blocker).**
  Caches now detect writes from other connections/processes via SQLite's
  data_version and rebuild safely; recall retries on mid-read external writes.
  The audit's 8-process hammer (779 KeyErrors in 60s on 0.4.0) now runs clean.
- **Concurrency hardening.** File-backed stores default to WAL journal mode
  with busy_timeout=30s and immediate write transactions; residual lock races
  retried with backoff. `journal_mode="DELETE"` restores strict
  single-file behavior.
- **pack() optimality (audit: Medium).** Exact 1-token-granularity knapsack
  when numpy is present (audit's 1000-trial probe: worst loss 23.29% -> 0.00%);
  pure-python fallback gains greedy refill (23.29% -> 16.47% worst observed)
  and is documented as near-optimal.
- **Broken numpy no longer breaks import (audit: Medium).** Soft import
  catches all exceptions.
- **Corrupt embedding blobs self-heal (audit: Medium).** Wrong-length blobs
  are re-embedded from content and persisted; NaN/Inf vectors are scrubbed in
  the numpy path.
- **Conflict detection (audit: Medium).** Weekday/month value mismatches now
  count like numeric mismatches, fixing the audit's false negative
  ("deadline is Friday" -> "deadline is now Monday").
- **README/MCP mismatch (audit: Medium).** Tool list corrected to all nine
  tools; MCP memory_store now rejects content over REMEMBRANE_MAX_CONTENT
  (default 100k chars).
- **Validation gaps (audit: Low).** namespace must be a non-empty string
  (was: raw IntegrityError on None); metadata must be JSON-serializable
  without NaN/Infinity (was: NaN accepted, datetime raised raw TypeError).
- **LangChain structured content (audit: Low).** List-of-parts message
  content stores its text, not str(list).
- Reopening an empty db file (e.g. after a kill during creation) repairs the
  schema.

### Added
- `python -m remembrane.bench` — measure recall/pack on your own machine; the
  README now publishes two reference tables (ours + the independent audit's)
  instead of implying portability.

### Changed
- CrewAI adapter documents honestly that it is not a StorageBackend subclass
  (native integration on the roadmap); adds get_record/count.
- README: concurrency section, WAL sidecar/NFS caveats, journal/export scope
  notes, CrewAI telemetry note.

## 0.4.0 — 2026-06-12

Driven by round 2 of independent auditing (adversarial claim verification).

### Fixed
- **pack() budget violation (audit: High).** Knapsack weights now round up and
  a final exact check enforces the cap: the token budget is a hard guarantee.
- **LangChain adapter (audit: High).** New `RemembraneChatMessageHistory`
  returns a real `BaseChatMessageHistory` for `RunnableWithMessageHistory`,
  verified against langchain-core 1.4. Legacy `RemembraneChatMemory` retained.
- **CrewAI adapter (audit: High).** Implements the current storage surface
  (delete/update/list_records, kwargs-tolerant search/save).
- **diff() directionality (audit: Medium).** `diff(b, a)` now inverts
  `diff(a, b)` instead of repeating it.
- **Corrupt journal payloads (audit: Medium).** Malformed JSON no longer
  crashes `log()`/`as_of()`; corrupt entries are surfaced as `_corrupt`.
- **Conflict precision (audit: Medium).** Markers split into strong negations
  vs weak change-verbs needing numeric corroboration; new `min_confidence`
  filter. Precision on the adversarial set: 0.56 -> 1.00 at unchanged recall.

### Added
- **Performance overhaul (audit: High).** Corpus caching, lazy row
  materialization, cached BM25 statistics, and an automatic numpy fast path
  (`pip install remembrane[fast]`). Versus the audit's measurements: recall at
  10k memories 738ms -> ~30ms; pack at 1k 1.9s -> ~17ms. README publishes
  measured tables instead of adjectives.
- `remembrane store --file PATH|-` for content beyond OS argv limits.
- `remembrane conflicts --min-confidence likely`.

### Changed
- README claims rewritten to measured, falsifiable statements (pack guarantee
  semantics, performance tables, adapter version notes, CLI scope notes).

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
