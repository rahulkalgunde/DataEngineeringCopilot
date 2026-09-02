# ADR-017: Qdrant 120s Timeout + Upsert Retry 30*2^attempt (Stash Drain)

## Status
Accepted — 2026-09-02

## Context
`git stash@{0}` "remaining unrelated dirty (qdrant retry etc) - keep for separate PR" contained an isolated `async_qdrant_store.py` fix from the `hybrid_bm25_hardening` + `pipeline_simplification` era: `AsyncQdrantClient` timeout `60→120s` and a new `_upsert_with_retry(max_retries=3, sleep 30*2^attempt)` (30s, 60s, 120s backoff). Meanwhile main advanced to `timeout=300s` (`44f3341 perf(rag): Tier 0+1 speed tuning`) and shipped `975d880/71d8aa1` BM25 alias-atomic cache copy + startup warning (`rebuild_bm25_cache_from_corpus`, `retrieval_prefetch_limit`, `rrf_weights`/`fusion DBSF`). A blind `git stash pop` would revert `DBSF`/`rrf_weights`/`prefetch_limit`/`rebuild_bm25_cache_from_corpus` — stale stash vs current `a965c41..134df22` diff showed `55` churn including deletions of those shipped features.

Plan `plans/2026-09-02_next_pending_1-4_plan.md:Task 4` required surgical cherry-pick: extract only the qdrant retry hunk via `git stash show -p stash@{0} -- data_engineering_copilot/infrastructure/async_qdrant_store.py > /tmp/qdrant_stash.patch` + `git apply --3way`, resolve `.gitignore research/` conflict (keep top-level `research/` ignore + existing `docs/research/`), deduplicate BM25 warning overlap (keep shipped warning, do not re-add namespace warning that duplicates `575d880` startup visibility), and keep `timeout=300` which already satisfies `>=120` (deduplicated, not downgraded).

Other stash hunks (`cooldown_aware_router.py` `embed_query` `input_type=query`, `embedding_cache.py` `provider:d` namespacing, `fallback_embedder.py` `embed_query` delegation, `rag_teaching_ui.py` `CachedEmbedder`, plus their test diffs) are intentionally left in stash for a separate PR — this ADR drains only the qdrant island.

## Decision
- Import `Any` (`from typing import Any, Literal, Self, cast`) for `_upsert_with_retry` signature.
- Add `AsyncQdrantVectorStore._upsert_with_retry(collection_name, points, max_retries=3)` with exponential backoff `asyncio.sleep(30 * 2**attempt)` for transient `yellow`/`connection` errors; logs `warning` per retry and `error` after exhaustion, re-raises last exception.
- `upsert_frozen_chunks` now delegates to `await self._upsert_with_retry(collection_name=self._collection_name, points=Batch(...))` instead of direct `self._client.upsert`.
- Keep `timeout=300` (current `44f3341`) — stash's `120` is superseded by larger `300` already deployed; draining the retry logic is the semantic carry.
- Keep `.gitignore` `research/` top-level ignore (already present at line 71, deduplicated — both `docs/research/` and `research/` now ignored).
- Leave `cooldown_aware_router` / `embedding_cache` / `fallback_embedder` / `rag_teaching_ui` hunks in stash (deferred to `feat(router): cooldown budget 60→120` or separate PR if not shipped via `916c91f`).

## Consequences
- Heavy `gen-build` upserts no longer kill the entire build on a single transient Qdrant `yellow` blip; retry spans up to `210s` (`30+60+120`) before failure.
- `upsert_chunks` (non-frozen path) retains direct `upsert` (no retry) — only generation-bound `upsert_frozen_chunks` is retried, matching stash scope.
- Timeout remains `300s` (covers `?wait=true` on `3072×2048` upserts); if a future reduction to `120` is desired it must be gated via benchmark, not via stash revert.

## Verification
- TDD: `tests/unit/test_qdrant_upsert_retry.py::test_upsert_with_retry_retries_3` fails `AttributeError` before patch, passes after (mocked `asyncio.sleep` to avoid 90s wall time; asserts `call_count==3` and sleeps `30,60`).
- Tier1: `ruff check --fix` → `ruff format` → `pyright` on `async_qdrant_store.py` + `pytest tests/unit/test_qdrant_upsert_retry.py tests/unit/test_async_qdrant_store*.py -v -n 0` PASS.
- Tier2: `ruff check data_engineering_copilot/ tests/ --fix` → `ruff format` → `pyright` → `pytest tests/unit/ -n 6` PASS.
- Stash hygiene: `git stash drop stash@{0}` only after commit green; `git stash list` empty verified.
