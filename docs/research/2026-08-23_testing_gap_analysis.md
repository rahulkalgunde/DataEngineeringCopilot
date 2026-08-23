# Testing Gap Analysis — External Research vs Current Suite (2026-08-23)

Companion to:
- `docs/research/2026-08-23_testing_strategy_external_research.md` (strategy-level, primary sources)
- `docs/research/rag_unit_testing_best_practices_2026-08-23.md` (ecosystem survey, prior session)
- Evidence base: fresh unit-suite coverage run (2026-08-23, branch coverage) + sessions/plans log mining

## Method

1. **External**: background agent researched pyramid layering, hermeticity,
   doubles contracts, coverage philosophy, xdist/asyncio practice against
   pytest/Fowler/testcontainers/Google sources → 18-item checklist.
2. **Internal**: full unit-suite coverage run; module↔test import mapping;
   mined `sessions/*.md`, `plans/*.md` for previously logged gaps.
3. **Comparison** below; fixes applied in-session, verified Tier 1 + Tier 2.

## Comparison verdict

DEC matches or exceeds industry practice on: strict markers, hermetic settings
(`make_settings()` + ambient-var fail-fast — best-in-class vs Haystack/LangChain/
LlamaIndex/RAGAS), REQUIRE_INFRA skip-vs-fail semantics, contract-pinned doubles,
worksteal xdist + `-n 0` debug path, per-test timeout (`fffaf34`), deterministic
`StubEmbedder` double, real-object seam tests, isolated eval harnesses.

Pyramid shape: ~2844 unit / integration+e2e+eval ≈ 93/5/2 — healthy, no ice-cream-cone.

## Gaps found → disposition

| # | Gap (evidence) | Severity | Disposition |
|---|---|---|---|
| G1 | `services/text_filter.py` at 20% cov — production ingestion-path logic, zero dedicated tests | High | **FIXED**: `tests/unit/test_text_filter.py` (13 tests, real DocumentChunk objects) → 98% |
| G2 | `infrastructure/provider_capabilities.py`: zero direct tests despite being contract-critical (AGENTS.md: gates generation params *silently*) | High | **FIXED**: `tests/unit/test_provider_capabilities.py` (fail-closed defaults, provider matrix, table invariants incl. seed⊆penalties, anthropic-absent) → 100% |
| G3 | `infrastructure/rst_parser.py` at 24% — pure parser, trivially testable | Medium | **FIXED**: `tests/unit/test_rst_parser.py` (URL gate, happy path, graceful-None via severe RST error) → 88%; learned contract: `html_to_markdown(min_words=40)` drops short docs |
| G4 | D2 from prior-session divergences still open: no hermetic socket guard (stray httpx/aiohttp could hit wire in CI) | High | **FIXED**: autouse `_block_external_sockets` in `tests/unit/conftest.py` — blocks non-loopback TCP (socket.socket factory + create_connection), allows loopback/AF_UNIX/integration-marked; suite green proves no false positives |
| G5 | No coverage floor anywhere (`fail_under = 0`); CI collects coverage.xml but never gates on it | Medium | **OPEN (recommendation)**: per Google guidance, gate *changed* code (~90%) not repo-wide; needs CI policy decision by owner |
| G6 | CLI entry modules barely covered (`cli_catalog` 7%, `cli_monitor` 10%, `cli_llm_probe` 12%, `cli.py` 28%) | Low-Med | OPEN — wiring-thin but huge surface; suggest CliRunner-based tests for top commands only |
| G7 | Evaluation harness internals thin (`langfuse_score_configs` 29%, `rerank_eval` 45%, `fast_eval` 52%) | Low-Med | OPEN — these have their own frozen-input harnesses (`eval-*`) serving as integration-level verification; unit pins lower priority |
| G8 | Mutation testing absent (coverage ≠ assertion strength) | Low | OPEN — scope mutmut to critical modules only if pursued |

Previously-logged gaps verified CLOSED this session (no action needed):
ingestion seam tests + chunker contract pairs + loud no_content events
(`plans/BUG_analysis_ingestion_silent_skip_2026-08-03.md` §11 items 1–7),
telemetry tracer tests, D1 StubEmbedder, D3 timeout.

## Session verification

- Tier 1 per file: ruff/format/pyright clean; targeted pytest green.
- Tier 2: full `tests/unit -n 6 --dist worksteal` → **2844 passed in 65.8s**
  (was 2805), socket guard active throughout, durations top-5 ≤ 13s.

## Follow-up queue (owner decisions)

1. ~~G5 coverage policy~~ **SHIPPED 2026-08-23 PM**: changed-code gate via
   diff-cover — `make test-unit-cov` + `make test-cov-gate` (`DIFF_THRESHOLD=90`),
   wired into CI `test-unit` after unit tests (checkout `fetch-depth: 0`).
   Repo-wide `fail_under` intentionally stays 0.
2. ~~G6/G7 CLI + eval-harness internals~~ **BACKFILLED 2026-08-23 PM**: 77 new
   unit tests — `run_rerank_eval` orchestration + adapter fallbacks,
   langfuse_score_configs seeding/reconciliation (fake clients),
   cli_catalog probe filtering/SKIP paths, cli_monitor formatters/dashboard/
   fetch_status retry semantics, cli_llm_probe payload/redaction/error-summary.
3. ~~G8 mutation testing absent~~ **PARTIALLY SHIPPED**: `mutmut>=3.0` dev dep +
   scoped `[tool.mutmut]` config + `make mutate` target. Baseline run blocked by
   upstream in-process runner bug → `plans/BLOCKER_2026-08-23_15-20.md`
   (evidence + escape hatches).
4. F1–F6 from `docs/rag_flaw_prevention_plan.md` (eval-side, unchanged).

**Dependency note:** pyproject/uv.lock gained diff-cover + mutmut ⇒ Docker image
dep-hash stale; run `make rebuild` before next app-stack session.
