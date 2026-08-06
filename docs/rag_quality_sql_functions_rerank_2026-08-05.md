# Plan: Spark RAG Quality — Eval Harness Bug, SQL Functions Corpus Gap, Rerank Pool

Date: 2026-08-05
Status: Approved — pending implementation

## Objective

Improve general-query RAG quality by fixing three coordinated issues found during
baseline diagnosis:

1. **Eval harness bug** — `evaluate_spark_dataset` builds the RAG service without
   initializing the reranker, so every published baseline silently measured the
   no-rerank pipeline.
2. **Corpus gap** — `docs/sql-ref-functions-builtin.md` is a Jekyll hub page whose
   body is only `{% include_api_gen ... %}` tags; the real SQL built-in function
   content lives in Scala `@ExpressionDescription` annotations that the corpus
   never indexes. Q3–5 of the Spark eval cannot succeed.
3. **Rerank pool truncation** — expected URLs land in the fused result at ranks
   beyond the current pool cap, so the reranker never sees them.

## Locked Decisions (user-confirmed 2026-08-05)

| # | Decision |
|---|----------|
| 1 | **Option A** — index Scala expression sources for Q3–5. Eval rows Q3–5 may change (approved override of the "stop on eval data change" condition). |
| 2 | Index **all 66** `@ExpressionDescription`-bearing files under `catalyst/expressions/`, selected via a resolver content filter (glob alone matches 133 files). |
| 3 | Rerank pool = **Change 3a**: `max(retrieval_top_k * 8, reranker_top_k * 5)` = 240. |
| 4 | **`_FUNC_` name resolution = parse `FunctionRegistry.scala`** (class-name heuristic insufficient: `ArrayJoin` → `array_join`, `ArraySort` → `array_sort`). |

## Change 1 — Fix eval harness (reranker was silently disabled)

**Problem:** `evaluate_spark_dataset` in `cli.py` builds via `build_rag_service()`
but never calls `await reranker.initialize()`. The production singleton
(`rag_service_singleton.py:44`) does. Every published baseline measured the
no-rerank pipeline.

**Fix:** Add `await service.reranker.initialize()` after service construction in
the spark-eval path. Result: provenance shows `reranker_enabled: true`.

**Test:** Unit test that the spark-eval path initializes the reranker; extend
`tests/unit/test_spark_eval_diagnostics.py`.

## Change 2 — Index SQL built-in function references (new `sql_function_ref` doc_type)

### Stream config (`config/spark_sources.json`)

```json
{
  "name": "sql_functions",
  "doc_type": "sql_function_ref",
  "language": "scala",
  "include": ["sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/**/*.scala"],
  "exclude": ["**/codegen/**", "**/test/**"],
  "chunking": "code"
}
```

- `_VALID_DOC_TYPES` in `config/settings.py:124` → add `"sql_function_ref"`.
- The stream `chunking` config field is validated but never read; `doc_type`
  alone selects the chunker. Value `code` chosen for schema consistency.

### Resolver content filter (`infrastructure/spark_source_resolver.py`)

- Add optional per-stream field `content_requires: list[str]`: a file is only
  included if its text contains every listed substring.
- `sql_functions` stream sets `content_requires: ["ExpressionDescription"]` so
  exactly the 66 annotated files index (67 non-annotated files under the glob
  are excluded).
- Verified counts: 133 non-codegen `.scala` under `catalyst/expressions/`;
  66 annotated (non-codegen); 0 annotated in codegen; all 66 annotated files
  contain a `case class` anchor.

### Chunker (`services/spark_chunker.py`)

Both entry points gain a `sql_function_ref` branch:
- `SparkChunker.chunk()` (async)
- `chunk_spark_document()` (sync)

**Annotation-aware splitter:**
1. Scan for `@ExpressionDescription(` blocks (multiline regex).
2. One chunk per annotation + the following `case class <Name>(...)` signature
   line.
3. Resolve SQL name(s) via a parsed `FunctionRegistry.scala` map:
   `expression[ArrayFilter]("filter")` → class `ArrayFilter` ↔ name `filter`.
   - Parser handles `expression[...]("name")`, `expressionBuilder[...]`,
     `expressionGeneratorOuter[...]`, `setAlias` aliases (`reduce` for
     `ArrayAggregate`), and `Some("3.4.0")` since-version args.
   - Verified: 426 `expression[` registrations, 27 `setAlias`, 16 with since.
4. Replace `_FUNC_` in annotation usage/examples with the resolved SQL name
   (canonical name; alias chunks get their alias name).
5. Keep annotation metadata (description / usage / examples, `group`, `since`,
   `deprecated`).
6. Annotation-free files → fall back to blank-line splitting (matches existing
   `_split_on_blank_lines` behavior).
7. Missing map entry → emit chunk with `_FUNC_` literal + metadata note (never
   drop content).

**`FunctionRegistry.scala` location:** resolved at build time from the pinned
tree (`data/spark_src/v4.0.0-b4ee7de0c9dc539a/`) or via the raw URL resolver.

### Eval row updates (`tests/evaluation/eval_dataset_spark.jsonl`, rows Q3–5)

- Replace `expected_urls` hub entry `docs/sql-ref-functions-builtin.md` with the
  `higherOrderFunctions.scala` raw URL
  (`https://raw.githubusercontent.com/apache/spark/{commit}/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/higherOrderFunctions.scala`).
- Update `expected_doc_types` (`api_reference` → `sql_function_ref`).

### Tests (Change 2)

- `tests/unit/test_spark_chunker.py`: annotation-aware splitting (chunk
  boundaries, `_FUNC_` replacement for filter/transform/aggregate, alias chunk
  `reduce`), fallback for annotation-free file.
- `tests/unit/test_spark_source_config.py`: new stream schema valid, content
  filter field accepted.
- `tests/unit/test_spark_source_resolver.py`: manifest includes exactly the 66
  annotated files with correct raw URLs + doc_type.
- `tests/unit/test_spark_eval_diagnostics.py`: Q3–5 rows now target the Scala
  source.

### Rebuild flow

`dec spark-build --generation <g>` → `dec spark-validate --generation <g>` →
`FORCE=1 dec spark-activate --generation <g>`.

**Rollout guard added (pre-build):** `_default_spark_generation()` previously
derived the generation name from `embedding` only, so the config change (new
`sql_functions` stream) would have produced the *same* generation identifier as
the then-active generation — a rebuild would have overwritten the live
collection in place. The name now hashes `{"embedding", "config"}` (canonical
`asdict`), so source-config changes yield a distinct generation automatically.
Covered by `test_default_generation_changes_when_config_changes` in
`tests/unit/test_cli_activation.py`.

## Change 3 — Widen rerank pool (Change 3a)

**Problem:** Q1 (`sql-ref-syntax-qry-select-window.md`) and Q6
(`python/pyspark/sql/column.py`) land in fused results at ranks 159/175 — past
the current pool cap 150 = `max(retrieval_top_k * 5, reranker_top_k * 5)` — so
the reranker never sees them.

**Fix:** `async_rag.py:462-465` → `max(retrieval_top_k * 8, reranker_top_k * 5)`
= 240.

**Test:** Unit test on the pool-size computation asserting 240 for the default
top-k.

## Verification & stop conditions

1. After Change 1, re-run cold `dec evaluate --spark --output-dir <d>` (clear
   `rag:cache:*` first) → provenance must show `reranker_enabled: true`.
2. After Change 3, Q1/Q6 expected URLs must appear within the pool (rank < 240).
3. After Change 2 + rebuild + activate, re-run eval → Q3–5 must retrieve
   `higherOrderFunctions.scala`; overall term/source/candidate scores should
   rise above the with-rerank baseline (term 0.860 / src 0.600 / cand 0.750).
4. **Stop and request human review if any eval rows beyond the pre-approved
   Q3–5 change must be modified.**
5. Full verification loop: `ruff` → `pyright` → unit tests (targeted first, then
   `make test-unit`).
6. Keep plan copies in `docs/` + `plans/` in sync; update `docs/cli_guide.md`
   if any `dec` help/output text changes.

## Baseline (for comparison)

| Variant | term | source | candidate | drops |
|---|---|---|---|---|
| Cold, no rerank (harness bug) | 0.668 | 0.500 | 0.750 | 3 |
| Cold, with rerank (cache cleared) | 0.860 | 0.600 | 0.750 | 2 |
