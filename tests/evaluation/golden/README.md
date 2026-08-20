# Golden Retrieval Evaluation Dataset

**Version:** 1.0  
**Created:** 2026-08-18  
**Total Queries:** 520 (500 in-scope + 20 out-of-scope)  
**Sources:** 5 (Apache Spark 4.0.0, Apache Airflow 3.3.1, Delta Lake v4.3.1, Claude Platform Docs, Claude Code Docs)

---

## Overview

This dataset provides a comprehensive golden set for evaluating retrieval quality in the DataEngineeringCopilot RAG system. Each query includes verified `expected_urls` that must be retrieved to answer the question correctly, enabling precise Recall@K and MRR measurement.

## Files

| File | Queries | Description |
|------|---------|-------------|
| `recall_spark.jsonl` | 100 | Apache Spark 4.0.0 (guides, API reference, code examples, SQL functions) |
| `recall_airflow.jsonl` | 100 | Apache Airflow 3.3.1 (core concepts, operators, scheduling, assets) |
| `recall_delta.jsonl` | 100 | Delta Lake v4.3.1 (batch, concurrency, optimizations, best practices) |
| `recall_claude_platform.jsonl` | 100 | Claude Platform API (Messages, prompt caching, tools, streaming) |
| `recall_claude_code.jsonl` | 100 | Claude Code (hooks, skills, MCP, subagents, permissions) |
| `recall_oos.jsonl` | 20 | Out-of-scope queries (should return no relevant results) |
| `recall_all.jsonl` | 500 | Merged in-scope queries (shuffled) for full benchmark |

---

## Intent Distribution

### Apache Spark 4.0.0 (100 queries)
| Intent | Count | Notes |
|--------|-------|-------|
| `api_lookup` | 25 | PySpark API signatures, parameters, class usage |
| `code_example` | 20 | Write jobs, implementations, patterns |
| `how_to` | 15 | Configuration, deployment, data sources |
| `factual` | 10 | Concepts, architecture, optimizers |
| `debugging` | 10 | OOM, exceptions, performance issues |
| `comparative` | 8 | YARN vs K8s, join strategies, APIs |
| `synthesis` | 7 | Multi-hop: config + API, streaming + SQL |
| `configuration` | 5 | Spark configs, tuning parameters |

### Apache Airflow 3.3.1 (100 queries)
| Intent | Count | Notes |
|--------|-------|-------|
| `how_to` | 25 | DAG creation, dynamic mapping, assets |
| `factual` | 20 | Core concepts, operators, XCom |
| `configuration` | 15 | Executors, schedulers, connections |
| `comparative` | 10 | Operator vs operator, executor types |
| `debugging` | 10 | Task failures, sensor timeouts |
| `synthesis` | 10 | Assets + scheduling, TaskFlow + XCom |
| `code_example` | 5 | DAG patterns, custom operators |
| `troubleshooting` | 5 | Scheduler stalls, duplicate runs |

### Delta Lake v4.3.1 (100 queries)
| Intent | Count | Notes |
|--------|-------|-------|
| `how_to` | 25 | Read/write, merge, vacuum, optimize, time travel |
| `factual` | 20 | ACID, MVCC, transaction log, Z-Order |
| `configuration` | 15 | Auto-optimize, CDC, retention, compaction |
| `debugging` | 10 | Concurrency conflicts, protocol errors |
| `synthesis` | 10 | Batch + streaming, Delta + Spark config |
| `comparative` | 10 | Delta vs Iceberg/Hudi, batch vs streaming |
| `troubleshooting` | 10 | Slow queries, storage growth, checkpoint issues |

### Claude Platform Docs (100 queries)
| Intent | Count | Notes |
|--------|-------|-------|
| `how_to` | 25 | API calls, streaming, tool use, prompt caching |
| `factual` | 20 | Models, constitutional AI, architecture |
| `configuration` | 15 | Rate limits, spend limits, compliance |
| `api_lookup` | 15 | Messages API, count_tokens, beta headers |
| `debugging` | 10 | Rate limits, auth errors, overloaded errors |
| `synthesis` | 10 | Prompt caching + tools, streaming + structured |
| `troubleshooting` | 5 | Cost overruns, latency, context overflow |

### Claude Code Docs (100 queries)
| Intent | Count | Notes |
|--------|-------|-------|
| `how_to` | 25 | Hooks, skills, MCP, subagents, permissions |
| `factual` | 20 | Architecture, LLM gateway, data usage |
| `configuration` | 15 | Settings, env vars, allowed tools |
| `troubleshooting` | 10 | Hook failures, skill loading, MCP disconnects |
| `debugging` | 10 | Permission denied, timeouts, context limits |
| `synthesis` | 10 | Hooks + skills, MCP + subagents |
| `code_example` | 10 | Hook scripts, SKILL.md, MCP servers |

---

## Schema

Each row in `recall_*.jsonl` follows the [eval_schema.py](../eval_schema.py) recall format:

```json
{
  "id": "spark-api_lookup-001",
  "question": "What is the signature of pyspark.sql.functions.col?",
  "expected_urls": [
    "https://raw.githubusercontent.com/apache/spark/.../builtin.py"
  ],
  "expected_terms": ["col", "pyspark.sql.functions", "Column"],
  "expected_doc_types": ["api_reference"],
  "expected_modules": ["pyspark.sql.functions"],
  "must_not_require": [],
  "forbidden_terms": [],
  "source_name": "Apache Spark 4.0.0",
  "doc_type": "api_reference",
  "intent": "api_lookup",
  "complexity": "single_hop",
  "abstraction": "specific"
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Stable slug identifier (lowercase, hyphens) |
| `question` | string | Yes | Natural language query |
| `expected_urls` | string[] | Yes* | URLs that MUST be retrieved to answer correctly |
| `expected_terms` | string[] | Yes | Key terms that should appear in retrieved chunks |
| `expected_doc_types` | string[] | No | Expected doc types: guide, api_reference, code_example, sql_function_ref |
| `expected_modules` | string[] | No | Expected module paths (e.g., pyspark.sql.functions) |
| `must_not_require` | string[] | No | Terms that should NOT be required in answer |
| `forbidden_terms` | string[] | No | Terms that indicate wrong answer |
| `source_name` | string | Yes | Source display name |
| `doc_type` | string | No | Primary doc type of expected content |
| `intent` | string | Yes | Query intent (see distribution above) |
| `complexity` | string | No | single_hop or multi_hop |
| `abstraction` | string | No | specific or abstract |
| `out_of_scope` | bool | No | If true, query is out-of-domain (only in recall_oos.jsonl) |

*For `out_of_scope=true` rows, `expected_urls` is empty.

---

## Corpus Alignment

This dataset is aligned with the pinned corpus generations:

| Source | Pinned Commit | Corpus Path |
|--------|---------------|-------------|
| Apache Spark 4.0.0 | `fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4` | `data/spark_corpus/spark-v4.0.0-fa33ea00-hybrid-*/` |
| Apache Airflow 3.3.1 | `3adbbe1c58e4532df1964cb7794805e763816ee8` | `data/pinned_corpus/pinned-*/` |
| Delta Lake v4.3.1 | `54ce02692567fc3e5107a3ae69ebeccdffa7edfe` | `data/pinned_corpus/pinned-*/` |
| Claude Platform Docs | `c2c813e171cb8d8c5f76bf1034aaf94304c267c8` | `data/pinned_corpus/pinned-*/` |
| Claude Code Docs | `339fdc31b01104c2b6357603464d26fc3ff77a03` | `data/pinned_corpus/pinned-*/` |

All `expected_urls` are verified to exist in the corresponding corpus generation.

---

## Usage

### Run Retrieval Benchmark
```bash
# Full benchmark (500 queries)
python -m data_engineering_copilot.evaluation.rag_optimization_benchmark \
  --dataset tests/evaluation/golden/recall_all.jsonl \
  --output benchmark_report.json

# Per-source benchmark
python -m data_engineering_copilot.evaluation.rag_optimization_benchmark \
  --dataset tests/evaluation/golden/recall_spark.jsonl \
  --output spark_benchmark.json
```

### Run Coverage Validation
```bash
# Verify all expected_urls exist in active generation
dec eval-coverage --generation spark-v4.0.0-fa33ea00-hybrid-20260808
```

### Run Fast Integrity Check
```bash
# Zero-LLM retrieval recall check
dec eval-fast --generation spark-v4.0.0-fa33ea00-hybrid-20260808
```

---

## Regeneration

To regenerate or expand the dataset:

```bash
# 1. Run synthetic generator (LLM-assisted)
python -m data_engineering_copilot.evaluation.synthetic_generator \
  --source spark \
  --count 50 \
  --output tests/evaluation/golden/recall_spark_new.jsonl

# 2. Review generated queries manually
# 3. Merge with existing (dedupe by id)
# 4. Re-trim to 100 per source
python scripts/trim_and_create_oos_v2.py

# 5. Re-validate coverage
dec eval-coverage
```

---

## Quality Gates

The [rag_optimization_benchmark.py](../rag_optimization_benchmark.py) enforces these gates when comparing against baseline:

| Metric | Gate |
|--------|------|
| Source Recall (global) | Δ ≥ -0.01 (no regression) |
| MRR (global) | Δ ≥ -0.02 (no regression) |
| Identifier Recall (api_lookup, code_example, debugging) | Δ ≥ +0.05 (improvement) |
| Generic Recall (factual, how_to) | Δ ≥ -0.01 (no regression) |
| Provider Calls | Δ ≤ -20% (reduction) |
| Duplicate Rate | Δ ≤ -10% (reduction) |

---

## Maintenance

- **Review quarterly**: Check for corpus drift (URLs moved, content changed)
- **Add new intents**: When new query patterns emerge in production
- **Expand OOS**: Add 2-4 new out-of-scope domains per quarter
- **Validate before release**: Run `dec eval-coverage` on every new generation

---

## Related Files

- `../eval_schema.py` — Schema validation
- `../eval_coverage.py` — Corpus coverage validation
- `../rag_optimization_benchmark.py` — Full retrieval benchmark
- `../fast_eval.py` — Zero-LLM integrity check (Layer 5: retrieval)
- `../../services/rag_evaluation.py` — RetrievalEvaluator (precision, recall, MRR)