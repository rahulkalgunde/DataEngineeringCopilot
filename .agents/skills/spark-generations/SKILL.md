---
name: spark-generations
description: Use for ANY task involving the DataEngineeringCopilot immutable index generation lifecycle — gen/spark manifest, build, validate, activate, rollback, reset, stale classification, pinned source or stream config, or Qdrant generation collections and alias switching. Triggers: generation, gen-manifest, gen-build, gen-validate, gen-activate, gen-rollback, gen-reset, gen-stale, spark-build, spark-validate, spark-activate, spark-rollback, pinned_sources.json, spark_sources.json, .index_state, alias, deterministic indexing.
---

# DataEngineeringCopilot Spark / Pinned Generations

Immutable corpus builds are versioned as **generations**. Each generation maps
to its own Qdrant collection; the logical `active_collection_alias`
(`data_engineering_docs`) is atomically repointed on activation. Runtime state
lives in `.index_state/active.json` + `history.jsonl`.

## Lifecycle (both Spark-native and combined pinned builds)

1. **manifest** — `dec spark-manifest` / `dec gen-manifest`: materialize all
   pinned sources (Spark / Airflow / Delta / Claude) + write a manifest.
   `dec spark-config-check` / `dec gen-config-check` validate config first.
2. **build** — `dec spark-build --generation <gen>` /
   `dec gen-build`: crawl/fetch, parse (native + rendered + RST), chunk
   (header-aware / api / code chunkers), enrich, embed, upsert into collection
   `data_engineering_docs__<gen>`. Does NOT activate.
3. **validate** — `dec spark-validate --generation <gen>` /
   `dec gen-validate`: point count, hybrid integrity, term/source recall;
   writes a **validation report** to `.index_state/`.
4. **activate** — `dec spark-activate` / `dec gen-activate`: refuses unless a
   passing validation report exists, then atomically repoints the alias.
   `gen-rollback` / `spark-rollback` switch back to a previous generation.
5. **maintenance** — `dec gen-reset` (purge alias + gen collections + state +
   BM25 caches, then reset), `dec gen-stale` (classify active/stale/orphan
   generations), `dec gen-config-check`.

`spark-*` commands operate on the Spark-native generation; `gen-*` operate on
the combined pinned generation. They share the same collection/alias/state
machinery in `infrastructure/async_qdrant_store.py` and the index builders.

## Source / stream config

`data_engineering_copilot/config/spark_sources.json` (Spark-only) and
`pinned_sources.json` (combined) share the stream schema. A source is
`type: "github"` (repository/ref/commit + streams) or `type: "url_index"`
(index_url/url_prefix/cache_dir). Each **stream**:
`name`, `doc_type` (`guide|api_reference|code_example|sql_function_ref`),
`language` (`conceptual|mixed|scala|...`), `chunking`
(`header_aware|api|code`), `include`/`exclude` globs, optional
`content_requires`. `doc_type`/`chunking` values are strictly validated —
unknown values raise `ValueError`.

- API refs: `python/pyspark/**/*.py`; SQL funcs: catalyst
  `expressions/**/*.scala` + `content_requires=["ExpressionDescription"]`.
- Spark Python guides come from `docs/**/*.md|*.rst` **and**
  `python/docs/source/{tutorial,user_guide,getting_started,development,migration_guide}/**/*.rst|md`.
- RST guides: `GithubSourcePreparer._chunk_manifest` converts RST headings to
  Markdown (`_rst_to_markdown_headings`) so the header-aware chunker splits
  correctly — mirror this for any new `.rst` guide stream.

## Gotchas

- Activation is **atomic alias repoint**; never build into the alias
  collection directly. `gen-reset` is destructive — requires infra.
- Generations are immutable: fixes require a new generation
  (rebuild → validate → activate), never in-place edits.
- Cache/BM25 state is scoped by generation — after switching, stale caches
  must be cleared (`dec clear-cache`, `dec gen-reset` handles it).
- Validation is a hard gate: `spark-activate` refuses without a passing report.
- `gen-stale` identifies orphaned collections for cleanup; verify against
  `history.jsonl` before deleting.
