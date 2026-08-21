# 014: GraphRAG and multi-hop decomposition — right-sized, not Microsoft-GraphRAG

Date: 2026-08-21 · Status: Accepted

## Context

Multi-hop questions ("how does X relate to Y across modes?") and entity-
neighborhood context were unsupported by pure vector retrieval. Microsoft
GraphRAG (arXiv:2404.16130) targets global sensemaking via community
summaries over LLM-derived entity graphs — heavy to build and maintain.

## Decision

Ship a right-sized pair:

1. **GraphRAG-lite**: ingestion-side LLM triplet extraction
   (`services/graph_extractor.py`) into a SQLite graph
   (`infrastructure/graph_store.py`); query-side entity extraction fetches
   1-hop neighbor triplets (`services/graph_traversal.py`) prepended as a
   bullet-list context block. No community summaries, no Leiden clustering —
   our workload is point-lookup QA over documentation, where the GraphRAG
   paper itself says vector retrieval remains the right default.
2. **Multi-hop decomposition**: `services/multi_hop_decomposer.py` plans a
   dependency-aware `QueryPlan` (steps + `depends_on`), executes steps
   sequentially with dependency-refined queries, and emits a
   "Multi-hop reasoning context:" block.

Both components are optional collaborators (absence = no-op) and fail open.

## Consequences

- Known limitations accepted: 1-hop traversal only (`depth` ignored), single
  `"concept"` node type, synchronous SQLite commits on the async path.
- Global/thematic corpus questions remain out of scope; if they become a
  requirement, adopt community summaries in a future ADR rather than growing
  this one.
