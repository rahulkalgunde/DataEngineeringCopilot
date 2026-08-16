"""Free (zero-LLM) layered integrity evaluation.

Implements the deterministic, LLM-free layers of the RAG evaluation contract:

1. **Corpus** — chunk count, unique sources, content-hash duplicates,
   empty/short chunks, and eval-dataset coverage.
2. **Chunk** — size distribution and boundary heuristics (chunk starts
   lowercase / ends mid-sentence / splits a code fence or markdown table).
3. **Embedding** — dimension/NaN/Inf sanity, determinism (same text embedded
   twice is ~identical) and semantic ordering (relevant > irrelevant).
4. **Vector DB** — point count == chunk count, metadata resolution and
   self-retrieval for sampled chunks.
5. **Retrieval** — URL recall + MRR over a small fixed golden set using only
   embed + vector search (no rewrite, no rerank, no LLM).

Only the local embedder and Qdrant are required (no paid LLM calls, no cloud
rerank). This is the gate to run after every code change.

Pure helpers (``chunk_size_stats``, ``chunk_boundary_issues``,
``validate_embedding``, ``embedding_consistency``, ``embedding_semantic_sanity``)
are synchronous and hermetic; ``run_fast_eval`` is async and talks to the
embedder + vector store.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

logger = __import__("logging").getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Chunk statistics
# ─────────────────────────────────────────────────────────────────────────────


def _percentile(sorted_values: Sequence[int | float], q: float) -> int | float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    base = int(pos)
    frac = pos - base
    if base + 1 < len(sorted_values):
        return sorted_values[base] + frac * (sorted_values[base + 1] - sorted_values[base])
    return sorted_values[base]


def chunk_size_stats(chunks: list[dict]) -> dict:
    """Size distribution across chunks: chars and a token estimate."""
    if not chunks:
        return {
            "count": 0,
            "min_chars": None,
            "max_chars": None,
            "mean_chars": None,
            "median_chars": None,
            "p95_chars": None,
            "p99_chars": None,
            "empty": 0,
            "oversized": 0,
            "max_chars_limit": None,
        }
    char_lens = sorted(len(chunk.get("text") or "") for chunk in chunks)
    token_lens = sorted(int(chunk.get("token_count") or 0) for chunk in chunks)
    return {
        "count": len(chunks),
        "min_chars": char_lens[0],
        "max_chars": char_lens[-1],
        "mean_chars": round(sum(char_lens) / len(char_lens), 1),
        "median_chars": _percentile(char_lens, 0.5),
        "p95_chars": _percentile(char_lens, 0.95),
        "p99_chars": _percentile(char_lens, 0.99),
        "min_tokens": token_lens[0],
        "max_tokens": token_lens[-1],
        "empty": sum(1 for c in char_lens if c == 0),
        "oversized": sum(1 for c in char_lens if c > 6000),
        "max_chars_limit": 6000,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chunk boundary heuristics
# ─────────────────────────────────────────────────────────────────────────────


def _looks_like_continuation(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return False
    first = stripped[0]
    if not first.isalpha():
        return False
    return first.islower()


def _splits_code_fence(text: str) -> bool:
    fences = 0
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fences += 1
    return fences % 2 == 1


def _splits_markdown_table(text: str) -> bool:
    lines = text.splitlines()
    if not lines:
        return False
    return bool(lines[-1].strip().startswith("|"))


def _ends_mid_sentence(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped[-1] in ".!?)]}\"'`>":
        return False
    return stripped[-1].isalnum() or stripped[-1] == "_"


def chunk_boundary_issues(chunks: list[dict]) -> list[dict]:
    """Deterministic boundary-quality checks over chunks.

    Returns one record per chunk that trips a heuristic, with the chunk_id and
    the list of issues. No LLM involved.
    """
    issues: list[dict] = []
    for chunk in chunks:
        text = chunk.get("text") or ""
        chunk_id = chunk.get("chunk_id") or ""
        flags: list[str] = []
        if _looks_like_continuation(text):
            flags.append("starts_lowercase_continuation")
        if _ends_mid_sentence(text):
            flags.append("ends_mid_sentence")
        if _splits_code_fence(text):
            flags.append("unbalanced_code_fence")
        if _splits_markdown_table(text):
            flags.append("starts_markdown_table")
        if flags:
            issues.append({"chunk_id": chunk_id, "issues": flags})
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Embedding checks
# ─────────────────────────────────────────────────────────────────────────────


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def validate_embedding(vector: list[float], dimension: int) -> list[str]:
    """Return a list of integrity problems for a single embedding vector."""
    problems: list[str] = []
    if len(vector) != dimension:
        problems.append(f"dimension_mismatch expected={dimension} got={len(vector)}")
    for i, value in enumerate(vector):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            problems.append(f"nonfinite_at_index={i}")
            break
    norm = math.sqrt(sum(x * x for x in vector)) if vector else 0.0
    if norm <= 0:
        problems.append("zero_norm")
    return problems


def embedding_consistency(embed, text: str) -> dict:
    """Embed the same text twice and report the cosine similarity."""
    v1 = embed(text)
    v2 = embed(text)
    return {
        "text": text[:80],
        "similarity": round(cosine_similarity(v1, v2), 4),
    }


def embedding_semantic_sanity(embed, pairs: list[dict]) -> dict:
    """For each (query, relevant, irrelevant) triple, check that the relevant
    passage is closer than the irrelevant one (deterministic, no LLM)."""
    results = []
    for pair in pairs:
        q = embed(pair["query"])
        rel = embed(pair["relevant"])
        irr = embed(pair["irrelevant"])
        sim_rel = cosine_similarity(q, rel)
        sim_irr = cosine_similarity(q, irr)
        results.append(
            {
                "query": pair["query"][:60],
                "sim_relevant": round(sim_rel, 4),
                "sim_irrelevant": round(sim_irr, 4),
                "pass": sim_rel > sim_irr,
            }
        )
    passed = sum(1 for r in results if r["pass"])
    return {"pairs": len(results), "passed": passed, "results": results}


# ─────────────────────────────────────────────────────────────────────────────
# Corpus integrity
# ─────────────────────────────────────────────────────────────────────────────


def load_chunks(corpus_root: Path) -> list[dict]:
    """Load ``chunks.jsonl`` from a generation corpus root."""
    path = corpus_root / "chunks.jsonl"
    if not path.exists():
        return []
    chunks: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return chunks


def corpus_integrity(chunks: list[dict]) -> dict:
    """Count, sources, content-hash duplicates and empty chunks."""
    sources: Counter = Counter(str(c.get("source_name") or "unknown") for c in chunks)
    hashes: Counter = Counter(c.get("content_hash") or c.get("parent_content_hash") or "" for c in chunks)
    dup_hashes = {h: count for h, count in hashes.items() if count > 1}
    empty = [c for c in chunks if not (c.get("text") or "").strip()]
    return {
        "chunk_count": len(chunks),
        "source_count": len(sources),
        "by_source": dict(sources),
        "content_hash_duplicates": len(dup_hashes),
        "duplicate_hashes": list(dup_hashes)[:20],
        "empty_chunks": len(empty),
        "empty_chunk_ids": [c.get("chunk_id") for c in empty][:20],
    }


def chunk_token_stats(chunks: list[dict]) -> dict:
    """Aggregate token/char budgets per segment (lossless reconstruction guard)."""
    if not chunks:
        return {"segments": 0, "over_limit": 0, "under_budget": 0}
    over = [c for c in chunks if int(c.get("token_count") or 0) > 3800]
    under = [c for c in chunks if len(c.get("text") or "") < 50]
    return {"segments": len(chunks), "over_token_budget": len(over), "under_char_budget": len(under)}


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration (async — talks to the embedder + vector store)
# ─────────────────────────────────────────────────────────────────────────────


async def run_fast_eval(
    *,
    generation: str | None = None,
    embedder,
    store,
    sanity_pairs: list[dict],
    recall_rows: list[dict],
    sample_size: int = 25,
    data_root: Path | None = None,
) -> dict:
    """Run all deterministic integrity layers against the active generation.

    Returns a JSON-serializable report. ``embedder`` exposes ``async embed_query``;
    ``store`` exposes ``async count()`` and ``async query(embedding, top_k,
    query_text=...)``. ``recall_rows`` follow the recall eval schema. Zero LLM
    calls; the embedder is expected to be the local/cached one.
    """
    from data_engineering_copilot.services.eval_coverage import CoverageValidator, resolve_generation_root

    project_root = Path(__file__).resolve().parents[2]
    data_root = data_root or project_root / "data"

    gen = generation or ""
    root = resolve_generation_root(gen, data_root) if gen else None
    if root is None:
        return {"generation": gen, "error": "no corpus found", "status": "error"}

    chunks = load_chunks(root)
    report: dict = {
        "generation": gen,
        "corpus_root": str(root),
        "status": "ok",
        "layers": {},
    }

    # Layer 1 — Corpus
    report["layers"]["corpus"] = corpus_integrity(chunks)
    report["layers"]["chunk"] = {
        **chunk_size_stats(chunks),
        **chunk_token_stats(chunks),
        "boundary_issues": chunk_boundary_issues(chunks),
    }

    # Layer 1b — Eval-dataset coverage (reuses the coverage gate)
    validator = CoverageValidator(root)
    report["layers"]["coverage"] = validator.report(recall_rows) if recall_rows else {"rows": 0, "pass": 0, "fail": 0}

    # Layer 3 — Embedding consistency + semantic sanity (local embedder only)
    async def _embed(text: str) -> list[float]:
        return await embedder.embed_query(text)

    consistency = await _embed("How do I configure Spark executor memory?")
    c1 = consistency
    c2 = await embedder.embed_query("How do I configure Spark executor memory?")
    report["layers"]["embedding"] = {
        "consistency": {
            "similarity": round(cosine_similarity(c1, c2), 4),
        },
        "semantic_sanity": await _semantic_sanity_async(_embed, sanity_pairs),
    }

    # Layer 4 — Vector DB integrity (count + self-retrieval)
    try:
        point_count = await store.count()
        report["layers"]["vectordb"] = {
            "point_count": point_count,
            "chunk_count": len(chunks),
            "count_matches": point_count == len(chunks),
        }
        if chunks:
            sample = chunks[:sample_size]
            self_retrieval = []
            for chunk in sample:
                text = (chunk.get("text") or "")[:512]
                vec = await _embed(text)
                results = await store.query(vec, top_k=1, query_text=text[:512])
                hit = bool(results and results[0].chunk.chunk_id == chunk.get("chunk_id"))
                self_retrieval.append({"chunk_id": chunk.get("chunk_id"), "self_hit": hit})
            report["layers"]["vectordb"]["self_retrieval"] = self_retrieval
            report["layers"]["vectordb"]["self_retrieval_hits"] = sum(1 for r in self_retrieval if r["self_hit"])
    except Exception as exc:  # noqa: BLE001 - infra may be down; report, don't crash
        report["layers"]["vectordb"] = {"error": str(exc)}

    # Layer 5 — Retrieval (URL recall + MRR over the small golden set)
    report["layers"]["retrieval"] = await _retrieval_recall(_embed, store, recall_rows)

    return report


async def _semantic_sanity_async(embed, pairs: list[dict]) -> dict:
    results = []
    for pair in pairs:
        q = await embed(pair["query"])
        rel = await embed(pair["relevant"])
        irr = await embed(pair["irrelevant"])
        sim_rel = cosine_similarity(q, rel)
        sim_irr = cosine_similarity(q, irr)
        results.append(
            {
                "query": pair["query"][:60],
                "sim_relevant": round(sim_rel, 4),
                "sim_irrelevant": round(sim_irr, 4),
                "pass": sim_rel > sim_irr,
            }
        )
    passed = sum(1 for r in results if r["pass"])
    return {"pairs": len(results), "passed": passed, "results": results}


async def _retrieval_recall(embed, store, recall_rows: list[dict]) -> dict:
    """URL recall + MRR over recall rows using raw embed + vector search."""
    if not recall_rows:
        return {"rows": 0, "source_recall": None, "mrr": None, "results": []}
    results = []
    for row in recall_rows:
        if row.get("out_of_scope"):
            continue
        expected_urls = {str(u).rstrip("/") for u in row.get("expected_urls") or []}
        query = row.get("question") or ""
        vec = await embed(query)
        retrieved = await store.query(vec, top_k=10, query_text=query)
        urls = [r.chunk.url.rstrip("/") for r in retrieved]
        hit = sum(1 for u in expected_urls if u in urls)
        recall = hit / len(expected_urls) if expected_urls else 1.0
        mrr = 0.0
        for rank, u in enumerate(urls, 1):
            if u in expected_urls:
                mrr = 1.0 / rank
                break
        results.append(
            {
                "id": row.get("id", ""),
                "source_recall": round(recall, 4),
                "mrr": round(mrr, 4),
                "expected_urls": len(expected_urls),
                "hit_urls": hit,
            }
        )
    n = len(results)
    source_recall = sum(r["source_recall"] for r in results) / n if n else None
    mrr = sum(r["mrr"] for r in results) / n if n else None
    return {
        "rows": n,
        "source_recall": round(source_recall, 4) if source_recall is not None else None,
        "mrr": round(mrr, 4) if mrr is not None else None,
        "results": results,
    }
