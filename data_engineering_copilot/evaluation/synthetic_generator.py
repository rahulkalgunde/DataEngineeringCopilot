"""Synthetic recall-eval generation from the active-generation corpus.

Two candidate paths:

- ``deterministic`` (default, offline): builds recall rows straight from the
  corpus chunks (headings / section titles). Corpus-grounded by construction.
- ``ragas`` (optional): Ragas ``TestsetGenerator`` over langchain documents
  (needs ``ragas`` + an LLM + embeddings). Gated behind an explicit flag.

Every candidate is filtered through :class:`CoverageValidator` before it can
be written, so synthetic rows can never target content that is not indexed.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "from",
    "at",
    "by",
    "as",
    "is",
    "are",
    "this",
    "that",
    "how",
    "what",
    "when",
    "where",
    "which",
    "who",
    "does",
    "do",
    "you",
    "your",
    "use",
    "using",
    "used",
    "set",
    "get",
    "make",
}


def chunks_for_source(generation_root: Path, source: str) -> list[dict]:
    """Load all chunks belonging to ``source`` from a generation corpus."""
    rows: list[dict] = []
    path = generation_root / "chunks.jsonl"
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (d.get("source_name") or "") == source:
            rows.append(d)
    return rows


def _heading_terms(chunk: dict, limit: int = 3) -> list[str]:
    heading = " ".join(chunk.get("heading_path") or []) or (chunk.get("title") or "")
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{2,}", heading) if w.lower() not in _STOPWORDS]
    seen: list[str] = []
    for w in words:
        if w.lower() not in {s.lower() for s in seen}:
            seen.append(w)
        if len(seen) >= limit:
            break
    return seen


def deterministic_candidates(chunks: list[dict], *, source: str, limit: int = 50) -> list[dict]:
    """Build recall rows from chunk headings — corpus-grounded by construction."""
    rows: list[dict] = []
    seen: set[str] = set()
    for chunk in chunks:
        url = (chunk.get("url") or "").strip().rstrip("/")
        if not url:
            continue
        title = chunk.get("title") or ""
        section = " ".join(chunk.get("heading_path") or [])
        heading = section or title
        if not heading or len(heading) < 8:
            continue
        terms = _heading_terms(chunk)
        text_lower = (chunk.get("text") or "").lower()
        terms = [t for t in terms if t.lower() in text_lower]
        if len(terms) < 2:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")[:40].strip("-")
        row_id = f"synthetic-{slug or 'row'}"
        if row_id in seen:
            row_id = f"{row_id}-{len(rows)}"
        seen.add(row_id)
        doc_type = chunk.get("doc_type") or "guide"
        rows.append(
            {
                "id": row_id,
                "question": f"What does '{heading}' cover in {source}?",
                "expected_urls": [url],
                "expected_terms": terms,
                "expected_doc_types": [doc_type],
                "expected_modules": [],
                "forbidden_terms": [],
                "source_name": source,
                "doc_type": doc_type,
                "intent": "factual",
                "complexity": "single_hop",
                "abstraction": "specific",
            }
        )
        if len(rows) >= limit:
            break
    return rows


def ragas_candidates(
    chunks: list[dict],
    *,
    source: str,
    llm,
    embeddings,
    testset_size: int = 25,
) -> list[dict]:
    """Generate candidates via Ragas ``TestsetGenerator`` (best-effort).

    Returns ``[]`` if ragas or its testset API is unavailable. The output
    mapping is version-sensitive; rows that cannot be mapped are dropped.
    """
    try:
        from data_engineering_copilot.services.ragas_evaluation import _install_vertexai_shim

        _install_vertexai_shim()
        from ragas.testset import TestsetGenerator  # type: ignore[import-not-found]  # optional

        generator = TestsetGenerator.from_langchain(llm=llm, embedding_model=embeddings)
        from langchain_core.documents import Document

        document_type = Document
    except (ImportError, AttributeError, TypeError) as exc:
        logger.info("ragas testset generation unavailable: %s", exc)
        return []

    docs = []
    for chunk in chunks:
        url = (chunk.get("url") or "").strip().rstrip("/")
        if not url:
            continue

        docs.append(
            document_type(
                page_content=(chunk.get("text") or "")[:4000],
                metadata={"source": url, "title": chunk.get("title") or ""},
            )
        )

    try:
        testset = generator.generate_with_langchain_docs(
            documents=docs[:200], testset_size=testset_size, raise_exceptions=False
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("ragas testset generation failed: %s", exc)
        return []

    rows: list[dict] = []
    samples = getattr(testset, "samples", testset if isinstance(testset, list) else [])
    for i, sample in enumerate(samples):
        user_input = getattr(sample, "user_input", None) or getattr(sample, "question", "")
        if not user_input:
            continue
        ref_contexts = getattr(sample, "reference_contexts", None) or []
        urls = [
            (getattr(ctx, "metadata", {}) or {}).get("source", "")
            for ctx in ref_contexts
            if isinstance(ctx, document_type)
        ]
        urls = [u for u in urls if u]
        if not urls:
            continue
        terms = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{2,}", str(user_input)) if w.lower() not in _STOPWORDS][
            :4
        ]
        rows.append(
            {
                "id": f"synthetic-ragas-{source[:12]}-{i:03d}",
                "question": str(user_input),
                "expected_urls": urls[:3],
                "expected_terms": terms,
                "expected_doc_types": ["guide"],
                "expected_modules": [],
                "forbidden_terms": [],
                "source_name": source,
                "doc_type": "guide",
                "intent": "factual",
                "complexity": "multi_hop",
                "abstraction": "specific",
            }
        )
    return rows


def gate_and_write(candidates: list[dict], path: Path, validator) -> int:
    """Filter candidates through the coverage validator and write survivors.

    Returns the number of rows written.
    """
    kept: list[dict] = []
    for row in candidates:
        verdict = validator.validate_row(row)
        if verdict["status"] != "pass":
            continue
        kept.append(row)
    with open(path, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(kept)


def generate(
    generation_root: Path,
    source: str,
    out_path: Path,
    *,
    limit: int = 50,
    ragas_llm=None,
    ragas_embeddings=None,
    testset_size: int = 25,
    validator=None,
) -> int:
    """Generate + gate a synthetic recall set for one source. Returns rows written."""
    from data_engineering_copilot.services.eval_coverage import CoverageValidator

    chunks = chunks_for_source(generation_root, source)
    if not chunks:
        logger.warning("no chunks for source %r in %s", source, generation_root)
        return 0

    validator = validator or CoverageValidator(generation_root)
    candidates: list[dict] = []
    if ragas_llm is not None and ragas_embeddings is not None:
        candidates = ragas_candidates(
            chunks, source=source, llm=ragas_llm, embeddings=ragas_embeddings, testset_size=testset_size
        )
    if not candidates:
        candidates = deterministic_candidates(chunks, source=source, limit=limit)
    return gate_and_write(candidates, out_path, validator)
