"""Offline chunking quality evaluation.

Loads the committed gold dataset, runs the requested chunkers, and reports
token-level IoU, excerpt precision, boundary similarity, and structural
fracture rate. No retrieval, reranking, or LLM calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path

from data_engineering_copilot.domain.models import ParsedDocument
from data_engineering_copilot.evaluation.chunking_metrics import (
    boundary_similarity,
    excerpt_precision,
    extract_markdown_structure,
    structural_fracture_rate,
    token_iou,
)
from data_engineering_copilot.services.chunker import DocumentChunker
from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker
from data_engineering_copilot.services.sentence_preserving_chunker import SentencePreservingChunker
from data_engineering_copilot.services.structured_data_chunker import StructuredDataChunker


def _build_chunker(strategy: str):
    if strategy == "recursive":
        return DocumentChunker(chunk_size_chars=1000, chunk_overlap_chars=100)
    if strategy == "sentence":
        return SentencePreservingChunker(max_tokens=3800, max_chars=6000)
    if strategy == "header":
        return HeaderAwareChunker(chunk_size_words=375, overlap_words=90, min_chunk_words=10)
    if strategy == "structured":
        return StructuredDataChunker(max_tokens=3800, max_chars=6000)
    raise ValueError(f"Unsupported strategy for chunking eval: {strategy!r}")


BUILTIN_STRATEGIES = ["recursive", "sentence", "header", "structured"]


def _load_gold(gold_source: str) -> list[dict]:
    base = Path("tests/evaluation/golden/chunking")
    files = []
    if gold_source in ("synthetic", "all"):
        files.append(base / "synthetic_gold.jsonl")
    if gold_source in ("human", "all"):
        files.append(base / "human_slice.jsonl")
    docs = []
    for path in files:
        if not path.exists():
            continue
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                docs.append(json.loads(line))
    return docs


def run_chunking_eval(strategy: str, gold_source: str, output_path: str) -> dict:
    gold_docs = _load_gold(gold_source)
    strategies = BUILTIN_STRATEGIES if strategy == "all" else [strategy]

    report: dict[str, dict] = {}
    for strat in strategies:
        chunker = _build_chunker(strat)
        ious: list[float] = []
        precs: list[float] = []
        bsims: list[float] = []
        fractures: list[float] = []
        for item in gold_docs:
            doc = ParsedDocument(source_name="eval", title="t", url="http://x", text=item["text"])
            chunks = chunker._sync_chunk(doc)
            ious.append(token_iou(item["text"], item["gold_spans"], chunks))
            precs.append(excerpt_precision(item["text"], item["gold_spans"], chunks))
            bsims.append(boundary_similarity(item["gold_spans"], chunks, len(item["text"])))
            nodes = extract_markdown_structure(item["text"])
            fractures.append(structural_fracture_rate(chunks, nodes))
        report[strat] = {
            "iou": sum(ious) / len(ious) if ious else 0.0,
            "precision": sum(precs) / len(precs) if precs else 0.0,
            "boundary_similarity": sum(bsims) / len(bsims) if bsims else 0.0,
            "fracture_rate": sum(fractures) / len(fractures) if fractures else 0.0,
            "doc_count": len(gold_docs),
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(report, fh, indent=2)

    return report
