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
from data_engineering_copilot.evaluation.chunking_quality import audit_chunks, gate_verdict
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
        from data_engineering_copilot.config.settings import AppSettings

        settings = AppSettings(skip_provider_check=True)
        return HeaderAwareChunker(
            chunk_size_words=settings.chunk_size_words,
            overlap_words=settings.chunk_overlap_words,
            min_chunk_words=int(settings.chunk_size_words * 0.1),
        )
    if strategy == "structured":
        return StructuredDataChunker(max_tokens=3800, max_chars=6000)
    raise ValueError(f"Unsupported strategy for chunking eval: {strategy!r}")


# Flaw-pattern #7 gate: fraction of headers/code-fences torn by a chunk
# boundary. Splitting prose mid-paragraph is normal; tearing a header line or
# a code fence detaches context (table from caption, code from its prose).
# Measured baseline 2026-08-23: all strategies score 0.0 on the synthetic
# corpus; 0.25 catches the onset of structural tearing while leaving room
# for a rare legitimate boundary landing inside a one-line node.
FRACTURE_GATE_THRESHOLD = 0.25


BUILTIN_STRATEGIES = ["recursive", "sentence", "header", "structured"]


# Corpus adequacy gate (local, hermetic, no retrieval/LLM).
# Grounded in the measured 2026-09-06 baseline (fence 0.0586, tiny 0.0170,
# oversized 0.0, median 91 words) amended by the post-re-cert measurements of
# both generations (old d3dbad402105: fence 0.0586 / tiny 0.0170 / sentence
# 0.311; new cd208afaf0f8: fence 0.1187 / tiny 0.0441 / sentence 0.2681).
# Thresholds are regression tripwires, not aspirational targets: each sits at
# or just above the best level a known-good chunker achieves today, so a
# reversion trips the gate. ADR-018 records the amendment (2026-09-06).
#
#   fence_fracture 0.08  - re-tightened 2026-09-07 (ADR-018 amendment 3). The
#                          nested-fence pairing rework (header_aware_chunker
#                          fence-regex indent `\s{0,3}` -> `[ \t]{0,3}`) fixed
#                          4-tick fences that were "paired" to a distant
#                          closer across a blank line, exposing fence-body
#                          headings as fake sections. Real-build re-cert on the
#                          rebuilt generation (79,760 chunks, same pinned
#                          sources): 0.0379 — down from the 73,017-chunk build
#                          floor 0.1108. Tripwire sits at 2x the measured floor
#                          (0.038) with headroom for the irreducible classes
#                          (genuinely-oversize fences > 3800 tok / ~14k chars,
#                          CP CodeGroup/Tab nested fences).
#   tiny_rate      0.02  - old-gen level 0.0170; rebuilt generation 0.0093
#   sentence       0.30  - best measured 0.2870 (rebuilt gen), above old 0.3110
#   table/oversized      - unchanged, both generations already pass
#
# The active generation (cd208afaf0f8) turned the existing gates green on
# 2026-09-06; fence was re-calibrated from the projected 0.10 to the measured
# 0.1108 (tripwire 0.12), then tightened to 0.08 after the fence-pairing fix
# re-cert measured 0.0379 (2026-09-07).
CORPUS_GATES = {
    "fence_fracture": 0.08,
    "tiny_rate": 0.02,
    "oversized_rate": 0.001,
    "table_fracture": 0.01,
    "sentence_fracture": 0.30,
}


def run_corpus_quality_gate(chunks_path: str, output_path: str) -> dict:
    """Stream *chunks_path* (a chunks.jsonl) and assert the corpus gate.

    Returns the audit report with an added ``gates`` verdict. Exit code
    responsibility is the caller's (CLI returns 1 when ``gates.pass`` is False).
    """
    rows = []
    with open(chunks_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    report = audit_chunks(rows)
    verdict = gate_verdict(
        report,
        fence_fracture=CORPUS_GATES["fence_fracture"],
        tiny_rate=CORPUS_GATES["tiny_rate"],
        oversized_rate=CORPUS_GATES["oversized_rate"],
        table_fracture=CORPUS_GATES["table_fracture"],
        sentence_fracture=CORPUS_GATES["sentence_fracture"],
    )
    report["gates"] = verdict
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(report, fh, indent=2)
    return report


# Repo-anchored (M13): eval gold lives under tests/evaluation/golden/chunking
# relative to THIS module, never to the caller's CWD.
_GOLD_DIR = Path(__file__).resolve().parents[2] / "tests" / "evaluation" / "golden" / "chunking"
_GOLD_FILES = {
    "synthetic": "synthetic_gold.jsonl",
    "human": "human_slice.jsonl",
    "corpus_slice": "corpus_slice.jsonl",
}


def _load_gold(gold_source: str) -> list[dict]:
    files = []
    if gold_source in ("synthetic", "all"):
        files.append(_GOLD_DIR / _GOLD_FILES["synthetic"])
    if gold_source in ("human", "all"):
        files.append(_GOLD_DIR / _GOLD_FILES["human"])
    if gold_source in ("corpus_slice", "all"):
        files.append(_GOLD_DIR / _GOLD_FILES["corpus_slice"])
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

    # Flaw-pattern #7 verdict: machine-readable so CI can gate without parsing prose.
    worst_fracture = max(m["fracture_rate"] for m in report.values())
    report["gates"] = {
        "fracture_ok": worst_fracture <= FRACTURE_GATE_THRESHOLD,
        "fracture_threshold": FRACTURE_GATE_THRESHOLD,
        "worst_fracture_rate": worst_fracture,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(report, fh, indent=2)

    return report
