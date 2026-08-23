"""Flaw-pattern #7 guards: context fragmentation (docs/rag_flaw_prevention_plan.md).

Three facets, each pinned so a chunker/eval regression fails CI:

1. **Header-carry invariant** — when ``HeaderAwareChunker`` splits a long
   section into multiple chunks, every continuation chunk keeps its parent
   heading in ``section_header`` / ``heading_path``. A fragment that loses its
   header is unretrievable as context ("table detached from its caption").
2. **Overlap ratio guard** — splitter overlap must stay within 5–20% of chunk
   size (Coverge: >20% overlap duplicates retrievals).
3. **Fracture-rate gate** — ``run_chunking_eval`` emits a ``gates`` verdict and
   the synthetic corpus includes a long multi-section doc so the metric is
   non-vacuous (previously all gold docs were <12 words → nothing could split
   → fracture always 0.0 and the gate would have measured nothing).
"""

from __future__ import annotations

import pytest

from data_engineering_copilot.domain.models import ParsedDocument
from data_engineering_copilot.evaluation.chunking_eval import run_chunking_eval
from data_engineering_copilot.services.chunker import DocumentChunker
from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker


def _doc(text: str) -> ParsedDocument:
    return ParsedDocument(source_name="test", title="t", url="http://x", text=text)


LONG_SECTION = "# Guide\n\n" + ("Intro sentence for the guide overview.\n\n" * 2)
# ~40 repeats of an 8-word paragraph ≈ 320 words → forces mid-section splits at
# chunk_size_words=50 while every fragment still belongs to "## Setup".
LONG_SECTION += "## Setup\n\n" + ("Run this step to configure the environment properly.\n\n" * 40)


class TestHeaderCarryInvariant:
    def test_continuation_chunks_keep_section_header(self):
        chunker = HeaderAwareChunker(chunk_size_words=50, overlap_words=10, min_chunk_words=3)
        chunks = chunker._sync_chunk(_doc(LONG_SECTION))
        setup_chunks = [c for c in chunks if "Run this step" in c.text]
        assert len(setup_chunks) >= 2, "doc must actually split under ## Setup for this invariant to bite"
        for chunk in setup_chunks:
            assert chunk.section_header == "Setup", (
                f"fragment lost its header context: {chunk.chunk_id} header={chunk.section_header!r}"
            )
            assert "Setup" in chunk.heading_path

    def test_pre_header_content_has_empty_header(self):
        chunker = HeaderAwareChunker(chunk_size_words=50, overlap_words=10, min_chunk_words=3)
        chunks = chunker._sync_chunk(_doc(LONG_SECTION))
        guide_chunks = [c for c in chunks if "guide overview" in c.text]
        assert guide_chunks, "preamble must survive"
        for chunk in guide_chunks:
            assert chunk.section_header == "Guide"


class TestOverlapRatioGuard:
    def test_eval_chunker_overlap_within_5_and_20_percent(self):
        from data_engineering_copilot.evaluation.chunking_eval import _build_chunker

        recursive = _build_chunker("recursive")
        assert isinstance(recursive, DocumentChunker)
        ratio = recursive.chunk_overlap_chars / recursive.chunk_size_chars
        assert 0.05 <= ratio <= 0.20, f"overlap {ratio:.0%} outside 5-20% (Coverge duplicate zone >20%)"


class TestFractureGate:
    def test_report_carries_gates_and_synthetic_passes(self, tmp_path):
        out = tmp_path / "chunking_eval.json"
        report = run_chunking_eval("all", "synthetic", str(out))
        assert "gates" in report, "eval must emit a machine-readable gates verdict"
        gates = report["gates"]
        # Measured baseline 2026-08-23: all four strategies score 0.0 (none tear
        # fences/headers); threshold leaves headroom to catch any tearing onset.
        assert gates["fracture_ok"] is True, f"fracture gate failed: {gates}"
        assert 0.0 < gates["fracture_threshold"] <= 0.5
        strategies = [k for k in report if isinstance(report[k], dict) and "fracture_rate" in report[k]]
        assert set(strategies) == {"recursive", "sentence", "header", "structured"}
        # Corpus must be big enough that splitters actually place boundaries
        # (otherwise the gate compares zero boundaries against zero nodes).
        assert max(m["iou"] for m in report.values() if "iou" in m) > 0.0

    @pytest.mark.parametrize("strategy", ["header", "sentence"])
    def test_header_strategy_nonzero_on_realistic_docs(self, strategy, tmp_path):
        out = tmp_path / "chunking_eval.json"
        report = run_chunking_eval(strategy, "synthetic", str(out))
        m = report[strategy]
        assert m["iou"] > 0.0 or m["fracture_rate"] > 0.0, (
            f"{strategy} strategy measures nothing on the corpus (silent zero-output flaw pattern)"
        )
