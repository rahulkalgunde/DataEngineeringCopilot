"""Tests for the verified mode-guardrail facts.

The core invariant: every fact shipped in ``mode_guardrails.py`` must be a
byte-exact substring of the pinned Spark documentation corpus.  This prevents
the guardrail block from ever contradicting the retrieved context.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data_engineering_copilot.services.mode_guardrails import (
    _MODE_FACTS,
    _MODE_PATTERNS,
    build_mode_guardrail_block,
    detect_modes,
)

CORPUS_DOCS = Path(__file__).resolve().parents[2] / "data" / "spark_src" / "v4.0.0-b4ee7de0c9dc539a" / "docs"


def test_every_fact_is_byte_exact_substring_of_corpus() -> None:
    for mode, (fact, doc_file) in _MODE_FACTS.items():
        doc_path = CORPUS_DOCS / Path(doc_file).name
        assert doc_path.is_file(), f"corpus doc missing: {doc_path}"
        corpus = doc_path.read_text(encoding="utf-8")
        assert fact in corpus, f"fact for {mode!r} is not a byte-exact substring of {doc_file}"


def test_detect_modes_yarn_and_k8s() -> None:
    assert detect_modes("How does dynamic allocation work on YARN or Kubernetes?") == [
        "yarn",
        "kubernetes",
    ]


def test_detect_modes_k8s_abbreviation() -> None:
    assert detect_modes("Spark on k8s") == ["kubernetes"]


def test_detect_modes_none_for_unrelated_question() -> None:
    assert detect_modes("How does Catalyst optimize a query?") == []


def test_build_block_includes_verified_facts_for_named_modes() -> None:
    block = build_mode_guardrail_block("How does dynamic allocation work on YARN or Kubernetes?")
    assert block is not None
    assert "## VERIFIED DOCUMENTATION FACTS" in block
    assert "[yarn]" in block
    assert "[kubernetes]" in block
    # Facts in the block must be exact corpus substrings.
    for mode in ("yarn", "kubernetes"):
        fact = _MODE_FACTS[mode][0]
        assert fact in block


def test_build_block_none_for_ordinary_query() -> None:
    assert build_mode_guardrail_block("What is Catalyst?") is None


@pytest.mark.parametrize("mode", sorted(_MODE_PATTERNS))
def test_all_modes_have_verified_facts(mode: str) -> None:
    assert mode in _MODE_FACTS, f"mode {mode!r} has no verified fact"
