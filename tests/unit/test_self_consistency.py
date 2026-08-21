"""Task 8: self-consistency sampling (dark flag) selection helper."""

from __future__ import annotations

from data_engineering_copilot.services.async_rag import select_most_consistent


def test_medoid_selection_prefers_majority_shape():
    candidates = [
        "df = spark.read.table('t')\ndf.show()",
        "df = spark.read.table('t')\ndf.show()",
        "completely different answer text",
    ]
    assert select_most_consistent(candidates) == candidates[0]


def test_single_candidate_returns_it():
    assert select_most_consistent(["only"]) == ["only"][0]


def test_all_identical_returns_first():
    cands = ["same text"] * 3
    assert select_most_consistent(cands) == "same text"
