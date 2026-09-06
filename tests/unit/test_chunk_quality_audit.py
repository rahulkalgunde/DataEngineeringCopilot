"""Unit tests for the hermetic corpus chunk-quality audit."""

import json

from data_engineering_copilot.evaluation.chunking_quality import (
    audit_chunks,
    ends_mid_sentence,
    fence_balance,
    gate_verdict,
    table_balance,
)


def _row(text, wc, **kw):
    row = {
        "text": text,
        "word_count": wc,
        "chunk_type": "text",
        "chunker_version": "header-aware-v1",
        "source_name": "s",
        "doc_type": "guide",
        "heading_path": (("H1", "H1"),),
        "file_path": "x.md",
    }
    row.update(kw)
    return row


def test_fence_balance():
    assert fence_balance("```python\ndef f(): pass\n```") is True
    assert fence_balance("```python\ndef f(): pass") is False


def test_table_balance():
    assert table_balance("<table><tr><td>a</td></tr></table>") is True
    assert table_balance("<table><tr><td>a</td>") is False
    assert table_balance("<tr><td>a</td></tr>") is True  # no <table> wrapper: skip
    assert table_balance("plain text") is True


def test_ends_mid_sentence():
    assert ends_mid_sentence("word ended here", "text") is True
    assert ends_mid_sentence("word ended here.", "text") is False
    assert ends_mid_sentence("```python\nreturn 1", "text") is False  # open fence
    assert ends_mid_sentence("anything at all", "code") is False
    assert ends_mid_sentence("", "text") is False


def test_audit_counts_tiny_fracture_and_oversized():
    chunks = [
        _row("fine chunk with enough words here", 6),
        _row("```python\ndef f(): pass", 5, heading_path=()),
        _row("x", 1),
        _row("a" * 90, 90),
    ]
    rep = audit_chunks(chunks, target_words=500)
    assert rep["total"] == 4
    assert rep["tiny_rate"] == 0.75  # 3 of 4 < 10 words
    assert rep["fence_fracture_rate"] == 0.25  # 1 of 4 unbalanced
    assert rep["oversized_rate"] == 0.0
    assert rep["heading_path_coverage"] == 3 / 4
    assert rep["word_count"]["median"] == 5.5
    # 3 of 4 rows end on sentence-internal chars => mid-sentence (H4 real metric);
    # the open-fenced row is excluded by the fence guard (its tail is code).
    assert rep["sentence_fracture_rate"] == 0.75
    # no HTML tables present
    assert rep["table_fracture_rate"] == 0.0


def test_audit_chunks_headings_optional():
    rep = audit_chunks([_row("hello world", 2, heading_path=())])
    assert rep["heading_path_coverage"] == 0.0


def test_audit_chunks_word_count_from_text_when_missing():
    rep = audit_chunks([_row("one two three", None)])
    assert rep["total"] == 1
    assert rep["word_count"]["median"] == 3


def test_audit_chunks_empty_input():
    rep = audit_chunks([])
    assert rep["total"] == 0
    assert rep["tiny_rate"] == 0.0
    assert rep["word_count"]["median"] == 0


def test_gate_verdict_passes_under_threshold():
    report = {
        "fence_fracture_rate": 0.01,
        "tiny_rate": 0.005,
        "oversized_rate": 0.0,
        "table_fracture_rate": 0.0,
        "sentence_fracture_rate": 0.0,
    }
    v = gate_verdict(report)
    assert v["pass"] is True


def test_gate_verdict_fails_on_fence_fracture():
    report = {
        "fence_fracture_rate": 0.07,
        "tiny_rate": 0.0,
        "oversized_rate": 0.0,
        "table_fracture_rate": 0.0,
        "sentence_fracture_rate": 0.0,
    }
    v = gate_verdict(report)
    assert v["pass"] is False
    assert v["checks"]["fence_fracture_rate"]["ok"] is False


def test_gate_verdict_fails_on_table_fracture_beyond_threshold():
    report = {
        "fence_fracture_rate": 0.0,
        "tiny_rate": 0.0,
        "oversized_rate": 0.0,
        "table_fracture_rate": 0.05,
        "sentence_fracture_rate": 0.0,
    }
    v = gate_verdict(report)
    assert v["pass"] is False
    assert v["checks"]["table_fracture_rate"]["ok"] is False


def test_corpus_gate_end_to_end(tmp_path):
    from data_engineering_copilot.evaluation.chunking_eval import run_corpus_quality_gate

    rows = [
        {
            "text": "good enough body text here",
            "word_count": 5,
            "chunk_type": "text",
            "chunker_version": "v1",
            "source_name": "s",
            "doc_type": "guide",
            "heading_path": [["H1", "H1"]],
            "file_path": "a.md",
        },
        {
            "text": "```python\ndef f():",
            "word_count": 4,
            "chunk_type": "text",
            "chunker_version": "v1",
            "source_name": "s",
            "doc_type": "guide",
            "heading_path": [["H1", "H1"]],
            "file_path": "b.md",
        },
    ]
    src = tmp_path / "chunks.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "out.json"
    report = run_corpus_quality_gate(str(src), str(out))
    assert report["total"] == 2
    assert report["fence_fracture_rate"] == 0.5
    assert report["gates"]["pass"] is False
    assert out.exists()
