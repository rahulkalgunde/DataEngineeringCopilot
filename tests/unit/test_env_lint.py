"""Gate G2: .env config-mutation class must never leak (see 2026-08-24 incident
where a substring revert silently flipped EVALUATION/ENRICHMENT pins)."""

from __future__ import annotations

import pytest

from scripts.lint_env import lint


def test_clean_file_passes():
    text = """
LLM_PROVIDER="openrouter"
OPENROUTER_API_KEY="k"
ANSWER_LLM_PROVIDER="openrouter"
EVALUATION_LLM_PROVIDER="groq"
GROQ_API_KEY="g"
ENRICHMENT_LLM_PROVIDER="ollama"
"""
    assert lint(text) == []


def test_duplicate_key_detected():
    errors = lint("FOO=1\nFOO=2\n")
    assert any("duplicate key FOO" in e for e in errors)


def test_unknown_provider_rejected():
    errors = lint('EVALUATION_LLM_PROVIDER="not_a_provider"\n')
    assert any("not a supported provider" in e for e in errors)


def test_missing_api_key_for_pinned_provider():
    errors = lint('EVALUATION_LLM_PROVIDER="groq"\n')
    assert any("GROQ_API_KEY is missing" in e for e in errors)


def test_empty_api_key_flagged():
    errors = lint('EVALUATION_LLM_PROVIDER="groq"\nGROQ_API_KEY=""\n')
    assert any("GROQ_API_KEY is empty" in e for e in errors)


def test_unquoted_trailing_content_flagged():
    errors = lint('EVALUATION_LLM_PROVIDER="groq" trailing junk\nGROQ_API_KEY="g"\n')
    assert any("trailing non-comment" in e for e in errors)


def test_quoted_inline_comment_ok():
    errors = lint('EVALUATION_LLM_PROVIDER="groq"  # speed head\nGROQ_API_KEY="g"\n')
    assert not any("trailing non-comment" in e for e in errors)


def test_keyless_providers_need_no_key():
    assert lint('LLM_PROVIDER="ollama"\n') == []
    assert lint('EMBEDDING_PROVIDER="local-hf"\n') == []


@pytest.mark.parametrize(
    "pin_line",
    [
        'EVALUATION_LLM_PROVIDER=openrouter  # was "opencodezen"',
        "ENRICHMENT_LLM_PROVIDER=openrouter  # Ox window ended",
    ],
)
def test_substring_corruption_pattern_caught(pin_line):
    """The exact 2026-08-24 corruption: pin flipped to openrouter by a
    substring replace; value unquoted + inline comment => flagged."""
    errors = lint(pin_line + "\n")
    assert errors, "corruption pattern must produce at least one violation"


def test_staleness_checker_flags_changed_source(tmp_path):
    import json as _json

    from scripts.check_derived_staleness import check

    src = tmp_path / "qa_x.jsonl"
    src.write_text('{"a":1}\n', encoding="utf-8")
    import hashlib

    prov = tmp_path / "probes.provenance.json"
    prov.write_text(
        _json.dumps({"sources": {str(src): hashlib.sha256(src.read_bytes()).hexdigest()[:12]}}),
        encoding="utf-8",
    )
    assert check(tmp_path) == []
    src.write_text('{"a":2}\n', encoding="utf-8")  # source changed
    out = check(tmp_path)
    assert any("STALE" in o for o in out)
