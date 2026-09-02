"""Task 5: RRF weights + bm25 b tuning contracts."""

from __future__ import annotations

from pathlib import Path


def test_tune_weights_script_exists() -> None:
    assert Path("scripts/tune_rrf_weights.py").exists()


def test_settings_has_bm25_b() -> None:
    from data_engineering_copilot.config.settings import AppSettings

    assert "bm25_b" in AppSettings.model_fields
    from tests.conftest import make_settings

    s = make_settings()
    assert s.bm25_b == 0.75


def test_settings_has_rrf_weights() -> None:
    from data_engineering_copilot.config.settings import AppSettings

    assert "rrf_dense_weight" in AppSettings.model_fields
    assert "rrf_sparse_weight" in AppSettings.model_fields
    from tests.conftest import make_settings

    s = make_settings()
    assert s.rrf_dense_weight == 1.0
    assert s.rrf_sparse_weight == 1.0


def test_settings_bm25_b_validation() -> None:
    from pydantic import ValidationError

    from tests.conftest import make_settings

    # valid
    s = make_settings(bm25_b=0.5)
    assert s.bm25_b == 0.5
    # invalid should raise
    import pytest

    with pytest.raises(ValidationError):
        make_settings(bm25_b=1.5)
    with pytest.raises(ValidationError):
        make_settings(bm25_b=-0.1)


def test_bm25_tokenizer_b_injectable_from_settings() -> None:
    from data_engineering_copilot.infrastructure.bm25_tokenizer import BM25Tokenizer

    # explicit b passed
    tok_explicit = BM25Tokenizer(b=0.5)
    assert tok_explicit._b == 0.5

    # b=None should pull from settings (default 0.75)
    tok_default = BM25Tokenizer(b=None)
    assert tok_default._b == 0.75

    # when settings.bm25_b overridden, tokenizer should follow
    # BM25Tokenizer reads settings at __init__ time
    from unittest.mock import patch

    # patch global settings object used inside tokenizer
    with patch("data_engineering_copilot.infrastructure.bm25_tokenizer.settings") as mock_settings:
        mock_settings.bm25_b = 0.5
        tok_patched = BM25Tokenizer(b=None)
        assert tok_patched._b == 0.5


def test_bm25_tokenizer_b_persists() -> None:
    from data_engineering_copilot.infrastructure.bm25_tokenizer import BM25Tokenizer

    tok = BM25Tokenizer(b=0.5, namespace=False)
    tok.fit(["hello world test document"])
    import tempfile
    from pathlib import Path as P

    with tempfile.TemporaryDirectory() as td:
        p = P(td) / "tok.json"
        tok.save(p)
        loaded = BM25Tokenizer.load(p)
        assert loaded._b == 0.5
        assert loaded._k1 == 1.2
