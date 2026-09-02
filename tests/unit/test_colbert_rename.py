"""Test lexical_ngram rename with colbert alias (ADR-013)."""

from __future__ import annotations

import pytest

from tests.conftest import make_settings

pytestmark = pytest.mark.unit


def test_lexical_ngram_alias() -> None:
    s = make_settings(reranker_type="lexical_ngram")
    assert s.reranker_type == "lexical_ngram"


def test_colbert_still_works_deprecated() -> None:
    with pytest.warns(DeprecationWarning, match="lexical_ngram"):
        s = make_settings(reranker_type="colbert")
        assert s.reranker_type == "colbert"


def test_colbert_header_is_lexical_proxy() -> None:
    import pathlib

    text = pathlib.Path("data_engineering_copilot/services/colbert_reranker.py").read_text(encoding="utf-8")
    assert "Char-3gram MaxSim lexical proxy" in text
    assert "NOT neural ColBERT" in text


def test_reranker_type_includes_lexical_ngram() -> None:
    # verify domain model also lists it
    from data_engineering_copilot.domain.models import RerankerType

    assert RerankerType.LEXICAL_NGRAM == "lexical_ngram"
    # colbert stays for back-compat
    assert RerankerType.COLBERT == "colbert"


def test_factory_routes_lexical_ngram() -> None:
    import pathlib

    text = pathlib.Path("data_engineering_copilot/factory.py").read_text(encoding="utf-8")
    assert 'reranker_type in ("colbert", "lexical_ngram")' in text or "lexical_ngram" in text
    assert "lexical_ngram proxy" in text.lower() or "lexical_ngram_proxy_not_neural" in text
