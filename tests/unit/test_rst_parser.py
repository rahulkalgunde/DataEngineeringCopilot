"""Behavioral tests for RstParser (reStructuredText ingestion parser).

Pins the URL-suffix gate, the happy path through docutils html5 rendering
into ParsedDocument, and the two graceful-None paths (severe RST error,
unsupported conversion output). Docutils behavior verified empirically:
section titles render as <h1 class="title"> inside an html_body fragment;
severe errors raise SystemMessage which the parser swallows -> None.
"""

from __future__ import annotations

import pytest

from data_engineering_copilot.domain.models import RawDocument
from data_engineering_copilot.infrastructure.rst_parser import RstParser


def _raw(url: str = "https://example.com/docs/guide.rst", body: str = "") -> RawDocument:
    # html_to_markdown drops results under 40 words (production contract),
    # so the default body carries enough prose to survive conversion.
    rst = (
        body
        if body
        else (
            "Spark Guide\n"
            "============\n\n"
            "Spark submits jobs lazily to the cluster. Transformations build "
            "a directed acyclic graph that the scheduler splits into stages, "
            "and each action materializes results across executors with "
            "shuffle boundaries computed between narrow dependency chains. "
            "Broadcast variables ship read-only shared state to every worker "
            "once per stage, avoiding repeated transfer costs during iterative "
            "graph algorithms where the same lookup table joins every round."
        )
    )
    return RawDocument(source_name="spark", url=url, html=rst)


class TestUrlGate:
    def test_non_rst_url_raises(self):
        with pytest.raises(ValueError, match="does not support"):
            RstParser().parse(_raw(url="https://example.com/docs/page.html"))

    @pytest.mark.parametrize("suffix", [".rst", ".rst.txt"])
    def test_supported_suffixes_accepted(self, suffix):
        doc = RstParser().parse(_raw(url=f"https://example.com/docs/guide{suffix}"))
        assert doc is not None


class TestHappyPath:
    def test_parses_title_and_text(self):
        doc = RstParser().parse(_raw())
        assert doc is not None
        assert doc.title == "Spark Guide"
        assert "Spark submits jobs lazily" in doc.text
        assert doc.source_name == "spark"
        assert doc.url == "https://example.com/docs/guide.rst"

    def test_title_falls_back_to_url_without_headings(self):
        raw = _raw(
            body=(
                "A plain paragraph without any section heading. The parser "
                "must fall back to the document URL as its title when the "
                "rendered fragment carries neither an h1 heading nor a "
                "title element, while still converting the remaining body "
                "prose into markdown text for indexing downstream."
            )
        )
        doc = RstParser().parse(raw)
        assert doc is not None
        assert doc.title == raw.url


class TestGracefulNonePaths:
    def test_severe_rst_error_returns_none(self):
        bad = "Title\n=====\n\n.. include:: /nonexistent/file.rst\n"
        assert RstParser().parse(_raw(body=bad)) is None
