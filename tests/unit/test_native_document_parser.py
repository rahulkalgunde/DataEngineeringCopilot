"""Phase 4 tests: native Markdown, RST, and code parsing."""

from __future__ import annotations

import pytest

from data_engineering_copilot.infrastructure.native_document_parser import NativeDocumentParser


@pytest.fixture
def parser() -> NativeDocumentParser:
    return NativeDocumentParser()


# ------------------------------------------------------------------
# Markdown parsing
# ------------------------------------------------------------------


def test_markdown_headings_survive(parser) -> None:
    md = """# Window Functions

## Ranking

Use dense_rank.

## Frames

Use rangeBetween.
"""
    doc = parser.parse_markdown(md, "docs/window.md")
    assert doc.title == "Window Functions"
    assert "# Window Functions" in doc.text
    assert "## Ranking" in doc.text
    assert len(doc.sections) >= 3
    assert doc.language == "conceptual"


def test_markdown_fenced_code_survives(parser) -> None:
    md = """# Example

```python
from pyspark.sql import Window
w = Window.partitionBy("a")
```
"""
    doc = parser.parse_markdown(md, "docs/example.md")
    assert "```python" in doc.text
    assert "Window.partitionBy" in doc.text


def test_markdown_title_falls_back_to_filename(parser) -> None:
    doc = parser.parse_markdown("some body text without heading", "docs/my_guide.md")
    assert doc.title == "my guide"


def test_liquid_directives_removed(parser) -> None:
    md = """# Guide

{% include_example sql/example.sql %}

Some text remains.
"""
    doc = parser.parse_markdown(md, "docs/guide.md")
    assert "{% include_example" not in doc.text
    assert "Some text remains." in doc.text


def test_html_comments_removed(parser) -> None:
    md = "# T\n<!-- build-only -->\nbody\n"
    doc = parser.parse_markdown(md, "docs/t.md")
    assert "build-only" not in doc.text
    assert "body" in doc.text


def test_frontmatter_stripped_from_markdown(parser) -> None:
    md = """---
layout: global
title: Spark SQL Reference
license: |
  Licensed to the Apache Software Foundation...
---
# Spark SQL Reference

Some body text.
"""
    doc = parser.parse_markdown(md, "docs/sql-ref.md")
    assert "layout: global" not in doc.text
    assert "Licensed to the Apache Software Foundation" not in doc.text
    assert "# Spark SQL Reference" in doc.text
    assert "Some body text." in doc.text
    # Title derives from the markdown heading, not frontmatter.
    assert doc.title == "Spark SQL Reference"


def test_frontmatter_absent_is_noop(parser) -> None:
    md = "# Plain\n\nJust body.\n"
    doc = parser.parse_markdown(md, "docs/plain.md")
    assert "# Plain" in doc.text


# ------------------------------------------------------------------
# RST parsing
# ------------------------------------------------------------------


def test_rst_headings_preserved(parser) -> None:
    rst = """Spark SQL Guide
================

Section One
-----------

Body text here.
"""
    doc = parser.parse_rst(rst, "docs/guide.rst")
    assert doc.title == "Spark SQL Guide"
    assert "Section One" in doc.text
    assert "Body text here." in doc.text


# ------------------------------------------------------------------
# Code parsing
# ------------------------------------------------------------------


def test_short_python_accepted(parser) -> None:
    code = "x = 1\n"
    doc = parser.parse_code(code, "examples/x.py", "python")
    assert doc.text == "x = 1\n"
    assert doc.language == "python"
    assert doc.line_count == 1


def test_python_with_defs_accepted(parser) -> None:
    code = "def foo(a, b):\n    return a + b\n\nprint(foo(1, 2))\n"
    doc = parser.parse_code(code, "examples/foo.py", "python")
    assert "def foo" in doc.text
    assert doc.line_count == 4


def test_invalid_python_preserved(parser) -> None:
    # Invalid syntax must not crash the parser; source is preserved.
    code = "def broken(:\n    pass\n"
    doc = parser.parse_code(code, "examples/broken.py", "python")
    assert "def broken" in doc.text


def test_parse_by_extension_python(parser, tmp_path) -> None:
    path = tmp_path / "foo.py"
    path.write_text("print('hi')\n")
    doc = parser.parse(path, "code_example")
    assert doc.language == "python"
    assert doc.text == "print('hi')\n"


def test_parse_by_extension_markdown(parser, tmp_path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("# Guide\nbody\n")
    doc = parser.parse(path, "guide")
    assert doc.title == "Guide"
    assert doc.language == "conceptual"


def test_parse_missing_file_raises(parser, tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        parser.parse(tmp_path / "missing.md", "guide")


def test_parse_unsupported_extension_raises(parser, tmp_path) -> None:
    path = tmp_path / "file.unknown"
    path.write_text("content")
    with pytest.raises(ValueError, match="Unsupported"):
        parser.parse(path, "guide")
