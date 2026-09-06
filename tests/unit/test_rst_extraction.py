"""Task 6 extraction-fidelity tests: docutils RST, literalinclude, JSX strip,
license preamble, Spark API breadcrumbs (H8/M/6.4). Hermetic, 0 LLM.
"""

from __future__ import annotations

import pytest

from data_engineering_copilot.services.github_source_preparer import _strip_apache_license
from data_engineering_copilot.services.rst_parser import rst_to_markdown, strip_jsx_wrappers

pytestmark = [pytest.mark.unit]


class TestRstToMarkdown:
    def test_section_hierarchy_becomes_heading_levels(self) -> None:
        text = "Outer\n=====\n\npar1\n\nInner\n-----\n\npar2\n"
        md = rst_to_markdown(text)
        assert "# Outer" in md
        assert "## Inner" in md
        assert md.index("# Outer") < md.index("## Inner")

    def test_overline_underline_title_separates_from_body(self) -> None:
        text = "====\nDoc\n====\n\nbody paragraph here\n"
        md = rst_to_markdown(text)
        assert "# Doc" in md
        assert "body paragraph here" in md

    def test_code_block_becomes_fenced_code(self) -> None:
        text = "Heading\n=======\n\n.. code-block:: python\n\n   print('hi')\n"
        md = rst_to_markdown(text)
        assert "```python" in md
        assert "print('hi')" in md
        assert "code-block" not in md

    def test_literal_block_becomes_fenced_code(self) -> None:
        text = "Using it\n\n::\n\n   timeout = 5\n"
        md = rst_to_markdown(text)
        assert "```" in md or "::" not in md
        assert "timeout = 5" in md

    def test_table_flattens_to_pipe_rows(self) -> None:
        text = (
            "H\n=\n\n"
            "+-------+--------+\n"
            "| name  | value  |\n"
            "+=======+========+\n"
            "| alpha | 1      |\n"
            "+-------+--------+\n"
        )
        md = rst_to_markdown(text)
        assert "| name" in md
        assert "| alpha" in md
        assert "| ----" in md

    def test_definition_list_keeps_terms(self) -> None:
        text = "Terms\n=====\n\nspark.sql.shuffle.partitions\n    Default partitions.\n"
        md = rst_to_markdown(text)
        assert "spark.sql.shuffle.partitions" in md

    def test_ragged_table_does_not_crash(self) -> None:
        text = "H\n=\n\n====  ====  ====\na     b     cd\n====  ====  ====\n1     2\n====  ====  ====\n"
        md = rst_to_markdown(text)
        assert "| a" in md

    def test_directive_boilerplate_is_gone(self) -> None:
        text = "A\n=\n\n.. toctree::\n   :maxdepth: 2\n\nintro\n"
        md = rst_to_markdown(text)
        assert "toctree" not in md


class TestLiteralInclude:
    def test_literalinclude_resolves_against_source_dir(self, tmp_path) -> None:
        (tmp_path / "snippet.py").write_text("import os\nprint(os.getcwd())\n")
        text = "Using\n=====\n\n.. literalinclude:: snippet.py\n   :language: python\n"
        md = rst_to_markdown(text, source_path=str(tmp_path / "page.rst"))
        assert "```python" in md
        assert "import os" in md

    def test_include_recursively_inlines_nested_page(self, tmp_path) -> None:
        (tmp_path / "part.rst").write_text("Part A\n======\n\nnested body text\n")
        text = "Main\n====\n\n.. include:: part.rst\n"
        md = rst_to_markdown(text, source_path=str(tmp_path / "main.rst"))
        assert "# Part A" in md
        assert "nested body text" in md

    def test_unresolvable_literalinclude_leaves_placeholder(self, tmp_path) -> None:
        text = "T\n=\n\n.. literalinclude:: missing.py\n"
        md = rst_to_markdown(text, source_path=str(tmp_path / "page.rst"))
        assert "not resolvable" in md


class TestStripJsxWrappers:
    def test_tabs_and_accordion_open_close_removed(self) -> None:
        text = (
            "<Tabs titles={['Sql', 'Python']}>\n"
            "  <Tab>SQL body</Tab>\n"
            "  <Accordion>folded</Accordion>\n"
            "</Tabs>\n"
            "## Real Heading\n"
            "plain paragraph\n"
        )
        cleaned = strip_jsx_wrappers(text)
        assert "<Tabs" not in cleaned
        assert "<Tab>" not in cleaned
        assert "<Accordion>" not in cleaned
        assert "SQL body" in cleaned
        assert "folded" in cleaned
        assert "## Real Heading" in cleaned
        assert "plain paragraph" in cleaned

    def test_self_closing_and_propped_tags_removed(self) -> None:
        text = '<Step number=1>\nbody kept\n<Code lang="sql" />\n</Step>\n'
        cleaned = strip_jsx_wrappers(text)
        assert "<Step" not in cleaned and "</Step>" not in cleaned
        assert "<Code" not in cleaned
        assert "body kept" in cleaned


class TestApacheLicenseStrip:
    FRONTMATTER = (
        "---\n"
        "layout: global\n"
        "title: Security\n"
        "license: |\n"
        "  Licensed to the Apache Software Foundation (ASF) under one or more\n"
        "  contributor license agreements.\n"
        "---\n"
    )

    def test_license_frontmatter_stripped(self) -> None:
        text = self.FRONTMATTER + "\n# Spark Security\n\nreal body\n"
        stripped = _strip_apache_license(text)
        assert "Licensed to the Apache" not in stripped
        assert "layout: global" not in stripped
        assert "# Spark Security" in stripped
        assert "real body" in stripped

    def test_nonlicense_frontmatter_left_alone(self) -> None:
        text = "---\ntitle: Plain\nauthor: x\n---\n\nbody\n"
        assert _strip_apache_license(text) == text

    def test_no_frontmatter_unchanged(self) -> None:
        text = "plain doc\n\nno license\n"
        assert _strip_apache_license(text) == text
