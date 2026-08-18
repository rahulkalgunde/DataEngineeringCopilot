from __future__ import annotations

from pathlib import Path

from data_engineering_copilot.infrastructure.html_to_markdown import html_to_markdown

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "html_markdown"


def _convert(name: str) -> str:
    html = (_FIXTURES / f"{name}.html").read_text(encoding="utf-8")
    result = html_to_markdown(html, min_words=1)
    assert result is not None, f"{name}.html converted to None"
    return result


def _golden(name: str) -> str:
    return (_FIXTURES / f"{name}.md").read_text(encoding="utf-8").strip()


# ------------------------------------------------------------------
# Golden fixture assertions (tests/fixtures/html_markdown/)
# ------------------------------------------------------------------


def test_golden_table_fixture():
    result = _convert("table")
    assert result == _golden("table"), "table conversion must match golden byte-for-byte"
    assert result.startswith("# Table Reference")
    assert "| Function | Partition | Order |" in result
    assert "| --- | --- | --- |" in result
    assert "| row\\_number | dept | salary desc |" in result


def test_golden_code_fixture():
    result = _convert("code")
    assert result == _golden("code"), "code conversion must match golden byte-for-byte"
    assert "```python" in result
    assert "```sql" in result
    assert 'w = Window.partitionBy("dept").orderBy("salary")' in result
    assert "SELECT dept, salary," in result


def test_golden_links_fixture():
    result = _convert("links")
    assert result == _golden("links"), "links conversion must match golden byte-for-byte"
    assert "[SQL Window Functions](https://spark.apache.org/docs/latest/sql-ref-window-functions.html)" in result
    assert '(https://spark.apache.org/docs/latest/api/python/index.html "Python API")' in result


def test_golden_navigation_fixture():
    result = _convert("navigation")
    assert result == _golden("navigation"), "navigation conversion must match golden byte-for-byte"
    assert result.startswith("# Navigation Heavy")
    for chrome in ("Docs Home", "Getting Started", "Related Pages", "Guide A", "Copyright notice", "analytics"):
        assert chrome not in result


def test_golden_malformed_fixture_is_deterministic():
    assert _convert("malformed") == _golden("malformed")
    html = (_FIXTURES / "malformed.html").read_text(encoding="utf-8")
    assert html_to_markdown(html, min_words=1) == html_to_markdown(html, min_words=1)


def test_golden_fixtures_are_stable_across_runs():
    for name in ("table", "code", "links", "malformed", "navigation"):
        assert _convert(name) == _golden(name), f"{name} golden drifted"


def test_basic_conversion():
    html = "<html><body><h1>Title</h1><p>Hello world test content enough words here to pass check.</p></body></html>"
    result = html_to_markdown(html, min_words=5)
    assert result is not None
    assert "# Title" in result
    assert "Hello world" in result


def test_strips_nav_footer():
    html = """<html><body>
    <nav>Navigation menu stuff</nav>
    <main><h1>Docs</h1><p>Main content with enough words to pass the minimum word count filter easily.</p></main>
    <footer>Footer links</footer>
    </body></html>"""
    result = html_to_markdown(html, min_words=5)
    assert result is not None
    assert "Navigation" not in result
    assert "Footer" not in result
    assert "Main content" in result


def test_preserves_code_blocks():
    html = """<html><body>
    <h1>Code Example</h1>
    <pre><code class="language-python">def hello():
    print("world")</code></pre>
    <p>This page has enough content to pass the word count filter for testing.</p>
    </body></html>"""
    result = html_to_markdown(html, min_words=5)
    assert result is not None
    assert "def hello" in result
    assert "print" in result


def test_min_words_filter():
    html = "<html><body><p>Too short.</p></body></html>"
    result = html_to_markdown(html, min_words=50)
    assert result is None


def test_code_language_detection():
    html = """<html><body>
    <h1>Example</h1>
    <pre><code class="language-python">x = 1</code></pre>
    <p>Enough content words here to pass the minimum word count filter for testing purposes only.</p>
    </body></html>"""
    result = html_to_markdown(html, min_words=5)
    assert result is not None
    assert "x = 1" in result
