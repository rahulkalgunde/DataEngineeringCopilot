from __future__ import annotations

from data_engineering_copilot.infrastructure.html_to_markdown import html_to_markdown


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
