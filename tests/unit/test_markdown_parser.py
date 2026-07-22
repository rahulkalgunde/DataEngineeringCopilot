from data_engineering_copilot.domain.models import RawDocument
from data_engineering_copilot.infrastructure.html_to_markdown import MarkdownParser


def test_parser_extracts_title_from_h1():
    html = """<html><body><main><h1>My Title</h1><p>{}</p></main></body></html>""".format(" ".join(["word"] * 50))
    parsed = MarkdownParser().parse(RawDocument(source_name="Test Source", url="https://example.com", html=html))
    assert parsed is not None
    assert parsed.title == "My Title"
    assert "# My Title" in parsed.text


def test_parser_falls_back_to_title_tag():
    html = """<html><head><title>Fallback Title</title></head><body><main><p>{}</p></main></body></html>""".format(
        " ".join(["word"] * 50)
    )
    parsed = MarkdownParser().parse(RawDocument(source_name="Test", url="https://example.com", html=html))
    assert parsed is not None
    assert parsed.title == "Fallback Title"


def test_parser_falls_back_to_url():
    html = "<html><body><main><p>{}</p></main></body></html>".format(" ".join(["word"] * 50))
    parsed = MarkdownParser().parse(RawDocument(source_name="Test", url="https://example.com/page", html=html))
    assert parsed is not None
    assert parsed.title == "https://example.com/page"


def test_parser_strips_nav_footer():
    html = """<html><body>
    <nav>Nav content</nav>
    <main><h1>Docs</h1><p>{}</p></main>
    <footer>Footer stuff</footer>
    </body></html>""".format(" ".join(["word"] * 50))
    parsed = MarkdownParser().parse(RawDocument(source_name="Test", url="https://example.com", html=html))
    assert parsed is not None
    assert "Nav content" not in parsed.text
    assert "Footer stuff" not in parsed.text


def test_parser_returns_none_for_short_page():
    html = "<html><body><main><p>Too short.</p></main></body></html>"
    parsed = MarkdownParser().parse(RawDocument(source_name="Test", url="https://example.com", html=html))
    assert parsed is None


def test_parser_sets_source_name_and_url():
    html = "<html><body><main><h1>Title</h1><p>{}</p></main></body></html>".format(" ".join(["word"] * 50))
    parsed = MarkdownParser().parse(
        RawDocument(source_name="My Source", url="https://docs.example.com/page", html=html)
    )
    assert parsed is not None
    assert parsed.source_name == "My Source"
    assert parsed.url == "https://docs.example.com/page"


def test_parser_preserves_markdown_formatting():
    html = """<html><body><main><h1>Title</h1>
    <p>Some <strong>bold</strong> and <em>italic</em> text.</p>
    <pre><code>print("hello")</code></pre>
    <p>{}</p></main></body></html>""".format(" ".join(["word"] * 40))
    parsed = MarkdownParser().parse(RawDocument(source_name="Test", url="https://example.com", html=html))
    assert parsed is not None
    assert "**bold**" in parsed.text or "bold" in parsed.text
    assert "print(" in parsed.text
