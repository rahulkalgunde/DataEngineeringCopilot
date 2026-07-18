from data_engineering_copilot.domain.models import RawDocument
from data_engineering_copilot.infrastructure.html_parser import DocumentationHtmlParser


def test_html_parser_extracts_title_and_main_text():
    html = """
    <html>
      <head><title>Fallback</title></head>
      <body>
        <nav>ignore me</nav>
        <main>
          <h1>Useful Page</h1>
          <p>{}</p>
        </main>
      </body>
    </html>
    """.format(" ".join(["content"] * 50))

    parsed = DocumentationHtmlParser().parse(
        RawDocument(source_name="Delta Lake Documentation", url="https://docs.delta.io/latest/", html=html)
    )

    assert parsed is not None
    assert parsed.title == "Useful Page"
    assert "ignore me" not in parsed.text
    assert "content" in parsed.text
