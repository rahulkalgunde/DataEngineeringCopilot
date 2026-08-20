"""Task 4/5 tests: SparkHtmlParser — main-content extraction from rendered HTML."""

from __future__ import annotations

from data_engineering_copilot.infrastructure.spark_html_parser import SparkHtmlParser

_COMMIT = "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4"


def _jekyll_parser() -> SparkHtmlParser:
    return SparkHtmlParser(
        content_root_selector="div#content",
        excluded_selectors=(".left-menu-wrapper", ".left-menu", "#markdown-toc", ".breadcrumb"),
    )


def _sphinx_parser() -> SparkHtmlParser:
    return SparkHtmlParser(
        content_root_selector="main",
        excluded_selectors=(".bd-breadcrumbs", ".prev-next-bottom", ".headerlink", ".related-topics"),
    )


def test_jekyll_selects_content_div_not_sidebar() -> None:
    html = """
    <html><body>
    <div class="left-menu-wrapper"><div class="left-menu">
      <h3>Spark SQL Guide</h3><ul><li><a href="sql-getting-started.html">Getting Started</a></li></ul>
    </div></div>
    <div class="container">
      <div class="content-with-sidebar mr-3" id="content">
        <h1 class="title">JDBC To Other Databases</h1>
        <p>Spark SQL includes a data source that can read data from databases using JDBC. It should be preferred over JdbcRDD because results are returned as DataFrames which can be processed with Spark SQL and joined with other tables.</p>
        <pre><code>df = spark.read.jdbc(url, "table")</code></pre>
      </div>
    </div>
    </body></html>
    """
    result = _jekyll_parser().parse(
        html, "https://spark.apache.org/docs/4.0.0/sql-data-sources-jdbc.html", "docs/sql-data-sources-jdbc.md"
    )
    assert result is not None
    assert result.title == "JDBC To Other Databases"
    assert "Getting Started" not in result.text  # sidebar nav absent
    assert "sql-getting-started.html" not in result.text
    assert "dataframe" in result.text.lower()
    assert result.canonical_url == "https://spark.apache.org/docs/4.0.0/sql-data-sources-jdbc.html"
    assert result.source_path == "docs/sql-data-sources-jdbc.md"


def test_jekyll_strips_markdown_toc() -> None:
    html = """
    <html><body>
    <div class="container"><div id="content">
      <h1 class="title">Configuration</h1>
      <ul id="markdown-toc"><li><a href="#spark-properties">Spark Properties</a></li></ul>
      <h2 id="spark-properties">Spark Properties</h2>
      <p>Spark configuration properties control nearly all application settings and are set by the application at runtime. This page covers the most common properties for tuning and deployment.</p>
    </div></div>
    </body></html>
    """
    result = _jekyll_parser().parse(
        html, "https://spark.apache.org/docs/4.0.0/configuration.html", "docs/configuration.md"
    )
    assert result is not None
    assert "Spark Properties" in result.text
    assert "markdown-toc" not in result.text


def test_jekyll_rejects_navigation_only() -> None:
    html = """
    <html><body>
    <div class="container"><div id="content">
      <h1 class="title">Page</h1>
      <p>Back</p>
    </div></div>
    </body></html>
    """
    result = _jekyll_parser().parse(html, "https://spark.apache.org/docs/4.0.0/back.html", "docs/back.md")
    assert result is None


def test_sphinx_selects_main_and_strips_breadcrumb() -> None:
    html = """
    <html><body>
    <div id="pst-skip-link"><a href="#main-content">Skip to main content</a></div>
    <main id="main-content" class="bd-main" role="main">
      <ul class="bd-breadcrumbs"><li>Spark</li><li>API Reference</li></ul>
      <article class="bd-article">
        <h1>pyspark.sql.functions.filter</h1>
        <dl class="py function"><dt>pyspark.sql.functions.filter</dt><dd>Returns an array of elements for which a predicate holds.</dd></dl>
        <nav id="pst-page-toc-nav" class="page-toc"><ul><li>Examples</li></ul></nav>
        <h2>Examples</h2>
        <pre><code>df.select(filter("items", lambda x: x.discount > 0.20))</code></pre>
      </article>
    </main>
    </body></html>
    """
    result = _sphinx_parser().parse(
        html,
        "https://spark.apache.org/docs/4.0.0/api/python/reference/pyspark.sql/api/pyspark.sql.functions.filter.html",
        "python/docs/source/reference/pyspark.sql/api/pyspark.sql.functions.filter.rst",
    )
    assert result is not None
    assert result.title == "pyspark.sql.functions.filter"
    assert "Breadcrumb" not in result.text and "API Reference" not in result.text
    assert "predicate holds" in result.text
    assert "page-toc" not in result.text
    assert result.content_hash


def test_sphinx_strips_source_links() -> None:
    html = """
    <html><body>
    <main class="bd-main">
      <article class="bd-article">
        <h1>pyspark.sql.functions.filter</h1>
        <dt><a class="reference internal" href="../../../_modules/...#filter"><span class="viewcode-link">[source]</span></a></dt>
        <p>Returns an array of elements for which a predicate holds in a given array with enough words here to pass.</p>
      </article>
    </main>
    </body></html>
    """
    result = _sphinx_parser().parse(html, "https://x/filter.html", "x.rst")
    assert result is not None
    assert "[source]" not in result.text
    assert "predicate holds" in result.text


def test_no_content_root_returns_none() -> None:
    html = "<html><body><div><p>nothing meaningful here.</p></div></body></html>"
    parser = SparkHtmlParser(content_root_selector="div#missing")
    result = parser.parse(html, "https://x", "x.md")
    assert result is None


def test_extra_strip_selectors_removes_sidebar() -> None:
    parser = SparkHtmlParser(
        content_root_selector="main",
        excluded_selectors=(),
        extra_strip_selectors=(".sidebar", ".toc"),
    )
    html = """
    <html><body><main>
      <h1>Guide</h1>
      <p>Main content with enough words to pass the minimum word count threshold easily.</p>
      <div class="sidebar"><ul><li>Sidebar link one</li><li>Sidebar link two</li></ul></div>
      <div class="toc"><p>Table of contents entry</p></div>
    </main></body></html>
    """
    result = parser.parse(html, "https://example.com/g", "g.html")
    assert result is not None
    assert "Main content" in result.text
    assert "Sidebar link" not in result.text
    assert "Table of contents" not in result.text


def test_headings_extracted() -> None:
    html = """
    <html><body>
    <div class="container"><div id="content">
      <h1 class="title">Structured Streaming</h1>
      <h2>Overview</h2>
      <p>Structured Streaming is a scalable fault-tolerant stream processing engine built on the Spark SQL engine. This overview explains the programming model.</p>
      <h3>Basic Concepts</h3>
      <p>Streams are modeled as unbounded tables. This is the fundamental model behind the API.</p>
    </div></div>
    </body></html>
    """
    result = _jekyll_parser().parse(html, "https://spark.apache.org/docs/4.0.0/streaming.html", "docs/streaming.md")
    assert result is not None
    levels = [level for _, level in result.headings]
    assert levels == [1, 2, 3]
    texts = [text for text, _ in result.headings]
    assert "Structured Streaming" in texts
    assert "Overview" in texts
