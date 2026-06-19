from data_engineering_copilot.config.settings import DocumentationSource
from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler


class FakeCrawler(DocumentationCrawler):
    def __init__(self, pages: dict[str, str]) -> None:
        super().__init__(timeout_seconds=1, delay_seconds=0)
        self.pages = pages

    def _download(self, url: str) -> str:
        return self.pages[url]


def test_crawler_resolves_relative_links_from_directory_url():
    source = DocumentationSource(
        name="Apache Spark Documentation",
        start_urls=("https://spark.apache.org/docs/latest/",),
        allowed_domains=("spark.apache.org",),
        url_prefixes=("https://spark.apache.org/docs/latest/",),
    )
    crawler = FakeCrawler(
        {
            "https://spark.apache.org/docs/latest/": """
                <html><body>
                    <main><p>{}</p></main>
                    <a href="index.html">Overview</a>
                    <a href="quick-start.html">Quick Start</a>
                </body></html>
            """.format("overview " * 40),
            "https://spark.apache.org/docs/latest/quick-start.html": """
                <html><body><main><p>{}</p></main></body></html>
            """.format("quickstart " * 40),
        }
    )

    documents = list(crawler.crawl(source, max_pages=10))

    assert [document.url for document in documents] == [
        "https://spark.apache.org/docs/latest/",
        "https://spark.apache.org/docs/latest/quick-start.html",
    ]


def test_crawler_deduplicates_trailing_slash_variants():
    source = DocumentationSource(
        name="Example Docs",
        start_urls=("https://example.com/docs/", "https://example.com/docs"),
        allowed_domains=("example.com",),
        url_prefixes=("https://example.com/docs/",),
    )
    crawler = FakeCrawler(
        {
            "https://example.com/docs/": """
                <html><body><main><p>{}</p></main></body></html>
            """.format("content " * 40),
        }
    )

    documents = list(crawler.crawl(source, max_pages=10))

    assert [document.url for document in documents] == ["https://example.com/docs/"]
