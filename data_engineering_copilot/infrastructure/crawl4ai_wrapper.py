import asyncio
from typing import List
from crawl4ai import AsyncWebCrawler

# Reuse the existing RawDocument dataclass from the project
from data_engineering_copilot.domain.models import RawDocument

class Crawl4AIWrapper:
    """
    Wrapper around Crawl4AI's AsyncWebCrawler that provides a
    synchronous‑style interface compatible with the existing
    DocumentationCrawler API used throughout the codebase.
    """

    def __init__(self, timeout_seconds: int = 15, delay_seconds: float = 0.2):
        self.timeout = timeout_seconds
        self.delay = delay_seconds

    async def _async_crawl(self, urls: List[str]) -> List[RawDocument]:
        async with AsyncWebCrawler(
            timeout=self.timeout,
            delay=self.delay,
            # Crawl4AI already respects robots.txt and sets a sensible user‑agent
            verbose=False,
        ) as crawler:
            tasks = [crawler.arun(url=url) for url in urls]
            results = await asyncio.gather(*tasks)

        raw_documents: List[RawDocument] = []
        for result in results:
            # AsyncWebCrawler returns a CrawlResult with .success, .url, .html
            if getattr(result, "success", False):
                raw_documents.append(
                    RawDocument(
                        source_name=\"Crawl4AI\",
                        url=result.url,
                        html=result.html,
                    )
                )
        return raw_documents

    def crawl(self, urls: List[str]) -> List[RawDocument]:
        """
        Public method matching the original DocumentationCrawler interface.
        Executes the async crawl in an event loop and returns a list of
        RawDocument objects.
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._async_crawl(urls))