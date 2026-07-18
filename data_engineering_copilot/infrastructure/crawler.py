from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, Iterable
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from data_engineering_copilot.config.settings import DocumentationSource
from data_engineering_copilot.domain.models import IngestionEvent, RawDocument

logger = logging.getLogger(__name__)


class LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value:
                self.links.append(value)


class DocumentationCrawler:
    def __init__(self, timeout_seconds: int, delay_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        self.delay_seconds = delay_seconds

    def crawl(
        self,
        source: DocumentationSource,
        max_pages: int,
        on_event: Callable[[IngestionEvent], None] | None = None,
    ) -> Iterable[RawDocument]:
        visited: set[str] = set()
        queued: set[str] = set()
        queue: deque[str] = deque(self._clean_url(url) for url in source.start_urls)
        queued.update(self._dedupe_key(url) for url in queue)
        logger.info(
            "Crawler started source=%s max_pages=%s start_urls=%s",
            source.name,
            max_pages,
            len(source.start_urls),
        )

        while queue and len(visited) < max_pages:
            url = queue.popleft()
            url_key = self._dedupe_key(url)
            if url_key in visited or not self._is_allowed(url, source):
                continue

            self._emit(
                on_event,
                IngestionEvent(
                    event_type="fetch_start",
                    source_name=source.name,
                    url=url,
                    message=f"Fetching HTML page: {url}",
                ),
            )
            try:
                html = self._download(url)
            except Exception as exc:
                visited.add(url_key)
                message = f"Skipping {url}: {exc}"
                logger.warning(
                    "Crawler fetch failed source=%s url=%s pages_fetched=%s error=%s",
                    source.name,
                    url,
                    len(visited),
                    exc,
                )
                self._emit(
                    on_event,
                    IngestionEvent(
                        event_type="fetch_error",
                        source_name=source.name,
                        url=url,
                        message=message,
                        pages_fetched=len(visited),
                        error=str(exc),
                    ),
                )
                continue

            visited.add(url_key)
            logger.info("Crawler fetch succeeded source=%s url=%s pages_fetched=%s", source.name, url, len(visited))
            self._emit(
                on_event,
                IngestionEvent(
                    event_type="fetch_success",
                    source_name=source.name,
                    url=url,
                    message=f"Fetched HTML page: {url}",
                    pages_fetched=len(visited),
                ),
            )
            yield RawDocument(source_name=source.name, url=url, html=html)

            for link in self._extract_links(html, url):
                link_key = self._dedupe_key(link)
                if link_key not in visited and link_key not in queued and self._is_allowed(link, source):
                    queue.append(link)
                    queued.add(link_key)

            time.sleep(self.delay_seconds)

        logger.info(
            "Crawler completed source=%s pages_visited=%s queued_remaining=%s",
            source.name,
            len(visited),
            len(queue),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
        reraise=True,
    )
    def _download(self, url: str) -> str:
        request = Request(url, headers={"User-Agent": "DataEngineeringCopilot/1.0"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                raise ValueError(f"unsupported content type: {content_type}")
            return response.read().decode("utf-8", errors="replace")

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        parser = LinkExtractor()
        parser.feed(html)
        links: list[str] = []
        for href in parser.links:
            if href.startswith(("mailto:", "tel:", "javascript:")):
                continue
            links.append(self._clean_url(urljoin(base_url, href)))
        return links

    def _clean_url(self, url: str) -> str:
        return urldefrag(url)[0]

    def _dedupe_key(self, url: str) -> str:
        clean_url = self._clean_url(url)
        parsed = urlparse(clean_url)
        if parsed.path.endswith("/index.html"):
            clean_url = clean_url[: -len("index.html")]
            parsed = urlparse(clean_url)
        if parsed.path == "/":
            return clean_url
        return clean_url.rstrip("/")

    def _is_allowed(self, url: str, source: DocumentationSource) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return False
        if parsed.netloc not in source.allowed_domains:
            return False
        return not (
            source.url_prefixes and not any(url.startswith(prefix.rstrip("/")) for prefix in source.url_prefixes)
        )

    def _emit(self, on_event: Callable[[IngestionEvent], None] | None, event: IngestionEvent) -> None:
        if on_event is not None:
            on_event(event)
