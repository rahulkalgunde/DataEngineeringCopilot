from __future__ import annotations

import asyncio
import time
import xml.etree.ElementTree as ET
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from html.parser import HTMLParser
from ipaddress import ip_address
from urllib.parse import urldefrag, urljoin, urlparse

import aiohttp
import redis.exceptions
import structlog
from bs4 import BeautifulSoup

from data_engineering_copilot.config.settings import DocumentationSource
from data_engineering_copilot.domain.models import IngestionEvent, RawDocument
from data_engineering_copilot.infrastructure.crawl_cache import CrawlCache
from data_engineering_copilot.infrastructure.crawl_db import CrawlRecord, PostgresCrawlFrontierDB

log = structlog.get_logger(__name__)

SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"

# Fatal errors that should not be retried — fail immediately
FATAL_ERROR_TYPES = (
    redis.exceptions.AuthenticationError,
    redis.exceptions.AuthorizationError,
)

# Maximum number of redirects to follow
MAX_REDIRECTS = 5

# Private/reserved IP ranges for SSRF protection
_PRIVATE_IP_PREFIXES = (
    "10.",
    "172.16.",
    "172.17.",
    "172.18.",
    "172.19.",
    "172.20.",
    "172.21.",
    "172.22.",
    "172.23.",
    "172.24.",
    "172.25.",
    "172.26.",
    "172.27.",
    "172.28.",
    "172.29.",
    "172.30.",
    "172.31.",
    "192.168.",
    "127.",
    "0.",
    "169.254.",
)


@dataclass
class CrawlMetrics:
    pages_discovered: int = 0
    pages_fetched: int = 0
    pages_failed: int = 0


@dataclass
class _DomainState:
    semaphore: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(1))
    priority: int = 1
    last_request_time: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class _LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value:
                self.links.append(value)


class AsyncDocumentationCrawler:
    """Async crawler: full GET per URL -> yield RawDocument.

    No conditional-GET fast path: every queued URL is fetched and the
    content-hash dedup in the ingestion pipeline decides whether re-indexing
    is required. This guarantees pages are never skipped based on stale cache
    state (e.g. after a Qdrant reset).
    """

    def __init__(
        self,
        frontier: PostgresCrawlFrontierDB,
        cache: CrawlCache,
        timeout_seconds: int = 15,
        delay_seconds: float = 0.5,
        concurrency: int = 20,
        max_concurrency: int = 40,
        max_retries: int = 3,
        retry_backoff_base: float = 1.0,
        user_agent: str = "DataEngineeringCopilot/1.0",
        thread_pool_size: int = 8,
        per_domain_concurrency: int = 3,
        priority_domains: dict[str, int] | None = None,
        priority_multipliers: dict[int, float] | None = None,
    ) -> None:
        self.frontier = frontier
        self.cache = cache
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.delay_seconds = delay_seconds
        self.concurrency = concurrency
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self.user_agent = user_agent
        self.thread_pool_size = thread_pool_size
        self.per_domain_concurrency = per_domain_concurrency
        self.priority_domains = priority_domains or {}
        self.priority_multipliers = priority_multipliers or {1: 1.0, 2: 2.0, 3: 4.0}
        self._domain_states: dict[str, _DomainState] = {}
        self._metrics = CrawlMetrics()
        self._executor = ThreadPoolExecutor(max_workers=thread_pool_size)
        self._fatal_error: BaseException | None = None

    def _get_domain_priority(self, domain: str) -> int:
        for pattern, priority in self.priority_domains.items():
            if domain == pattern or domain.endswith("." + pattern):
                return priority
        return 1

    def _get_priority_multiplier(self, priority: int) -> float:
        return self.priority_multipliers.get(priority, 1.0)

    def _get_domain_state(self, url: str, source_priority: int = 1) -> _DomainState:
        domain = urlparse(url).netloc
        if domain not in self._domain_states:
            source_prio = source_priority
            dict_prio = self._get_domain_priority(domain)
            priority = max(source_prio, dict_prio)
            state = _DomainState(priority=priority)
            self._domain_states[domain] = state
            self._recalculate_all_semaphores()
        return self._domain_states[domain]

    def _recalculate_all_semaphores(self) -> None:
        total_weight = 0.0
        weights: dict[str, float] = {}
        for dom, state in self._domain_states.items():
            w = self._get_priority_multiplier(state.priority)
            weights[dom] = w
            total_weight += w
        for dom, state in self._domain_states.items():
            share = weights[dom] / total_weight if total_weight > 0 else 1.0
            slots = max(1, int(self.max_concurrency * share))
            slots = min(slots, self.per_domain_concurrency)
            state.semaphore = asyncio.Semaphore(slots)

    async def _enforce_delay(self, domain_state: _DomainState) -> None:
        async with domain_state.lock:
            now = time.monotonic()
            elapsed = now - domain_state.last_request_time
            if elapsed < self.delay_seconds:
                await asyncio.sleep(self.delay_seconds - elapsed)
            domain_state.last_request_time = time.monotonic()

    async def crawl(
        self,
        source: DocumentationSource,
        max_pages: int = 80,
        on_event: Callable[[IngestionEvent], None] | None = None,
    ) -> AsyncIterator[RawDocument]:
        if self.frontier._db is None:
            await self.frontier.initialize()
        await self._seed_frontier(source, max_pages)

        # Single persistent network session context maximizes connection pooling efficiency
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            queue: asyncio.Queue[CrawlRecord] = asyncio.Queue()
            results_queue: asyncio.Queue[RawDocument | None] = asyncio.Queue()

            yielded_count = 0
            total_attempted = 0
            max_attempted = max(max_pages * 3, 200)
            in_flight = 0

            async def worker() -> None:
                nonlocal in_flight
                while True:
                    record = await queue.get()
                    try:
                        doc = await self._process_url(session, record, source, on_event)
                    except Exception as exc:
                        if isinstance(exc, FATAL_ERROR_TYPES):
                            self._fatal_error = exc
                            log.error(
                                "crawler.fatal_error",
                                error_type=type(exc).__name__,
                                error=str(exc),
                                url=record.url,
                            )
                            in_flight -= 1
                            queue.task_done()
                            break
                        log.warning(
                            "crawler.worker_exception",
                            error_type=type(exc).__name__,
                            error=str(exc),
                            url=record.url,
                        )
                        doc = None
                    await results_queue.put(doc)
                    in_flight -= 1
                    queue.task_done()

            # Spin up static pool of long-running concurrent worker tasks
            workers = [asyncio.create_task(worker()) for _ in range(self.concurrency)]

            try:
                while yielded_count < max_pages and total_attempted < max_attempted:
                    # Dynamically adjust and feed work slots based on current depth capacities
                    if queue.qsize() < self.concurrency:
                        needed = self.concurrency - queue.qsize()
                        records = await self.frontier.get_pending(source.name, limit=needed)

                        if not records and queue.empty() and results_queue.empty() and in_flight <= 0:
                            break

                        for record in records:
                            if total_attempted >= max_attempted:
                                break
                            claimed = await self.frontier.claim(record.url_hash)
                            if claimed is not None:
                                in_flight += 1
                                await queue.put(claimed)
                                total_attempted += 1

                    # Non-blocking, predictable yielding mechanism
                    if not results_queue.empty() or queue.empty():
                        doc = await results_queue.get()
                        results_queue.task_done()
                        if doc is not None and yielded_count < max_pages:
                            yield doc
                            yielded_count += 1
                    else:
                        await asyncio.sleep(0.01)

            finally:
                # Direct cancellation cleanup of workers on extraction conclusion or exception
                for w in workers:
                    w.cancel()
                await asyncio.gather(*workers, return_exceptions=True)

            # Fail-fast: raise fatal errors instead of returning 0 results silently
            if self._fatal_error is not None:
                raise self._fatal_error

        log.info(
            "crawler.completed",
            yielded=yielded_count,
            total_attempted=total_attempted,
            metrics=self._metrics.__dict__,
        )

    async def _process_url(
        self,
        session: aiohttp.ClientSession,
        record: CrawlRecord,
        source: DocumentationSource,
        on_event: Callable[[IngestionEvent], None] | None,
    ) -> RawDocument | None:
        domain_state = self._get_domain_state(record.url, source_priority=source.priority)
        async with domain_state.semaphore:
            await self._enforce_delay(domain_state)

            result = await self._phase2_get(session, record)
            if result is None:
                return None

            html, content_type = result

            await self._extract_and_discover(record, html, source)
            self._metrics.pages_fetched += 1

            self._emit(
                on_event,
                IngestionEvent(
                    event_type="fetch_success",
                    source_name=record.source_name,
                    message=f"Fetched: {record.url}",
                    url=record.url,
                    pages_fetched=self._metrics.pages_fetched,
                ),
            )

            return RawDocument(
                source_name=record.source_name,
                url=record.url,
                html=html,
                content_type=content_type,
            )

    def _is_private_ip(self, url: str) -> bool:
        """Check if URL resolves to a private/reserved IP (SSRF protection)."""
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            if not hostname:
                return True
            # Check if it's an IP address
            try:
                ip = ip_address(hostname)
                return ip.is_private or ip.is_reserved or ip.is_loopback or ip.is_link_local
            except ValueError:
                # Not an IP address — check for common internal hostnames
                return hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1")
        except Exception:
            return True

    async def _phase2_get(self, session: aiohttp.ClientSession, record: CrawlRecord) -> tuple[str, str] | None:
        # SSRF protection: reject private IPs
        if self._is_private_ip(record.url):
            await self.frontier.mark_failed(record.url_hash, "SSRF: private/reserved IP")
            self._metrics.pages_failed += 1
            return None

        redirect_count = 0
        current_url = record.url

        for attempt in range(self.max_retries):
            try:
                async with session.get(
                    current_url,
                    headers={"User-Agent": self.user_agent},
                    max_redirects=MAX_REDIRECTS,
                ) as resp:
                    # Handle redirects manually to count them
                    if resp.status in (301, 302, 303, 307, 308):
                        redirect_count += 1
                        if redirect_count > MAX_REDIRECTS:
                            await self.frontier.mark_failed(record.url_hash, f"Too many redirects: {redirect_count}")
                            self._metrics.pages_failed += 1
                            return None
                        location = resp.headers.get("Location", "")
                        if location:
                            current_url = urljoin(current_url, location)
                            continue
                        # No Location header — treat as error
                        await self.frontier.mark_failed(record.url_hash, f"Redirect without Location: {resp.status}")
                        self._metrics.pages_failed += 1
                        return None

                    # Handle 410 Gone
                    if resp.status == 410:
                        await self.frontier.mark_gone(record.url_hash)
                        self._metrics.pages_failed += 1
                        return None

                    # Handle 429 Too Many Requests with Retry-After
                    if resp.status == 429:
                        retry_after = resp.headers.get("Retry-After", "")
                        try:
                            wait_time = float(retry_after)
                        except (ValueError, TypeError):
                            wait_time = self.retry_backoff_base * (2**attempt)
                        await asyncio.sleep(wait_time)
                        continue

                    if resp.status != 200:
                        await self.frontier.mark_failed(record.url_hash, f"HTTP {resp.status}")
                        self._metrics.pages_failed += 1
                        return None

                    content_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
                    if "text/html" not in content_type and "text/plain" not in content_type:
                        await self.frontier.mark_failed(record.url_hash, f"Not HTML: {content_type}")
                        self._metrics.pages_failed += 1
                        return None

                    html = await resp.text()
                    await self.cache.set_headers(
                        record.url_hash,
                        status=resp.status,
                        etag=resp.headers.get("ETag"),
                        last_modified=resp.headers.get("Last-Modified"),
                    )
                    return (html, content_type)
            except Exception as exc:
                if attempt < self.max_retries - 1:
                    backoff = self.retry_backoff_base * (2**attempt)
                    await asyncio.sleep(backoff)
                else:
                    await self.frontier.mark_failed(record.url_hash, str(exc))
                    self._metrics.pages_failed += 1
                    return None
        return None

    async def _seed_frontier(self, source: DocumentationSource, max_pages: int) -> None:
        seed_urls: list[str] = []
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            sitemap_urls = await self._try_sitemap(session, source)
        if sitemap_urls:
            seed_urls.extend(sitemap_urls)
        for url in source.start_urls:
            cleaned = self._clean_url(url)
            if cleaned not in seed_urls:
                seed_urls.append(cleaned)
        for url in seed_urls:
            await self.frontier.discover(
                url=self._clean_url(url),
                source_name=source.name,
                parent_hash=None,
                depth=0,
            )

    async def _try_sitemap(self, session: aiohttp.ClientSession, source: DocumentationSource) -> list[str] | None:
        parsed_start = urlparse(source.start_urls[0])
        sitemap_url = f"{parsed_start.scheme}://{parsed_start.netloc}/sitemap.xml"
        try:
            async with session.get(sitemap_url, headers={"User-Agent": self.user_agent}) as resp:
                if resp.status != 200:
                    return None
                raw = await resp.text()
        except Exception:
            return None
        return self._parse_sitemap(raw, source)

    def _parse_sitemap(self, raw_xml: str, source: DocumentationSource) -> list[str] | None:
        try:
            root = ET.fromstring(raw_xml)
        except ET.ParseError:
            return None
        entries: list[tuple[str, str]] = []
        for url_elem in root.iter(f"{{{SITEMAP_NS}}}url"):
            loc_elem = url_elem.find(f"{{{SITEMAP_NS}}}loc")
            if loc_elem is None or not loc_elem.text:
                continue
            loc = loc_elem.text.strip()
            lastmod_elem = url_elem.find(f"{{{SITEMAP_NS}}}lastmod")
            lastmod = lastmod_elem.text.strip() if lastmod_elem is not None and lastmod_elem.text else ""
            entries.append((lastmod, loc))
        filtered = [
            loc for lastmod, loc in entries if any(loc.startswith(prefix.rstrip("/")) for prefix in source.url_prefixes)
        ]
        if not filtered:
            return None
        with_dates = [(lm, loc) for lm, loc in entries if loc in filtered and lm]
        without_dates = [(lm, loc) for lm, loc in entries if loc in filtered and not lm]
        with_dates.sort(key=lambda x: x[0], reverse=True)
        return [loc for _, loc in with_dates] + [loc for _, loc in without_dates]

    async def _extract_and_discover(self, record: CrawlRecord, html: str, source: DocumentationSource) -> None:
        # Offload synchronous, CPU-heavy parser feed entirely to background threads
        loop = asyncio.get_running_loop()
        links = await loop.run_in_executor(self._executor, self._extract_links, html, record.url)

        for link_url in links:
            if self._is_allowed(link_url, source):
                child_hash = await self.frontier.discover(
                    url=link_url,
                    source_name=record.source_name,
                    parent_hash=record.url_hash,
                    depth=record.depth + 1,
                )
                if child_hash:
                    self._metrics.pages_discovered += 1
                    await self.frontier.add_edge(record.url_hash, child_hash)

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        # Phase 1: Fast regex-free HTMLParser
        parser = _LinkExtractor()
        try:
            parser.feed(html)
            raw_hrefs = parser.links
        except Exception:
            log.warning("crawler.html_parser_failed", extra={"url": base_url})
            raw_hrefs = []

        # Phase 2: Fallback to BeautifulSoup if HTMLParser extracted absolutely nothing
        # but the document has substantial length.
        if not raw_hrefs and len(html) > 500:
            log.debug("crawler.link_extractor_fallback_triggered", url=base_url)
            try:
                soup = BeautifulSoup(html, "html.parser")
                raw_hrefs = [a["href"] for a in soup.find_all("a", href=True) if a.get("href")]
            except Exception:
                log.exception("crawler.link_extractor_fallback_failed", url=base_url)

        # Phase 3: Clean and filter scheme anomalies
        links: list[str] = []
        for href in raw_hrefs:
            href_str = str(href)
            if href_str.startswith(("mailto:", "tel:", "javascript:", "#")):
                continue
            links.append(self._clean_url(urljoin(base_url, href_str)))
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

    @staticmethod
    def _emit(
        on_event: Callable[[IngestionEvent], None] | None,
        event: IngestionEvent,
    ) -> None:
        if on_event is not None:
            on_event(event)

    def shutdown(self) -> None:
        """Gracefully terminates thread executor allocations."""
        self._executor.shutdown(wait=False)

    async def close(self) -> None:
        """Close the crawler, releasing thread executor allocations."""
        self.shutdown()
