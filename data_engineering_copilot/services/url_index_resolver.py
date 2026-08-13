"""Config-driven ``llms.txt`` index resolver.

Materializes the markdown files behind a pinned ``url_index`` source (e.g. the
Claude Platform/Code docs) into a cache directory and returns a deterministic
manifest of entries. Reuses the pure parsing helpers from the Claude ingestion
path (``parse_llms_index`` / ``strip_frontmatter``) so index semantics stay
identical between the interim command and the generic pipeline.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from data_engineering_copilot.config.settings import PinnedSourceConfig
from data_engineering_copilot.services.claude_docs_ingestion import parse_llms_index

logger = logging.getLogger(__name__)

_BACKOFF_SECONDS = (1.0, 2.0, 4.0)
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_USER_AGENT = "DataEngineeringCopilot/1.0"
_MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class UrlIndexEntry:
    """A single doc page referenced by an ``llms.txt`` index."""

    title: str
    url: str
    relative_path: str


@dataclass(frozen=True)
class UrlIndexManifest:
    """Deterministic enumeration of the pages resolved from an index."""

    source_name: str
    slug: str
    index_url: str
    url_prefix: str
    root: Path
    entries: tuple[UrlIndexEntry, ...]
    manifest_hash: str


def _url_to_relpath(url: str, url_prefix: str) -> str:
    if url.startswith(url_prefix):
        return url[len(url_prefix) :]
    return urlparse(url).path.lstrip("/")


class UrlIndexResolver:
    """Fetch and cache the markdown files behind a pinned ``url_index`` source.

    Parameters
    ----------
    config:
        A ``PinnedSourceConfig`` of type ``url_index``.
    cache_dir:
        Directory under which the source's ``cache_dir`` subtree is materialized.
    """

    def __init__(self, config: PinnedSourceConfig, cache_dir: Path) -> None:
        if config.type != "url_index":
            raise ValueError(f"UrlIndexResolver requires type='url_index', got {config.type!r}")
        self._config = config
        self._cache_dir = Path(cache_dir) / config.cache_dir

    async def resolve(self, client: httpx.AsyncClient | None = None) -> UrlIndexManifest:
        """Download (or reuse) the indexed markdown files and return a manifest.

        Already-cached files are skipped so re-runs are idempotent. Returns a
        deterministic manifest including the canonical entry list.
        """
        entries = await self._fetch_entries(client)
        await self._download_missing(entries, client)
        manifest = UrlIndexManifest(
            source_name=self._config.name,
            slug=self._config.slug,
            index_url=self._config.index_url,
            url_prefix=self._config.url_prefix,
            root=self._cache_dir,
            entries=tuple(entries),
            manifest_hash=self._hash_entries(entries),
        )
        logger.info(
            "url_index_resolver.resolve source=%s entries=%d root=%s",
            self._config.name,
            len(entries),
            self._cache_dir,
        )
        return manifest

    async def _fetch_entries(self, client: httpx.AsyncClient | None) -> list[UrlIndexEntry]:
        async def _get(client: httpx.AsyncClient) -> str:
            resp = await client.get(self._config.index_url, headers={"User-Agent": _USER_AGENT})
            resp.raise_for_status()
            return resp.text

        if client is not None:
            text = await _get(client)
        else:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as own_client:
                text = await _get(own_client)

        pairs = parse_llms_index(text, self._config.url_prefix)
        return [
            UrlIndexEntry(
                title=title,
                url=url,
                relative_path=_url_to_relpath(url, self._config.url_prefix),
            )
            for title, url in pairs
        ]

    async def _download_missing(self, entries: list[UrlIndexEntry], client: httpx.AsyncClient | None) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        missing = [e for e in entries if not (self._cache_dir / e.relative_path).is_file()]
        if not missing:
            return

        semaphore = asyncio.Semaphore(8)

        async def _fetch_one(entry: UrlIndexEntry) -> None:
            out_path = self._cache_dir / entry.relative_path
            async with semaphore:
                for attempt in range(len(_BACKOFF_SECONDS) + 1):
                    try:
                        resp = await active.get(entry.url, headers={"User-Agent": _USER_AGENT})
                        if resp.status_code in _RETRYABLE_STATUSES and (attempt + 1) < len(_BACKOFF_SECONDS):
                            await asyncio.sleep(_backoff(attempt, resp.headers.get("Retry-After")))
                            continue
                        resp.raise_for_status()
                        if len(resp.content) > _MAX_DOWNLOAD_BYTES:
                            logger.warning("url_index_resolver.oversized url=%s", entry.url)
                            return
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_bytes(resp.content)
                        return
                    except httpx.HTTPStatusError as exc:
                        if exc.response.status_code in _RETRYABLE_STATUSES and (attempt + 1) < len(_BACKOFF_SECONDS):
                            await asyncio.sleep(_backoff(attempt))
                            continue
                        logger.warning(
                            "url_index_resolver.fetch_failed status=%s url=%s",
                            exc.response.status_code,
                            entry.url,
                        )
                        return
                    except (httpx.TransportError, httpx.TimeoutException):
                        if (attempt + 1) < len(_BACKOFF_SECONDS):
                            await asyncio.sleep(_backoff(attempt))
                            continue
                        logger.warning("url_index_resolver.fetch_transport_failed url=%s", entry.url)
                        return

        async def _fetch_all() -> None:
            await asyncio.gather(*(_fetch_one(entry) for entry in missing))

        if client is not None:
            active = client
            await _fetch_all()
        else:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as own_client:
                active = own_client
                await _fetch_all()

    def _hash_entries(self, entries: list[UrlIndexEntry]) -> str:
        canonical = [{"title": e.title, "url": e.url, "relative_path": e.relative_path} for e in entries]
        return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _backoff(attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return _BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)]
