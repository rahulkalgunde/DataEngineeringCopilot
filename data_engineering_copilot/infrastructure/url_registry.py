"""Redis-backed URL registry for crawl state persistence.

Stores ``url → html_hash`` per documentation source so that subsequent
ingestion runs can skip HTTP downloads for unchanged pages.

Redis key: ``crawl:url_registry:{source_name}`` (hash)
"""

from __future__ import annotations

import json
import time

import structlog

log = structlog.get_logger(__name__)


class UrlRegistry:
    """Per-source URL state store backed by a Redis hash.

    Each field in the hash maps a URL to a JSON object::

        { "html_hash": "<sha256>", "discovered_at": <unix_ts> }

    The registry has **no TTL** — entries persist until explicitly cleared
    via :meth:`clear` (typically called by ``reset-index``).
    """

    def __init__(self, redis_client: object, source_name: str) -> None:
        self._redis = redis_client
        self._key = f"crawl:url_registry:{source_name}"
        self._source_name = source_name

    # ------------------------------------------------------------------ #
    # Read                                                                #
    # ------------------------------------------------------------------ #

    def get_html_hash(self, url: str) -> str | None:
        """Return the stored ``html_hash`` for *url*, or ``None``."""
        raw = self._redis.hget(self._key, url)  # type: ignore[union-attr]
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            record = json.loads(raw)
            return record.get("html_hash")
        except (json.JSONDecodeError, AttributeError):
            return None

    def get_all_urls(self) -> dict[str, str]:
        """Return ``{url: html_hash}`` for every registered URL."""
        raw: dict[bytes, bytes] = self._redis.hgetall(self._key)  # type: ignore[union-attr]
        result: dict[str, str] = {}
        for url_bytes, value_bytes in raw.items():
            url = url_bytes.decode("utf-8") if isinstance(url_bytes, bytes) else url_bytes
            value = value_bytes.decode("utf-8") if isinstance(value_bytes, bytes) else value_bytes
            try:
                record = json.loads(value)
                result[url] = record.get("html_hash", "")
            except (json.JSONDecodeError, AttributeError):
                result[url] = value
        return result

    def count(self) -> int:
        """Return the number of URLs in the registry."""
        return self._redis.hlen(self._key)  # type: ignore[union-attr]

    # ------------------------------------------------------------------ #
    # Write                                                               #
    # ------------------------------------------------------------------ #

    def set_html_hash(self, url: str, html_hash: str) -> None:
        """Store or update the *html_hash* for *url*."""
        record = json.dumps(
            {
                "html_hash": html_hash,
                "discovered_at": time.time(),
            }
        )
        self._redis.hset(self._key, url, record)  # type: ignore[union-attr]

    def clear(self) -> None:
        """Remove all entries for this source."""
        self._redis.delete(self._key)  # type: ignore[union-attr]
        log.info("url_registry.cleared", source=self._source_name)
