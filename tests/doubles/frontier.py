"""In-memory crawl frontier double (PostgresCrawlFrontierDB behavior).

Reproduces the state machine (DISCOVERED → FETCHING → PROCESSED/FAILED/SKIPPED)
and retry/rediscovery semantics of the PostgreSQL frontier without a database,
so crawler/ingestion logic tests run offline.
"""

from __future__ import annotations

import hashlib
import time

from data_engineering_copilot.infrastructure.crawl_db import CrawlRecord, CrawlState


class InMemoryFrontierDB:
    """Thread-of-record replacement for PostgresCrawlFrontierDB in tests."""

    def __init__(self) -> None:
        self._records: dict[str, CrawlRecord] = {}
        self._edges: dict[str, set[str]] = {}
        self._closed = False

    @staticmethod
    def hash_url(url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()

    async def initialize(self) -> None:
        self._closed = False

    async def close(self) -> None:
        self._closed = True

    async def discover(
        self,
        url: str,
        source_name: str,
        parent_hash: str | None,
        depth: int,
        max_attempts: int = 3,
    ) -> str | None:
        url_hash = self.hash_url(url)
        now = time.time()
        existing = self._records.get(url_hash)
        if existing is None:
            self._records[url_hash] = CrawlRecord(
                url_hash=url_hash,
                url=url,
                source_name=source_name,
                state=CrawlState.DISCOVERED.value,
                parent_hash=parent_hash,
                depth=depth,
                etag=None,
                last_modified=None,
                attempts=0,
                last_error=None,
                created_at=now,
                updated_at=now,
            )
            if parent_hash is not None:
                self._edges.setdefault(parent_hash, set()).add(url_hash)
            return url_hash
        if existing.state == CrawlState.FAILED.value and existing.attempts < max_attempts:
            self._records[url_hash] = _with_state(existing, CrawlState.DISCOVERED.value, depth, parent_hash, now)
            if parent_hash is not None:
                self._edges.setdefault(parent_hash, set()).add(url_hash)
            return url_hash
        return None

    async def claim(self, url_hash: str) -> CrawlRecord | None:
        existing = self._records.get(url_hash)
        if existing is None or existing.state != CrawlState.DISCOVERED.value:
            return None
        claimed = _with_state(existing, CrawlState.FETCHING.value, existing.depth, existing.parent_hash, time.time())
        self._records[url_hash] = CrawlRecord(
            url_hash=claimed.url_hash,
            url=claimed.url,
            source_name=claimed.source_name,
            state=claimed.state,
            parent_hash=claimed.parent_hash,
            depth=claimed.depth,
            etag=claimed.etag,
            last_modified=claimed.last_modified,
            attempts=existing.attempts + 1,
            last_error=claimed.last_error,
            created_at=claimed.created_at,
            updated_at=claimed.updated_at,
        )
        return self._records[url_hash]

    async def mark_processed(self, url_hash: str) -> None:
        self._records[url_hash] = _with_state(self._records[url_hash], CrawlState.PROCESSED.value)

    async def mark_failed(self, url_hash: str, error: str) -> None:
        record = self._records[url_hash]
        self._records[url_hash] = CrawlRecord(
            url_hash=record.url_hash,
            url=record.url,
            source_name=record.source_name,
            state=CrawlState.FAILED.value,
            parent_hash=record.parent_hash,
            depth=record.depth,
            etag=record.etag,
            last_modified=record.last_modified,
            attempts=record.attempts,
            last_error=error,
            created_at=record.created_at,
            updated_at=time.time(),
        )

    async def mark_skipped(self, url_hash: str) -> None:
        self._records[url_hash] = _with_state(self._records[url_hash], CrawlState.SKIPPED.value)

    async def get_pending(self, source_name: str, limit: int = 50) -> list[CrawlRecord]:
        pending = sorted(
            (
                r
                for r in self._records.values()
                if r.source_name == source_name and r.state == CrawlState.DISCOVERED.value
            ),
            key=lambda r: (r.depth, r.created_at),
        )
        return pending[:limit]

    async def get_record(self, url_hash: str) -> CrawlRecord | None:
        return self._records.get(url_hash)

    async def reset_stranded(self, stale_after_seconds: float = 1800.0) -> int:
        cutoff = time.time() - stale_after_seconds
        count = 0
        for url_hash, record in list(self._records.items()):
            if record.state == CrawlState.FETCHING.value and record.updated_at < cutoff:
                self._records[url_hash] = _with_state(record, CrawlState.DISCOVERED.value)
                count += 1
        return count

    async def all_urls(self, source_name: str) -> list[str]:
        return [r.url for r in self._records.values() if r.source_name == source_name]

    async def reactivate_missing(self, source_name: str, indexed_urls: set[str], max_attempts: int = 3) -> int:
        count = 0
        for url_hash, record in list(self._records.items()):
            if record.source_name != source_name:
                continue
            if record.state not in (CrawlState.PROCESSED.value, CrawlState.FAILED.value):
                continue
            if record.state == CrawlState.FAILED.value and record.attempts >= max_attempts:
                continue
            if record.url in indexed_urls:
                continue
            self._records[url_hash] = _with_state(record, CrawlState.DISCOVERED.value)
            count += 1
        return count

    async def add_edge(self, parent_hash: str, child_hash: str) -> None:
        self._edges.setdefault(parent_hash, set()).add(child_hash)

    async def get_edges(self, parent_hash: str) -> list[str]:
        return sorted(self._edges.get(parent_hash, set()))

    async def rediscover_children(self, parent_hash: str, source_name: str, depth: int) -> int:
        count = 0
        for child_hash in self._edges.get(parent_hash, set()):
            record = self._records.get(child_hash)
            if record is None or record.state not in (CrawlState.PROCESSED.value, CrawlState.FAILED.value):
                continue
            self._records[child_hash] = _with_state(record, CrawlState.DISCOVERED.value, depth=depth)
            count += 1
        return count

    async def stats(self, source_name: str | None = None) -> dict[str, int]:
        result: dict[str, int] = {}
        for record in self._records.values():
            if source_name and record.source_name != source_name:
                continue
            result[record.state] = result.get(record.state, 0) + 1
        return result

    async def drop_all(self) -> None:
        self._records.clear()
        self._edges.clear()


def _with_state(
    record: CrawlRecord,
    state: str,
    depth: int | None = None,
    parent_hash: str | None = None,
    updated_at: float | None = None,
) -> CrawlRecord:
    return CrawlRecord(
        url_hash=record.url_hash,
        url=record.url,
        source_name=record.source_name,
        state=state,
        parent_hash=parent_hash if parent_hash is not None else record.parent_hash,
        depth=depth if depth is not None else record.depth,
        etag=record.etag,
        last_modified=record.last_modified,
        attempts=record.attempts,
        last_error=record.last_error,
        created_at=record.created_at,
        updated_at=updated_at if updated_at is not None else time.time(),
    )
