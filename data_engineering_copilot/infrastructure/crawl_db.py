from __future__ import annotations

import hashlib
import logging
import time
from enum import StrEnum

import asyncpg

log = logging.getLogger(__name__)


PG_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS crawl_frontier (
    url_hash      TEXT PRIMARY KEY,
    url           TEXT NOT NULL,
    source_name   TEXT NOT NULL,
    state         TEXT NOT NULL DEFAULT 'DISCOVERED',
    parent_hash   TEXT,
    depth         INTEGER NOT NULL DEFAULT 0,
    etag          TEXT,
    last_modified TEXT,
    attempts      INTEGER NOT NULL DEFAULT 0,
    last_error    TEXT,
    created_at    DOUBLE PRECISION NOT NULL,
    updated_at    DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_frontier_state
    ON crawl_frontier(source_name, state, depth, created_at);

CREATE TABLE IF NOT EXISTS sitemap_edges (
    parent_hash TEXT NOT NULL,
    child_hash  TEXT NOT NULL,
    PRIMARY KEY (parent_hash, child_hash),
    FOREIGN KEY (parent_hash) REFERENCES crawl_frontier(url_hash),
    FOREIGN KEY (child_hash)  REFERENCES crawl_frontier(url_hash)
);
"""


class CrawlState(StrEnum):
    DISCOVERED = "DISCOVERED"
    FETCHING = "FETCHING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


class PostgresCrawlFrontierDB:
    """Async PostgreSQL manager for the crawl frontier and sitemap edges."""

    def __init__(self, dsn: str, pool_min_size: int = 2, pool_max_size: int = 10) -> None:
        self._dsn = dsn
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._pool: asyncpg.Pool | None = None

    @property
    def _db(self) -> asyncpg.Pool | None:
        return self._pool

    async def initialize(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._pool_min_size,
            max_size=self._pool_max_size,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(PG_SCHEMA_SQL)
        reset_count = await self.reset_stranded()
        if reset_count:
            log.warning("crawler.crash_recovery_reset count=%d", reset_count)
        log.info("crawler.frontier_initialized PostgreSQL dsn=%s", self._dsn)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    @staticmethod
    def hash_url(url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()

    async def discover(
        self,
        url: str,
        source_name: str,
        parent_hash: str | None,
        depth: int,
    ) -> str | None:
        assert self._pool is not None
        url_hash = self.hash_url(url)
        now = time.time()
        async with self._pool.acquire() as conn:
            result = await conn.fetchrow(
                """INSERT INTO crawl_frontier
                   (url_hash, url, source_name, state, parent_hash, depth, created_at, updated_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                   ON CONFLICT(url_hash) DO NOTHING
                   RETURNING url_hash""",
                url_hash,
                url,
                source_name,
                CrawlState.DISCOVERED.value,
                parent_hash,
                depth,
                now,
                now,
            )
            if result:
                if parent_hash is not None:
                    await conn.execute(
                        "INSERT INTO sitemap_edges (parent_hash, child_hash) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                        parent_hash,
                        url_hash,
                    )
                return url_hash
            row = await conn.fetchrow(
                "SELECT state FROM crawl_frontier WHERE url_hash = $1",
                url_hash,
            )
            # Only re-discover FAILED pages (retry transient errors).
            # PROCESSED pages must stay PROCESSED — re-discovering them
            # creates infinite loops in cyclic link graphs (e.g. docs sites).
            if row and row["state"] == CrawlState.FAILED.value:
                await conn.execute(
                    "UPDATE crawl_frontier SET state = $1, parent_hash = $2, depth = $3, updated_at = $4 WHERE url_hash = $5",
                    CrawlState.DISCOVERED.value,
                    parent_hash,
                    depth,
                    now,
                    url_hash,
                )
                if parent_hash is not None:
                    await conn.execute(
                        "INSERT INTO sitemap_edges (parent_hash, child_hash) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                        parent_hash,
                        url_hash,
                    )
                return url_hash
            return None

    async def claim(self, url_hash: str) -> CrawlRecord | None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            now = time.time()
            row = await conn.fetchrow(
                "UPDATE crawl_frontier SET state = $1, updated_at = $2, attempts = attempts + 1 "
                "WHERE url_hash = $3 AND state = $4 RETURNING *",
                CrawlState.FETCHING.value,
                now,
                url_hash,
                CrawlState.DISCOVERED.value,
            )
            if row is None:
                return None
            return _pg_row_to_record(row)

    async def mark_processed(self, url_hash: str) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            now = time.time()
            await conn.execute(
                "UPDATE crawl_frontier SET state = $1, updated_at = $2 WHERE url_hash = $3",
                CrawlState.PROCESSED.value,
                now,
                url_hash,
            )

    async def mark_failed(self, url_hash: str, error: str) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            now = time.time()
            await conn.execute(
                "UPDATE crawl_frontier SET state = $1, last_error = $2, updated_at = $3 WHERE url_hash = $4",
                CrawlState.FAILED.value,
                error,
                now,
                url_hash,
            )

    async def get_pending(self, source_name: str, limit: int = 50) -> list[CrawlRecord]:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM crawl_frontier WHERE source_name = $1 AND state = $2 ORDER BY depth ASC, created_at ASC LIMIT $3",
                source_name,
                CrawlState.DISCOVERED.value,
                limit,
            )
            return [_pg_row_to_record(row) for row in rows]

    async def get_record(self, url_hash: str) -> CrawlRecord | None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM crawl_frontier WHERE url_hash = $1",
                url_hash,
            )
            if row is None:
                return None
            return _pg_row_to_record(row)

    async def reset_stranded(self) -> int:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE crawl_frontier SET state = $1 WHERE state = $2",
                CrawlState.DISCOVERED.value,
                CrawlState.FETCHING.value,
            )
            return int(result.split()[1]) if result and result.startswith("UPDATE") else 0

    async def add_edge(self, parent_hash: str, child_hash: str) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO sitemap_edges (parent_hash, child_hash) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                parent_hash,
                child_hash,
            )

    async def get_edges(self, parent_hash: str) -> list[str]:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT child_hash FROM sitemap_edges WHERE parent_hash = $1",
                parent_hash,
            )
            return [row["child_hash"] for row in rows]

    async def rediscover_children(self, parent_hash: str, source_name: str, depth: int) -> int:
        """Re-discover children of a 304-cached page for BFS expansion."""
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT child_hash FROM sitemap_edges WHERE parent_hash = $1",
                parent_hash,
            )
            if not rows:
                return 0
            rediscovered = 0
            now = time.time()
            for row in rows:
                child_hash = row["child_hash"]
                result = await conn.execute(
                    "UPDATE crawl_frontier SET state = $1, depth = $2, updated_at = $3 "
                    "WHERE url_hash = $4 AND state IN ($5, $6)",
                    CrawlState.DISCOVERED.value,
                    depth,
                    now,
                    child_hash,
                    CrawlState.PROCESSED.value,
                    CrawlState.FAILED.value,
                )
                if result and result.startswith("UPDATE"):
                    count = int(result.split()[1])
                    rediscovered += count
            return rediscovered

    async def stats(self, source_name: str | None = None) -> dict[str, int]:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            if source_name:
                rows = await conn.fetch(
                    "SELECT state, COUNT(*)::int as cnt FROM crawl_frontier WHERE source_name = $1 GROUP BY state",
                    source_name,
                )
            else:
                rows = await conn.fetch(
                    "SELECT state, COUNT(*)::int as cnt FROM crawl_frontier GROUP BY state",
                )
            return {row["state"]: row["cnt"] for row in rows}

    async def drop_all(self) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS sitemap_edges CASCADE")
            await conn.execute("DROP TABLE IF EXISTS crawl_frontier CASCADE")


class CrawlRecord:
    """Immutable snapshot of a crawl_frontier row."""

    __slots__ = (
        "url_hash",
        "url",
        "source_name",
        "state",
        "parent_hash",
        "depth",
        "etag",
        "last_modified",
        "attempts",
        "last_error",
        "created_at",
        "updated_at",
    )

    def __init__(
        self,
        url_hash: str,
        url: str,
        source_name: str,
        state: str,
        parent_hash: str | None,
        depth: int,
        etag: str | None,
        last_modified: str | None,
        attempts: int,
        last_error: str | None,
        created_at: float,
        updated_at: float,
    ) -> None:
        self.url_hash = url_hash
        self.url = url
        self.source_name = source_name
        self.state = state
        self.parent_hash = parent_hash
        self.depth = depth
        self.etag = etag
        self.last_modified = last_modified
        self.attempts = attempts
        self.last_error = last_error
        self.created_at = created_at
        self.updated_at = updated_at


def _pg_row_to_record(row: asyncpg.Record) -> CrawlRecord:
    return CrawlRecord(
        url_hash=row["url_hash"],
        url=row["url"],
        source_name=row["source_name"],
        state=row["state"],
        parent_hash=row["parent_hash"],
        depth=row["depth"],
        etag=row["etag"],
        last_modified=row["last_modified"],
        attempts=row["attempts"],
        last_error=row["last_error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
