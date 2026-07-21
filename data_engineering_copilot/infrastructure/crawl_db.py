from __future__ import annotations

import hashlib
import logging
import time
from enum import StrEnum

import aiosqlite

log = logging.getLogger(__name__)

SCHEMA_SQL = """
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
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
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


class CrawlFrontierDB:
    """Async SQLite manager for the crawl frontier and sitemap edges."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()
        reset_count = await self.reset_stranded()
        if reset_count:
            log.warning("crawler.crash_recovery_reset count=%d", reset_count)
        log.info("crawler.frontier_initialized db_path=%s", self.db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

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
        """Insert a URL as DISCOVERED if not already present.

        Returns the url_hash if newly inserted, None if already known.
        """
        assert self._db is not None
        url_hash = self.hash_url(url)
        now = time.time()
        try:
            await self._db.execute(
                """INSERT INTO crawl_frontier
                   (url_hash, url, source_name, state, parent_hash, depth, created_at, updated_at)
                   VALUES (?, ?, ?, 'DISCOVERED', ?, ?, ?, ?)""",
                (url_hash, url, source_name, parent_hash, depth, now, now),
            )
            if parent_hash is not None:
                await self._db.execute(
                    "INSERT OR IGNORE INTO sitemap_edges (parent_hash, child_hash) VALUES (?, ?)",
                    (parent_hash, url_hash),
                )
            await self._db.commit()
        except aiosqlite.IntegrityError:
            return None
        return url_hash

    async def claim(self, url_hash: str) -> CrawlRecord | None:
        """Atomically set state → FETCHING. Returns None if not DISCOVERED."""
        assert self._db is not None
        now = time.time()
        cursor = await self._db.execute(
            """UPDATE crawl_frontier
               SET state = 'FETCHING', updated_at = ?, attempts = attempts + 1
               WHERE url_hash = ? AND state = 'DISCOVERED'""",
            (now, url_hash),
        )
        await self._db.commit()
        if cursor.rowcount == 0:
            return None
        return await self.get_record(url_hash)

    async def mark_processed(self, url_hash: str) -> None:
        assert self._db is not None
        now = time.time()
        await self._db.execute(
            "UPDATE crawl_frontier SET state = 'PROCESSED', updated_at = ? WHERE url_hash = ?",
            (now, url_hash),
        )
        await self._db.commit()

    async def mark_failed(self, url_hash: str, error: str) -> None:
        assert self._db is not None
        now = time.time()
        await self._db.execute(
            """UPDATE crawl_frontier
               SET state = 'FAILED', last_error = ?, updated_at = ?
               WHERE url_hash = ?""",
            (error, now, url_hash),
        )
        await self._db.commit()

    async def get_pending(self, source_name: str, limit: int = 50) -> list[CrawlRecord]:
        assert self._db is not None
        cursor = await self._db.execute(
            """SELECT * FROM crawl_frontier
               WHERE source_name = ? AND state = 'DISCOVERED'
               ORDER BY depth ASC, created_at ASC
               LIMIT ?""",
            (source_name, limit),
        )
        rows = await cursor.fetchall()
        return [_row_to_record(row) for row in rows]

    async def get_record(self, url_hash: str) -> CrawlRecord | None:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM crawl_frontier WHERE url_hash = ?",
            (url_hash,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    async def reset_stranded(self) -> int:
        """Reset FETCHING → DISCOVERED on boot. Returns count of reset records."""
        assert self._db is not None
        cursor = await self._db.execute("UPDATE crawl_frontier SET state = 'DISCOVERED' WHERE state = 'FETCHING'")
        await self._db.commit()
        return cursor.rowcount

    async def add_edge(self, parent_hash: str, child_hash: str) -> None:
        assert self._db is not None
        await self._db.execute(
            "INSERT OR IGNORE INTO sitemap_edges (parent_hash, child_hash) VALUES (?, ?)",
            (parent_hash, child_hash),
        )
        await self._db.commit()

    async def get_edges(self, parent_hash: str) -> list[str]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT child_hash FROM sitemap_edges WHERE parent_hash = ?",
            (parent_hash,),
        )
        rows = await cursor.fetchall()
        return [row["child_hash"] for row in rows]

    async def stats(self, source_name: str | None = None) -> dict[str, int]:
        assert self._db is not None
        if source_name:
            cursor = await self._db.execute(
                "SELECT state, COUNT(*) as cnt FROM crawl_frontier WHERE source_name = ? GROUP BY state",
                (source_name,),
            )
        else:
            cursor = await self._db.execute("SELECT state, COUNT(*) as cnt FROM crawl_frontier GROUP BY state")
        rows = await cursor.fetchall()
        return {row["state"]: row["cnt"] for row in rows}


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


def _row_to_record(row: aiosqlite.Row) -> CrawlRecord:
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
