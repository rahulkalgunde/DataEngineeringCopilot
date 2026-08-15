"""In-memory Redis double for hermetic tests.

Implements the async subset of the redis client that infrastructure relies on
(``hget``/``hset``/``delete``/``pipeline``), so AsyncUrlRegistry and similar
components can be tested offline.  Keys and fields are normalized to ``str``.
"""

from __future__ import annotations

from typing import cast

_Command = tuple[str, tuple[object, ...]]


class _StubPipeline:
    def __init__(self, store: _StubRedis) -> None:
        self._store = store
        self._commands: list[_Command] = []

    def hget(self, key: str | bytes, field: str | bytes) -> _StubPipeline:
        self._commands.append(("hget", (key, field)))
        return self

    def hset(self, key: str | bytes, field: str | bytes, value: str | bytes) -> _StubPipeline:
        self._commands.append(("hset", (key, field, value)))
        return self

    def expire(self, key: str | bytes, time: int) -> _StubPipeline:
        self._commands.append(("expire", (key, time)))
        return self

    async def execute(self) -> list[bytes | None | int]:
        results: list[bytes | None | int] = []
        for cmd, args in self._commands:
            if cmd == "hget":
                key = cast("str | bytes", args[0])
                field = cast("str | bytes", args[1])
                results.append(self._store.hget_sync(key, field))
            elif cmd == "hset":
                key = cast("str | bytes", args[0])
                field = cast("str | bytes", args[1])
                value = cast("str | bytes", args[2])
                self._store.hset_sync(key, field, value)
                results.append(1)
            elif cmd == "expire":
                key = cast("str | bytes", args[0])
                ttl = cast(int, args[1])
                self._store.expire_sync(key, ttl)
                results.append(1)
        return results


class _StubRedis:
    """Async in-memory hash store with a compatible pipeline API."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, bytes]] = {}
        self._strings: dict[str, bytes] = {}
        self._lists: dict[str, list[bytes]] = {}
        self._ttls: dict[str, int] = {}

    @staticmethod
    def _norm(value: str | bytes) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else value

    def hget_sync(self, key: str | bytes, field: str | bytes) -> bytes | None:
        return self._data.get(self._norm(key), {}).get(self._norm(field))

    def hset_sync(self, key: str | bytes, field: str | bytes, value: str | bytes) -> None:
        self._data.setdefault(self._norm(key), {})[self._norm(field)] = (
            value.encode("utf-8") if isinstance(value, str) else value
        )

    def expire_sync(self, key: str | bytes, ttl: int) -> None:
        """No-op TTL tracking (in-memory store has no expiry)."""
        self._ttls[self._norm(key)] = ttl

    async def hget(self, key: str | bytes, field: str | bytes) -> bytes | None:
        return self.hget_sync(key, field)

    async def hset(self, key: str | bytes, field: str | bytes, value: str | bytes) -> None:
        self.hset_sync(key, field, value)

    async def expire(self, key: str | bytes, time: int) -> bool:
        self.expire_sync(key, time)
        return True

    async def delete(self, *keys: str | bytes) -> int:
        removed = 0
        for key in keys:
            removed += self._data.pop(self._norm(key), None) is not None
            removed += self._strings.pop(self._norm(key), None) is not None
            removed += self._lists.pop(self._norm(key), None) is not None
        return removed

    async def set(self, key: str | bytes, value: str | bytes) -> None:
        self._strings[self._norm(key)] = value.encode("utf-8") if isinstance(value, str) else value

    async def get(self, key: str | bytes) -> bytes | None:
        return self._strings.get(self._norm(key))

    async def rpush(self, key: str | bytes, *values: str | bytes) -> int:
        norm = self._norm(key)
        if norm not in self._lists:
            self._lists[norm] = []
        for value in values:
            self._lists[norm].append(value.encode("utf-8") if isinstance(value, str) else value)
        return len(self._lists[norm])

    async def ltrim(self, key: str | bytes, start: int, end: int) -> None:
        norm = self._norm(key)
        items = self._lists.get(norm, [])
        self._lists[norm] = items[start : end + 1] if end != -1 else items[start:]

    async def lrange(self, key: str | bytes, start: int, end: int) -> list[bytes]:
        items = self._lists.get(self._norm(key), [])
        if end == -1:
            return items[start:]
        return items[start : end + 1]

    def pipeline(self, transaction: bool = True) -> _StubPipeline:
        return _StubPipeline(self)

    def close(self) -> None:
        pass

    def dump(self) -> dict[str, dict[str, str]]:
        """Human-readable snapshot for assertions (bytes decoded to str)."""
        return {
            key: {field: value.decode("utf-8") for field, value in hashes.items()} for key, hashes in self._data.items()
        }
