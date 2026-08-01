"""In-memory Redis double for hermetic tests.

Implements the async subset of the redis client that infrastructure relies on
(``hget``/``hset``/``delete``/``pipeline``), so AsyncUrlRegistry and similar
components can be tested offline.  Keys and fields are normalized to ``str``.
"""

from __future__ import annotations

_Command = tuple[str, tuple[str | bytes, ...]]


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

    async def execute(self) -> list[bytes | None | int]:
        results: list[bytes | None | int] = []
        for cmd, args in self._commands:
            if cmd == "hget":
                key, field = args
                results.append(self._store.hget_sync(key, field))
            elif cmd == "hset":
                key, field, value = args
                self._store.hset_sync(key, field, value)
                results.append(1)
        return results


class _StubRedis:
    """Async in-memory hash store with a compatible pipeline API."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, bytes]] = {}

    @staticmethod
    def _norm(value: str | bytes) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else value

    def hget_sync(self, key: str | bytes, field: str | bytes) -> bytes | None:
        return self._data.get(self._norm(key), {}).get(self._norm(field))

    def hset_sync(self, key: str | bytes, field: str | bytes, value: str | bytes) -> None:
        self._data.setdefault(self._norm(key), {})[self._norm(field)] = (
            value.encode("utf-8") if isinstance(value, str) else value
        )

    async def hget(self, key: str | bytes, field: str | bytes) -> bytes | None:
        return self.hget_sync(key, field)

    async def hset(self, key: str | bytes, field: str | bytes, value: str | bytes) -> None:
        self.hset_sync(key, field, value)

    async def delete(self, *keys: str | bytes) -> int:
        removed = 0
        for key in keys:
            removed += self._data.pop(self._norm(key), None) is not None
        return removed

    def pipeline(self, transaction: bool = True) -> _StubPipeline:
        return _StubPipeline(self)

    def close(self) -> None:
        pass

    def dump(self) -> dict[str, dict[str, str]]:
        """Human-readable snapshot for assertions (bytes decoded to str)."""
        return {
            key: {field: value.decode("utf-8") for field, value in hashes.items()} for key, hashes in self._data.items()
        }
