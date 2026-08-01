from __future__ import annotations

from typing import Self

import httpx


class SafeAsyncClientMixin:
    """Mixin providing a single shared ``httpx.AsyncClient``.

    Subclasses must implement ``close()`` to clean up the client.
    Requirements on ``self``:
        - ``self.base_url`` — str
        - ``self.timeout_seconds`` — int | float
    """

    _client: httpx.AsyncClient | None = None
    base_url: str = ""
    timeout_seconds: int | float = 30.0
    connect_timeout_seconds: int | float | None = None
    pool_timeout_seconds: int | float | None = None

    def _make_client_kwargs(self) -> dict:
        return {}

    async def _get_safe_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            kwargs = self._make_client_kwargs()
            timeout = httpx.Timeout(
                self.timeout_seconds,
                connect=self.connect_timeout_seconds,
                pool=self.pool_timeout_seconds,
            )
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=timeout,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
                **kwargs,
            )
        return self._client

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
