from __future__ import annotations

import asyncio
import contextlib
from typing import Self

import httpx


class SafeAsyncClientMixin:
    """Mixin ensuring clean httpx client closing across changing asyncio event loops.

    All client classes that lazy-init an ``httpx.AsyncClient`` should inherit
    this mixin and call ``await self._get_safe_client()`` instead of
    reimplementing the loop-aware pattern.

    Requirements on ``self``:
        - ``self.base_url`` — str
        - ``self.timeout_seconds`` — int | float
    """

    _client: httpx.AsyncClient | None = None
    _loop_id: int | None = None
    base_url: str = ""
    timeout_seconds: int | float = 30.0

    def _make_client_kwargs(self) -> dict:
        return {}

    async def _get_safe_client(self) -> httpx.AsyncClient:
        current_loop = id(asyncio.get_running_loop())

        if self._client is not None and self._loop_id != current_loop:
            old = self._client
            self._client = None
            with contextlib.suppress(Exception):
                asyncio.create_task(old.aclose())

        if self._client is None or self._client.is_closed:
            kwargs = self._make_client_kwargs()
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout_seconds),
                **kwargs,
            )
            self._loop_id = current_loop

        return self._client

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
