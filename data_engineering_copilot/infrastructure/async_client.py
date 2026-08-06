from __future__ import annotations

import asyncio
import contextlib
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
    _client_loop: asyncio.AbstractEventLoop | None = None
    base_url: str = ""
    timeout_seconds: int | float = 30.0
    connect_timeout_seconds: int | float | None = None
    pool_timeout_seconds: int | float | None = None

    def _make_client_kwargs(self) -> dict:
        return {}

    async def _get_safe_client(self) -> httpx.AsyncClient:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        # An httpx.AsyncClient is bound to the event loop it was created on.
        # Reusing it from a different loop (e.g. RAGAS evaluates metrics in a
        # thread pool, each sync bridge call running its own asyncio.run loop)
        # raises "Event loop is closed". Recreate whenever the running loop
        # changed, even if the client was never explicitly closed.
        if (
            self._client is None
            or self._client.is_closed
            or (current_loop is not None and self._client_loop is not current_loop)
        ):
            if self._client is not None:
                # The old client's loop may already be closed; dropping the
                # reference is sufficient to stop reusing it.
                with contextlib.suppress(Exception):
                    await self._client.aclose()
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
            self._client_loop = current_loop
        return self._client

    async def close(self) -> None:
        """Close the underlying client. Overridden by subclasses."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
