"""Connection management for edgeloop backends.

Provides a shared, configurable HTTP connection pool that backends
use instead of creating their own httpx clients. This enables:
- Connection reuse across multiple backends/agents
- Centralized timeout and retry configuration
- Proper lifecycle management (single close point)
- Future: Unix socket support for local-only servers
"""

import logging
from contextlib import asynccontextmanager

import httpx

logger = logging.getLogger(__name__)


class Connection:
    """Managed HTTP connection to a local LLM server.

    Usage:
        conn = Connection("http://localhost:11434")
        # ... pass to backends ...
        await conn.close()

    Or as async context manager:
        async with Connection("http://localhost:11434") as conn:
            backend = OllamaBackend(model="qwen3:0.6b", connection=conn)
            ...
    """

    def __init__(
        self,
        endpoint: str,
        timeout: float = 120.0,
        max_keepalive: int = 5,
        max_connections: int = 10,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_keepalive_connections=max_keepalive,
                max_connections=max_connections,
            ),
        )
        self._request_count = 0
        self._closed = False

    @property
    def client(self) -> httpx.AsyncClient:
        """The underlying httpx client. Backends use this directly."""
        if self._closed:
            raise RuntimeError("Connection is closed")
        return self._client

    @property
    def request_count(self) -> int:
        """Total requests made through this connection."""
        return self._request_count

    def track_request(self):
        """Called by backends to track request count."""
        self._request_count += 1

    async def health_check(self) -> bool:
        """Check if the server is reachable."""
        try:
            r = await self._client.get(f"{self.endpoint}/")
            return r.status_code < 500
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def close(self):
        """Close the connection pool."""
        if not self._closed:
            await self._client.aclose()
            self._closed = True
            logger.debug("Connection to %s closed (%d requests)", self.endpoint, self._request_count)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()
