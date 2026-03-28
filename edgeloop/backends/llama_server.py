"""llama-server backend — talks to llama.cpp's HTTP server.

llama-server automatically reuses KV cache when the prompt shares
a common prefix with the previous request on the same slot.
Uses cache_prompt=true and id_slot for cache pinning.
"""

import json
import logging
from typing import AsyncIterator

import httpx

from edgeloop.cache import CacheStats
from edgeloop.connection import Connection

logger = logging.getLogger(__name__)


class LlamaServerBackend:
    """Backend for llama-server (llama.cpp HTTP API).

    Args:
        endpoint: llama-server URL (e.g., http://localhost:8080)
        slot_id: Pin to a specific KV cache slot.
        connection: Optional shared Connection. Created internally if not provided.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        endpoint: str,
        slot_id: int | None = None,
        connection: Connection | None = None,
        timeout: float = 120.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.slot_id = slot_id
        if connection is not None:
            self._conn = connection
            self._owns_conn = False
        else:
            self._conn = Connection(endpoint, timeout=timeout)
            self._owns_conn = True
        self._last_stats: CacheStats | None = None

    @property
    def last_cache_stats(self) -> CacheStats | None:
        return self._last_stats

    async def complete(
        self,
        prompt: str,
        stop: list[str] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        messages: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens from llama-server /completion.

        Uses raw prompt with cache_prompt=true for prefix-based KV reuse.
        """
        body: dict = {
            "prompt": prompt,
            "stream": True,
            "cache_prompt": True,
            "n_predict": max_tokens,
            "temperature": temperature,
        }

        if self.slot_id is not None:
            body["id_slot"] = self.slot_id

        if stop:
            body["stop"] = stop

        logger.debug("POST %s/completion (slot=%s, prompt_chars=%d)", self.endpoint, self.slot_id, len(prompt))
        self._conn.track_request()

        try:
            async with self._conn.client.stream(
                "POST", f"{self.endpoint}/completion", json=body
            ) as response:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data = line[6:]
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get("stop", False):
                        self._record_stats(chunk)
                        return

                    content = chunk.get("content", "")
                    if content:
                        yield content

        except (httpx.ConnectError, OverflowError, OSError) as e:
            raise ConnectionError(f"Cannot connect to llama-server at {self.endpoint}: {e}") from e
        except BaseException as e:
            if isinstance(e, BaseExceptionGroup):
                raise ConnectionError(f"Cannot connect to llama-server at {self.endpoint}: {e}") from e
            raise

    def _record_stats(self, done_chunk: dict):
        """Extract cache stats from llama-server's timings."""
        timings = done_chunk.get("timings", {})
        prompt_n = timings.get("prompt_n", 0)
        prompt_ms = timings.get("prompt_ms", 0)
        predicted_n = timings.get("predicted_n", 0)
        predicted_ms = timings.get("predicted_ms", 0)

        # llama-server reports only the tokens it actually processed (not cached ones)
        # so prompt_n IS the number of newly-prefilled tokens
        self._last_stats = CacheStats(
            prompt_tokens=prompt_n,
            generated_tokens=predicted_n,
            prefill_ms=prompt_ms,
            generation_ms=predicted_ms,
            cache_hit_tokens=0,  # llama-server doesn't report this separately
        )

        logger.info(
            "Done: prefill=%d tok (%.0fms), gen=%d tok",
            prompt_n, prompt_ms, predicted_n,
        )

    async def token_count(self, text: str) -> int:
        """Count tokens via llama-server /tokenize endpoint."""
        try:
            response = await self._conn.client.post(
                f"{self.endpoint}/tokenize",
                json={"content": text},
            )
            response.raise_for_status()
            return len(response.json()["tokens"])
        except httpx.ConnectError as e:
            raise ConnectionError(f"Cannot connect to llama-server at {self.endpoint}: {e}") from e

    async def close(self):
        if self._owns_conn:
            await self._conn.close()
