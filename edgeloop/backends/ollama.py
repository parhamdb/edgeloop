"""Ollama backend — talks to Ollama's /api/chat endpoint.

Uses structured messages so Ollama reuses KV cache from previous turns.
Only new tokens at the end get prefilled — the shared prefix stays cached.
"""

import json
import logging
from typing import AsyncIterator

import httpx

from edgeloop.cache import CacheStats
from edgeloop.connection import Connection

logger = logging.getLogger(__name__)


class OllamaBackend:
    """Backend for Ollama.

    Args:
        model: Ollama model name (e.g., "qwen2.5-coder:7b")
        endpoint: Ollama server URL (default http://localhost:11434)
        connection: Optional shared Connection. Created internally if not provided.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        endpoint: str = "http://localhost:11434",
        connection: Connection | None = None,
        timeout: float = 120.0,
    ):
        self.model = model
        self.endpoint = endpoint.rstrip("/")
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
        """Stream tokens from Ollama.

        Prefers structured messages (/api/chat) for KV cache reuse.
        Falls back to raw prompt (/api/generate) if no messages provided.
        """
        body: dict = {
            "model": self.model,
            "stream": True,
            "think": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        if stop:
            body["options"]["stop"] = stop

        if messages is not None:
            body["messages"] = messages
            url = f"{self.endpoint}/api/chat"
            response_field = "message"
        else:
            body["prompt"] = prompt
            url = f"{self.endpoint}/api/generate"
            response_field = None

        logger.debug("POST %s (model=%s, msgs=%d)", url, self.model, len(messages) if messages else 0)
        self._conn.track_request()

        try:
            async with self._conn.client.stream("POST", url, json=body) as response:
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get("done", False):
                        self._record_stats(chunk)
                        return

                    if response_field:
                        content = chunk.get(response_field, {}).get("content", "")
                    else:
                        content = chunk.get("response", "")

                    if content:
                        yield content

        except httpx.ConnectError as e:
            raise ConnectionError(f"Cannot connect to Ollama at {self.endpoint}: {e}") from e

    def _record_stats(self, done_chunk: dict):
        """Extract cache stats from Ollama's done response."""
        pe_count = done_chunk.get("prompt_eval_count", 0)
        pe_dur = done_chunk.get("prompt_eval_duration", 0)
        ev_count = done_chunk.get("eval_count", 0)
        ev_dur = done_chunk.get("eval_duration", 0)

        pe_ms = pe_dur / 1e6 if pe_dur else 0
        ev_ms = ev_dur / 1e6 if ev_dur else 0
        ev_tps = (ev_count / (ev_dur / 1e9)) if ev_dur > 0 else 0

        # Ollama reports total prompt tokens, but the prefill time tells us
        # how many were actually re-evaluated vs cached.
        # Heuristic: if prefill_ms < 20ms for >50 tokens, most were cached.
        estimated_new_tokens = int(pe_ms * 10) if pe_ms > 0 else pe_count  # ~10 tok/ms on GPU
        cache_hits = max(0, pe_count - estimated_new_tokens)

        self._last_stats = CacheStats(
            prompt_tokens=pe_count,
            generated_tokens=ev_count,
            prefill_ms=pe_ms,
            generation_ms=ev_ms,
            cache_hit_tokens=cache_hits,
        )

        logger.info(
            "Done: prefill=%d tok (%.0fms, ~%d cached), gen=%d tok (%.0f tok/s)",
            pe_count, pe_ms, cache_hits, ev_count, ev_tps,
        )

    async def token_count(self, text: str) -> int:
        """Approximate token count."""
        return len(text) // 4

    async def close(self):
        if self._owns_conn:
            await self._conn.close()
