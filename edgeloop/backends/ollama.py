"""Ollama backend — talks to Ollama's /api/chat endpoint.

Uses structured messages so Ollama reuses KV cache from previous turns.
Only new tokens at the end get prefilled — the shared prefix stays cached.

Thinking mode support:
  Qwen3 and other reasoning models produce a thinking phase (<think> tags)
  before the actual response. When thinking=True:
  - Thinking tokens stream in the 'thinking' field, content is empty
  - Once thinking ends, content tokens stream in the 'content' field
  - Both are yielded to the caller, but thinking tokens are NOT parsed
    for tool calls (the agent only parses the content portion)
  - Thinking consumes extra tokens — the budget is automatically expanded

  When thinking=False (default for tool-calling agents):
  - No thinking tokens are produced
  - All tokens go directly to content
  - Faster and more token-efficient for tool-calling scenarios
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
        thinking: Enable thinking/reasoning mode for models that support it.
                  When True, thinking tokens are generated before the response.
                  Default False — better for tool calling (saves tokens).
        connection: Optional shared Connection. Created internally if not provided.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        endpoint: str = "http://localhost:11434",
        thinking: bool = False,
        connection: Connection | None = None,
        timeout: float = 120.0,
    ):
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.thinking = thinking
        if connection is not None:
            self._conn = connection
            self._owns_conn = False
        else:
            self._conn = Connection(endpoint, timeout=timeout)
            self._owns_conn = True
        self._last_stats: CacheStats | None = None
        self._last_thinking: str | None = None  # thinking text from last call

    @property
    def last_cache_stats(self) -> CacheStats | None:
        return self._last_stats

    @property
    def last_thinking(self) -> str | None:
        """The thinking/reasoning text from the last completion, if any."""
        return self._last_thinking

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

        When thinking mode is enabled:
        - Token budget is expanded (thinking tokens + response tokens)
        - Only response content is yielded (not thinking tokens)
        - Thinking text is stored in self.last_thinking for inspection
        """
        # Expand budget if thinking is on — thinking typically uses 50-200 tokens
        effective_max = max_tokens + 300 if self.thinking else max_tokens

        body: dict = {
            "model": self.model,
            "stream": True,
            "think": self.thinking,
            "options": {
                "num_predict": effective_max,
                "temperature": temperature,
            },
        }

        if stop:
            body["options"]["stop"] = stop

        if messages is not None:
            body["messages"] = messages
            url = f"{self.endpoint}/api/chat"
            is_chat = True
        else:
            body["prompt"] = prompt
            url = f"{self.endpoint}/api/generate"
            is_chat = False

        logger.debug(
            "POST %s (model=%s, msgs=%d, think=%s, max_tok=%d)",
            url, self.model, len(messages) if messages else 0,
            self.thinking, effective_max,
        )
        self._conn.track_request()

        thinking_parts: list[str] = []
        self._last_thinking = None

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
                        if thinking_parts:
                            self._last_thinking = "".join(thinking_parts)
                            logger.debug(
                                "Thinking: %d chars: %s",
                                len(self._last_thinking),
                                self._last_thinking[:100],
                            )
                        self._record_stats(chunk, len(thinking_parts))
                        return

                    if is_chat:
                        msg = chunk.get("message", {})
                        thinking = msg.get("thinking", "")
                        content = msg.get("content", "")
                    else:
                        thinking = ""
                        content = chunk.get("response", "")

                    # Collect thinking tokens but don't yield them
                    if thinking:
                        thinking_parts.append(thinking)
                        continue

                    # Yield only actual response content
                    if content:
                        yield content

        except httpx.ConnectError as e:
            raise ConnectionError(f"Cannot connect to Ollama at {self.endpoint}: {e}") from e

    def _record_stats(self, done_chunk: dict, thinking_token_chunks: int):
        """Extract cache stats from Ollama's done response."""
        pe_count = done_chunk.get("prompt_eval_count", 0)
        pe_dur = done_chunk.get("prompt_eval_duration", 0)
        ev_count = done_chunk.get("eval_count", 0)
        ev_dur = done_chunk.get("eval_duration", 0)

        pe_ms = pe_dur / 1e6 if pe_dur else 0
        ev_ms = ev_dur / 1e6 if ev_dur else 0
        ev_tps = (ev_count / (ev_dur / 1e9)) if ev_dur > 0 else 0

        estimated_new_tokens = int(pe_ms * 10) if pe_ms > 0 else pe_count
        cache_hits = max(0, pe_count - estimated_new_tokens)

        self._last_stats = CacheStats(
            prompt_tokens=pe_count,
            generated_tokens=ev_count,
            prefill_ms=pe_ms,
            generation_ms=ev_ms,
            cache_hit_tokens=cache_hits,
        )

        think_info = f", thinking_chunks={thinking_token_chunks}" if thinking_token_chunks else ""
        logger.info(
            "Done: prefill=%d tok (%.0fms, ~%d cached), gen=%d tok (%.0f tok/s)%s",
            pe_count, pe_ms, cache_hits, ev_count, ev_tps, think_info,
        )

    async def token_count(self, text: str) -> int:
        """Approximate token count."""
        return len(text) // 4

    async def close(self):
        if self._owns_conn:
            await self._conn.close()
