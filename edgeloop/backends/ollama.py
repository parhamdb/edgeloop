"""Ollama backend — talks to Ollama's /api/chat endpoint.

Uses structured messages so Ollama can reuse the KV cache from previous
turns. On each call, Ollama only prefills NEW tokens at the end — the
shared prefix stays cached.

Measured improvement: 14x prefill speedup (55ms cold → 4ms warm).
"""

import json
import logging
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)


class OllamaBackend:
    """Backend for Ollama.

    Endpoint: http://host:port (default port 11434)
    Model: Any model installed in Ollama (e.g., qwen2.5-coder:7b)
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        endpoint: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def complete(
        self,
        prompt: str,
        stop: list[str] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        messages: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens from Ollama.

        Prefers structured messages (/api/chat) over raw prompt (/api/generate).
        Structured messages enable KV cache reuse across turns.
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

        # Use /api/chat with messages for KV cache reuse
        if messages is not None:
            body["messages"] = messages
            url = f"{self.endpoint}/api/chat"
            response_field = "message"
        else:
            body["prompt"] = prompt
            url = f"{self.endpoint}/api/generate"
            response_field = None

        logger.debug(
            "POST %s (model=%s, msgs=%d)",
            url, self.model, len(messages) if messages else 0,
        )

        try:
            async with self._client.stream("POST", url, json=body) as response:
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get("done", False):
                        pe_count = chunk.get("prompt_eval_count", 0)
                        pe_dur = chunk.get("prompt_eval_duration", 0)
                        ev_count = chunk.get("eval_count", 0)
                        ev_dur = chunk.get("eval_duration", 0)
                        pe_ms = pe_dur / 1e6 if pe_dur else 0
                        ev_tps = (ev_count / (ev_dur / 1e9)) if ev_dur > 0 else 0
                        logger.info(
                            "Done: prefill=%d tok (%.0fms), gen=%d tok (%.0f tok/s)",
                            pe_count, pe_ms, ev_count, ev_tps,
                        )
                        return

                    if response_field:
                        content = chunk.get(response_field, {}).get("content", "")
                    else:
                        content = chunk.get("response", "")

                    if content:
                        yield content

        except httpx.ConnectError as e:
            raise ConnectionError(f"Cannot connect to Ollama at {self.endpoint}: {e}") from e

    async def token_count(self, text: str) -> int:
        """Approximate token count (Ollama has no tokenize endpoint)."""
        return len(text) // 4

    async def close(self):
        await self._client.aclose()
