"""LLM backend protocol and llama-server/Ollama implementations."""

import json
import logging
from typing import AsyncIterator, Protocol, runtime_checkable

import httpx

logger = logging.getLogger(__name__)


@runtime_checkable
class Backend(Protocol):
    """Protocol for LLM backends.

    Backends accept either a raw prompt string OR structured messages.
    The agent prefers messages (better for cache reuse), but falls back
    to raw prompt for backends that don't support messages.
    """

    async def complete(
        self,
        prompt: str,
        stop: list[str] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        messages: list[dict] | None = None,
    ) -> AsyncIterator[str]: ...

    async def token_count(self, text: str) -> int: ...


class LlamaServerBackend:
    """Backend that talks to llama-server's HTTP API.

    Uses /completion endpoint with cache_prompt=true for prefix caching.
    llama-server automatically reuses KV cache when the prompt shares
    a common prefix with the previous request on the same slot.
    """

    def __init__(
        self,
        endpoint: str,
        slot_id: int | None = None,
        timeout: float = 120.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.slot_id = slot_id
        self._client = httpx.AsyncClient(timeout=timeout)

    async def complete(
        self,
        prompt: str,
        stop: list[str] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        messages: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Stream completion tokens from llama-server.

        Uses raw prompt (not messages) since llama-server's /completion
        endpoint does prefix-based KV cache matching automatically.
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

        logger.debug(
            "POST %s/completion (slot=%s, prompt_chars=%d)",
            self.endpoint, self.slot_id, len(prompt),
        )

        try:
            async with self._client.stream(
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
                        logger.debug("Stream complete")
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

    async def token_count(self, text: str) -> int:
        """Count tokens using llama-server's /tokenize endpoint."""
        try:
            response = await self._client.post(
                f"{self.endpoint}/tokenize",
                json={"content": text},
            )
            response.raise_for_status()
            return len(response.json()["tokens"])
        except httpx.ConnectError as e:
            raise ConnectionError(f"Cannot connect to llama-server at {self.endpoint}: {e}") from e

    async def close(self):
        await self._client.aclose()


class OllamaBackend:
    """Backend that talks to Ollama's /api/chat endpoint.

    Uses structured messages (not raw prompts) so Ollama can reuse
    the KV cache from previous turns. On each call, Ollama only
    prefills NEW tokens at the end — the shared prefix stays cached.

    This is the key optimization for local models: a 5-turn conversation
    with 500 total tokens only prefills ~50 new tokens per turn, not 500.
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
        """Stream completion tokens from Ollama.

        Prefers 'messages' (structured chat) over 'prompt' (raw text).
        Structured messages let Ollama reuse KV cache across turns.
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
            # Fallback to /api/generate with raw prompt
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
                        # Log cache stats
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

                    # Extract content from chat vs generate format
                    if response_field:
                        content = chunk.get(response_field, {}).get("content", "")
                    else:
                        content = chunk.get("response", "")

                    if content:
                        yield content

        except httpx.ConnectError as e:
            raise ConnectionError(f"Cannot connect to Ollama at {self.endpoint}: {e}") from e

    async def token_count(self, text: str) -> int:
        """Approximate token count."""
        return len(text) // 4

    async def close(self):
        await self._client.aclose()
