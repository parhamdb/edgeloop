"""LLM backend protocol and llama-server implementation."""

import json
import logging
from typing import AsyncIterator, Protocol, runtime_checkable

import httpx

logger = logging.getLogger(__name__)


@runtime_checkable
class Backend(Protocol):
    """Protocol for LLM backends. Implement complete() and token_count()."""

    async def complete(
        self,
        prompt: str,
        stop: list[str] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]: ...

    async def token_count(self, text: str) -> int: ...


class LlamaServerBackend:
    """Backend that talks to llama-server's HTTP API."""

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
    ) -> AsyncIterator[str]:
        """Stream completion tokens from llama-server."""
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

        logger.debug("POST %s/completion (slot=%s, tokens=%d)", self.endpoint, self.slot_id, max_tokens)

        try:
            async with self._client.stream(
                "POST", f"{self.endpoint}/completion", json=body
            ) as response:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data = line[6:]  # Strip "data: " prefix
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        logger.debug("Skipping non-JSON SSE line: %s", data[:50])
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
            # ExceptionGroup from anyio when port is invalid or connection fails
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
        """Close the HTTP client."""
        await self._client.aclose()


class OllamaBackend:
    """Backend that talks to Ollama's HTTP API."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        endpoint: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)
        self._context: list[int] | None = None  # Ollama's KV cache token context

    async def complete(
        self,
        prompt: str,
        stop: list[str] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Stream completion tokens from Ollama."""
        body: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        if stop:
            body["options"]["stop"] = stop

        # Disable thinking mode for Qwen3 models — we want direct responses
        # for tool calling reliability. Thinking wastes tokens on small models.
        body["think"] = False

        # Pass back the context from the previous response for KV cache reuse
        if self._context is not None:
            body["context"] = self._context

        logger.debug("POST %s/api/generate (model=%s, tokens=%d)", self.endpoint, self.model, max_tokens)

        try:
            async with self._client.stream(
                "POST", f"{self.endpoint}/api/generate", json=body
            ) as response:
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug("Skipping non-JSON line: %s", line[:50])
                        continue

                    # Capture context for KV cache reuse on next call
                    if chunk.get("done", False):
                        self._context = chunk.get("context")
                        logger.debug("Stream complete, context tokens: %d",
                                     len(self._context) if self._context else 0)
                        return

                    content = chunk.get("response", "")
                    if content:
                        yield content

        except httpx.ConnectError as e:
            raise ConnectionError(f"Cannot connect to Ollama at {self.endpoint}: {e}") from e

    async def token_count(self, text: str) -> int:
        """Approximate token count (Ollama doesn't have a tokenize endpoint)."""
        # Rough heuristic: ~4 chars per token for English text
        return len(text) // 4

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
