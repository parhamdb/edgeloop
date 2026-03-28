"""llama-server backend — talks to llama.cpp's HTTP server.

llama-server automatically reuses KV cache when the prompt shares
a common prefix with the previous request on the same slot.
Uses cache_prompt=true and id_slot for cache pinning.
"""

import json
import logging
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)


class LlamaServerBackend:
    """Backend for llama-server (llama.cpp HTTP API).

    Endpoint: http://host:port (default port 8080)
    Slot pinning: Use slot_id to pin conversations to specific KV cache slots.
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
        """Stream tokens from llama-server /completion.

        Uses raw prompt with cache_prompt=true. llama-server matches
        the prompt prefix against the KV cache on the assigned slot
        and only re-evaluates new tokens.
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
                        # Log cache stats from llama-server
                        timings = chunk.get("timings", {})
                        if timings:
                            logger.info(
                                "Done: prefill=%d tok (%.0fms), gen=%d tok",
                                timings.get("prompt_n", 0),
                                timings.get("prompt_ms", 0),
                                timings.get("predicted_n", 0),
                            )
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
        """Count tokens via llama-server /tokenize endpoint."""
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
