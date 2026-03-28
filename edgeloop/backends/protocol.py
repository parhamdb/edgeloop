"""Backend protocol — the contract all LLM backends must satisfy."""

from typing import AsyncIterator, Protocol, runtime_checkable

from edgeloop.cache import CacheStats


@runtime_checkable
class Backend(Protocol):
    """Protocol for LLM backends.

    Any object with these methods satisfies the protocol.
    No inheritance needed — just implement the methods.

    To implement a new backend:
        class MyBackend:
            async def complete(self, prompt, stop=None, temperature=0.7,
                             max_tokens=1024, messages=None) -> AsyncIterator[str]:
                ...yield tokens...

            async def token_count(self, text) -> int:
                return len(my_tokenizer.encode(text))

            @property
            def last_cache_stats(self) -> CacheStats | None:
                return self._last_stats
    """

    async def complete(
        self,
        prompt: str,
        stop: list[str] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        messages: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Stream completion tokens."""
        ...

    async def token_count(self, text: str) -> int:
        """Count tokens in text."""
        ...

    @property
    def last_cache_stats(self) -> CacheStats | None:
        """Cache stats from the most recent complete() call.

        Returns None if no completion has been done yet.
        Backends should populate this after each streaming response completes.
        """
        ...
