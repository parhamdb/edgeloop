"""Custom backend — implement the Backend protocol for any LLM.

The Backend protocol requires two methods:
    complete()    → stream tokens
    token_count() → count tokens in text

Optionally:
    last_cache_stats → CacheStats from last call
"""

import asyncio
from typing import AsyncIterator
from edgeloop import Agent, tool
from edgeloop.cache import CacheStats


class EchoBackend:
    """Example backend that echoes input. Replace with your LLM."""

    def __init__(self):
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
        # Your LLM call goes here
        last_msg = messages[-1]["content"] if messages else prompt[-200:]
        yield f"Echo: {last_msg[:100]}"
        self._last_stats = CacheStats(prompt_tokens=len(prompt) // 4)

    async def token_count(self, text: str) -> int:
        return len(text) // 4


@tool
def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"


async def main():
    agent = Agent(backend=EchoBackend(), tools=[greet])
    print(await agent.run("Say hello to Alice"))


if __name__ == "__main__":
    asyncio.run(main())
