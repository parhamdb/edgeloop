"""Example: using a custom backend with edgeloop.

Shows how to implement the Backend protocol for any LLM server.
"""

import asyncio
from typing import AsyncIterator
from edgeloop import Agent, tool
from edgeloop.cache import CacheStats


class EchoBackend:
    """A fake backend that echoes tool calls for testing.

    Replace the complete() method with your own LLM server logic.
    """

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
        # A real backend would call your LLM here
        # For demo, just echo back what the user asked
        if messages:
            last_msg = messages[-1]["content"] if messages else prompt
        else:
            last_msg = prompt[-200:]

        response = f"I received your message. You said: {last_msg[:100]}"
        yield response

        self._last_stats = CacheStats(prompt_tokens=len(prompt) // 4)

    async def token_count(self, text: str) -> int:
        return len(text) // 4


@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


async def main():
    agent = Agent(
        backend=EchoBackend(),
        tools=[greet],
    )
    response = await agent.run("Say hello to Alice")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
