"""Shared connection — reuse HTTP connections across agents.

Useful when running multiple agents against the same Ollama server.
"""

import asyncio
from edgeloop import Agent, Connection, OllamaBackend, tool


@tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


async def main():
    # One connection shared by two agents
    async with Connection("http://localhost:11434") as conn:
        fast = OllamaBackend(model="qwen3:0.6b", connection=conn)
        smart = OllamaBackend(model="qwen2.5-coder:7b", connection=conn)

        agent_fast = Agent(backend=fast, tools=[add])
        agent_smart = Agent(backend=smart, tools=[add])

        r1 = await agent_fast.run("Add 10 and 20.")
        print(f"Fast agent:  {r1}")

        r2 = await agent_smart.run("Add 10 and 20.")
        print(f"Smart agent: {r2}")

        print(f"\nTotal requests through shared connection: {conn.request_count}")


if __name__ == "__main__":
    asyncio.run(main())
