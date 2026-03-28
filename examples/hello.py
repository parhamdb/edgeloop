"""Minimal edgeloop example.

Usage:
    1. Start llama-server: llama-server -m model.gguf --port 8080
    2. Run: python examples/hello.py
"""

import asyncio
from edgeloop import Agent, tool


@tool
def read_file(path: str) -> str:
    """Read a file from disk and return its contents."""
    with open(path) as f:
        return f.read()


@tool
def list_files(directory: str = ".") -> str:
    """List files in a directory."""
    import os
    entries = os.listdir(directory)
    return "\n".join(entries)


async def main():
    agent = Agent(
        endpoint="http://localhost:8080",
        tools=[read_file, list_files],
        template="chatml",
        log_level="INFO",
    )

    response = await agent.run("List the files in the current directory, then read README.md if it exists.")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
