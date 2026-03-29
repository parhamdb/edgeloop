"""Quick start — agent with tools on a local LLM.

Ollama:
    ollama pull qwen2.5-coder:7b
    python examples/hello.py

llama-server:
    llama-server -m model.gguf --port 8080 -ngl 99
    python examples/hello.py --llama
"""

import asyncio
import os
import sys
from edgeloop import Agent, tool


@tool
def read_file(path: str) -> str:
    """Read a file from disk."""
    with open(path) as f:
        return f.read()


@tool
def list_files(directory: str = ".") -> str:
    """List files in a directory."""
    return "\n".join(sorted(os.listdir(directory)))


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Examples: '2+3', '100/4'."""
    return str(eval(expression))


async def main():
    if "--llama" in sys.argv:
        agent = Agent(
            endpoint="http://localhost:8080",
            tools=[read_file, list_files, calculator],
            template="chatml",
            log_level="INFO",
        )
        print("Backend: llama-server on :8080\n")
    else:
        agent = Agent(
            model="qwen2.5-coder:7b",
            tools=[read_file, list_files, calculator],
            log_level="INFO",
        )
        print("Backend: Ollama (qwen2.5-coder:7b)\n")

    response = await agent.run(
        "List files in the current directory, then calculate 123 * 456."
    )
    print(f"Agent: {response}\n")
    print(f"Cache: {agent.cache.summary()}")


if __name__ == "__main__":
    asyncio.run(main())
