"""Minimal edgeloop example — agent with tools on a local LLM.

Usage with Ollama:
    ollama pull qwen2.5-coder:7b
    python examples/hello.py

Usage with llama-server:
    llama-server -m model.gguf --port 8080 -ngl 99
    python examples/hello.py --llama
"""

import asyncio
import sys
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
    return "\n".join(sorted(os.listdir(directory)))


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Examples: '2+3', '10*5'."""
    return str(eval(expression))


async def main():
    use_llama = "--llama" in sys.argv

    if use_llama:
        agent = Agent(
            endpoint="http://localhost:8080",
            tools=[read_file, list_files, calculator],
            template="chatml",
            log_level="INFO",
        )
        print("Using llama-server on :8080")
    else:
        agent = Agent(
            model="qwen2.5-coder:7b",
            tools=[read_file, list_files, calculator],
            log_level="INFO",
        )
        print("Using Ollama (qwen2.5-coder:7b)")

    print()
    response = await agent.run(
        "List the files in the current directory. "
        "Then calculate 123 * 456 using the calculator."
    )
    print(f"\nAgent: {response}")
    print(f"\nCache stats: {agent.cache.summary()}")


if __name__ == "__main__":
    asyncio.run(main())
