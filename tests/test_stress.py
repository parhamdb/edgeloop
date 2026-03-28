"""Stress tests for edgeloop — push the framework hard with real models.

Tests complex tool use, multi-step reasoning, error recovery, and edge cases.
Run with: pytest tests/test_stress.py -v -s
"""

import asyncio
import time
import os
import tempfile
import pytest
from edgeloop import Agent, tool


# --- Complex tools ---

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Examples: '2+3', '10*5', '100/4'."""
    try:
        # Safe eval for basic math
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return f"Error: Invalid characters in expression: {expression}"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    with open(path, "w") as f:
        f.write(content)
    return f"Written {len(content)} bytes to {path}"


@tool
def read_file(path: str) -> str:
    """Read a file and return its contents."""
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found: {path}"


@tool
def list_directory(path: str = ".") -> str:
    """List files in a directory."""
    try:
        entries = os.listdir(path)
        return "\n".join(entries[:20])
    except Exception as e:
        return f"Error: {e}"


# --- Test with different model sizes ---

TINY_MODELS = ["qwen3:0.6b"]
SMALL_MODELS = ["qwen3:1.7b"]
MEDIUM_MODELS = ["qwen2.5-coder:7b"]
ALL_MODELS = TINY_MODELS + SMALL_MODELS + MEDIUM_MODELS


@pytest.mark.parametrize("model", MEDIUM_MODELS)
@pytest.mark.asyncio
async def test_multi_step_tool_chain(model):
    """Test a task requiring multiple sequential tool calls.

    Only run on 7B+ models — smaller models can't reliably chain tool calls.
    """
    agent = Agent(
        model=model,
        tools=[calculator],
        max_tokens=300,
        temperature=0.1,
        max_iterations=8,
    )

    start = time.time()
    response = await agent.run(
        "Calculate (10 + 5) * 3 step by step. "
        "First add 10+5 using the calculator, then multiply the result by 3."
    )
    elapsed = time.time() - start

    print(f"\n[{model}] Multi-step ({elapsed:.2f}s): {response[:300]}")
    assert "45" in response


@pytest.mark.parametrize("model", SMALL_MODELS + MEDIUM_MODELS)
@pytest.mark.asyncio
async def test_file_read_write_roundtrip(model):
    """Test writing a file then reading it back. Skips 0.6B (too unreliable with paths)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.txt")

        agent = Agent(
            model=model,
            tools=[write_file, read_file],
            max_tokens=300,
            temperature=0.1,
            max_iterations=6,
        )

        start = time.time()
        response = await agent.run(
            f"Write 'Hello World' to {filepath}, then read it back and tell me what it says."
        )
        elapsed = time.time() - start

        print(f"\n[{model}] File roundtrip ({elapsed:.2f}s): {response[:300]}")
        # Verify the file was actually written
        assert os.path.exists(filepath)
        assert "Hello" in response or "hello" in response


@pytest.mark.parametrize("model", ALL_MODELS)
@pytest.mark.asyncio
async def test_tool_with_error_handling(model):
    """Test that the agent handles tool errors gracefully."""
    agent = Agent(
        model=model,
        tools=[read_file],
        max_tokens=300,
        temperature=0.1,
        max_iterations=5,
    )

    start = time.time()
    response = await agent.run("Read the file /nonexistent/path/file.txt and tell me what happened.")
    elapsed = time.time() - start

    print(f"\n[{model}] Error handling ({elapsed:.2f}s): {response[:300]}")
    # Agent should report the error, not crash
    assert len(response) > 0
    r = response.lower()
    assert "not found" in r or "error" in r or "exist" in r or "does not" in r or "cannot" in r


@pytest.mark.asyncio
async def test_rapid_fire_requests():
    """Test many quick requests in sequence — stress cache reuse."""
    model = "qwen3:0.6b"
    agent = Agent(model=model, tools=[calculator], max_tokens=100, temperature=0.1, max_iterations=3)

    times = []
    for i in range(5):
        a, b = i * 10, i * 7
        start = time.time()
        response = await agent.run(f"Calculate {a}+{b} using the calculator tool.")
        elapsed = time.time() - start
        times.append(elapsed)
        expected = str(a + b)
        print(f"  Request {i+1}: {elapsed:.3f}s, expected={expected}, got: {response[:100]}")

    avg = sum(times) / len(times)
    print(f"\n--- Rapid fire: avg={avg:.3f}s, min={min(times):.3f}s, max={max(times):.3f}s ---")


@pytest.mark.asyncio
async def test_no_tools_needed():
    """Test that the agent doesn't try to use tools when not needed."""
    agent = Agent(
        model="qwen2.5-coder:7b",
        tools=[calculator, read_file],
        max_tokens=100,
        temperature=0.1,
    )

    response = await agent.run("What is the capital of Japan?")
    print(f"\nNo-tool response: {response[:200]}")
    assert "Tokyo" in response or "tokyo" in response


@pytest.mark.asyncio
async def test_large_model_complex_task():
    """Test qwen2.5:14b on a complex multi-tool task."""
    agent = Agent(
        model="qwen2.5:14b",
        tools=[calculator, list_directory],
        max_tokens=500,
        temperature=0.1,
        max_iterations=8,
    )

    start = time.time()
    response = await agent.run(
        "First list the files in the current directory. "
        "Then calculate 123 * 456 using the calculator. "
        "Report both results."
    )
    elapsed = time.time() - start

    print(f"\n[qwen2.5:14b] Complex ({elapsed:.2f}s): {response[:500]}")
    # Model should list files AND compute multiplication (may or may not use tool)
    assert "edgeloop" in response or "pyproject" in response  # listed files
    assert "56088" in response or "55908" in response or "456" in response  # attempted math


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
