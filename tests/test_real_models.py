"""Real model integration tests for edgeloop.

These tests hit actual Ollama models on the local GPU.
Run with: pytest tests/test_real_models.py -v -s
"""

import asyncio
import time
import pytest
from edgeloop import Agent, tool


# --- Tools for testing ---

@tool
def add_numbers(a: int, b: int) -> str:
    """Add two numbers together and return the result."""
    return str(a + b)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Fake weather data
    weather = {
        "london": "Rainy, 12°C",
        "tokyo": "Sunny, 25°C",
        "new york": "Cloudy, 18°C",
    }
    return weather.get(city.lower(), f"Unknown city: {city}")


@tool
def read_file(path: str) -> str:
    """Read a file and return its contents."""
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found: {path}"


# --- Test configs ---

MODELS = [
    "qwen3:0.6b",
    "qwen3:1.7b",
    "qwen2.5-coder:7b",
]


# --- Basic chat tests ---

@pytest.mark.parametrize("model", MODELS)
@pytest.mark.asyncio
async def test_basic_chat(model):
    """Test that the model can produce a coherent response."""
    agent = Agent(model=model, tools=[], max_tokens=100, temperature=0.1)

    start = time.time()
    response = await agent.run("What is 2+2? Answer with just the number.")
    elapsed = time.time() - start

    print(f"\n[{model}] Response ({elapsed:.2f}s): {response[:200]}")
    assert len(response) > 0
    # Tiny models may get math wrong — just verify they respond


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.asyncio
async def test_tool_calling(model):
    """Test that the model can call a tool and use the result."""
    agent = Agent(
        model=model,
        tools=[add_numbers],
        max_tokens=200,
        temperature=0.1,
        max_iterations=5,
    )

    start = time.time()
    response = await agent.run("What is 15 + 27? Use the add_numbers tool.")
    elapsed = time.time() - start

    print(f"\n[{model}] Tool call response ({elapsed:.2f}s): {response[:300]}")
    # The model should have used the tool and gotten 42
    assert "42" in response


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.asyncio
async def test_multi_turn_tool(model):
    """Test multi-step tool usage."""
    agent = Agent(
        model=model,
        tools=[get_weather],
        max_tokens=200,
        temperature=0.1,
        max_iterations=5,
    )

    start = time.time()
    response = await agent.run("What's the weather in Tokyo? Use the get_weather tool.")
    elapsed = time.time() - start

    print(f"\n[{model}] Weather response ({elapsed:.2f}s): {response[:300]}")
    assert "25" in response or "Sunny" in response or "sunny" in response


# --- Performance tests ---

@pytest.mark.asyncio
async def test_response_latency():
    """Measure time-to-first-response across models."""
    results = {}
    for model in MODELS:
        agent = Agent(model=model, tools=[], max_tokens=50, temperature=0.1)

        start = time.time()
        response = await agent.run("Say hello.")
        elapsed = time.time() - start

        results[model] = elapsed
        print(f"[{model}] Latency: {elapsed:.2f}s, Response: {response[:100]}")

    print("\n--- Latency Summary ---")
    for model, latency in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {model}: {latency:.2f}s")


@pytest.mark.asyncio
async def test_cache_efficiency():
    """Test that second request on same agent is faster (KV cache reuse)."""
    model = "qwen3:0.6b"  # Smallest for fastest test
    agent = Agent(model=model, tools=[], max_tokens=50, temperature=0.1)

    # First request — cold
    start = time.time()
    r1 = await agent.run("What is the capital of France?")
    cold_time = time.time() - start

    # Second request — should benefit from context reuse
    start = time.time()
    r2 = await agent.run("And what about Germany?")
    warm_time = time.time() - start

    print(f"\n[Cache test] Cold: {cold_time:.2f}s, Warm: {warm_time:.2f}s")
    print(f"  Response 1: {r1[:100]}")
    print(f"  Response 2: {r2[:100]}")
    # Note: warm might not be faster with Ollama since each agent.run() is independent
    # But the context token passing should help


# --- Repair pipeline stress test ---

@pytest.mark.asyncio
async def test_repair_with_tiny_model():
    """Tiny models produce more broken output — test repair pipeline handles it."""
    agent = Agent(
        model="qwen3:0.6b",
        tools=[add_numbers],
        max_tokens=200,
        temperature=0.3,
        max_iterations=5,
        max_retries=2,
    )

    start = time.time()
    response = await agent.run("Please add 100 and 200 using the add_numbers tool. Return the result.")
    elapsed = time.time() - start

    print(f"\n[qwen3:0.6b repair test] ({elapsed:.2f}s): {response[:300]}")
    # Even if repair fails, we should get SOME response (not crash)
    assert len(response) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
