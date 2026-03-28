"""Backend comparison benchmark: Ollama vs llama-server.

Compares KV cache behavior, latency, and throughput on the same tasks.
Requires:
  - Ollama running on :11434 with qwen3:0.6b
  - llama-server running on :8081 with qwen2.5-0.5b-instruct (similar size)

Run: python tests/bench_backends.py
"""

import asyncio
import time
import json
from edgeloop import Agent, tool
from edgeloop.backends import OllamaBackend, LlamaServerBackend
from edgeloop.connection import Connection


@tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


@tool
def multiply(a: int, b: int) -> str:
    """Multiply two numbers."""
    return str(a * b)


async def bench_single_turn(name, agent, prompt, n_runs=3):
    """Measure single-turn latency."""
    times = []
    for _ in range(n_runs):
        start = time.time()
        response = await agent.run(prompt)
        elapsed = time.time() - start
        times.append(elapsed)
    avg = sum(times) / len(times)
    print(f"  [{name}] avg={avg:.3f}s  (runs: {', '.join(f'{t:.3f}' for t in times)})")
    return avg


async def bench_tool_roundtrip(name, agent, n_runs=3):
    """Measure tool call roundtrip (2 LLM calls: tool call + final answer)."""
    times = []
    for i in range(n_runs):
        a, b = 10 + i * 7, 20 + i * 3
        start = time.time()
        response = await agent.run(f"Add {a} and {b} using the add tool.")
        elapsed = time.time() - start
        expected = str(a + b)
        correct = expected in response
        times.append({"time": elapsed, "correct": correct})
    avg = sum(t["time"] for t in times) / len(times)
    correct_count = sum(1 for t in times if t["correct"])
    print(f"  [{name}] avg={avg:.3f}s  correct={correct_count}/{n_runs}")
    return avg


async def bench_multi_tool(name, agent):
    """Measure multi-step tool chaining."""
    start = time.time()
    response = await agent.run(
        "First add 15 and 25 using the add tool. "
        "Then multiply the result by 3 using the multiply tool."
    )
    elapsed = time.time() - start
    has_120 = "120" in response
    print(f"  [{name}] {elapsed:.3f}s  correct={'120' in response}  response: {response[:100]}")
    return elapsed


async def bench_cache_within_run(name, agent):
    """Measure KV cache benefit within a single agent.run() (multiple iterations)."""
    start = time.time()
    response = await agent.run("Add 10 and 20. Then add 30 and 40. Report both results.")
    elapsed = time.time() - start

    stats = agent.cache.summary()
    print(f"  [{name}] {elapsed:.3f}s  requests={stats['total_requests']}  "
          f"cache_hit={stats['cache_hit_ratio']:.0%}  last_prefill={stats['last_prefill_ms']:.0f}ms")
    return elapsed


async def bench_cache_across_runs(name, make_agent):
    """Measure KV cache benefit across separate agent.run() calls."""
    agent = make_agent()
    times = []
    for i in range(5):
        start = time.time()
        await agent.run(f"What is {i + 1} plus {i + 2}?")
        elapsed = time.time() - start
        times.append(elapsed)

    print(f"  [{name}] runs: {', '.join(f'{t:.3f}' for t in times)}")
    print(f"  [{name}] cold={times[0]:.3f}s  warm_avg={sum(times[1:])/4:.3f}s  speedup={times[0]/max(sum(times[1:])/4, 0.001):.1f}x")
    return times


async def main():
    print("=" * 70)
    print("BACKEND COMPARISON: Ollama vs llama-server")
    print("=" * 70)
    print("Ollama: qwen3:0.6b on :11434")
    print("llama-server: qwen2.5-0.5b-instruct on :8081")
    print()

    def make_ollama(**kwargs):
        return Agent(model="qwen3:0.6b", tools=kwargs.get("tools", []),
                     max_tokens=300, temperature=0.1, max_iterations=8)

    def make_llama(**kwargs):
        return Agent(endpoint="http://localhost:8081", tools=kwargs.get("tools", []),
                     max_tokens=300, temperature=0.1, max_iterations=8, template="chatml")

    # --- Test 1: Simple response (no tools) ---
    print("--- Test 1: Simple response (no tools) ---")
    await bench_single_turn("ollama", make_ollama(), "Say hello in one word.", n_runs=5)
    await bench_single_turn("llama ", make_llama(), "Say hello in one word.", n_runs=5)
    print()

    # --- Test 2: Tool call roundtrip ---
    print("--- Test 2: Tool call roundtrip ---")
    await bench_tool_roundtrip("ollama", make_ollama(tools=[add]), n_runs=5)
    await bench_tool_roundtrip("llama ", make_llama(tools=[add]), n_runs=5)
    print()

    # --- Test 3: Multi-step tool chain ---
    print("--- Test 3: Multi-step tool chain (add then multiply) ---")
    await bench_multi_tool("ollama", make_ollama(tools=[add, multiply]))
    await bench_multi_tool("llama ", make_llama(tools=[add, multiply]))
    print()

    # --- Test 4: KV cache within a run ---
    print("--- Test 4: KV cache efficiency within a run ---")
    await bench_cache_within_run("ollama", make_ollama(tools=[add]))
    await bench_cache_within_run("llama ", make_llama(tools=[add]))
    print()

    # --- Test 5: KV cache across runs ---
    print("--- Test 5: KV cache across separate runs ---")
    await bench_cache_across_runs("ollama", make_ollama)
    await bench_cache_across_runs("llama ", make_llama)
    print()

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
