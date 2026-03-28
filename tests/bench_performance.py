"""Performance benchmarks for edgeloop.

Measures TTFT, total latency, tokens/sec, and cache behavior.
Run with: python tests/bench_performance.py
"""

import asyncio
import time
import json
from edgeloop import Agent, tool
from edgeloop.backend import OllamaBackend


@tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


async def bench_ttft(model: str, prompt: str, n_runs: int = 3):
    """Measure time-to-first-token using raw backend."""
    import httpx

    times = []
    for _ in range(n_runs):
        async with httpx.AsyncClient(timeout=30.0) as client:
            start = time.time()
            first_token_time = None

            async with client.stream(
                "POST",
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True,
                    "think": False,
                    "options": {"num_predict": 50, "temperature": 0.1},
                },
            ) as response:
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get("response") and first_token_time is None:
                        first_token_time = time.time() - start
                    if chunk.get("done"):
                        total_time = time.time() - start
                        eval_count = chunk.get("eval_count", 0)
                        prompt_eval_count = chunk.get("prompt_eval_count", 0)
                        eval_duration = chunk.get("eval_duration", 1)  # nanoseconds
                        prompt_eval_duration = chunk.get("prompt_eval_duration", 1)
                        break

            tokens_per_sec = (eval_count / (eval_duration / 1e9)) if eval_duration > 0 else 0
            prompt_tokens_per_sec = (prompt_eval_count / (prompt_eval_duration / 1e9)) if prompt_eval_duration > 0 else 0
            times.append({
                "ttft": first_token_time,
                "total": total_time,
                "gen_tps": tokens_per_sec,
                "prompt_tps": prompt_tokens_per_sec,
                "prompt_tokens": prompt_eval_count,
                "gen_tokens": eval_count,
            })

    return times


async def bench_agent_roundtrip(model: str, prompt: str, tools: list, n_runs: int = 3):
    """Measure full agent roundtrip time."""
    times = []
    for _ in range(n_runs):
        agent = Agent(model=model, tools=tools, max_tokens=200, temperature=0.1, max_iterations=5)
        start = time.time()
        response = await agent.run(prompt)
        elapsed = time.time() - start
        times.append({"time": elapsed, "response": response[:100]})
    return times


async def bench_cache_reuse(model: str):
    """Measure if consecutive requests benefit from KV cache."""
    agent = Agent(model=model, tools=[], max_tokens=50, temperature=0.1)

    # First request — cold
    start = time.time()
    r1 = await agent.run("Tell me about Paris.")
    cold = time.time() - start

    # Second request — should reuse context
    start = time.time()
    r2 = await agent.run("Now tell me about London.")
    warm = time.time() - start

    # Third request
    start = time.time()
    r3 = await agent.run("And Tokyo?")
    warmer = time.time() - start

    return {"cold": cold, "warm": warm, "warmer": warmer}


async def main():
    models = ["qwen3:0.6b", "qwen3:1.7b", "qwen2.5-coder:7b"]

    print("=" * 70)
    print("EDGELOOP PERFORMANCE BENCHMARK")
    print("=" * 70)

    # --- Raw TTFT ---
    print("\n--- Time-to-First-Token (raw backend) ---")
    short_prompt = "Hello, how are you?"
    long_prompt = "You are a helpful assistant. " * 50 + "\nUser: What is the meaning of life?"

    for model in models:
        print(f"\n  [{model}]")
        for name, prompt in [("short", short_prompt), ("long", long_prompt)]:
            times = await bench_ttft(model, prompt, n_runs=3)
            avg_ttft = sum(t["ttft"] for t in times if t["ttft"]) / max(len(times), 1)
            avg_gen_tps = sum(t["gen_tps"] for t in times) / len(times)
            avg_prompt_tps = sum(t["prompt_tps"] for t in times) / len(times)
            print(f"    {name:6s}: TTFT={avg_ttft:.3f}s  gen={avg_gen_tps:.0f} tok/s  prefill={avg_prompt_tps:.0f} tok/s  prompt_tokens={times[0]['prompt_tokens']}")

    # --- Agent roundtrip ---
    print("\n--- Agent Roundtrip (with tool call) ---")
    for model in models:
        times = await bench_agent_roundtrip(model, "Add 10 and 20 using the add tool.", [add], n_runs=3)
        avg = sum(t["time"] for t in times) / len(times)
        print(f"  [{model}] avg={avg:.3f}s  response: {times[0]['response']}")

    # --- Cache reuse ---
    print("\n--- Cache Reuse (3 consecutive requests) ---")
    for model in models:
        cache = await bench_cache_reuse(model)
        print(f"  [{model}] cold={cache['cold']:.3f}s  warm={cache['warm']:.3f}s  warmer={cache['warmer']:.3f}s")

    # --- System prompt overhead ---
    print("\n--- System Prompt Overhead ---")
    for model in models:
        # Without tools (short system prompt)
        agent_no_tools = Agent(model=model, tools=[], max_tokens=50, temperature=0.1)
        start = time.time()
        await agent_no_tools.run("Hi")
        no_tools_time = time.time() - start

        # With tools (longer system prompt)
        agent_with_tools = Agent(model=model, tools=[add], max_tokens=50, temperature=0.1)
        start = time.time()
        await agent_with_tools.run("Hi")
        with_tools_time = time.time() - start

        print(f"  [{model}] no_tools={no_tools_time:.3f}s  with_tools={with_tools_time:.3f}s  overhead={with_tools_time-no_tools_time:.3f}s")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
