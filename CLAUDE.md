# edgeloop

Minimal agentic framework for local LLMs — edge-first, KV cache optimized.

## Project structure

```
edgeloop/                     # Source (1,336 lines total)
├── agent.py                  # Agent class, ReAct loop, chat templates (ChatML/Llama3/Mistral)
├── tools.py                  # @tool decorator → JSON schema from type hints, async execution
├── repair.py                 # JSON extraction/repair, fuzzy tool match, positional arg mapping
├── cache.py                  # CacheManager + CacheStats — tracks prefill efficiency, advises truncation
├── connection.py             # Shared HTTP connection pool for backends
├── cli.py                    # Click CLI: `edgeloop chat`
├── backend.py                # Re-exports from backends/
├── __init__.py               # Public API
└── backends/
    ├── protocol.py           # Backend protocol: complete() + token_count() + last_cache_stats
    ├── ollama.py             # Ollama /api/chat — structured messages for KV cache reuse
    └── llama_server.py       # llama-server /completion — cache_prompt + slot pinning

tests/                        # 50 unit tests + real model tests
├── test_tools.py             # @tool schema generation, execution, timeout
├── test_repair.py            # JSON repair, fuzzy match, coercion (22 tests)
├── test_backend.py           # LlamaServerBackend mock tests
├── test_agent.py             # Agent loop with mock backend
├── test_cli.py               # CLI help, tool loading
├── test_real_models.py       # Real GPU tests: basic chat, tool calling, cache
├── test_stress.py            # Multi-step chains, file I/O, rapid fire
├── test_scenarios.py         # 12 real-world scenarios across 4 backends
├── bench_performance.py      # TTFT, throughput, cache efficiency
└── bench_backends.py         # Ollama vs llama-server comparison

examples/
├── hello.py                  # Quick start with Ollama or llama-server
└── custom_backend.py         # How to implement Backend protocol
```

## Dependencies

Runtime: `httpx`, `click`. Dev: `pytest`, `pytest-asyncio`. Nothing else.

## Architecture decisions

- **Single async process.** No multiprocessing, no thread pools beyond tool execution.
- **Backend protocol uses `typing.Protocol`** — no inheritance needed. Any object with `complete()` and `token_count()` satisfies it.
- **Ollama uses `/api/chat` not `/api/generate`** — structured messages enable KV cache reuse across turns (14x prefill speedup measured).
- **System prompt built once at `Agent.__init__`** — deterministic, never changes. Maximizes prefix cache hits.
- **Compact tool schema format** — `name(param:type)` instead of full JSON schema. ~40% fewer tokens.
- **Thinking mode disabled by default** — Qwen3 thinking consumes 50-200 tokens before responding, doesn't improve tool-calling accuracy, and is 3-5x slower. Opt-in via `thinking=True`.

## Key patterns

- Tool calls use `{"tool": "name", "arguments": {"param": "value"}}` JSON format
- Repair pipeline: extract JSON → fix syntax → fuzzy match tool name → coerce types → positional arg fallback
- Agent loop: build prompt → call backend → parse/repair → tool call or return → append to history (never rewrite)
- Each backend reports `CacheStats` after completion; agent's `CacheManager` tracks lifetime metrics

## Running tests

```bash
source .venv/bin/activate

# Unit tests (fast, no GPU)
pytest tests/test_tools.py tests/test_repair.py tests/test_backend.py tests/test_agent.py tests/test_cli.py -v

# Real model tests (needs Ollama with qwen3:0.6b, qwen3:1.7b, qwen2.5-coder:7b)
pytest tests/test_real_models.py tests/test_stress.py -v -s

# Full scenario matrix (needs Ollama + optionally llama-server on :8082)
python tests/test_scenarios.py
```

## Models tested

- qwen3:0.6b — works for single tool calls, fails multi-step chains
- qwen3:1.7b — reliable for most scenarios
- qwen2.5-coder:7b — reliable, best Ollama option
- qwen2.5:14b — reliable, slower
- qwen2.5-0.5b via llama-server — fastest backend, reliable
- qwen2.5-1.5b via llama-server — fastest + reliable

## Performance reference (RTX 4070)

- Ollama 0.6b: 80ms simple, 210ms tool roundtrip
- Ollama 7b: 190ms simple, 470ms tool roundtrip
- llama-server 1.5b: 34ms simple, 143ms tool roundtrip
- llama-server is 2-12x faster than Ollama (no Go middleware overhead)
