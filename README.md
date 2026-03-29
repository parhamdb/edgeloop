# edgeloop

Minimal agentic framework for local LLMs. Built for edge devices, optimized for speed.

edgeloop runs agent loops against local models (llama.cpp, Ollama) with KV cache-aware prompt management and output repair for small models. Two dependencies. 1,300 lines of Python. Sub-second tool calls on consumer GPUs.

```python
from edgeloop import Agent, tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

agent = Agent(model="qwen2.5-coder:7b", tools=[calculator])
# await agent.run("What is 847 + 253?")  → calls calculator → "1100"
```

## Why

Cloud agentic frameworks are designed for powerful API models. On local hardware the problems are different:

- **Prefill dominates latency.** Long conversations cause KV cache thrashing. edgeloop uses structured messages, stable prompt prefixes, and append-only history to maximize cache reuse (measured: 14x prefill speedup).
- **Small models break tool calls.** Malformed JSON, wrong argument names, hallucinated tools. edgeloop includes a repair pipeline that fixes common mistakes before execution.
- **Every token counts.** edgeloop uses a compact tool schema format (~40% fewer tokens than JSON schema) and tracks context budget to truncate before overflow.

## Install

```bash
pip install -e .
```

Runtime dependencies: `httpx`, `click`. That's it.

## Usage

### Library

```python
import asyncio
from edgeloop import Agent, tool

@tool
def read_file(path: str) -> str:
    """Read a file from disk."""
    with open(path) as f:
        return f.read()

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

async def main():
    # Ollama backend (default)
    agent = Agent(model="qwen2.5-coder:7b", tools=[read_file, calculator])
    print(await agent.run("What is 123 * 456? Use the calculator."))

    # llama-server backend (2-12x faster)
    agent = Agent(
        endpoint="http://localhost:8080",
        tools=[read_file, calculator],
        template="chatml",
    )
    print(await agent.run("Read pyproject.toml and tell me the version."))

asyncio.run(main())
```

### CLI

```bash
# Ollama
edgeloop chat --endpoint http://localhost:11434 --tools ./my_tools.py

# llama-server
edgeloop chat --endpoint http://localhost:8080 --tools ./my_tools.py --template chatml
```

### Custom backend

Any object with `complete()` and `token_count()` works:

```python
class MyBackend:
    async def complete(self, prompt, stop=None, temperature=0.7,
                      max_tokens=1024, messages=None):
        async for token in my_llm.stream(prompt):
            yield token

    async def token_count(self, text):
        return len(text) // 4

agent = Agent(backend=MyBackend(), tools=[...])
```

## Architecture

```
edgeloop/
├── agent.py              # Agent class, ReAct loop, prompt templates
├── tools.py              # @tool decorator, schema extraction, execution
├── repair.py             # JSON repair, fuzzy matching, type coercion
├── cache.py              # KV cache tracking, context budget, truncation
├── connection.py         # Shared HTTP connection pool
├── cli.py                # Click CLI
└── backends/
    ├── protocol.py       # Backend protocol (2 methods)
    ├── ollama.py         # Ollama /api/chat with KV cache reuse
    └── llama_server.py   # llama-server /completion with slot pinning
```

### Agent loop

```
User message → Build prompt → Send to backend (streaming)
                                      ↓
                               Parse + repair output
                                      ↓
                              Tool call? → Execute → Append result → Loop
                                  │
                                  no → Return response
```

### KV cache strategy

1. **Stable prefix.** System prompt built once at init. Same config = same prefix = cache hit.
2. **Append-only.** New turns append to history. Backend only prefills new tokens at the end.
3. **Structured messages.** Ollama gets `[system, user, assistant, ...]` via `/api/chat` so it can match the prefix and skip cached tokens.
4. **Slot pinning.** llama-server gets `id_slot` + `cache_prompt=true` for explicit prefix-based caching.
5. **Truncation.** When context fills up, oldest turns are dropped while system prompt stays pinned.

### Output repair

Runs on every LLM response. Handles what small models get wrong:

1. **JSON extraction** — markdown fences, XML tags, raw JSON in text
2. **JSON repair** — trailing commas, single quotes, unmatched braces
3. **Fuzzy tool match** — `"red_file"` → `"read_file"` (Levenshtein ≤ 2)
4. **Positional arg mapping** — wrong param names but right values
5. **Type coercion** — `"42"` → `42` when schema says integer
6. **Retry** — if repair fails, inject error and ask the model to try again

## Configuration

```python
Agent(
    # Backend (one of)
    model="qwen2.5-coder:7b",           # → OllamaBackend
    endpoint="http://localhost:8080",     # → LlamaServerBackend
    backend=MyBackend(),                  # → custom

    # Behavior
    tools=[my_tool],
    system_prompt="You are helpful.",
    template="chatml",          # chatml | llama3 | mistral
    max_tokens=4096,            # context budget
    max_iterations=10,          # max tool call loops
    max_retries=2,              # retries on broken output
    temperature=0.7,
    slot_id=0,                  # KV cache slot (llama-server)
    thinking=False,             # enable reasoning mode (Qwen3)
    log_level="WARNING",        # DEBUG | INFO | WARNING | ERROR
)
```

## Cache monitoring

```python
agent = Agent(model="qwen3:1.7b", tools=[...])
await agent.run("Do something with tools")

print(agent.cache.summary())
# {'total_requests': 3, 'cache_hit_ratio': 0.72,
#  'current_context_tokens': 245, 'max_context_tokens': 4096,
#  'last_prefill_ms': 15.0}
```

## Thinking mode

Qwen3 and other reasoning models produce a thinking phase before responding. edgeloop handles this:

```python
agent = Agent(model="qwen3:1.7b", thinking=True, tools=[...])
await agent.run("Complex problem")

# Thinking text available for inspection
print(agent._backend.last_thinking)
```

Thinking is off by default for agentic use — it consumes extra tokens (3-5x slower) without improving tool-calling accuracy. Enable it for reasoning-heavy tasks that don't need tools.

## Performance

Tested on RTX 4070 (12GB):

| Model | Simple | Tool roundtrip | Multi-step |
|-------|--------|---------------|------------|
| qwen3:0.6b (Ollama) | 80ms | 210ms | 290ms |
| qwen3:1.7b (Ollama) | 93ms | 268ms | 437ms |
| qwen2.5-coder:7b (Ollama) | 190ms | 470ms | 1.2s |
| qwen2.5-1.5b (llama-server) | **34ms** | **143ms** | **197ms** |

llama-server is 2-12x faster than Ollama on the same hardware — no middleware overhead.

### Scenario tests (12 scenarios, 4 backends)

| Backend | Pass rate | Avg latency |
|---------|-----------|-------------|
| llama-server 1.5B | 12/12 | 0.305s |
| Ollama 7B | 12/12 | 1.181s |
| Ollama 1.7B | 11/12 | 0.456s |
| Ollama 0.6B | 11/12 | 0.326s |

Scenarios: calculator, chained math, file I/O, JSON extraction, text search, directory listing, error recovery, no-tool-needed, long output, shell commands, multi-tool selection, cross-tool context.

## Development

```bash
git clone https://github.com/parhamdb/edgeloop.git
cd edgeloop
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Unit tests (no GPU needed)
pytest tests/test_*.py -v

# Real model tests (needs Ollama running)
pytest tests/test_real_models.py tests/test_stress.py -v -s

# Scenario tests (needs Ollama + optionally llama-server)
python tests/test_scenarios.py

# Benchmarks
python tests/bench_performance.py
python tests/bench_backends.py
```

## License

MIT
