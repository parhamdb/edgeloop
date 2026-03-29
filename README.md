# edgeloop

Minimal agentic framework for local LLMs. Built for edge devices, optimized for speed.

edgeloop runs agent loops against local models (llama.cpp, Ollama) with aggressive KV cache management and output repair for small models. Two dependencies. Five core files. Sub-second tool calls on consumer GPUs.

## Why edgeloop

Cloud agentic frameworks (Strands, LangChain, CrewAI) are designed for powerful API models. On local hardware:

- **Prefill dominates latency** — long conversations thrash the KV cache
- **Small models break tool calls** — malformed JSON, wrong argument names, hallucinated tools
- **Every token counts** — bloated system prompts waste prefill time

edgeloop treats these as first-class problems, not afterthoughts.

## Quick Start

```bash
pip install -e .
```

### As a library

```python
import asyncio
from edgeloop import Agent, tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

async def main():
    # With Ollama
    agent = Agent(model="qwen2.5-coder:7b", tools=[calculator])
    print(await agent.run("What is 847 + 253?"))

    # With llama-server
    agent = Agent(endpoint="http://localhost:8080", tools=[calculator])
    print(await agent.run("What is 847 + 253?"))

asyncio.run(main())
```

### As a CLI

```bash
# Interactive chat with Ollama
edgeloop chat --endpoint http://localhost:11434 --tools ./my_tools.py

# Interactive chat with llama-server
edgeloop chat --endpoint http://localhost:8080 --tools ./my_tools.py --template chatml
```

## Architecture

```
edgeloop/
├── agent.py                # Agent class, ReAct loop, chat templates
├── cache.py                # KV cache tracking and optimization
├── connection.py           # Shared HTTP connection pool
├── tools.py                # @tool decorator, schema generation
├── repair.py               # JSON repair, fuzzy matching, type coercion
├── cli.py                  # Click CLI
└── backends/
    ├── protocol.py         # Backend protocol (2 methods)
    ├── ollama.py           # Ollama /api/chat backend
    └── llama_server.py     # llama-server /completion backend
```

### Agent Loop

```
User message
    ↓
Build prompt (system + tool schemas + history)
    ↓
Send to backend (streaming) ←─────────────┐
    ↓                                      │
Parse output → repair if malformed         │
    ↓                                      │
Tool call? ──yes──→ execute → append ──────┘
    │
    no
    ↓
Return response
```

### KV Cache Strategy

Every design decision serves prefill efficiency:

1. **Stable prompt prefix** — system prompt + tool schemas built once at init, never changes. Same agent config = same prefix = automatic cache hit.
2. **Append-only history** — new turns append to the end. The backend only prefills new tokens.
3. **Structured messages** — Ollama backend uses `/api/chat` (not `/api/generate`) so the server matches the message prefix and skips cached tokens.
4. **Slot pinning** — llama-server backend uses `id_slot` + `cache_prompt=true` for explicit prefix-based caching.
5. **Context truncation** — when approaching the limit, oldest turns are dropped while the system prompt stays pinned.

Measured improvement: **14x prefill speedup** (55ms cold → 4ms warm) on consecutive requests.

### Output Repair

Small models (0.6B-3B) regularly produce broken tool calls. The repair pipeline runs on every response:

1. **JSON extraction** — finds JSON in markdown fences, XML tags, or raw text
2. **JSON repair** — fixes trailing commas, single quotes, unmatched braces
3. **Fuzzy tool matching** — corrects `"red_file"` → `"read_file"` (Levenshtein distance ≤ 2)
4. **Positional arg mapping** — when model uses wrong param names but right values
5. **Type coercion** — casts `"42"` → `42` if schema says integer
6. **Retry with feedback** — if repair fails, injects error into context and retries

## Tools

Tools are plain Python functions with a decorator:

```python
from edgeloop import tool

@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web and return results."""
    # Your implementation here
    return results
```

The `@tool` decorator:
- Extracts function signature → JSON schema for the LLM
- Extracts docstring → tool description
- Wraps execution with timeout and error capture
- Does **not** alter the function's normal behavior — still callable as a regular function

## Backends

### Ollama (recommended for getting started)

```python
agent = Agent(model="qwen2.5-coder:7b")
```

Uses Ollama's `/api/chat` endpoint. Install models with `ollama pull`.

### llama-server (recommended for performance)

```python
agent = Agent(endpoint="http://localhost:8080", template="chatml")
```

2-12x faster than Ollama due to no middleware overhead. Start with:
```bash
llama-server -m model.gguf --port 8080 -ngl 99 -c 4096
```

### Custom backend

Implement two methods:

```python
class MyBackend:
    async def complete(self, prompt, stop=None, temperature=0.7,
                      max_tokens=1024, messages=None):
        async for token in my_llm.stream(prompt):
            yield token

    async def token_count(self, text):
        return len(text) // 4  # or use a real tokenizer

agent = Agent(backend=MyBackend(), tools=[...])
```

### Shared connections

```python
from edgeloop import Connection, OllamaBackend, Agent

conn = Connection("http://localhost:11434", max_keepalive=5)
backend = OllamaBackend(model="qwen2.5-coder:7b", connection=conn)
agent = Agent(backend=backend, tools=[...])
```

## Configuration

```python
Agent(
    # Backend (pick one)
    model="qwen2.5-coder:7b",           # Ollama
    endpoint="http://localhost:8080",     # llama-server
    backend=MyBackend(),                  # Custom

    # Agent behavior
    tools=[my_tool],                      # Tool functions
    system_prompt="You are helpful.",      # Custom system prompt
    template="chatml",                    # Chat template: chatml, llama3, mistral
    max_tokens=4096,                      # Context budget
    max_iterations=10,                    # Max tool call loops
    max_retries=2,                        # Retries on broken output
    temperature=0.7,                      # Sampling temperature
    slot_id=0,                            # Pin to KV cache slot (llama-server)
    log_level="WARNING",                  # DEBUG, INFO, WARNING, ERROR
)
```

## Cache Monitoring

```python
agent = Agent(model="qwen3:1.7b", tools=[...])
await agent.run("Do something with tools")

print(agent.cache.summary())
# {
#   'total_requests': 3,
#   'cache_hit_ratio': 0.72,
#   'current_context_tokens': 245,
#   'max_context_tokens': 4096,
#   'last_prefill_ms': 15.0,
# }
```

## Performance

Tested on RTX 4070 (12GB VRAM):

### Latency by model size

| Model | Simple response | Tool roundtrip | Multi-step |
|-------|----------------|---------------|------------|
| qwen3:0.6b (Ollama) | 80ms | 210ms | 290ms |
| qwen3:1.7b (Ollama) | 93ms | 268ms | 437ms |
| qwen2.5-coder:7b (Ollama) | 190ms | 470ms | 1.2s |
| qwen2.5-1.5b (llama-server) | **34ms** | **143ms** | **197ms** |

### Backend comparison (similar model sizes)

| Metric | Ollama 1.7B | llama-server 1.5B | Speedup |
|--------|-------------|-------------------|---------|
| Simple response | 93ms | 34ms | 2.7x |
| Tool roundtrip | 268ms | 143ms | 1.9x |
| Cold start | 682ms | 55ms | 12x |

### Scenario test results (12 scenarios, 4 backends)

| Backend | Pass Rate | Avg Latency |
|---------|-----------|-------------|
| llama-server 1.5B | **12/12** | **0.305s** |
| Ollama 7B | **12/12** | 1.181s |
| Ollama 1.7B | 11/12 | 0.456s |
| Ollama 0.6B | 11/12 | 0.326s |

## Dependencies

- `httpx` — async HTTP client
- `click` — CLI

That's it. No LangChain, no Pydantic, no YAML parsers, no async frameworks.

## Development

```bash
git clone https://github.com/parhamdb/edgeloop.git
cd edgeloop
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
