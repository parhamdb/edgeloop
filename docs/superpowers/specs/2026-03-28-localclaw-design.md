# edgeloop: Minimal Agentic Framework for Local LLMs

## Context

Cloud agentic frameworks (Strands, OpenClaw, Hermes Agent) waste cycles on local hardware. Prefill dominates latency — long conversations thrash the KV cache and perceived tokens/second tanks. Small local models also botch tool calls regularly.

**edgeloop** is a minimal agentic framework built for this reality. It's a small, focused library (~5 core files) that does one thing well: run an agent loop against a local LLM with maximum efficiency and robust error handling.

**Target:** Embedded systems, low-power devices, edge hardware. Minimum dependencies, minimum RAM, maximum speed.

## Developer Experience

### Library (primary)

```python
from edgeloop import Agent, tool

@tool
def read_file(path: str) -> str:
    """Read a file from disk."""
    return open(path).read()

agent = Agent(
    endpoint="http://localhost:8080",  # llama-server
    tools=[read_file],
)

response = agent.run("What's in README.md?")
```

### CLI

```bash
edgeloop chat --endpoint http://localhost:8080 --tools ./my_tools.py
```

## Architecture

No layers, no abstractions beyond what's needed. Five modules:

```
edgeloop/
├── agent.py      # Agent class + ReAct loop
├── backend.py    # LLM backend (llama-server HTTP, extensible)
├── tools.py      # @tool decorator, schema generation, execution
├── repair.py     # Output parsing, JSON repair, tool-call fixing
├── cli.py        # Terminal chat interface
└── __init__.py   # Public API
```

### Agent Loop

```
User message
    ↓
Build prompt (system + tool schemas + history + message)
    ↓
Send to LLM backend (streaming) ←──────────┐
    ↓                                       │
Parse output → repair if malformed          │
    ↓                                       │
Tool call? ──yes──→ execute → append result─┘
    │
    no
    ↓
Return response
```

Max iterations guard (default 10). Every step logged at DEBUG level.

## Core: KV Cache Efficiency

The framework's reason for existing. Every design decision serves cache efficiency:

1. **Stable prompt prefix**: System prompt + tool definitions are built deterministically. Same agent config = same token prefix = automatic cache hit on llama-server. No randomness, no timestamps, no varying whitespace.

2. **Append-only history**: Never rewrite earlier messages. New turns append to the end. This maximizes prefix overlap with what's already in the KV cache — the backend only prefills new tokens.

3. **Token counting**: Track prompt token count locally. When approaching the model's context limit, truncate oldest turns (keeping system prompt intact) rather than letting the backend overflow and re-prefill from scratch.

4. **Slot reuse hints**: When talking to llama-server, use the `id_slot` parameter to pin conversations to specific cache slots. Avoids cross-conversation cache eviction.

These aren't optional optimizations — they're built into the core prompt builder and can't be turned off.

## Core: Output Repair

Local models (7B, 13B) regularly produce broken tool calls. The repair pipeline runs on every LLM response:

1. **Extract tool calls**: Find JSON blocks in the output. Handle common wrapping (markdown fences, XML tags, natural language around JSON).
2. **JSON repair**: Fix trailing commas, single quotes, unmatched braces, unquoted keys.
3. **Fuzzy tool matching**: If the model writes `"tool": "red_file"` but only `read_file` exists, correct it (edit distance ≤ 2).
4. **Schema coercion**: Cast `"42"` → `42` if schema says integer. Strip extra fields. Add defaults for missing optional fields.
5. **Retry on failure**: If repair fails, inject the error message into context and ask the model to try again. Max 2 retries.

## Core: Backend Interface

Minimal protocol — just what the agent loop needs:

```python
class Backend:
    async def complete(self, messages, tools, slot_id=None) -> AsyncIterator[str]:
        """Stream completion. slot_id hints cache slot to reuse."""

    async def token_count(self, text: str) -> int:
        """Count tokens for context budget tracking."""
```

That's it. Two methods. Default implementation talks to llama-server's `/completion` endpoint.

Adding a new backend (Ollama, vLLM) = implement these two methods.

## Tool System

Tools are Python functions with a decorator:

```python
@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web and return results."""
    # ...
```

The `@tool` decorator:
- Extracts the function signature → generates JSON schema for the LLM
- Extracts the docstring → becomes the tool description
- Wraps execution with timeout and error capture

No YAML configs, no executor types, no plugin system. A tool is a function. Import it, pass it to `Agent(tools=[...])`.

For the CLI, `--tools ./my_tools.py` imports the module and collects all `@tool`-decorated functions.

## Logging

Structured logging via Python's `logging` module (no extra dependency). Every component logs at appropriate levels:

- **INFO**: Agent turn start/end, tool execution, final response
- **DEBUG**: Full prompts sent to LLM, raw LLM output, repair actions taken, token counts, cache slot decisions
- **WARNING**: Repair triggered, retry triggered, token budget exceeded
- **ERROR**: Tool execution failure, backend connection failure

Format: `timestamp level module message` — parseable, greppable. No fancy formatters.

Configure via `EDGELOOP_LOG_LEVEL=DEBUG` env var or `Agent(log_level="DEBUG")`.

## Dependencies

- `httpx` — async HTTP client for llama-server communication
- `click` — CLI (lighter than Typer, no Pydantic dependency)

That's it. Two runtime dependencies. No FastAPI, no Pydantic, no YAML parsers, no async frameworks.

The framework uses `asyncio` from stdlib. `httpx` is the only non-stdlib import in the core library.

## Project Structure

```
edgeloop/
├── pyproject.toml
├── README.md
├── edgeloop/
│   ├── __init__.py      # Public API: Agent, tool, Backend
│   ├── agent.py         # Agent class, ReAct loop, token budgeting
│   ├── backend.py       # Backend protocol + LlamaServerBackend
│   ├── tools.py         # @tool decorator, schema gen, execution
│   ├── repair.py        # JSON repair, fuzzy match, schema coercion
│   └── cli.py           # Click CLI: `edgeloop chat`
├── tests/
│   ├── test_agent.py
│   ├── test_repair.py
│   ├── test_tools.py
│   └── test_backend.py
└── examples/
    └── hello.py         # Minimal working example
```

## Verification

1. **test_repair.py**: Feed known-broken JSON/tool-call outputs, verify repair produces valid results. This is the most critical test — it protects against local model quirks.
2. **test_tools.py**: Verify `@tool` generates correct schemas, handles type coercion, respects timeouts.
3. **test_agent.py**: Mock backend, verify the loop executes tools, respects max iterations, handles retries.
4. **test_backend.py**: Mock HTTP responses from llama-server, verify streaming, slot_id passing, token counting.
5. **Smoke test**: Point at a real llama-server with a small model, run `edgeloop chat`, have a multi-turn conversation with tool use.
