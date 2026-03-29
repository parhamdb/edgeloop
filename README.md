# edgeloop

Minimal agentic framework for local LLMs. Single static binary, config-driven, tools as CLI commands.

Deploy a 2.6MB binary + a TOML config to any edge device. It talks to llama-server, Ollama, or any OpenAI-compatible API, executes tools as shell commands, and streams responses over CLI, WebSocket, MQTT, or Unix/TCP sockets.

## Quick Start

```bash
cargo build --release
./target/release/edgeloop --config edgeloop.toml
```

## Config

Everything is in TOML. No code needed.

```toml
transports = ["cli"]
tool_packages = ["tools/filesystem"]

[agent]
system_prompt = "You are a helpful assistant."
template = "chatml"          # chatml | llama3 | mistral
max_tokens = 4096
temperature = 0.7

[backend]
type = "ollama"              # ollama | llama-server | openai
endpoint = "http://localhost:11434"
model = "qwen2.5-coder:7b"

[transport.cli]
prompt = "you> "
```

## Tools as CLI Commands

Tools are shell commands declared in TOML. No SDK, no plugins.

```toml
# tools/filesystem/tools.toml
[[tools]]
name = "read_file"
description = "Read a file from disk"
command = "cat {path}"
[tools.parameters]
path = { type = "string", required = true }

[[tools]]
name = "shell"
description = "Run a shell command"
command = "sh -c '{command}'"
timeout = 30
[tools.parameters]
command = { type = "string", required = true }
```

The agent substitutes `{param}` in the command, spawns `sh -c`, captures stdout.

## Feature Flags

Compile only what you need:

```bash
# Full build (all backends + transports)
cargo build --release --features full

# Minimal for OpenWrt (~1.5MB)
cargo build --release --no-default-features --features "llama-server,cli-transport"

# Home automation
cargo build --release --no-default-features --features "ollama,mqtt,websocket"
```

| Feature | What it adds |
|---------|-------------|
| `ollama` | Ollama /api/chat backend |
| `llama-server` | llama-server /completion backend |
| `openai` | OpenAI-compatible API backend |
| `cli-transport` | Interactive terminal REPL |
| `websocket` | WebSocket server transport |
| `mqtt` | MQTT pub/sub transport |
| `unix-socket` | Unix domain socket transport |
| `tcp-socket` | TCP socket transport |

## Architecture

```
src/
├── main.rs              # CLI entry, config load, wire everything
├── config.rs            # TOML parsing, env var expansion, includes
├── agent.rs             # ReAct loop, chat templates, prompt building
├── repair.rs            # JSON repair, fuzzy matching, type coercion
├── tool.rs              # Subprocess executor, arg substitution
├── cache.rs             # KV cache tracking, context budget
├── message.rs           # Message, OutputEvent types
├── backend/
│   ├── mod.rs           # Backend trait + factory
│   ├── ollama.rs        # Ollama /api/chat (stub — Plan 2)
│   ├── llama_server.rs  # llama-server /completion (stub — Plan 2)
│   └── openai.rs        # OpenAI-compatible (stub — Plan 2)
└── transport/
    ├── mod.rs           # Transport trait + factory
    ├── cli.rs           # Interactive stdin/stdout
    ├── websocket.rs     # (stub — Plan 3)
    ├── mqtt.rs          # (stub — Plan 3)
    └── socket.rs        # (stub — Plan 3)
```

### Agent Loop

```
Message in (from transport)
  → Build prompt (system + tool schemas + history)
  → Apply chat template (ChatML/Llama3/Mistral)
  → Stream from backend
  → Repair pipeline (extract JSON → fix syntax → fuzzy match → coerce types)
  → Tool call? → spawn subprocess → append result → loop
  → Plain text? → return to transport
```

### Output Repair

Small local models produce broken tool calls. The repair pipeline fixes them:

1. **JSON extraction** — markdown fences, XML tags, raw brace matching
2. **JSON repair** — single quotes, trailing commas, unmatched braces
3. **Fuzzy tool match** — Levenshtein distance ≤ 2
4. **Positional arg mapping** — wrong param names but right values
5. **Type coercion** — `"42"` → `42` when schema says integer

### KV Cache

- System prompt built once (stable prefix for cache hits)
- History append-only (maximizes prefix overlap)
- Structured messages for Ollama (server-side cache reuse)
- Slot pinning for llama-server
- Auto-truncation at 80% context budget

## Binary Size

| Build | Size |
|-------|------|
| Default (ollama + llama-server + cli) | 2.6MB |
| Minimal (llama-server + cli only) | ~1.5MB |
| Full (all features) | ~5MB |

## Cross-Compilation

```bash
# Install cross
cargo install cross

# ARM64 (Pi 4/5, Jetson)
cross build --release --target aarch64-unknown-linux-musl

# ARMv7 (Pi 3)
cross build --release --target armv7-unknown-linux-musleabihf

# MIPS (OpenWrt routers)
cross build --release --target mips-unknown-linux-musl --no-default-features --features "llama-server,cli-transport"
```

All targets produce fully static musl binaries — no libc dependency.

## Development

```bash
cargo build           # debug build
cargo test            # run all tests (34 tests)
cargo build --release # optimized binary
```

## Status

**Plan 1 (Core) — Complete:**
- Config loading with env vars and includes
- Repair pipeline (JSON fix, fuzzy match, coercion)
- Cache manager with truncation
- Tool executor (subprocess + timeout)
- Agent loop with chat templates
- CLI transport
- Backend trait with mock for testing

**Plan 2 (Backends) — Next:**
- Ollama /api/chat with streaming
- llama-server /completion with SSE
- OpenAI-compatible API

**Plan 3 (Transports) — After Plan 2:**
- WebSocket, MQTT, Unix/TCP sockets

**Plan 4 (Cross-compile + CI) — After Plan 3:**
- Cross.toml, Makefile, feature flag matrix

## License

MIT
