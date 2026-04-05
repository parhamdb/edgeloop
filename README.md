# edgeloop

Minimal agentic framework for local LLMs. Single static binary, config-driven, tools as CLI commands.

Deploy a 5MB binary + a TOML config to any edge device. Talks to Ollama, llama-server, or any OpenAI-compatible API. Executes tools as shell commands. Streams responses over CLI, WebSocket, MQTT, or Unix/TCP sockets.

## Quick Start

```bash
cargo build --release --features full
./target/release/edgeloop --config edgeloop.toml
```

## How It Works

1. You write a TOML config: which LLM backend, which tools (as shell commands), which I/O transports
2. edgeloop starts, loads config, connects to the LLM, listens on configured transports
3. Messages come in (CLI, WebSocket, MQTT, socket), the agent loop calls the LLM, executes tools, returns responses
4. The repair pipeline fixes broken JSON from small models automatically

## Config

```toml
transports = ["cli"]
tool_packages = ["tools/filesystem"]

[agent]
system_prompt = "You are a helpful assistant."
template = "chatml"          # chatml | llama3 | mistral
max_tokens = 4096
temperature = 0.7
stream_tokens = true         # stream tokens to transport as they arrive (for TTS)

[backend]
type = "ollama"              # ollama | llama-server | openai
endpoint = "http://localhost:11434"
model = "qwen2.5-coder:7b"

[transport.cli]
prompt = "you> "
```

See `examples/` for more configs: home automation (MQTT+WS), OpenWrt (minimal), cloud (OpenAI).

## Tools

Tools are shell commands in TOML. No SDK, no plugins, no code.

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

Parameters are substituted into the command template. stdout is returned to the agent.

## Backends

| Backend | Config `type` | What it connects to |
|---------|--------------|-------------------|
| **Ollama** | `ollama` | Local/remote Ollama server |
| **llama-server** | `llama-server` | llama.cpp HTTP server with KV cache slot pinning + multimodal |
| **vLLM** | `vllm` | vLLM server with guided decoding and prefix caching |
| **OpenAI** | `openai` | OpenAI, Together, Groq, OpenRouter, any compatible API |

```toml
# Remote Ollama
[backend]
type = "ollama"
endpoint = "http://192.168.1.50:11434"
model = "qwen2.5-coder:7b"

# llama-server with slot pinning
[backend]
type = "llama-server"
endpoint = "http://localhost:8080"
slot_id = 0

# OpenAI-compatible
[backend]
type = "openai"
endpoint = "https://api.openai.com/v1"
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"
```

## Transports

| Transport | Config name | Protocol |
|-----------|-----------|----------|
| **CLI** | `cli` | Plain text stdin/stdout |
| **WebSocket** | `websocket` | JSON frames |
| **MQTT** | `mqtt` | JSON on pub/sub topics |
| **Unix socket** | `unix` | Newline-delimited JSON |
| **TCP socket** | `tcp` | Newline-delimited JSON |

Multiple transports run simultaneously:
```toml
transports = ["cli", "websocket", "mqtt"]
```

JSON protocol (WebSocket/MQTT/sockets):
```json
→ {"message": "What is 2+2?", "session": "abc"}
← {"type": "done", "content": "4", "session": "abc"}
```

Multimodal messages (images — inline, file path, or URL):
```json
→ {"message": "What do you see?", "session": "abc", "images": [
    {"b64": "/9j/4AAQ...", "description": "Inline base64"},
    {"path": "/tmp/photo.jpg", "description": "Local file"},
    {"url": "http://127.0.0.1:8080/image/42", "description": "HTTP reference"}
  ]}
```

Each image needs one of `b64`, `path`, or `url`. Edgeloop resolves file/URL references at request time (reads file or HTTP GET, base64 encodes). Optional fields: `description`, `mime_type` (inferred from extension/content-type if omitted, defaults to `image/jpeg`). Supported by all backends: Ollama (native `images` field), OpenAI/vLLM (content-array format), and llama-server (`image_data` on `/completion` — requires `--mmproj` flag).

## Token Streaming

Enable `stream_tokens = true` in `[agent]` to stream tokens as they arrive from the LLM backend. Useful for TTS or real-time UI updates.

```toml
[agent]
stream_tokens = true
```

With streaming enabled, transports emit progressive events before the final `done`:
```json
← {"type": "token", "content": "I saw", "session": "abc"}
← {"type": "token", "content": " your mug", "session": "abc"}
← {"type": "token", "content": " near the chair.", "session": "abc"}
← {"type": "done", "content": "I saw your mug near the chair.", "session": "abc"}
```

During tool execution, `tool_call` and `tool_result` events are also streamed:
```json
← {"type": "tool_call", "tool": "find_object", "arguments": {"name": "mug"}, "session": "abc"}
← {"type": "tool_result", "tool": "find_object", "result": "near the chair", "session": "abc"}
```

Token events use non-blocking `try_send` — if a transport consumer is slow, tokens are dropped rather than blocking the LLM stream. The `done` event always contains the complete response.

## Feature Flags

Compile only what you need:

```bash
# Default (ollama + llama-server + cli)
cargo build --release

# Full (all backends + transports) — 5.0MB
cargo build --release --features full

# Minimal for OpenWrt
cargo build --release --no-default-features --features "llama-server,cli-transport"

# Home automation
cargo build --release --no-default-features --features "ollama,mqtt,websocket"
```

## Config Features

- **Env var expansion**: `${VAR}` or `${VAR:-default}` in any string value
- **Config includes**: `include = ["secrets.toml"]` — merge multiple TOML files
- **Tool packages**: `tool_packages = ["tools/filesystem", "tools/custom"]` — modular tool sets

## Performance

Tested on RTX 4070 with Ollama:

| Model | Simple response | Tool roundtrip |
|-------|----------------|---------------|
| qwen3:0.6b | 93ms (warm) | 858ms |
| qwen3:1.7b | 160ms (warm) | 890ms |
| qwen2.5-coder:7b | 164ms (warm) | 694ms |

WebSocket roundtrip: 117ms end-to-end.

## Architecture

```
src/
├── main.rs              # CLI entry, config → agent → transports
├── config.rs            # TOML parsing, env vars, includes, tool packages
├── agent.rs             # ReAct loop, token streaming, chat templates, history truncation
├── repair.rs            # JSON repair, fuzzy match, type coercion
├── tool.rs              # Subprocess executor, arg substitution, timeout
├── cache.rs             # KV cache tracking, context budget
├── message.rs           # Message, ImageAttachment, OutputEvent types
├── backend/
│   ├── mod.rs           # Backend trait
│   ├── openai_compat.rs # Shared multimodal wire types
│   ├── ollama.rs        # /api/chat NDJSON streaming, native images
│   ├── llama_server.rs  # /completion SSE streaming + slot pinning + image_data
│   ├── openai.rs        # /v1/chat/completions SSE, multimodal
│   └── vllm.rs          # vLLM: guided decoding, prefix cache, multimodal
└── transport/
    ├── mod.rs           # Transport trait
    ├── cli.rs           # stdin/stdout REPL
    ├── websocket.rs     # tokio-tungstenite
    ├── mqtt.rs          # rumqttc
    └── socket.rs        # Unix + TCP
```

## Cross-Compilation

```bash
cargo install cross

# ARM64 (Pi 4/5, Jetson)
cross build --release --target aarch64-unknown-linux-musl --features full

# MIPS (OpenWrt)
cross build --release --target mips-unknown-linux-musl --no-default-features --features "llama-server,cli-transport"
```

All targets: fully static musl binaries.

## Development

```bash
cargo build                          # debug build
cargo test --bin edgeloop            # unit tests (85)
cargo test --test integration_test   # integration tests (5, needs Ollama)
cargo test --test full_test          # full tests (10, needs llama-server/Ollama)
cargo test --test benchmark -- --nocapture  # benchmarks (needs Ollama)
cargo build --release --features full       # release binary
```

## License

MIT
