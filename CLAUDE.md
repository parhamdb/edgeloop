# edgeloop

Rust binary — minimal agentic framework for local LLMs. Config-driven, tools as CLI commands, feature-gated.

## Project structure

```
src/                                    # ~1,700 lines Rust
├── main.rs                             # clap CLI, config load, wire agent + transports
├── lib.rs                              # Public module exports for integration tests
├── config.rs                           # TOML structs, ${VAR:-default} env expansion, includes, tool loading
├── agent.rs                            # ReAct loop, parallel tool calls, 4 chat templates, truncation
├── repair.rs                           # JSON extract/fix, Levenshtein fuzzy match, positional arg mapping
├── tool.rs                             # sh -c subprocess executor, {arg} substitution, timeout
├── cache.rs                            # CacheStats + CacheManager — prefill tracking, truncation at 80%
├── message.rs                          # Message, ImageAttachment, OutputEvent, IncomingRequest
├── backend/
│   ├── mod.rs                          # Backend trait (stream_completion + token_count + last_cache_stats)
│   ├── openai_compat.rs               # Shared wire types for OpenAI-compatible APIs (multimodal)
│   ├── ollama.rs                       # /api/chat NDJSON streaming, thinking mode, native images
│   ├── llama_server.rs                 # /completion SSE, cache_prompt + id_slot, /tokenize
│   ├── openai.rs                       # /v1/chat/completions SSE, multimodal content-array
│   └── vllm.rs                         # vLLM backend: guided decoding, prefix cache stats, multimodal
└── transport/
    ├── mod.rs                          # Transport trait (serve + name), factory
    ├── cli.rs                          # stdin/stdout REPL
    ├── websocket.rs                    # tokio-tungstenite, JSON frames
    ├── mqtt.rs                         # rumqttc pub/sub, reconnection
    └── socket.rs                       # Unix domain + TCP, newline-delimited JSON

tests/
├── integration_test.rs                 # Real Ollama: chat, tool call, file read, no-tool, perf
└── benchmark.rs                        # Latency across 3 models, tool roundtrip, rapid fire

tools/                                  # Example tool packages
├── filesystem/tools.toml               # read_file, write_file, list_dir
└── system/tools.toml                   # shell, find_files

examples/                               # Example configs
├── home-automation.toml                # MQTT + WebSocket + Ollama
├── minimal-openwrt.toml                # CLI + remote llama-server
├── cloud-openai.toml                   # CLI + WebSocket + OpenAI API
├── local-vllm.toml                     # CLI + vLLM with guided decoding
└── local-gemma4.toml                   # CLI + Ollama + Gemma 4 26B-A4B
```

## Building

```bash
cargo build                             # debug, default features
cargo build --release --features full   # all backends + transports (5.0MB)
cargo build --release                   # default: ollama + llama-server + cli (4.4MB)
cargo test --bin edgeloop               # 71 unit tests
cargo test --test integration_test      # 5 integration tests (needs Ollama)
cargo test --test benchmark -- --nocapture  # performance benchmarks
```

## Feature flags

default: ollama, llama-server, cli-transport
Backends: ollama, llama-server, openai, vllm
Transports: cli-transport, websocket, mqtt, unix-socket, tcp-socket

## Key patterns

- Tools are `sh -c` subprocesses. `{param}` substitution in command template from TOML.
- Backend trait: `stream_completion()` returns `BoxStream<Result<String>>`, `token_count()`, `last_cache_stats()`.
- Transport trait: `serve(handler)` — handler is `Arc<dyn Fn(TransportRequest)>`.
- Non-CLI transports use JSON protocol: `{"message":"...","session":"..."}` → `{"type":"done","content":"...","session":"..."}`.
- Multimodal images: incoming requests can include `"images": [{"b64":"...","description":"...","mime_type":"image/jpeg"}]`. Images flow transport → agent → backend. OpenAI/vLLM use content-array format (`image_url` parts), Ollama uses native `images` field, llama-server degrades to text descriptions. Shared wire types in `backend/openai_compat.rs`.
- Repair pipeline: `repair_tool_calls()` handles single `{...}` and array `[{...}, ...]`; delegates to `parse_single_tool_call()`. Fuzzy match via Levenshtein, positional arg coercion. Legacy `repair_tool_call()` returns the first result.
- Agent loop: build prompt → stream backend → repair → tool(s) or return. Append-only history.
- Parallel tool calls: opt-in via `parallel_tools = true` in `[agent]`. LLM emits a JSON array; tools execute concurrently via `tokio::task::JoinSet`; all results are batched into one user message. Recommended for 7B+ models. Default is false.
- All backends use `reqwest` with `rustls-tls` — no OpenSSL, clean static linking.

## Performance (RTX 4070, Ollama)

- qwen3:0.6b: 93ms simple, 858ms tool roundtrip
- qwen2.5-coder:7b: 164ms simple, 694ms tool roundtrip
- WebSocket end-to-end: 117ms

## Dependencies

Runtime: tokio, reqwest (rustls), serde/serde_json, toml, clap, regex, tracing, async-trait, futures, anyhow
Optional: tokio-tungstenite (websocket), rumqttc (mqtt)
