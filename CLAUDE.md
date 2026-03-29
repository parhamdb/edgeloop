# edgeloop

Rust binary — minimal agentic framework for local LLMs. Config-driven, tools as CLI commands, feature-gated backends and transports.

## Project structure

```
src/                              # ~800 lines Rust
├── main.rs                       # CLI (clap), config load, wire agent + transport
├── config.rs                     # TOML structs, env var expansion ${VAR:-default}, includes, tool package loading
├── agent.rs                      # ReAct loop, 3 chat templates (ChatML/Llama3/Mistral), truncation
├── repair.rs                     # JSON extract/fix, Levenshtein fuzzy match, positional arg mapping, type coercion
├── tool.rs                       # Subprocess executor — substitute {args} in command, spawn sh -c, timeout
├── cache.rs                      # CacheStats + CacheManager — prefill tracking, truncation threshold
├── message.rs                    # Message (role+content), OutputEvent (token/tool_call/done/error), IncomingRequest
├── backend/
│   ├── mod.rs                    # Backend trait (stream_completion + token_count + last_cache_stats), MockBackend, factory
│   ├── ollama.rs                 # Stub — Plan 2
│   ├── llama_server.rs           # Stub — Plan 2
│   └── openai.rs                 # Stub — Plan 2
└── transport/
    ├── mod.rs                    # Transport trait (serve + name), factory, TransportRequest/RequestHandler
    ├── cli.rs                    # Interactive stdin/stdout REPL
    ├── websocket.rs              # Stub — Plan 3
    ├── mqtt.rs                   # Stub — Plan 3
    └── socket.rs                 # Stub — Plan 3

tools/                            # Example tool packages (TOML)
├── filesystem/tools.toml         # read_file, write_file, list_dir
└── system/tools.toml             # shell, find_files

edgeloop.toml                     # Example main config
Cargo.toml                        # Features: ollama, llama-server, openai, cli-transport, websocket, mqtt, unix-socket, tcp-socket
```

## Key patterns

- **Tools are CLI commands.** Defined in TOML, executed via `sh -c`, args substituted with `{param}` templating.
- **Feature flags gate backends and transports.** Core (agent, repair, config, tool) always compiled. `cargo build --no-default-features --features "llama-server,cli-transport"` for minimal binary.
- **Backend trait** has 3 methods: `stream_completion()`, `token_count()`, `last_cache_stats()`. Uses `async_trait` + `BoxStream`.
- **Transport trait** has `serve(handler)` — receives messages, dispatches to agent, streams OutputEvents back.
- **Repair pipeline**: extract_json → repair_json → fuzzy_match_tool → coerce_arguments. Handles broken JSON, wrong tool names, wrong arg types.
- **Chat templates**: ChatML (Qwen), Llama3, Mistral — selected via config `template` field.
- **Config env vars**: `${VAR}` and `${VAR:-default}` expanded at load time.

## Building

```bash
cargo build                    # debug
cargo build --release          # release (2.6MB)
cargo test                     # 34 tests
cargo build --features full    # all features
```

## Tests

```bash
cargo test                           # all 34 tests
cargo test config::tests             # config parsing (5 tests)
cargo test repair::tests             # repair pipeline (16 tests)
cargo test cache::tests              # cache manager (3 tests)
cargo test tool::tests               # tool executor (6 tests)
cargo test agent::tests              # agent loop with mock (4 tests)
```

## Dependencies

Runtime: tokio, reqwest (rustls), serde, serde_json, toml, clap, regex, tracing, async-trait, futures, anyhow.
Optional: tokio-tungstenite (websocket), rumqttc (mqtt).
All pure Rust — no C deps, clean static linking via musl.
