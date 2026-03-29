# Changelog

## 0.1.0 — 2026-03-29

### Rust Rewrite
Complete rewrite from Python prototype to Rust. Single static binary, config-driven.

### Backends
- **Ollama** — /api/chat with NDJSON streaming, thinking mode support, KV cache stats
- **llama-server** — /completion with SSE streaming, cache_prompt + slot pinning, /tokenize
- **OpenAI** — /v1/chat/completions SSE, works with any OpenAI-compatible API

### Transports
- **CLI** — interactive stdin/stdout REPL
- **WebSocket** — JSON frames over tokio-tungstenite
- **MQTT** — pub/sub via rumqttc with auto-reconnection
- **Unix socket** — newline-delimited JSON
- **TCP socket** — newline-delimited JSON

### Core
- **Agent loop** — ReAct pattern with configurable iterations and retries
- **Chat templates** — ChatML, Llama3, Mistral
- **Repair pipeline** — JSON extraction, syntax repair, fuzzy tool matching (Levenshtein), positional arg mapping, type coercion
- **Tool executor** — subprocess with `{param}` substitution and timeout
- **KV cache manager** — prefill tracking, context budget, auto-truncation
- **Config** — TOML with env var expansion (`${VAR:-default}`) and includes

### Feature Flags
- Compile-time selection of backends and transports
- Default: ollama + llama-server + cli (4.4MB)
- Full: all features (5.0MB)
- Minimal: llama-server + cli (~2MB)

### Testing
- 45 unit tests (config, repair, cache, tool, agent)
- 5 integration tests (real Ollama)
- 3 benchmark tests (latency across models)

### Performance (RTX 4070)
- Simple response: 93-164ms (warm)
- Tool roundtrip: 694-890ms
- WebSocket end-to-end: 117ms

### Previous (Python prototype)
The Python version (v0.0.x) served as the design prototype:
- Tested across 4 backends, 12 scenarios, 96% pass rate
- Identified key issues: Qwen3 thinking mode, KV cache reuse, compact prompts
- All findings incorporated into the Rust version
