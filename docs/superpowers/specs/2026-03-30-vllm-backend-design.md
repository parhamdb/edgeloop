# vLLM Backend Design

## Overview

Dedicated vLLM backend (`src/backend/vllm.rs`) that leverages vLLM-specific optimizations unavailable through the generic OpenAI backend. Follows the existing pattern: own module, own feature flag, implements the `Backend` trait.

## Motivation

The generic `openai.rs` backend works with vLLM but leaves performance on the table:
- `cache_hit_tokens` is always 0 (vLLM reports it, we don't parse it)
- `token_count()` uses `len/4` approximation (vLLM has `/tokenize`)
- No guided decoding (vLLM can enforce valid JSON at decode time, eliminating repair failures)
- No vLLM-specific sampling params (`min_tokens`, `repetition_penalty`, `top_k`, `min_p`, `stop_token_ids`)

## Architecture

### Module & Feature Flag

- New file: `src/backend/vllm.rs`
- Feature flag: `vllm` (added to `full` feature set)
- Registered in `backend/mod.rs` via `#[cfg(feature = "vllm")]`
- Backend type string: `"vllm"`

### VllmBackend Struct

```rust
pub struct VllmBackend {
    client: reqwest::Client,
    endpoint: String,          // default: http://localhost:8000
    model: String,
    api_key: Option<String>,   // optional — local vLLM doesn't need auth
    seed: Option<i64>,
    min_tokens: Option<usize>,
    repetition_penalty: Option<f64>,
    top_k: Option<i32>,
    min_p: Option<f64>,
    stop_token_ids: Option<Vec<u32>>,
    truncate_prompt_tokens: Option<usize>,
    guided_mode: Option<String>,       // "json", "regex", "grammar", "choice", "none"
    guided_pattern: Option<String>,    // explicit pattern override
    last_cache_stats: Mutex<Option<CacheStats>>,
}
```

### Config Fields

Eight new `Option` fields on `BackendConfig` (all `#[serde(default)]`, zero impact on other backends):

| Field | Type | Description |
|---|---|---|
| `min_tokens` | `Option<usize>` | Minimum tokens before allowing stop |
| `repetition_penalty` | `Option<f64>` | >1.0 discourages repeats |
| `top_k` | `Option<i32>` | Top-k sampling (-1 = disabled) |
| `min_p` | `Option<f64>` | Min probability threshold |
| `stop_token_ids` | `Option<Vec<u32>>` | Stop on token IDs |
| `truncate_prompt_tokens` | `Option<usize>` | Server-side prompt truncation |
| `guided_mode` | `Option<String>` | Guided decoding mode |
| `guided_pattern` | `Option<String>` | Explicit pattern override |

## Guided Decoding

The biggest optimization. vLLM enforces output structure at decode time via `guided_json` in `extra_body`.

### Schema Generation

When tools are available and `guided_mode = "json"`, the backend auto-generates a JSON schema from tool definitions:

```json
{
  "type": "object",
  "properties": {
    "tool": { "type": "string", "enum": ["calculator", "read_file"] },
    "arguments": { "type": "object" }
  },
  "required": ["tool", "arguments"]
}
```

This is sent as `extra_body.guided_json` in the chat completion request.

### Behavior

- `guided_mode = "json"` (default when tools present): auto-generate schema from tool defs
- `guided_mode = "regex"`: use `guided_pattern` as regex
- `guided_mode = "grammar"`: use `guided_pattern` as EBNF grammar
- `guided_mode = "choice"`: use `guided_pattern` as comma-separated choices
- `guided_mode = "none"` or no tools: no guided decoding, free-form output
- Guided decoding is only applied when tools are configured. Plain chat never uses it.

### Interaction with Repair Pipeline

Guided decoding makes the happy path near-100% reliable. The repair pipeline still runs as fallback — if guided output somehow fails to parse, repair handles it. No changes to `repair.rs`.

## Cache Stats

vLLM returns prefix cache hit counts in the usage response:

```json
{
  "usage": {
    "prompt_tokens": 142,
    "completion_tokens": 47,
    "prompt_tokens_details": { "cached_tokens": 128 }
  }
}
```

Parsed into `CacheStats`:
- `prompt_tokens` from `usage.prompt_tokens`
- `generated_tokens` from `usage.completion_tokens`
- `cache_hit_tokens` from `usage.prompt_tokens_details.cached_tokens`
- `prefill_ms: 0.0`, `generation_ms: 0.0` (vLLM doesn't report timing in API response)

## Tokenization

`token_count()` uses vLLM's `/tokenize` endpoint:
- Request: `POST /tokenize` with `{"model": "...", "prompt": "text"}`
- Response: `{"tokens": [...], "count": N}`
- Fallback: `len / 4` if endpoint unreachable (same as `llama_server.rs`)

## Streaming

SSE streaming via `/v1/chat/completions` with `stream: true` and `stream_options: {"include_usage": true}`. Same SSE parsing as `openai.rs` — parse `data: ` lines, extract `delta.content`, capture usage from final chunk.

### Request Body

Standard OpenAI fields plus vLLM `extra_body`:

```json
{
  "model": "Qwen/Qwen2.5-Coder-7B",
  "messages": [...],
  "stream": true,
  "temperature": 0.1,
  "max_tokens": 4096,
  "stop": [...],
  "seed": 42,
  "stream_options": {"include_usage": true},
  "extra_body": {
    "min_tokens": 5,
    "repetition_penalty": 1.1,
    "top_k": 40,
    "min_p": 0.05,
    "stop_token_ids": [151645],
    "truncate_prompt_tokens": 3800,
    "guided_json": { ... }
  }
}
```

## Files to Create/Modify

| File | Action | Description |
|---|---|---|
| `src/backend/vllm.rs` | Create | Full backend: struct, SSE streaming, guided decoding, tokenization, unit tests |
| `src/backend/mod.rs` | Edit | Add `pub mod vllm`, register `"vllm"` in `create_backend()` |
| `src/config.rs` | Edit | Add 8 new `Option` fields to `BackendConfig` |
| `Cargo.toml` | Edit | Add `vllm` feature flag, include in `full` |
| `tests/vllm_integration.rs` | Create | Integration tests (requires running vLLM) |
| `examples/local-vllm.toml` | Create | Example config |

## Testing

### Unit Tests (in `vllm.rs`)

- `test_new_default_endpoint` — defaults to `http://localhost:8000`
- `test_new_trailing_slash_stripped`
- `test_new_optional_api_key` — works with and without API key
- `test_last_cache_stats_initially_none`
- `test_token_count_fallback` — falls back to len/4
- `test_sse_chunk_deserialization` — delta content parsing
- `test_usage_with_cached_tokens` — `prompt_tokens_details.cached_tokens` parsed correctly
- `test_generate_tool_schema` — JSON schema from ToolDef list
- `test_request_body_includes_extra_body` — guided_json, min_tokens, etc.
- `test_cache_stats_written_and_read`

### Integration Tests (separate file, needs vLLM server)

- `test_simple_chat` — basic response
- `test_tool_call_guided` — tool call with guided decoding, verify valid JSON
- `test_cache_hit_ratio` — second request should show cache hits > 0
- `test_exact_token_count` — `/tokenize` returns correct count

## Example Config

```toml
# examples/local-vllm.toml
transports = ["cli"]

[agent]
system_prompt = "You are a helpful assistant."
template = "chatml"

[backend]
type = "vllm"
endpoint = "http://localhost:8000"
model = "Qwen/Qwen2.5-Coder-7B"
seed = 42
guided_mode = "json"
min_tokens = 5
repetition_penalty = 1.05
```
