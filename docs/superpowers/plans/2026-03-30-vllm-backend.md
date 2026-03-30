# vLLM Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated vLLM backend with guided decoding, prefix cache stats, exact tokenization, and vLLM-specific sampling parameters.

**Architecture:** New `src/backend/vllm.rs` behind a `vllm` feature flag, following the same pattern as `ollama.rs`, `llama_server.rs`, and `openai.rs`. Each backend is self-contained — own struct, own request/response types, implements `Backend` trait. No shared abstractions between backends.

**Tech Stack:** Rust, tokio, reqwest (rustls), serde/serde_json, futures, async-trait — all already in Cargo.toml.

---

### Task 1: Add config fields and feature flag

**Files:**
- Modify: `Cargo.toml:8-18` (features section)
- Modify: `src/config.rs:37-95` (BackendConfig struct)

- [ ] **Step 1: Add `vllm` feature flag to Cargo.toml**

In `Cargo.toml`, add `vllm = []` to the features section and include it in `full`:

```toml
[features]
default = ["ollama", "llama-server", "cli-transport"]
ollama = []
llama-server = []
openai = []
vllm = []
cli-transport = []
websocket = ["dep:tokio-tungstenite"]
mqtt = ["dep:rumqttc"]
unix-socket = []
tcp-socket = []
full = ["ollama", "llama-server", "openai", "vllm", "cli-transport", "websocket", "mqtt", "unix-socket", "tcp-socket"]
```

- [ ] **Step 2: Add vLLM-specific fields to BackendConfig**

In `src/config.rs`, add these fields to `BackendConfig` after the existing `cache_reuse` field:

```rust
    /// vLLM: Minimum tokens to generate before allowing stop.
    /// Prevents premature stops on short tool calls.
    #[serde(default)]
    pub min_tokens: Option<usize>,

    /// vLLM: Repetition penalty (>1.0 discourages repeats).
    #[serde(default)]
    pub repetition_penalty: Option<f64>,

    /// vLLM: Top-k sampling (-1 = all tokens).
    #[serde(default)]
    pub top_k: Option<i32>,

    /// vLLM: Minimum probability threshold (alternative to top_p).
    #[serde(default)]
    pub min_p: Option<f64>,

    /// vLLM: Stop generation on specific token IDs (not just strings).
    #[serde(default)]
    pub stop_token_ids: Option<Vec<u32>>,

    /// vLLM: Server-side prompt truncation to N tokens.
    #[serde(default)]
    pub truncate_prompt_tokens: Option<usize>,

    /// vLLM: Guided decoding mode — "json", "regex", "grammar", "choice", or "none".
    /// When "json", auto-generates JSON schema from tool definitions.
    #[serde(default)]
    pub guided_mode: Option<String>,

    /// vLLM: Explicit guided decoding pattern (regex string, grammar, or choice list).
    /// Overrides auto-generated schema when set.
    #[serde(default)]
    pub guided_pattern: Option<String>,
```

- [ ] **Step 3: Update all BackendConfig literals in tests to include new fields**

Every test that constructs a `BackendConfig` directly needs the new fields. Search for `BackendConfig {` across the codebase. Each one needs these additional fields:

```rust
min_tokens: None, repetition_penalty: None, top_k: None, min_p: None,
stop_token_ids: None, truncate_prompt_tokens: None, guided_mode: None, guided_pattern: None,
```

Files to update:
- `src/backend/ollama.rs` (test `test_new`)
- `src/backend/llama_server.rs` (tests `make_config`)
- `src/config.rs` (test `test_parse_full_config` — only if struct literal is used)
- `tests/integration_test.rs` (`make_ollama_config`)
- `tests/benchmark.rs` (`make_agent`)
- `tests/speculative_bench.rs` (`make_agent`)
- `tests/long_chat_test.rs` (`BackendConfig` literal)
- `tests/cache_test.rs` (`BackendConfig` literal)

- [ ] **Step 4: Run tests to verify nothing broke**

Run: `cargo test --bin edgeloop 2>&1`
Expected: all 45 unit tests pass

Run: `cargo check --test integration_test --test benchmark --test speculative_bench 2>&1`
Expected: compiles clean

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/config.rs src/backend/ollama.rs src/backend/llama_server.rs tests/integration_test.rs tests/benchmark.rs tests/speculative_bench.rs tests/long_chat_test.rs tests/cache_test.rs
git commit -m "feat: add vllm feature flag and config fields for vLLM backend"
```

---

### Task 2: Create vLLM backend — struct, constructor, types

**Files:**
- Create: `src/backend/vllm.rs`

This task creates the file with the struct, constructor, request/response types, and guided JSON schema generation. The `Backend` trait impl comes in the next task.

- [ ] **Step 1: Write unit tests for struct and schema generation**

Create `src/backend/vllm.rs` with the struct, constructor, types, and the `#[cfg(test)]` module at the bottom:

```rust
use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Mutex;
use reqwest::Client;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::backend::Backend;
use crate::cache::CacheStats;
use crate::config::{BackendConfig, ToolDef};
use crate::message::Message;

pub struct VllmBackend {
    client: Client,
    endpoint: String,
    model: String,
    api_key: Option<String>,
    seed: Option<i64>,
    min_tokens: Option<usize>,
    repetition_penalty: Option<f64>,
    top_k: Option<i32>,
    min_p: Option<f64>,
    stop_token_ids: Option<Vec<u32>>,
    truncate_prompt_tokens: Option<usize>,
    guided_mode: Option<String>,
    guided_pattern: Option<String>,
    tool_schema: Option<Value>,
    last_cache_stats: Mutex<Option<CacheStats>>,
}

impl VllmBackend {
    pub fn new(config: &BackendConfig, tools: &[ToolDef]) -> Self {
        let endpoint = if config.endpoint.is_empty() {
            "http://localhost:8000".to_string()
        } else {
            config.endpoint.trim_end_matches('/').to_string()
        };

        let api_key = config.api_key_env.as_deref()
            .and_then(|env_var| std::env::var(env_var).ok())
            .filter(|k| !k.is_empty());

        let tool_schema = if !tools.is_empty() {
            let mode = config.guided_mode.as_deref().unwrap_or("json");
            if mode == "json" {
                Some(generate_tool_schema(tools))
            } else {
                None
            }
        } else {
            None
        };

        Self {
            client: Client::new(),
            endpoint,
            model: config.model.clone(),
            api_key,
            seed: config.seed,
            min_tokens: config.min_tokens,
            repetition_penalty: config.repetition_penalty,
            top_k: config.top_k,
            min_p: config.min_p,
            stop_token_ids: config.stop_token_ids.clone(),
            truncate_prompt_tokens: config.truncate_prompt_tokens,
            guided_mode: config.guided_mode.clone(),
            guided_pattern: config.guided_pattern.clone(),
            tool_schema,
            last_cache_stats: Mutex::new(None),
        }
    }
}

/// Generate a JSON schema for tool calls from tool definitions.
/// The schema enforces: {"tool": "<name>", "arguments": {...}}
fn generate_tool_schema(tools: &[ToolDef]) -> Value {
    let tool_names: Vec<Value> = tools.iter()
        .map(|t| Value::String(t.name.clone()))
        .collect();

    serde_json::json!({
        "type": "object",
        "properties": {
            "tool": {
                "type": "string",
                "enum": tool_names
            },
            "arguments": {
                "type": "object"
            }
        },
        "required": ["tool", "arguments"]
    })
}

// ----- Request / Response types -----

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    temperature: f64,
    max_tokens: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
    stream_options: StreamOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    extra_body: Option<Value>,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Deserialize, Debug)]
struct SseDelta {
    content: Option<String>,
}

#[derive(Deserialize, Debug)]
struct SseChoice {
    delta: SseDelta,
}

#[derive(Deserialize, Debug)]
struct PromptTokensDetails {
    cached_tokens: Option<u64>,
}

#[derive(Deserialize, Debug)]
struct SseUsage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Deserialize, Debug)]
struct SseChunk {
    choices: Option<Vec<SseChoice>>,
    usage: Option<SseUsage>,
}

#[derive(Deserialize)]
struct TokenizeResponse {
    tokens: Vec<Value>,
    #[serde(default)]
    count: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::config::ParamDef;

    fn make_config(endpoint: &str) -> BackendConfig {
        BackendConfig {
            backend_type: "vllm".into(),
            endpoint: endpoint.into(),
            model: "Qwen/Qwen2.5-Coder-7B".into(),
            slot_id: None, n_keep: None, keep_alive: None, thinking: false,
            grammar: None, seed: Some(42), num_ctx: None, cache_reuse: None,
            api_key_env: None,
            min_tokens: Some(5), repetition_penalty: Some(1.1),
            top_k: Some(40), min_p: Some(0.05),
            stop_token_ids: Some(vec![151645]),
            truncate_prompt_tokens: Some(3800),
            guided_mode: Some("json".into()),
            guided_pattern: None,
        }
    }

    fn make_tool(name: &str) -> ToolDef {
        let mut params = HashMap::new();
        params.insert("expression".into(), ParamDef {
            param_type: "string".into(), required: true, default: None,
        });
        ToolDef {
            name: name.into(),
            description: format!("Test tool: {}", name),
            command: "echo test".into(),
            stdin: None, timeout: 10, workdir: None,
            env: HashMap::new(), parameters: params,
        }
    }

    #[test]
    fn test_new_default_endpoint() {
        let mut cfg = make_config("");
        cfg.guided_mode = None;
        let backend = VllmBackend::new(&cfg, &[]);
        assert_eq!(backend.endpoint, "http://localhost:8000");
    }

    #[test]
    fn test_new_trailing_slash_stripped() {
        let cfg = make_config("http://localhost:8000/");
        let backend = VllmBackend::new(&cfg, &[]);
        assert_eq!(backend.endpoint, "http://localhost:8000");
    }

    #[test]
    fn test_new_optional_api_key() {
        let cfg = make_config("http://localhost:8000");
        let backend = VllmBackend::new(&cfg, &[]);
        assert!(backend.api_key.is_none());
    }

    #[test]
    fn test_last_cache_stats_initially_none() {
        let cfg = make_config("http://localhost:8000");
        let backend = VllmBackend::new(&cfg, &[]);
        assert!(backend.last_cache_stats().is_none());
    }

    #[test]
    fn test_generate_tool_schema() {
        let tools = vec![make_tool("calculator"), make_tool("read_file")];
        let schema = generate_tool_schema(&tools);
        let tool_enum = &schema["properties"]["tool"]["enum"];
        assert_eq!(tool_enum[0], "calculator");
        assert_eq!(tool_enum[1], "read_file");
        assert_eq!(schema["required"][0], "tool");
        assert_eq!(schema["required"][1], "arguments");
    }

    #[test]
    fn test_tool_schema_stored_when_json_mode() {
        let cfg = make_config("http://localhost:8000");
        let tools = vec![make_tool("calculator")];
        let backend = VllmBackend::new(&cfg, &tools);
        assert!(backend.tool_schema.is_some());
        let schema = backend.tool_schema.as_ref().unwrap();
        assert_eq!(schema["properties"]["tool"]["enum"][0], "calculator");
    }

    #[test]
    fn test_no_tool_schema_when_no_tools() {
        let cfg = make_config("http://localhost:8000");
        let backend = VllmBackend::new(&cfg, &[]);
        assert!(backend.tool_schema.is_none());
    }

    #[test]
    fn test_no_tool_schema_when_mode_none() {
        let mut cfg = make_config("http://localhost:8000");
        cfg.guided_mode = Some("none".into());
        let tools = vec![make_tool("calculator")];
        let backend = VllmBackend::new(&cfg, &tools);
        assert!(backend.tool_schema.is_none());
    }

    #[test]
    fn test_sse_chunk_deserialization() {
        let raw = r#"{"choices":[{"delta":{"content":"Hello"}}]}"#;
        let chunk: SseChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.choices.unwrap()[0].delta.content.as_ref().unwrap(), "Hello");
    }

    #[test]
    fn test_usage_with_cached_tokens() {
        let raw = r#"{"usage":{"prompt_tokens":142,"completion_tokens":47,"prompt_tokens_details":{"cached_tokens":128}}}"#;
        let chunk: SseChunk = serde_json::from_str(raw).unwrap();
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(142));
        assert_eq!(usage.completion_tokens, Some(47));
        assert_eq!(usage.prompt_tokens_details.unwrap().cached_tokens, Some(128));
    }

    #[test]
    fn test_usage_without_cached_tokens() {
        let raw = r#"{"usage":{"prompt_tokens":100,"completion_tokens":20}}"#;
        let chunk: SseChunk = serde_json::from_str(raw).unwrap();
        let usage = chunk.usage.unwrap();
        assert!(usage.prompt_tokens_details.is_none());
    }

    #[test]
    fn test_cache_stats_written_and_read() {
        let cfg = make_config("http://localhost:8000");
        let backend = VllmBackend::new(&cfg, &[]);
        {
            let mut guard = backend.last_cache_stats.lock().unwrap();
            *guard = Some(CacheStats {
                prompt_tokens: 142,
                generated_tokens: 47,
                prefill_ms: 0.0,
                generation_ms: 0.0,
                cache_hit_tokens: 128,
            });
        }
        let stats = backend.last_cache_stats().unwrap();
        assert_eq!(stats.prompt_tokens, 142);
        assert_eq!(stats.cache_hit_tokens, 128);
    }

    #[test]
    fn test_build_extra_body() {
        let cfg = make_config("http://localhost:8000");
        let tools = vec![make_tool("calculator")];
        let backend = VllmBackend::new(&cfg, &tools);
        let extra = backend.build_extra_body();
        let extra = extra.unwrap();
        assert_eq!(extra["min_tokens"], 5);
        assert_eq!(extra["repetition_penalty"], 1.1);
        assert_eq!(extra["top_k"], 40);
        assert_eq!(extra["min_p"], 0.05);
        assert_eq!(extra["stop_token_ids"][0], 151645);
        assert_eq!(extra["truncate_prompt_tokens"], 3800);
        assert!(extra.get("guided_json").is_some());
    }

    #[test]
    fn test_build_extra_body_minimal() {
        let mut cfg = make_config("http://localhost:8000");
        cfg.min_tokens = None;
        cfg.repetition_penalty = None;
        cfg.top_k = None;
        cfg.min_p = None;
        cfg.stop_token_ids = None;
        cfg.truncate_prompt_tokens = None;
        cfg.guided_mode = None;
        let backend = VllmBackend::new(&cfg, &[]);
        let extra = backend.build_extra_body();
        assert!(extra.is_none());
    }
}
```

- [ ] **Step 2: Add the `build_extra_body` method**

Add this method to the `impl VllmBackend` block, after `new()`:

```rust
    /// Build the vLLM-specific `extra_body` JSON for the chat request.
    /// Returns None if no vLLM-specific params are set.
    fn build_extra_body(&self) -> Option<Value> {
        let mut extra = serde_json::Map::new();

        if let Some(v) = self.min_tokens {
            extra.insert("min_tokens".into(), Value::Number(v.into()));
        }
        if let Some(v) = self.repetition_penalty {
            extra.insert("repetition_penalty".into(), serde_json::json!(v));
        }
        if let Some(v) = self.top_k {
            extra.insert("top_k".into(), Value::Number(v.into()));
        }
        if let Some(v) = self.min_p {
            extra.insert("min_p".into(), serde_json::json!(v));
        }
        if let Some(ref v) = self.stop_token_ids {
            extra.insert("stop_token_ids".into(), serde_json::json!(v));
        }
        if let Some(v) = self.truncate_prompt_tokens {
            extra.insert("truncate_prompt_tokens".into(), Value::Number(v.into()));
        }

        // Guided decoding
        match self.guided_mode.as_deref() {
            Some("json") => {
                if let Some(ref schema) = self.tool_schema {
                    extra.insert("guided_json".into(), schema.clone());
                }
            }
            Some("regex") => {
                if let Some(ref pattern) = self.guided_pattern {
                    extra.insert("guided_regex".into(), Value::String(pattern.clone()));
                }
            }
            Some("grammar") => {
                if let Some(ref pattern) = self.guided_pattern {
                    extra.insert("guided_grammar".into(), Value::String(pattern.clone()));
                }
            }
            Some("choice") => {
                if let Some(ref pattern) = self.guided_pattern {
                    let choices: Vec<Value> = pattern.split(',')
                        .map(|s| Value::String(s.trim().to_string()))
                        .collect();
                    extra.insert("guided_choice".into(), Value::Array(choices));
                }
            }
            _ => {}
        }

        if extra.is_empty() {
            None
        } else {
            Some(Value::Object(extra))
        }
    }
```

- [ ] **Step 3: Run tests to verify struct and schema tests pass**

Run: `cargo test --bin edgeloop --features vllm 2>&1`
Expected: all tests pass including the new vllm tests

- [ ] **Step 4: Commit**

```bash
git add src/backend/vllm.rs
git commit -m "feat(vllm): add backend struct, config, guided JSON schema generation"
```

---

### Task 3: Implement Backend trait — streaming, cache stats, tokenization

**Files:**
- Modify: `src/backend/vllm.rs` (add `Backend` impl)

- [ ] **Step 1: Add the Backend trait implementation**

Add this `impl Backend for VllmBackend` block after the `build_extra_body` method:

```rust
#[async_trait]
impl Backend for VllmBackend {
    fn stream_completion(
        &self,
        _prompt: &str,
        messages: &[Message],
        temperature: f64,
        max_tokens: usize,
        stop: &[String],
    ) -> BoxStream<'_, Result<String>> {
        let url = format!("{}/v1/chat/completions", self.endpoint);
        let body = ChatRequest {
            model: self.model.clone(),
            messages: messages.to_vec(),
            stream: true,
            temperature,
            max_tokens,
            stop: stop.to_vec(),
            stream_options: StreamOptions { include_usage: true },
            seed: self.seed,
            extra_body: self.build_extra_body(),
        };

        let (tx, rx) = mpsc::channel::<Result<String>>(64);
        let client = self.client.clone();
        let api_key = self.api_key.clone();

        let (stats_tx, stats_rx) = mpsc::channel::<CacheStats>(1);

        tokio::spawn(async move {
            let mut request = client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(&body);

            if let Some(ref key) = api_key {
                request = request.bearer_auth(key);
            }

            let response = match request.send().await {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(Err(anyhow::anyhow!("Failed to connect to vLLM: {}", e))).await;
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let body_text = response.text().await.unwrap_or_default();
                let _ = tx
                    .send(Err(anyhow::anyhow!("vLLM error {}: {}", status, body_text)))
                    .await;
                return;
            }

            let mut byte_stream = response.bytes_stream();
            let mut buf = String::new();
            let mut prompt_tokens: usize = 0;
            let mut completion_tokens: usize = 0;
            let mut cache_hit_tokens: usize = 0;

            'outer: while let Some(chunk_result) = byte_stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(Err(anyhow::anyhow!("Stream error: {}", e))).await;
                        return;
                    }
                };

                let text = match std::str::from_utf8(&chunk) {
                    Ok(t) => t.to_string(),
                    Err(e) => {
                        let _ = tx.send(Err(anyhow::anyhow!("UTF-8 decode error: {}", e))).await;
                        return;
                    }
                };

                buf.push_str(&text);

                loop {
                    if let Some(newline_pos) = buf.find('\n') {
                        let line = buf[..newline_pos].trim_end_matches('\r').to_string();
                        buf = buf[newline_pos + 1..].to_string();

                        if line.is_empty() {
                            continue;
                        }

                        let data = match line.strip_prefix("data: ") {
                            Some(d) => d.to_string(),
                            None => continue,
                        };

                        if data == "[DONE]" {
                            let stats = CacheStats {
                                prompt_tokens,
                                generated_tokens: completion_tokens,
                                prefill_ms: 0.0,
                                generation_ms: 0.0,
                                cache_hit_tokens,
                            };
                            let _ = stats_tx.send(stats).await;
                            break 'outer;
                        }

                        let sse_chunk: SseChunk = match serde_json::from_str(&data) {
                            Ok(c) => c,
                            Err(_) => {
                                if let Ok(v) = serde_json::from_str::<Value>(&data) {
                                    if let Some(err) = v.get("error") {
                                        let _ = tx
                                            .send(Err(anyhow::anyhow!("vLLM error: {}", err)))
                                            .await;
                                        return;
                                    }
                                }
                                continue;
                            }
                        };

                        // Capture usage with prefix cache stats
                        if let Some(usage) = &sse_chunk.usage {
                            if let Some(pt) = usage.prompt_tokens {
                                prompt_tokens = pt as usize;
                            }
                            if let Some(ct) = usage.completion_tokens {
                                completion_tokens = ct as usize;
                            }
                            if let Some(ref details) = usage.prompt_tokens_details {
                                if let Some(cached) = details.cached_tokens {
                                    cache_hit_tokens = cached as usize;
                                }
                            }
                        }

                        // Yield delta content tokens
                        if let Some(choices) = &sse_chunk.choices {
                            for choice in choices {
                                if let Some(content) = &choice.delta.content {
                                    if !content.is_empty() {
                                        if tx.send(Ok(content.clone())).await.is_err() {
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        break;
                    }
                }
            }

            // EOF without [DONE]
            let stats = CacheStats {
                prompt_tokens,
                generated_tokens: completion_tokens,
                prefill_ms: 0.0,
                generation_ms: 0.0,
                cache_hit_tokens,
            };
            let _ = stats_tx.send(stats).await;
        });

        let stats_ptr: *const Mutex<Option<CacheStats>> = &self.last_cache_stats;
        let token_stream = ReceiverStream::new(rx);

        let wrapped = VllmTokenStream {
            inner: token_stream,
            stats_rx,
            stats_ptr,
            done: false,
        };

        Box::pin(wrapped)
    }

    async fn token_count(&self, text: &str) -> Result<usize> {
        let url = format!("{}/tokenize", self.endpoint);
        match self.client
            .post(&url)
            .json(&serde_json::json!({ "model": &self.model, "prompt": text }))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<TokenizeResponse>().await {
                    Ok(body) => Ok(body.count.unwrap_or(body.tokens.len())),
                    Err(_) => Ok(text.len() / 4),
                }
            }
            _ => Ok(text.len() / 4),
        }
    }

    fn last_cache_stats(&self) -> Option<CacheStats> {
        self.last_cache_stats.lock().unwrap().clone()
    }
}
```

- [ ] **Step 2: Add the VllmTokenStream wrapper**

Add this after the `Backend` impl, before `#[cfg(test)]`:

```rust
// Stream wrapper that updates cache stats when the inner stream ends.
struct VllmTokenStream {
    inner: ReceiverStream<Result<String>>,
    stats_rx: mpsc::Receiver<CacheStats>,
    stats_ptr: *const Mutex<Option<CacheStats>>,
    done: bool,
}

// SAFETY: stats_ptr points to a field of VllmBackend which is Send + Sync.
// The stream's lifetime is tied to &'_ VllmBackend so the pointer is valid
// for the duration of the stream.
unsafe impl Send for VllmTokenStream {}
unsafe impl Sync for VllmTokenStream {}

impl futures::Stream for VllmTokenStream {
    type Item = Result<String>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        use std::task::Poll;

        if self.done {
            return Poll::Ready(None);
        }

        match std::pin::Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
            Poll::Ready(None) => {
                self.done = true;
                if let Ok(stats) = self.stats_rx.try_recv() {
                    unsafe {
                        (*self.stats_ptr).lock().unwrap().replace(stats);
                    }
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
```

- [ ] **Step 3: Add token_count fallback test**

Add this test to the existing `#[cfg(test)] mod tests` block:

```rust
    #[tokio::test]
    async fn test_token_count_fallback() {
        let cfg = make_config("http://127.0.0.1:19999");
        let backend = VllmBackend::new(&cfg, &[]);
        let text = "hello world 1234";
        let count = backend.token_count(text).await.unwrap();
        assert_eq!(count, text.len() / 4);
    }
```

- [ ] **Step 4: Run tests**

Run: `cargo test --bin edgeloop --features vllm 2>&1`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add src/backend/vllm.rs
git commit -m "feat(vllm): implement Backend trait — streaming, prefix cache stats, tokenization"
```

---

### Task 4: Register backend in mod.rs and update create_backend

**Files:**
- Modify: `src/backend/mod.rs:8-39`
- Modify: `src/backend/mod.rs:30-39` (create_backend function)

- [ ] **Step 1: Add vllm module declaration**

In `src/backend/mod.rs`, add after the openai module declaration:

```rust
#[cfg(feature = "vllm")]
pub mod vllm;
```

- [ ] **Step 2: Update create_backend to accept tools parameter**

The vLLM backend needs tools to generate the guided JSON schema. Update `create_backend` signature and add the vllm match arm. The other backends ignore the tools param:

```rust
pub fn create_backend(config: &crate::config::BackendConfig, tools: &[crate::config::ToolDef]) -> Result<Box<dyn Backend>> {
    match config.backend_type.as_str() {
        #[cfg(feature = "ollama")]
        "ollama" => Ok(Box::new(ollama::OllamaBackend::new(config))),
        #[cfg(feature = "llama-server")]
        "llama-server" => Ok(Box::new(llama_server::LlamaServerBackend::new(config))),
        #[cfg(feature = "openai")]
        "openai" => Ok(Box::new(openai::OpenAIBackend::new(config)?)),
        #[cfg(feature = "vllm")]
        "vllm" => Ok(Box::new(vllm::VllmBackend::new(config, tools))),
        other => anyhow::bail!("Unknown or disabled backend: '{}'", other),
    }
}
```

- [ ] **Step 3: Update callers of create_backend**

Search for `create_backend(` in `src/main.rs` and pass the tools slice. Typically this is something like:

```rust
// Before:
let backend = backend::create_backend(&config.backend)?;
// After:
let backend = backend::create_backend(&config.backend, &tools)?;
```

Find the exact call site by grepping `create_backend` in `src/main.rs` and update it.

- [ ] **Step 4: Run tests**

Run: `cargo test --bin edgeloop --features vllm 2>&1`
Expected: all tests pass

Run: `cargo build --features vllm 2>&1`
Expected: compiles clean

- [ ] **Step 5: Commit**

```bash
git add src/backend/mod.rs src/main.rs
git commit -m "feat(vllm): register vllm backend in mod.rs, update create_backend signature"
```

---

### Task 5: Add example config

**Files:**
- Create: `examples/local-vllm.toml`

- [ ] **Step 1: Create example config file**

```toml
# vLLM backend — local GPU inference with guided decoding
#
# Start vLLM:
#   python -m vllm.entrypoints.openai.api_server \
#     --model Qwen/Qwen2.5-Coder-7B \
#     --enable-prefix-caching \
#     --enable-prompt-tokens-details \
#     --dtype auto
#
# Run edgeloop:
#   cargo run --features vllm -- --config examples/local-vllm.toml

transports = ["cli"]
tool_packages = ["tools/filesystem"]

[agent]
system_prompt = "You are a helpful assistant. Use tools when needed."
template = "chatml"
max_tokens = 4096
temperature = 0.1

[backend]
type = "vllm"
endpoint = "http://localhost:8000"
model = "Qwen/Qwen2.5-Coder-7B"
seed = 42
guided_mode = "json"
min_tokens = 5
repetition_penalty = 1.05

[cache]
max_context = 4096
truncation_threshold = 0.8
```

- [ ] **Step 2: Commit**

```bash
git add examples/local-vllm.toml
git commit -m "docs: add vLLM example config"
```

---

### Task 6: Add integration tests

**Files:**
- Create: `tests/vllm_integration.rs`

- [ ] **Step 1: Create integration test file**

These tests require a running vLLM server. They are skipped by default (run with `cargo test --test vllm_integration --features vllm`).

```rust
//! Integration tests for vLLM backend — requires running vLLM server.
//!
//! Start vLLM:
//!   python -m vllm.entrypoints.openai.api_server \
//!     --model Qwen/Qwen2.5-Coder-7B \
//!     --enable-prefix-caching \
//!     --enable-prompt-tokens-details
//!
//! Run:
//!   VLLM_ENDPOINT=http://localhost:8000 cargo test --test vllm_integration --features vllm -- --nocapture

#[cfg(test)]
mod vllm_integration {
    use std::collections::HashMap;
    use std::sync::Arc;

    fn vllm_endpoint() -> Option<String> {
        std::env::var("VLLM_ENDPOINT").ok()
    }

    fn vllm_model() -> String {
        std::env::var("VLLM_MODEL").unwrap_or_else(|_| "Qwen/Qwen2.5-Coder-7B".into())
    }

    fn make_config() -> edgeloop::config::BackendConfig {
        edgeloop::config::BackendConfig {
            backend_type: "vllm".into(),
            endpoint: vllm_endpoint().unwrap_or_else(|| "http://localhost:8000".into()),
            model: vllm_model(),
            slot_id: None, n_keep: None, keep_alive: None, thinking: false,
            grammar: None, seed: Some(42), num_ctx: None, cache_reuse: None,
            api_key_env: None,
            min_tokens: None, repetition_penalty: None, top_k: None, min_p: None,
            stop_token_ids: None, truncate_prompt_tokens: None,
            guided_mode: Some("json".into()), guided_pattern: None,
        }
    }

    fn make_agent_config() -> (edgeloop::config::AgentConfig, edgeloop::config::CacheConfig) {
        let agent = edgeloop::config::AgentConfig {
            system_prompt: "You are a helpful assistant.".into(),
            template: "chatml".into(),
            max_tokens: 4096,
            max_iterations: 8,
            max_retries: 2,
            temperature: 0.1,
        };
        let cache = edgeloop::config::CacheConfig {
            max_context: 4096,
            truncation_threshold: 0.8,
        };
        (agent, cache)
    }

    fn make_calculator() -> edgeloop::config::ToolDef {
        let mut params = HashMap::new();
        params.insert("expression".into(), edgeloop::config::ParamDef {
            param_type: "string".into(), required: true, default: None,
        });
        edgeloop::config::ToolDef {
            name: "calculator".into(),
            description: "Evaluate a math expression".into(),
            command: "python3 -c 'print(eval(\"{expression}\"))'".into(),
            stdin: None, timeout: 10, workdir: None,
            env: HashMap::new(), parameters: params,
        }
    }

    #[tokio::test]
    async fn test_simple_chat() {
        if vllm_endpoint().is_none() {
            eprintln!("Skipping: VLLM_ENDPOINT not set");
            return;
        }

        let backend_cfg = make_config();
        let (agent_cfg, cache_cfg) = make_agent_config();
        let backend = Arc::new(edgeloop::backend::vllm::VllmBackend::new(&backend_cfg, &[]));
        let agent = edgeloop::agent::Agent::new(backend, vec![], &agent_cfg, &cache_cfg);

        let response = agent.run("What is 2+2? Answer with just the number.").await;
        assert!(response.contains("4"), "Expected '4' in response: {}", response);
    }

    #[tokio::test]
    async fn test_tool_call_guided() {
        if vllm_endpoint().is_none() {
            eprintln!("Skipping: VLLM_ENDPOINT not set");
            return;
        }

        let backend_cfg = make_config();
        let (agent_cfg, cache_cfg) = make_agent_config();
        let tools = vec![make_calculator()];
        let backend = Arc::new(edgeloop::backend::vllm::VllmBackend::new(&backend_cfg, &tools));
        let agent = edgeloop::agent::Agent::new(backend, tools, &agent_cfg, &cache_cfg);

        let response = agent.run("What is 123 * 456? Use the calculator tool.").await;
        let clean = response.replace(",", "").replace(" ", "");
        assert!(clean.contains("56088"), "Expected '56088' in response: {}", response);
    }

    #[tokio::test]
    async fn test_cache_hit_ratio() {
        if vllm_endpoint().is_none() {
            eprintln!("Skipping: VLLM_ENDPOINT not set");
            return;
        }

        let backend_cfg = make_config();
        let (agent_cfg, cache_cfg) = make_agent_config();
        let backend = Arc::new(edgeloop::backend::vllm::VllmBackend::new(&backend_cfg, &[]));
        let agent = edgeloop::agent::Agent::new(backend, vec![], &agent_cfg, &cache_cfg);

        // First request primes the cache
        let _ = agent.run("Say hello.").await;
        // Second request should hit prefix cache (same system prompt)
        let _ = agent.run("Say goodbye.").await;

        let cache = agent.cache.lock().await;
        println!("Cache hit ratio: {:.1}%", cache.overall_cache_hit_ratio() * 100.0);
        // With prefix caching enabled, second request should have some cache hits
        // We don't assert a specific ratio since it depends on vLLM config
    }

    #[tokio::test]
    async fn test_exact_token_count() {
        if vllm_endpoint().is_none() {
            eprintln!("Skipping: VLLM_ENDPOINT not set");
            return;
        }

        let backend_cfg = make_config();
        let backend = edgeloop::backend::vllm::VllmBackend::new(&backend_cfg, &[]);
        let count = backend.token_count("Hello, world!").await.unwrap();
        // Exact tokenization should give a small number (3-5 tokens), not len/4 = 3
        println!("Token count for 'Hello, world!': {}", count);
        assert!(count > 0 && count < 20, "Unexpected token count: {}", count);
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add tests/vllm_integration.rs
git commit -m "test: add vLLM integration tests (requires running vLLM server)"
```

---

### Task 7: Final verification

**Files:** None (verification only)

- [ ] **Step 1: Run all unit tests**

Run: `cargo test --bin edgeloop --features vllm 2>&1`
Expected: all unit tests pass (45 existing + ~12 new vllm tests)

- [ ] **Step 2: Verify all feature combinations compile**

Run: `cargo check --features vllm 2>&1`
Expected: compiles clean

Run: `cargo check --features full 2>&1`
Expected: compiles clean

Run: `cargo check 2>&1` (default features, no vllm)
Expected: compiles clean

- [ ] **Step 3: Verify build size**

Run: `cargo build --release --features full 2>&1 && ls -la target/release/edgeloop`
Expected: binary size stays reasonable (under 6MB)

- [ ] **Step 4: Commit any fixups if needed**
