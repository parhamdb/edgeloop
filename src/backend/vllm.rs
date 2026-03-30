use anyhow::Result;
use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Mutex;
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

        let api_key = config
            .api_key_env
            .as_deref()
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
                    let choices: Vec<Value> = pattern
                        .split(',')
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
}

/// Generate a JSON schema for tool calls from tool definitions.
/// The schema enforces: {"tool": "<name>", "arguments": {...}}
fn generate_tool_schema(tools: &[ToolDef]) -> Value {
    let tool_names: Vec<Value> = tools
        .iter()
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

// ----- Backend implementation -----

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
            stream_options: StreamOptions {
                include_usage: true,
            },
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
                    let _ = tx
                        .send(Err(anyhow::anyhow!("Failed to connect to vLLM: {}", e)))
                        .await;
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
                        let _ = tx
                            .send(Err(anyhow::anyhow!("UTF-8 decode error: {}", e)))
                            .await;
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
        match self
            .client
            .post(&url)
            .json(&serde_json::json!({ "model": &self.model, "prompt": text }))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => match resp.json::<TokenizeResponse>().await {
                Ok(body) => Ok(body.count.unwrap_or(body.tokens.len())),
                Err(_) => Ok(text.len() / 4),
            },
            _ => Ok(text.len() / 4),
        }
    }

    fn last_cache_stats(&self) -> Option<CacheStats> {
        self.last_cache_stats.lock().unwrap().clone()
    }
}

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

// --- Unit tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ParamDef;
    use std::collections::HashMap;

    fn make_config(endpoint: &str) -> BackendConfig {
        BackendConfig {
            backend_type: "vllm".into(),
            endpoint: endpoint.into(),
            model: "Qwen/Qwen2.5-Coder-7B".into(),
            slot_id: None,
            n_keep: None,
            keep_alive: None,
            thinking: false,
            grammar: None,
            seed: Some(42),
            num_ctx: None,
            cache_reuse: None,
            api_key_env: None,
            min_tokens: Some(5),
            repetition_penalty: Some(1.1),
            top_k: Some(40),
            min_p: Some(0.05),
            stop_token_ids: Some(vec![151645]),
            truncate_prompt_tokens: Some(3800),
            guided_mode: Some("json".into()),
            guided_pattern: None,
        }
    }

    fn make_tool(name: &str) -> ToolDef {
        let mut params = HashMap::new();
        params.insert(
            "expression".into(),
            ParamDef {
                param_type: "string".into(),
                required: true,
                default: None,
            },
        );
        ToolDef {
            name: name.into(),
            description: format!("Test tool: {}", name),
            command: "echo test".into(),
            stdin: None,
            timeout: 10,
            workdir: None,
            env: HashMap::new(),
            parameters: params,
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
        assert_eq!(
            chunk.choices.unwrap()[0].delta.content.as_ref().unwrap(),
            "Hello"
        );
    }

    #[test]
    fn test_usage_with_cached_tokens() {
        let raw = r#"{"usage":{"prompt_tokens":142,"completion_tokens":47,"prompt_tokens_details":{"cached_tokens":128}}}"#;
        let chunk: SseChunk = serde_json::from_str(raw).unwrap();
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(142));
        assert_eq!(usage.completion_tokens, Some(47));
        assert_eq!(
            usage.prompt_tokens_details.unwrap().cached_tokens,
            Some(128)
        );
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

    #[tokio::test]
    async fn test_token_count_fallback() {
        let cfg = make_config("http://127.0.0.1:19999");
        let backend = VllmBackend::new(&cfg, &[]);
        let text = "hello world 1234";
        let count = backend.token_count(text).await.unwrap();
        assert_eq!(count, text.len() / 4);
    }
}
