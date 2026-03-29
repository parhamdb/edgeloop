use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

use crate::backend::Backend;
use crate::cache::CacheStats;
use crate::config::BackendConfig;
use crate::message::Message;

pub struct OllamaBackend {
    client: reqwest::Client,
    endpoint: String,
    model: String,
    thinking: bool,
    last_cache_stats: Arc<Mutex<Option<CacheStats>>>,
}

impl OllamaBackend {
    pub fn new(config: &BackendConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            endpoint: config.endpoint.clone(),
            model: config.model.clone(),
            thinking: config.thinking,
            last_cache_stats: Arc::new(Mutex::new(None)),
        }
    }
}

// --- Request / Response types ---

#[derive(Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    messages: &'a [OllamaMessage],
    stream: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    think: bool,
    options: OllamaOptions,
}

#[derive(Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct OllamaOptions {
    num_predict: usize,
    temperature: f64,
}

#[derive(Deserialize, Debug)]
struct OllamaChatChunk {
    message: Option<OllamaChunkMessage>,
    done: bool,
    // Present only on the final done chunk
    prompt_eval_count: Option<u64>,
    eval_count: Option<u64>,
    prompt_eval_duration: Option<u64>, // nanoseconds
    eval_duration: Option<u64>,        // nanoseconds
}

#[derive(Deserialize, Debug)]
struct OllamaChunkMessage {
    content: Option<String>,
    // `thinking` field is present when think=true
    #[allow(dead_code)]
    thinking: Option<String>,
}

// Internal state for the unfold-based stream.
enum StreamState {
    // Haven't made the HTTP call yet.
    Init {
        client: reqwest::Client,
        url: String,
        body_json: Vec<u8>,
        cache_stats_ref: Arc<Mutex<Option<CacheStats>>>,
    },
    // Streaming bytes from the response.
    Running {
        byte_stream: Box<dyn futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin + Send>,
        line_buf: String,
        cache_stats_ref: Arc<Mutex<Option<CacheStats>>>,
        // Tokens accumulated from the current byte chunk that haven't been yielded yet.
        pending: Vec<String>,
    },
    Done,
}

// --- Backend impl ---

#[async_trait]
impl Backend for OllamaBackend {
    fn stream_completion(
        &self,
        _prompt: &str,
        messages: &[Message],
        temperature: f64,
        max_tokens: usize,
        _stop: &[String],
    ) -> BoxStream<'_, Result<String>> {
        let url = format!("{}/api/chat", self.endpoint.trim_end_matches('/'));
        let ollama_messages: Vec<OllamaMessage> = messages
            .iter()
            .map(|m| OllamaMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect();

        let body = OllamaChatRequest {
            model: &self.model,
            messages: &ollama_messages,
            stream: true,
            think: self.thinking,
            options: OllamaOptions {
                num_predict: max_tokens,
                temperature,
            },
        };

        let body_json = match serde_json::to_vec(&body) {
            Ok(b) => b,
            Err(e) => {
                return Box::pin(futures::stream::once(async move {
                    Err(anyhow::anyhow!("Failed to serialize request: {}", e))
                }));
            }
        };

        let cache_stats_ref = Arc::clone(&self.last_cache_stats);
        let client = self.client.clone();

        let initial_state = StreamState::Init {
            client,
            url,
            body_json,
            cache_stats_ref,
        };

        let stream = futures::stream::unfold(initial_state, |state| async move {
            match state {
                StreamState::Done => None,

                StreamState::Init { client, url, body_json, cache_stats_ref } => {
                    let response = client
                        .post(&url)
                        .header("Content-Type", "application/json")
                        .body(body_json)
                        .send()
                        .await;

                    let response = match response {
                        Ok(r) => r,
                        Err(e) => {
                            return Some((
                                Err(anyhow::anyhow!("ConnectionError: could not connect to Ollama at {}: {}", url, e)),
                                StreamState::Done,
                            ));
                        }
                    };

                    if !response.status().is_success() {
                        let status = response.status();
                        let body = response.text().await.unwrap_or_default();
                        return Some((
                            Err(anyhow::anyhow!("Ollama API error {}: {}", status, body)),
                            StreamState::Done,
                        ));
                    }

                    let byte_stream = Box::new(response.bytes_stream());
                    process_running(byte_stream, String::new(), cache_stats_ref, Vec::new()).await
                }

                StreamState::Running { byte_stream, line_buf, cache_stats_ref, pending } => {
                    process_running(byte_stream, line_buf, cache_stats_ref, pending).await
                }
            }
        });

        Box::pin(stream)
    }

    async fn token_count(&self, text: &str) -> Result<usize> {
        Ok(text.len() / 4)
    }

    fn last_cache_stats(&self) -> Option<CacheStats> {
        self.last_cache_stats.lock().ok()?.clone()
    }
}

type ByteStream = Box<dyn futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin + Send>;

async fn process_running(
    mut byte_stream: ByteStream,
    mut line_buf: String,
    cache_stats_ref: Arc<Mutex<Option<CacheStats>>>,
    mut pending: Vec<String>,
) -> Option<(Result<String>, StreamState)> {
    // If there are buffered tokens from a previous chunk, yield one now.
    if !pending.is_empty() {
        let token = pending.remove(0);
        let next = StreamState::Running { byte_stream, line_buf, cache_stats_ref, pending };
        return Some((Ok(token), next));
    }

    // Pull the next byte chunk from the HTTP stream.
    loop {
        let chunk = match byte_stream.next().await {
            Some(Ok(c)) => c,
            Some(Err(e)) => {
                return Some((Err(anyhow::anyhow!("Stream error: {}", e)), StreamState::Done));
            }
            None => {
                // HTTP stream ended without a done chunk — treat as finished.
                return None;
            }
        };

        let text = match std::str::from_utf8(&chunk) {
            Ok(t) => t.to_owned(),
            Err(e) => {
                return Some((Err(anyhow::anyhow!("UTF-8 error: {}", e)), StreamState::Done));
            }
        };

        line_buf.push_str(&text);

        // Parse all complete lines in the buffer.
        let mut tokens_from_chunk: Vec<String> = Vec::new();
        let mut done_seen = false;

        while let Some(pos) = line_buf.find('\n') {
            let line: String = line_buf.drain(..=pos).collect();
            let line = line.trim().to_owned();
            if line.is_empty() {
                continue;
            }

            let parsed: OllamaChatChunk = match serde_json::from_str(&line) {
                Ok(p) => p,
                Err(e) => {
                    return Some((
                        Err(anyhow::anyhow!("JSON parse error: {} (line: {})", e, line)),
                        StreamState::Done,
                    ));
                }
            };

            if parsed.done {
                let stats = CacheStats {
                    prompt_tokens: parsed.prompt_eval_count.unwrap_or(0) as usize,
                    generated_tokens: parsed.eval_count.unwrap_or(0) as usize,
                    prefill_ms: parsed.prompt_eval_duration.unwrap_or(0) as f64 / 1_000_000.0,
                    generation_ms: parsed.eval_duration.unwrap_or(0) as f64 / 1_000_000.0,
                    cache_hit_tokens: 0,
                };
                if let Ok(mut guard) = cache_stats_ref.lock() {
                    *guard = Some(stats);
                }
                done_seen = true;
                break;
            }

            // Yield content tokens; skip thinking tokens.
            if let Some(msg) = &parsed.message {
                if let Some(content) = &msg.content {
                    if !content.is_empty() {
                        tokens_from_chunk.push(content.clone());
                    }
                }
                // msg.thinking is intentionally ignored.
            }
        }

        if done_seen {
            // Emit any tokens collected before the done marker, then terminate.
            if tokens_from_chunk.is_empty() {
                return None;
            }
            let first = tokens_from_chunk.remove(0);
            // remaining tokens would be lost if we return Done — but done_seen means the
            // stream is over, so any remaining pending tokens are already drained into tokens_from_chunk.
            // Return Done after we've yielded them via pending.
            let next = if tokens_from_chunk.is_empty() {
                StreamState::Done
            } else {
                StreamState::Running {
                    byte_stream,
                    line_buf,
                    cache_stats_ref,
                    pending: tokens_from_chunk,
                }
            };
            return Some((Ok(first), next));
        }

        if !tokens_from_chunk.is_empty() {
            let first = tokens_from_chunk.remove(0);
            let next = StreamState::Running {
                byte_stream,
                line_buf,
                cache_stats_ref,
                pending: tokens_from_chunk,
            };
            return Some((Ok(first), next));
        }

        // No tokens yet — loop and read another byte chunk.
    }
}

// --- Unit tests ---

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(endpoint: &str, model: &str, thinking: bool) -> BackendConfig {
        BackendConfig {
            backend_type: "ollama".into(),
            endpoint: endpoint.into(),
            model: model.into(),
            slot_id: None,
            thinking,
            api_key_env: None,
        }
    }

    #[test]
    fn test_new_stores_config() {
        let cfg = make_config("http://localhost:11434", "llama3", false);
        let backend = OllamaBackend::new(&cfg);
        assert_eq!(backend.endpoint, "http://localhost:11434");
        assert_eq!(backend.model, "llama3");
        assert!(!backend.thinking);
    }

    #[test]
    fn test_new_thinking_flag() {
        let cfg = make_config("http://localhost:11434", "qwen3:0.6b", true);
        let backend = OllamaBackend::new(&cfg);
        assert!(backend.thinking);
    }

    #[test]
    fn test_last_cache_stats_initially_none() {
        let cfg = make_config("http://localhost:11434", "llama3", false);
        let backend = OllamaBackend::new(&cfg);
        assert!(backend.last_cache_stats().is_none());
    }

    #[tokio::test]
    async fn test_token_count_heuristic() {
        let cfg = make_config("http://localhost:11434", "llama3", false);
        let backend = OllamaBackend::new(&cfg);
        let count = backend.token_count("hello world 1234").await.unwrap();
        assert_eq!(count, "hello world 1234".len() / 4);
    }

    #[test]
    fn test_cache_stats_written_and_read() {
        let cfg = make_config("http://localhost:11434", "llama3", false);
        let backend = OllamaBackend::new(&cfg);

        {
            let mut guard = backend.last_cache_stats.lock().unwrap();
            *guard = Some(CacheStats {
                prompt_tokens: 42,
                generated_tokens: 10,
                prefill_ms: 5.0,
                generation_ms: 20.0,
                cache_hit_tokens: 0,
            });
        }

        let stats = backend.last_cache_stats().unwrap();
        assert_eq!(stats.prompt_tokens, 42);
        assert_eq!(stats.generated_tokens, 10);
    }

    #[test]
    fn test_done_chunk_duration_conversion() {
        // nanoseconds → milliseconds: 5_000_000 ns = 5.0 ms
        let ns: u64 = 5_000_000;
        let ms = ns as f64 / 1_000_000.0;
        assert!((ms - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_request_serialization() {
        let messages = vec![OllamaMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        let req = OllamaChatRequest {
            model: "llama3",
            messages: &messages,
            stream: true,
            think: false,
            options: OllamaOptions {
                num_predict: 256,
                temperature: 0.7,
            },
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "llama3");
        assert_eq!(json["stream"], true);
        assert_eq!(json["options"]["num_predict"], 256);
        // think=false should be skipped (skip_serializing_if)
        assert!(json.get("think").is_none());
    }

    #[test]
    fn test_request_serialization_with_thinking() {
        let messages: Vec<OllamaMessage> = vec![];
        let req = OllamaChatRequest {
            model: "qwen3",
            messages: &messages,
            stream: true,
            think: true,
            options: OllamaOptions {
                num_predict: 128,
                temperature: 0.5,
            },
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["think"], true);
    }

    #[test]
    fn test_done_chunk_deserialization() {
        let raw = r#"{"done":true,"prompt_eval_count":50,"eval_count":20,"prompt_eval_duration":3000000,"eval_duration":8000000}"#;
        let chunk: OllamaChatChunk = serde_json::from_str(raw).unwrap();
        assert!(chunk.done);
        assert_eq!(chunk.prompt_eval_count, Some(50));
        assert_eq!(chunk.eval_count, Some(20));
        let prefill_ms = chunk.prompt_eval_duration.unwrap() as f64 / 1_000_000.0;
        assert!((prefill_ms - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_token_chunk_deserialization() {
        let raw = r#"{"message":{"role":"assistant","content":"Hello"},"done":false}"#;
        let chunk: OllamaChatChunk = serde_json::from_str(raw).unwrap();
        assert!(!chunk.done);
        let msg = chunk.message.unwrap();
        assert_eq!(msg.content.unwrap(), "Hello");
    }

    #[test]
    fn test_thinking_chunk_deserialization() {
        let raw = r#"{"message":{"role":"assistant","content":"","thinking":"let me think"},"done":false}"#;
        let chunk: OllamaChatChunk = serde_json::from_str(raw).unwrap();
        assert!(!chunk.done);
        let msg = chunk.message.unwrap();
        // content is empty — we would not yield it
        assert_eq!(msg.content.unwrap_or_default(), "");
        assert_eq!(msg.thinking.unwrap(), "let me think");
    }
}
