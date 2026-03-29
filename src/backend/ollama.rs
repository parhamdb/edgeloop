use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::backend::Backend;
use crate::cache::CacheStats;
use crate::config::BackendConfig;
use crate::message::Message;

pub struct OllamaBackend {
    client: reqwest::Client,
    endpoint: String,
    model: String,
    thinking: bool,
    last_cache_stats: Mutex<Option<CacheStats>>,
}

impl OllamaBackend {
    pub fn new(config: &BackendConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            endpoint: config.endpoint.trim_end_matches('/').to_string(),
            model: config.model.clone(),
            thinking: config.thinking,
            last_cache_stats: Mutex::new(None),
        }
    }
}

// --- Request / Response types ---

#[derive(Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
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
    prompt_eval_count: Option<u64>,
    eval_count: Option<u64>,
    prompt_eval_duration: Option<u64>, // nanoseconds
    eval_duration: Option<u64>,        // nanoseconds
}

#[derive(Deserialize, Debug)]
struct OllamaChunkMessage {
    content: Option<String>,
    #[allow(dead_code)]
    thinking: Option<String>,
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
        let url = format!("{}/api/chat", self.endpoint);
        let body = OllamaChatRequest {
            model: self.model.clone(),
            messages: messages
                .iter()
                .map(|m| OllamaMessage { role: m.role.clone(), content: m.content.clone() })
                .collect(),
            stream: true,
            think: self.thinking,
            options: OllamaOptions { num_predict: max_tokens, temperature },
        };

        let (tx, rx) = mpsc::channel::<Result<String>>(64);
        let (stats_tx, stats_rx) = mpsc::channel::<CacheStats>(1);
        let client = self.client.clone();

        tokio::spawn(async move {
            let response = match client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx
                        .send(Err(anyhow::anyhow!(
                            "ConnectionError: could not connect to Ollama at {}: {}",
                            url,
                            e
                        )))
                        .await;
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let body_text = response.text().await.unwrap_or_default();
                let _ = tx
                    .send(Err(anyhow::anyhow!("Ollama API error {}: {}", status, body_text)))
                    .await;
                return;
            }

            let mut byte_stream = response.bytes_stream();
            let mut buf = String::new();

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

                        let parsed: OllamaChatChunk = match serde_json::from_str(&line) {
                            Ok(p) => p,
                            Err(e) => {
                                let _ = tx
                                    .send(Err(anyhow::anyhow!(
                                        "JSON parse error: {} (line: {})",
                                        e,
                                        line
                                    )))
                                    .await;
                                return;
                            }
                        };

                        if parsed.done {
                            let stats = CacheStats {
                                prompt_tokens: parsed.prompt_eval_count.unwrap_or(0) as usize,
                                generated_tokens: parsed.eval_count.unwrap_or(0) as usize,
                                prefill_ms: parsed.prompt_eval_duration.unwrap_or(0) as f64
                                    / 1_000_000.0,
                                generation_ms: parsed.eval_duration.unwrap_or(0) as f64
                                    / 1_000_000.0,
                                cache_hit_tokens: 0,
                            };
                            let _ = stats_tx.send(stats).await;
                            break 'outer;
                        }

                        if let Some(msg) = &parsed.message {
                            if let Some(content) = &msg.content {
                                if !content.is_empty() {
                                    if tx.send(Ok(content.clone())).await.is_err() {
                                        return;
                                    }
                                }
                            }
                        }
                    } else {
                        break;
                    }
                }
            }

            let _ = stats_tx
                .send(CacheStats {
                    prompt_tokens: 0,
                    generated_tokens: 0,
                    prefill_ms: 0.0,
                    generation_ms: 0.0,
                    cache_hit_tokens: 0,
                })
                .await;
        });

        let stats_ptr: *const Mutex<Option<CacheStats>> = &self.last_cache_stats;
        let token_stream = ReceiverStream::new(rx);
        let wrapped = OllamaTokenStream { inner: token_stream, stats_rx, stats_ptr, done: false };
        Box::pin(wrapped)
    }

    async fn token_count(&self, text: &str) -> Result<usize> {
        Ok(text.len() / 4)
    }

    fn last_cache_stats(&self) -> Option<CacheStats> {
        self.last_cache_stats.lock().unwrap().clone()
    }
}

struct OllamaTokenStream {
    inner: ReceiverStream<Result<String>>,
    stats_rx: mpsc::Receiver<CacheStats>,
    stats_ptr: *const Mutex<Option<CacheStats>>,
    done: bool,
}

unsafe impl Send for OllamaTokenStream {}
unsafe impl Sync for OllamaTokenStream {}

impl futures::Stream for OllamaTokenStream {
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
    fn test_new_thinking_flag_and_trailing_slash_stripped() {
        let cfg = make_config("http://localhost:11434/", "qwen3:0.6b", true);
        let backend = OllamaBackend::new(&cfg);
        assert!(backend.thinking);
        assert_eq!(backend.endpoint, "http://localhost:11434");
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
        let text = "hello world 1234";
        let count = backend.token_count(text).await.unwrap();
        assert_eq!(count, text.len() / 4);
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
    fn test_done_chunk_duration_ns_to_ms() {
        let ns: u64 = 5_000_000;
        let ms = ns as f64 / 1_000_000.0;
        assert!((ms - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_request_serialization_no_thinking() {
        let body = OllamaChatRequest {
            model: "llama3".into(),
            messages: vec![OllamaMessage { role: "user".into(), content: "hi".into() }],
            stream: true,
            think: false,
            options: OllamaOptions { num_predict: 256, temperature: 0.7 },
        };
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["model"], "llama3");
        assert_eq!(json["stream"], true);
        assert_eq!(json["options"]["num_predict"], 256);
        assert!(json.get("think").is_none());
    }

    #[test]
    fn test_request_serialization_with_thinking() {
        let body = OllamaChatRequest {
            model: "qwen3".into(),
            messages: vec![],
            stream: true,
            think: true,
            options: OllamaOptions { num_predict: 128, temperature: 0.5 },
        };
        let json = serde_json::to_value(&body).unwrap();
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
        let gen_ms = chunk.eval_duration.unwrap() as f64 / 1_000_000.0;
        assert!((gen_ms - 8.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_token_chunk_deserialization() {
        let raw = r#"{"message":{"role":"assistant","content":"Hello"},"done":false}"#;
        let chunk: OllamaChatChunk = serde_json::from_str(raw).unwrap();
        assert!(!chunk.done);
        let msg = chunk.message.unwrap();
        assert_eq!(msg.content.unwrap(), "Hello");
        assert!(msg.thinking.is_none());
    }

    #[test]
    fn test_thinking_chunk_deserialization() {
        let raw = r#"{"message":{"role":"assistant","content":"","thinking":"let me think"},"done":false}"#;
        let chunk: OllamaChatChunk = serde_json::from_str(raw).unwrap();
        assert!(!chunk.done);
        let msg = chunk.message.unwrap();
        assert_eq!(msg.content.unwrap_or_default(), "");
        assert_eq!(msg.thinking.unwrap(), "let me think");
    }
}
