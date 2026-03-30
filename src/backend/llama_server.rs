use async_trait::async_trait;
use futures::stream::BoxStream;
use anyhow::Result;
use serde::Deserialize;
use std::sync::{Arc, Mutex};

use crate::backend::Backend;
use crate::cache::CacheStats;
use crate::config::BackendConfig;
use crate::message::Message;

pub struct LlamaServerBackend {
    client: reqwest::Client,
    endpoint: String,
    slot_id: Option<usize>,
    n_keep: Option<i32>,
    last_stats: Arc<Mutex<Option<CacheStats>>>,
}

impl LlamaServerBackend {
    pub fn new(config: &BackendConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            endpoint: config.endpoint.trim_end_matches('/').to_owned(),
            slot_id: config.slot_id,
            n_keep: config.n_keep,
            last_stats: Arc::new(Mutex::new(None)),
        }
    }
}

#[derive(Deserialize)]
struct CompletionChunk {
    content: String,
    #[serde(default)]
    stop: bool,
    #[serde(default)]
    timings: Option<Timings>,
}

#[derive(Deserialize)]
struct Timings {
    #[serde(default)]
    prompt_n: usize,
    #[serde(default)]
    prompt_ms: f64,
    #[serde(default)]
    predicted_n: usize,
    #[serde(default)]
    predicted_ms: f64,
}

#[derive(Deserialize)]
struct TokenizeResponse {
    tokens: Vec<serde_json::Value>,
}

#[async_trait]
impl Backend for LlamaServerBackend {
    fn stream_completion(
        &self,
        prompt: &str,
        _messages: &[Message],
        temperature: f64,
        max_tokens: usize,
        stop: &[String],
    ) -> BoxStream<'_, Result<String>> {
        use tokio_stream::wrappers::ReceiverStream;

        let url = format!("{}/completion", self.endpoint);
        let mut body = serde_json::json!({
            "prompt": prompt,
            "stream": true,
            "cache_prompt": true,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": stop,
        });
        if let Some(slot_id) = self.slot_id {
            body["id_slot"] = serde_json::json!(slot_id);
        }
        if let Some(n_keep) = self.n_keep {
            body["n_keep"] = serde_json::json!(n_keep);
        }

        let client = self.client.clone();
        let stats_ref = Arc::clone(&self.last_stats);

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String>>(64);

        tokio::spawn(async move {
            let response = match client.post(&url).json(&body).send().await {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(Err(anyhow::anyhow!("Failed to connect to llama-server: {}", e))).await;
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let body_text = response.text().await.unwrap_or_default();
                let _ = tx.send(Err(anyhow::anyhow!(
                    "llama-server returned {}: {}", status, body_text
                ))).await;
                return;
            }

            use futures::StreamExt;
            let mut byte_stream = response.bytes_stream();
            let mut buf = String::new();

            while let Some(chunk_result) = byte_stream.next().await {
                let bytes = match chunk_result {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = tx.send(Err(anyhow::anyhow!("Stream read error: {}", e))).await;
                        return;
                    }
                };
                let text = match std::str::from_utf8(&bytes) {
                    Ok(s) => s.to_owned(),
                    Err(e) => {
                        let _ = tx.send(Err(anyhow::anyhow!("UTF-8 decode error: {}", e))).await;
                        return;
                    }
                };
                buf.push_str(&text);

                loop {
                    match buf.find('\n') {
                        None => break,
                        Some(pos) => {
                            let line: String = buf[..pos].trim_end_matches('\r').to_owned();
                            buf = buf[pos + 1..].to_owned();

                            if line.is_empty() {
                                continue;
                            }

                            let data = match line.strip_prefix("data: ") {
                                Some(d) => d.to_owned(),
                                None => continue,
                            };

                            let chunk: CompletionChunk = match serde_json::from_str(&data) {
                                Ok(c) => c,
                                Err(e) => {
                                    let _ = tx.send(Err(anyhow::anyhow!(
                                        "Failed to parse SSE chunk '{}': {}", data, e
                                    ))).await;
                                    return;
                                }
                            };

                            if chunk.stop {
                                if let Some(timings) = chunk.timings {
                                    let stats = CacheStats {
                                        prompt_tokens: timings.prompt_n,
                                        generated_tokens: timings.predicted_n,
                                        prefill_ms: timings.prompt_ms,
                                        generation_ms: timings.predicted_ms,
                                        cache_hit_tokens: 0,
                                    };
                                    if let Ok(mut guard) = stats_ref.lock() {
                                        *guard = Some(stats);
                                    }
                                }
                                return;
                            }

                            if !chunk.content.is_empty() {
                                if tx.send(Ok(chunk.content)).await.is_err() {
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        });

        Box::pin(ReceiverStream::new(rx))
    }

    async fn token_count(&self, text: &str) -> Result<usize> {
        let url = format!("{}/tokenize", self.endpoint);
        match self
            .client
            .post(&url)
            .json(&serde_json::json!({ "content": text }))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => match resp.json::<TokenizeResponse>().await {
                Ok(body) => Ok(body.tokens.len()),
                Err(_) => Ok(text.len() / 4),
            },
            _ => Ok(text.len() / 4),
        }
    }

    fn last_cache_stats(&self) -> Option<CacheStats> {
        self.last_stats.lock().ok()?.clone()
    }
}

// --- Unit tests ---

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(endpoint: &str, slot_id: Option<usize>) -> BackendConfig {
        BackendConfig {
            backend_type: "llama-server".into(),
            endpoint: endpoint.into(),
            model: String::new(),
            slot_id,
            n_keep: None,
            keep_alive: None,
            thinking: false,
            api_key_env: None,
        }
    }

    #[test]
    fn test_new_stores_config() {
        let cfg = make_config("http://localhost:8080", Some(2));
        let backend = LlamaServerBackend::new(&cfg);
        assert_eq!(backend.endpoint, "http://localhost:8080");
        assert_eq!(backend.slot_id, Some(2));
    }

    #[test]
    fn test_new_trailing_slash_stripped() {
        let cfg = make_config("http://localhost:8080/", None);
        let backend = LlamaServerBackend::new(&cfg);
        assert_eq!(backend.endpoint, "http://localhost:8080");
    }

    #[test]
    fn test_last_cache_stats_initially_none() {
        let cfg = make_config("http://localhost:8080", None);
        let backend = LlamaServerBackend::new(&cfg);
        assert!(backend.last_cache_stats().is_none());
    }

    #[tokio::test]
    async fn test_token_count_fallback() {
        // No server running — should fall back to len/4.
        let cfg = make_config("http://127.0.0.1:19999", None);
        let backend = LlamaServerBackend::new(&cfg);
        let text = "hello world 1234";
        let count = backend.token_count(text).await.unwrap();
        assert_eq!(count, text.len() / 4);
    }

    #[test]
    fn test_cache_stats_written_and_read() {
        let cfg = make_config("http://localhost:8080", None);
        let backend = LlamaServerBackend::new(&cfg);
        {
            let mut guard = backend.last_stats.lock().unwrap();
            *guard = Some(CacheStats {
                prompt_tokens: 100,
                generated_tokens: 30,
                prefill_ms: 12.5,
                generation_ms: 80.0,
                cache_hit_tokens: 0,
            });
        }
        let stats = backend.last_cache_stats().unwrap();
        assert_eq!(stats.prompt_tokens, 100);
        assert_eq!(stats.generated_tokens, 30);
    }

    #[test]
    fn test_completion_chunk_deserialization() {
        let raw = r#"{"content":"Hello","stop":false}"#;
        let chunk: CompletionChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.content, "Hello");
        assert!(!chunk.stop);
        assert!(chunk.timings.is_none());
    }

    #[test]
    fn test_stop_chunk_with_timings() {
        let raw = r#"{"content":"","stop":true,"timings":{"prompt_n":42,"prompt_ms":15.0,"predicted_n":10,"predicted_ms":50.0}}"#;
        let chunk: CompletionChunk = serde_json::from_str(raw).unwrap();
        assert!(chunk.stop);
        let t = chunk.timings.unwrap();
        assert_eq!(t.prompt_n, 42);
        assert!((t.prompt_ms - 15.0).abs() < f64::EPSILON);
        assert_eq!(t.predicted_n, 10);
        assert!((t.predicted_ms - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_request_body_includes_slot_id() {
        let slot_id: usize = 3;
        let mut body = serde_json::json!({
            "prompt": "hi",
            "stream": true,
            "cache_prompt": true,
            "n_predict": 512,
            "temperature": 0.7,
            "stop": [],
        });
        body["id_slot"] = serde_json::json!(slot_id);
        assert_eq!(body["id_slot"], 3);
        assert_eq!(body["cache_prompt"], true);
    }
}
