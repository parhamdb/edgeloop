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
    keep_alive: Option<String>,
    num_ctx: Option<usize>,
    seed: Option<i64>,
    last_stats: Mutex<Option<CacheStats>>,
}

impl OllamaBackend {
    pub fn new(config: &BackendConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            endpoint: config.endpoint.trim_end_matches('/').to_string(),
            model: config.model.clone(),
            thinking: config.thinking,
            keep_alive: config.keep_alive.clone(),
            num_ctx: config.num_ctx,
            seed: config.seed,
            last_stats: Mutex::new(None),
        }
    }
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMsg>,
    stream: bool,
    think: bool,
    options: ChatOptions,
    /// How long to keep model loaded. Number (seconds) or string ("30m").
    /// -1 = forever. Omitted = Ollama default.
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct ChatMsg {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatOptions {
    num_predict: usize,
    temperature: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_ctx: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
}

#[derive(Deserialize)]
struct ChatChunk {
    message: Option<ChunkMessage>,
    done: bool,
    prompt_eval_count: Option<u64>,
    eval_count: Option<u64>,
    prompt_eval_duration: Option<u64>,
    eval_duration: Option<u64>,
}

#[derive(Deserialize)]
struct ChunkMessage {
    content: Option<String>,
    #[allow(dead_code)]
    thinking: Option<String>,
}

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
        let body = ChatRequest {
            model: self.model.clone(),
            messages: messages.iter().map(|m| ChatMsg {
                role: m.role.clone(),
                content: m.content.clone(),
            }).collect(),
            stream: true,
            think: self.thinking,
            options: ChatOptions { num_predict: max_tokens, temperature, num_ctx: self.num_ctx, seed: self.seed },
            keep_alive: self.keep_alive.as_ref().map(|v| {
                // Try to parse as integer first (e.g., "-1", "0", "300")
                if let Ok(n) = v.parse::<i64>() {
                    serde_json::Value::Number(n.into())
                } else {
                    // Otherwise send as string (e.g., "30m", "1h")
                    serde_json::Value::String(v.clone())
                }
            }),
        };

        let client = self.client.clone();
        let (tx, rx) = mpsc::channel::<Result<String>>(64);

        // Clone a reference to last_stats for the spawned task
        let stats_mutex = &self.last_stats as *const Mutex<Option<CacheStats>>;

        // SAFETY: The OllamaBackend outlives the stream because the Agent holds
        // an Arc<dyn Backend> and awaits the stream within the same run() call.
        let stats_ptr = stats_mutex as usize;

        tokio::spawn(async move {
            let response = match client.post(&url).json(&body).send().await {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(Err(anyhow::anyhow!("Cannot connect to Ollama: {}", e))).await;
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                let _ = tx.send(Err(anyhow::anyhow!("Ollama error {}: {}", status, text))).await;
                return;
            }

            let mut stream = response.bytes_stream();
            let mut buf = String::new();

            while let Some(chunk) = stream.next().await {
                let bytes = match chunk {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = tx.send(Err(anyhow::anyhow!("Stream error: {}", e))).await;
                        return;
                    }
                };

                buf.push_str(&String::from_utf8_lossy(&bytes));

                // Process complete lines
                while let Some(pos) = buf.find('\n') {
                    let line = buf[..pos].trim().to_string();
                    buf = buf[pos + 1..].to_string();

                    if line.is_empty() { continue; }

                    let parsed: ChatChunk = match serde_json::from_str(&line) {
                        Ok(p) => p,
                        Err(_) => continue, // skip unparseable lines
                    };

                    if parsed.done {
                        // Record cache stats
                        let stats = CacheStats {
                            prompt_tokens: parsed.prompt_eval_count.unwrap_or(0) as usize,
                            generated_tokens: parsed.eval_count.unwrap_or(0) as usize,
                            prefill_ms: parsed.prompt_eval_duration.unwrap_or(0) as f64 / 1_000_000.0,
                            generation_ms: parsed.eval_duration.unwrap_or(0) as f64 / 1_000_000.0,
                            cache_hit_tokens: 0,
                        };
                        // SAFETY: see above — backend outlives stream
                        unsafe {
                            let mutex = &*(stats_ptr as *const Mutex<Option<CacheStats>>);
                            *mutex.lock().unwrap() = Some(stats);
                        }
                        return; // stream complete
                    }

                    if let Some(msg) = &parsed.message {
                        if let Some(content) = &msg.content {
                            if !content.is_empty() {
                                if tx.send(Ok(content.clone())).await.is_err() {
                                    return; // receiver dropped
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
        Ok(text.len() / 4)
    }

    fn last_cache_stats(&self) -> Option<CacheStats> {
        self.last_stats.lock().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let config = BackendConfig {
            backend_type: "ollama".into(),
            endpoint: "http://localhost:11434".into(),
            model: "qwen3:0.6b".into(),
            slot_id: None,
            n_keep: None,
            keep_alive: None,
            thinking: false, grammar: None, seed: None, num_ctx: None, cache_reuse: None,
            api_key_env: None,
            min_tokens: None, repetition_penalty: None, top_k: None, min_p: None,
            stop_token_ids: None, truncate_prompt_tokens: None, guided_mode: None, guided_pattern: None,
        };
        let backend = OllamaBackend::new(&config);
        assert_eq!(backend.model, "qwen3:0.6b");
        assert_eq!(backend.endpoint, "http://localhost:11434");
    }

    #[test]
    fn test_parse_chunk() {
        let json = r#"{"message":{"role":"assistant","content":"Hello"},"done":false}"#;
        let chunk: ChatChunk = serde_json::from_str(json).unwrap();
        assert!(!chunk.done);
        assert_eq!(chunk.message.unwrap().content.unwrap(), "Hello");
    }

    #[test]
    fn test_parse_done_chunk() {
        let json = r#"{"done":true,"prompt_eval_count":42,"eval_count":10,"prompt_eval_duration":5000000,"eval_duration":3000000}"#;
        let chunk: ChatChunk = serde_json::from_str(json).unwrap();
        assert!(chunk.done);
        assert_eq!(chunk.prompt_eval_count, Some(42));
        assert_eq!(chunk.eval_duration, Some(3000000));
    }
}
