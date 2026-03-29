use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Mutex;
use reqwest::Client;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::backend::Backend;
use crate::cache::CacheStats;
use crate::config::BackendConfig;
use crate::message::Message;

pub struct OpenAIBackend {
    client: Client,
    endpoint: String,
    model: String,
    api_key: String,
    last_cache_stats: Mutex<Option<CacheStats>>,
}

impl OpenAIBackend {
    pub fn new(config: &BackendConfig) -> Result<Self> {
        let env_var = config.api_key_env.as_deref().unwrap_or("OPENAI_API_KEY");
        let api_key = std::env::var(env_var)
            .with_context(|| format!("API key env var '{}' not set", env_var))?;

        if api_key.is_empty() {
            anyhow::bail!("API key env var '{}' is empty", env_var);
        }

        let endpoint = if config.endpoint.is_empty() {
            "https://api.openai.com/v1".to_string()
        } else {
            config.endpoint.trim_end_matches('/').to_string()
        };

        let client = Client::new();

        Ok(Self {
            client,
            endpoint,
            model: config.model.clone(),
            api_key,
            last_cache_stats: Mutex::new(None),
        })
    }
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
struct SseUsage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
}

#[derive(Deserialize, Debug)]
struct SseChunk {
    choices: Option<Vec<SseChoice>>,
    usage: Option<SseUsage>,
}

// ----- Backend implementation -----

#[async_trait]
impl Backend for OpenAIBackend {
    fn stream_completion(
        &self,
        _prompt: &str,
        messages: &[Message],
        temperature: f64,
        max_tokens: usize,
        stop: &[String],
    ) -> BoxStream<'_, Result<String>> {
        let url = format!("{}/chat/completions", self.endpoint);
        let body = ChatRequest {
            model: self.model.clone(),
            messages: messages.to_vec(),
            stream: true,
            temperature,
            max_tokens,
            stop: stop.to_vec(),
            stream_options: StreamOptions { include_usage: true },
        };

        let (tx, rx) = mpsc::channel::<Result<String>>(64);
        let client = self.client.clone();
        let api_key = self.api_key.clone();

        // We need to send stats back; use a second one-shot channel to retrieve them.
        let (stats_tx, stats_rx) = mpsc::channel::<CacheStats>(1);

        tokio::spawn(async move {
            let response = match client
                .post(&url)
                .bearer_auth(&api_key)
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(Err(anyhow::anyhow!("Request failed: {}", e))).await;
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let body_text = response.text().await.unwrap_or_default();
                let _ = tx
                    .send(Err(anyhow::anyhow!("API error {}: {}", status, body_text)))
                    .await;
                return;
            }

            let mut byte_stream = response.bytes_stream();
            let mut buf = String::new();
            let mut prompt_tokens: usize = 0;
            let mut completion_tokens: usize = 0;

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

                // Process all complete lines in the buffer
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
                                cache_hit_tokens: 0,
                            };
                            let _ = stats_tx.send(stats).await;
                            break 'outer;
                        }

                        // Parse SSE data as JSON
                        let sse_chunk: SseChunk = match serde_json::from_str(&data) {
                            Ok(c) => c,
                            Err(_) => {
                                // Check for API-level error object
                                if let Ok(v) = serde_json::from_str::<Value>(&data) {
                                    if let Some(err) = v.get("error") {
                                        let _ = tx
                                            .send(Err(anyhow::anyhow!("API error: {}", err)))
                                            .await;
                                        return;
                                    }
                                }
                                continue;
                            }
                        };

                        // Capture usage from final chunk (stream_options: include_usage)
                        if let Some(usage) = &sse_chunk.usage {
                            if let Some(pt) = usage.prompt_tokens {
                                prompt_tokens = pt as usize;
                            }
                            if let Some(ct) = usage.completion_tokens {
                                completion_tokens = ct as usize;
                            }
                        }

                        // Yield delta content tokens
                        if let Some(choices) = &sse_chunk.choices {
                            for choice in choices {
                                if let Some(content) = &choice.delta.content {
                                    if !content.is_empty() {
                                        if tx.send(Ok(content.clone())).await.is_err() {
                                            // Receiver dropped; stop.
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        break; // No complete line yet — wait for more data
                    }
                }
            }

            // EOF without explicit [DONE] — record whatever stats we have
            let stats = CacheStats {
                prompt_tokens,
                generated_tokens: completion_tokens,
                prefill_ms: 0.0,
                generation_ms: 0.0,
                cache_hit_tokens: 0,
            };
            let _ = stats_tx.send(stats).await;
        });

        // Wrap the receiver in a stream; store stats when available via a separate future.
        // Because last_cache_stats is on &self we need a different approach:
        // we build a stream that, after the token stream ends, updates cache stats.
        // We do this by chaining: token stream then a "finalizer" future stored separately.
        //
        // Simplest approach: collect stats lazily from the stats_rx by storing a
        // shared Arc<Mutex<...>>. But that requires Arc. Instead we use the fact that
        // last_cache_stats is already a Mutex on self — we capture a raw pointer.
        //
        // Safe because: self is `&'_ self` and the BoxStream has the same lifetime '_.
        // The stream is polled inside the same lifetime scope as self.
        let stats_ptr: *const Mutex<Option<CacheStats>> = &self.last_cache_stats;

        let token_stream = ReceiverStream::new(rx);

        // We wrap the stream to intercept completion: use a custom stream wrapper.
        let wrapped = TokenStreamWithStats {
            inner: token_stream,
            stats_rx,
            stats_ptr,
            done: false,
        };

        Box::pin(wrapped)
    }

    async fn token_count(&self, text: &str) -> Result<usize> {
        // Rough approximation: 1 token ≈ 4 chars
        Ok(text.len() / 4)
    }

    fn last_cache_stats(&self) -> Option<CacheStats> {
        self.last_cache_stats.lock().unwrap().clone()
    }
}

// A wrapper stream that updates cache stats when the inner stream ends.
struct TokenStreamWithStats {
    inner: ReceiverStream<Result<String>>,
    stats_rx: mpsc::Receiver<CacheStats>,
    stats_ptr: *const Mutex<Option<CacheStats>>,
    done: bool,
}

// SAFETY: stats_ptr points to a field of OpenAIBackend which is Send + Sync.
// The stream's lifetime is tied to &'_ OpenAIBackend so the pointer is valid
// for the duration of the stream.
unsafe impl Send for TokenStreamWithStats {}
unsafe impl Sync for TokenStreamWithStats {}

impl futures::Stream for TokenStreamWithStats {
    type Item = Result<String>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        use std::task::Poll;

        if self.done {
            return Poll::Ready(None);
        }

        // Poll the inner token stream first
        match std::pin::Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
            Poll::Ready(None) => {
                // Inner stream ended — try to grab stats
                self.done = true;
                // Non-blocking try_recv
                if let Ok(stats) = self.stats_rx.try_recv() {
                    // SAFETY: pointer valid for lifetime '_ of the Backend reference.
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
