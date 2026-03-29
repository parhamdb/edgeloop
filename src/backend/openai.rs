use async_trait::async_trait;
use futures::stream::BoxStream;
use anyhow::Result;
use crate::backend::Backend;
use crate::cache::CacheStats;
use crate::config::BackendConfig;
use crate::message::Message;

pub struct OpenAIBackend;
impl OpenAIBackend {
    pub fn new(_config: &BackendConfig) -> Result<Self> { Ok(Self) }
}

#[async_trait]
impl Backend for OpenAIBackend {
    fn stream_completion(&self, _prompt: &str, _messages: &[Message], _temperature: f64, _max_tokens: usize, _stop: &[String]) -> BoxStream<'_, Result<String>> {
        Box::pin(futures::stream::once(async { anyhow::bail!("OpenAIBackend not yet implemented") }))
    }
    async fn token_count(&self, text: &str) -> Result<usize> { Ok(text.len() / 4) }
    fn last_cache_stats(&self) -> Option<CacheStats> { None }
}
