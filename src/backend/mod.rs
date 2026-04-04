use async_trait::async_trait;
use futures::stream::BoxStream;
use anyhow::Result;

use crate::cache::CacheStats;
use crate::message::Message;

pub(crate) mod openai_compat;

#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "llama-server")]
pub mod llama_server;
#[cfg(feature = "openai")]
pub mod openai;
#[cfg(feature = "vllm")]
pub mod vllm;

#[async_trait]
pub trait Backend: Send + Sync {
    fn stream_completion(
        &self,
        prompt: &str,
        messages: &[Message],
        temperature: f64,
        max_tokens: usize,
        stop: &[String],
    ) -> BoxStream<'_, Result<String>>;

    async fn token_count(&self, text: &str) -> Result<usize>;
    fn last_cache_stats(&self) -> Option<CacheStats>;
}

#[allow(unused_variables)]
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

#[cfg(test)]
pub mod mock {
    use super::*;
    use futures::stream;
    use std::sync::Mutex;

    pub struct MockBackend {
        responses: Mutex<Vec<String>>,
    }

    impl MockBackend {
        pub fn new(responses: Vec<String>) -> Self {
            Self { responses: Mutex::new(responses) }
        }
    }

    #[async_trait]
    impl Backend for MockBackend {
        fn stream_completion(&self, _prompt: &str, _messages: &[Message], _temperature: f64, _max_tokens: usize, _stop: &[String]) -> BoxStream<'_, Result<String>> {
            let mut responses = self.responses.lock().unwrap();
            let text = if responses.is_empty() { String::new() } else { responses.remove(0) };
            Box::pin(stream::once(async move { Ok(text) }))
        }
        async fn token_count(&self, text: &str) -> Result<usize> { Ok(text.len() / 4) }
        fn last_cache_stats(&self) -> Option<CacheStats> { None }
    }
}
