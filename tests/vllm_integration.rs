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
    use edgeloop::backend::Backend;

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
            parallel_tools: false, stream_tokens: false,
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

        let response = agent.run("What is 2+2? Answer with just the number.", &[], None, "test").await;
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

        let response = agent.run("What is 123 * 456? Use the calculator tool.", &[], None, "test").await;
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
        let _ = agent.run("Say hello.", &[], None, "test").await;
        // Second request should hit prefix cache (same system prompt)
        let _ = agent.run("Say goodbye.", &[], None, "test").await;

        let cache = agent.cache.lock().await;
        println!("Cache hit ratio: {:.1}%", cache.overall_cache_hit_ratio() * 100.0);
    }

    #[tokio::test]
    async fn test_exact_token_count() {
        if vllm_endpoint().is_none() {
            eprintln!("Skipping: VLLM_ENDPOINT not set");
            return;
        }

        let backend_cfg = make_config();
        let backend: edgeloop::backend::vllm::VllmBackend = edgeloop::backend::vllm::VllmBackend::new(&backend_cfg, &[]);
        let count = backend.token_count("Hello, world!").await.unwrap();
        println!("Token count for 'Hello, world!': {}", count);
        assert!(count > 0 && count < 20, "Unexpected token count: {}", count);
    }
}
