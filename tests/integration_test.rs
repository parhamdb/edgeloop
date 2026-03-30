//! Integration tests — require Ollama running with qwen2.5-coder:7b

use std::collections::HashMap;
use std::sync::Arc;

// We test the full pipeline: backend → agent → tool execution
// These tests hit real Ollama so they're slow and need the server running.

#[cfg(test)]
mod ollama_integration {
    use super::*;

    fn make_ollama_config() -> (edgeloop::config::AgentConfig, edgeloop::config::CacheConfig, edgeloop::config::BackendConfig) {
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
        let backend_cfg = edgeloop::config::BackendConfig {
            backend_type: "ollama".into(),
            endpoint: "http://localhost:11434".into(),
            model: "qwen2.5-coder:7b".into(),
            slot_id: None,
            n_keep: None,
            keep_alive: None,
            thinking: false,
            api_key_env: None,
        };
        (agent, cache, backend_cfg)
    }

    fn make_tool(name: &str, cmd: &str, params: Vec<(&str, &str, bool)>) -> edgeloop::config::ToolDef {
        let mut parameters = HashMap::new();
        for (pname, ptype, required) in params {
            parameters.insert(pname.to_string(), edgeloop::config::ParamDef {
                param_type: ptype.to_string(),
                required,
                default: None,
            });
        }
        edgeloop::config::ToolDef {
            name: name.to_string(),
            description: format!("Test tool: {}", name),
            command: cmd.to_string(),
            stdin: None,
            timeout: 10,
            workdir: None,
            env: HashMap::new(),
            parameters,
        }
    }

    #[tokio::test]
    async fn test_simple_chat() {
        let (agent_cfg, cache_cfg, backend_cfg) = make_ollama_config();
        let backend = Arc::new(edgeloop::backend::ollama::OllamaBackend::new(&backend_cfg));
        let agent = edgeloop::agent::Agent::new(backend, vec![], &agent_cfg, &cache_cfg);

        let response = agent.run("What is 2+2? Answer with just the number.").await;
        assert!(response.contains("4"), "Expected '4' in response: {}", response);
    }

    #[tokio::test]
    async fn test_tool_call() {
        let (agent_cfg, cache_cfg, backend_cfg) = make_ollama_config();
        let backend = Arc::new(edgeloop::backend::ollama::OllamaBackend::new(&backend_cfg));
        let calculator = make_tool("calculator", "python3 -c 'print(eval(\"{expression}\"))'", vec![("expression", "string", true)]);
        let agent = edgeloop::agent::Agent::new(backend, vec![calculator], &agent_cfg, &cache_cfg);

        let response = agent.run("What is 123 * 456? Use the calculator tool.").await;
        let clean = response.replace(",", "").replace(" ", "");
        assert!(clean.contains("56088"), "Expected '56088' in response: {}", response);
    }

    #[tokio::test]
    async fn test_file_tool() {
        let (agent_cfg, cache_cfg, backend_cfg) = make_ollama_config();
        let backend = Arc::new(edgeloop::backend::ollama::OllamaBackend::new(&backend_cfg));
        let read_file = make_tool("read_file", "cat {path}", vec![("path", "string", true)]);
        let agent = edgeloop::agent::Agent::new(backend, vec![read_file], &agent_cfg, &cache_cfg);

        let response = agent.run("Read the file Cargo.toml and tell me the package name.").await;
        assert!(response.contains("edgeloop"), "Expected 'edgeloop' in response: {}", response);
    }

    #[tokio::test]
    async fn test_no_tool_needed() {
        let (agent_cfg, cache_cfg, backend_cfg) = make_ollama_config();
        let backend = Arc::new(edgeloop::backend::ollama::OllamaBackend::new(&backend_cfg));
        let calculator = make_tool("calculator", "python3 -c 'print(eval(\"{expression}\"))'", vec![("expression", "string", true)]);
        let agent = edgeloop::agent::Agent::new(backend, vec![calculator], &agent_cfg, &cache_cfg);

        let response = agent.run("What is the capital of Japan?").await;
        assert!(response.to_lowercase().contains("tokyo"), "Expected 'tokyo' in response: {}", response);
    }

    #[tokio::test]
    async fn test_multiple_turns_performance() {
        let (agent_cfg, cache_cfg, backend_cfg) = make_ollama_config();
        let backend = Arc::new(edgeloop::backend::ollama::OllamaBackend::new(&backend_cfg));
        let agent = edgeloop::agent::Agent::new(backend, vec![], &agent_cfg, &cache_cfg);

        let start = std::time::Instant::now();
        let _ = agent.run("Say hello.").await;
        let first = start.elapsed();

        let start = std::time::Instant::now();
        let _ = agent.run("Say goodbye.").await;
        let second = start.elapsed();

        println!("First: {:?}, Second: {:?}", first, second);
        // Both should complete in reasonable time
        assert!(first.as_secs() < 10, "First response too slow: {:?}", first);
        assert!(second.as_secs() < 10, "Second response too slow: {:?}", second);
    }
}
