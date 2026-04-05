//! Full integration tests — llama-server backend, parallel tool calls, Gemma 4 template
//!
//! Requires:
//!   - Ollama running with gemma4:26b (for parallel tool tests)
//!   - llama-server on port 8090 (for llama-server backend tests)

use std::collections::HashMap;
use std::sync::Arc;

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tool(name: &str, cmd: &str, params: Vec<(&str, &str, bool)>) -> edgeloop::config::ToolDef {
        let mut parameters = HashMap::new();
        for (pname, ptype, required) in params {
            parameters.insert(pname.to_string(), edgeloop::config::ParamDef {
                param_type: ptype.to_string(), required, default: None,
            });
        }
        edgeloop::config::ToolDef {
            name: name.to_string(),
            description: format!("Test tool: {}", name),
            command: cmd.to_string(),
            stdin: None, timeout: 10, workdir: None,
            env: HashMap::new(), parameters,
        }
    }

    fn backend_config(backend_type: &str, endpoint: &str, model: &str) -> edgeloop::config::BackendConfig {
        edgeloop::config::BackendConfig {
            backend_type: backend_type.into(),
            endpoint: endpoint.into(),
            model: model.into(),
            slot_id: None, n_keep: None, keep_alive: None, thinking: false,
            grammar: None, seed: None, num_ctx: None, cache_reuse: None, api_key_env: None,
            min_tokens: None, repetition_penalty: None, top_k: None, min_p: None,
            stop_token_ids: None, truncate_prompt_tokens: None, guided_mode: None, guided_pattern: None,
        }
    }

    fn backend_config_thinking(backend_type: &str, endpoint: &str, model: &str) -> edgeloop::config::BackendConfig {
        let mut cfg = backend_config(backend_type, endpoint, model);
        cfg.thinking = true;
        cfg
    }

    // ─── llama-server backend ───────────────────────────────────────

    fn llama_server_available() -> bool {
        std::net::TcpStream::connect("127.0.0.1:8090").is_ok()
    }

    #[tokio::test]
    async fn test_llama_server_simple_chat() {
        if !llama_server_available() {
            eprintln!("Skipping: llama-server not running on :8090");
            return;
        }
        let agent_cfg = edgeloop::config::AgentConfig {
            system_prompt: "You are a helpful assistant.".into(),
            template: "chatml".into(),
            max_tokens: 4096, max_iterations: 8, max_retries: 2, temperature: 0.1,
            parallel_tools: false, stream_tokens: false,
        };
        let cache_cfg = edgeloop::config::CacheConfig { max_context: 4096, truncation_threshold: 0.8 };
        let backend_cfg = backend_config("llama-server", "http://localhost:8090", "");
        let backend = Arc::new(edgeloop::backend::llama_server::LlamaServerBackend::new(&backend_cfg));
        let agent = edgeloop::agent::Agent::new(backend, vec![], &agent_cfg, &cache_cfg);

        let response = agent.run("What is 2+2? Answer with just the number.", &[], None, "test").await;
        println!("[llama-server simple] {}", response);
        assert!(response.contains("4"), "Expected '4' in: {}", response);
    }

    #[tokio::test]
    async fn test_llama_server_tool_call() {
        if !llama_server_available() {
            eprintln!("Skipping: llama-server not running on :8090");
            return;
        }
        let agent_cfg = edgeloop::config::AgentConfig {
            system_prompt: "You are a helpful assistant.".into(),
            template: "chatml".into(),
            max_tokens: 4096, max_iterations: 8, max_retries: 2, temperature: 0.1,
            parallel_tools: false, stream_tokens: false,
        };
        let cache_cfg = edgeloop::config::CacheConfig { max_context: 4096, truncation_threshold: 0.8 };
        let backend_cfg = backend_config("llama-server", "http://localhost:8090", "");
        let backend = Arc::new(edgeloop::backend::llama_server::LlamaServerBackend::new(&backend_cfg));
        let calculator = make_tool("calculator", "python3 -c 'print(eval(\"{expression}\"))'", vec![("expression", "string", true)]);
        let agent = edgeloop::agent::Agent::new(backend, vec![calculator], &agent_cfg, &cache_cfg);

        let response = agent.run("What is 123 * 456? Use the calculator tool.", &[], None, "test").await;
        println!("[llama-server tool] {}", response);
        let clean = response.replace(",", "").replace(" ", "");
        assert!(clean.contains("56088"), "Expected '56088' in: {}", response);
    }

    #[tokio::test]
    async fn test_llama_server_multimodal_image() {
        if !llama_server_available() {
            eprintln!("Skipping: llama-server not running on :8090");
            return;
        }
        let agent_cfg = edgeloop::config::AgentConfig {
            system_prompt: "You are a helpful assistant. Describe images precisely.".into(),
            template: "chatml".into(),
            max_tokens: 4096, max_iterations: 8, max_retries: 2, temperature: 0.1,
            parallel_tools: false, stream_tokens: false,
        };
        let cache_cfg = edgeloop::config::CacheConfig { max_context: 4096, truncation_threshold: 0.8 };
        let backend_cfg = backend_config("llama-server", "http://localhost:8090", "");
        let backend = Arc::new(edgeloop::backend::llama_server::LlamaServerBackend::new(&backend_cfg));
        let agent = edgeloop::agent::Agent::new(backend, vec![], &agent_cfg, &cache_cfg);

        // Create a small 1x1 red PNG (67 bytes) as base64
        use base64::Engine;
        let red_pixel_png: [u8; 67] = [
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
            0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xDE, 0x00, 0x00, 0x00,
            0x0C, 0x49, 0x44, 0x41, 0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
            0x00, 0x00, 0x03, 0x00, 0x01, 0x36, 0x28, 0x19, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        let b64 = base64::engine::general_purpose::STANDARD.encode(&red_pixel_png);

        let images = vec![edgeloop::message::ImageAttachment {
            b64,
            description: Some("a small red image".into()),
            mime_type: Some("image/png".into()),
        }];

        let response = agent.run("What do you see in this image? Describe it.", &images, None, "test").await;
        println!("[llama-server multimodal] {}", response);
        // The model should produce some response about the image rather than saying it can't see anything
        assert!(!response.is_empty(), "Response should not be empty");
        // Should NOT contain the typical "I can't see" fallback from text-only mode
        let lower = response.to_lowercase();
        assert!(
            !lower.contains("i don't have") && !lower.contains("i cannot see") && !lower.contains("no image"),
            "Model appears to not have received the image: {}",
            response
        );
    }

    // ─── Ollama parallel tool calls ─────────────────────────────────

    fn ollama_available() -> bool {
        std::net::TcpStream::connect("127.0.0.1:11434").is_ok()
    }

    #[tokio::test]
    async fn test_ollama_parallel_tool_calls() {
        if !ollama_available() {
            eprintln!("Skipping: Ollama not running on :11434");
            return;
        }
        let agent_cfg = edgeloop::config::AgentConfig {
            system_prompt: "You are a helpful assistant. When asked for multiple pieces of info, call multiple tools at once using a JSON array.".into(),
            template: "chatml".into(),
            max_tokens: 4096, max_iterations: 8, max_retries: 2, temperature: 0.1,
            parallel_tools: true, stream_tokens: false,
        };
        let cache_cfg = edgeloop::config::CacheConfig { max_context: 4096, truncation_threshold: 0.8 };
        let backend_cfg = backend_config_thinking("ollama", "http://localhost:11434", "gemma4:26b");
        let backend = Arc::new(edgeloop::backend::ollama::OllamaBackend::new(&backend_cfg));
        let date_tool = make_tool("get_date", "date +%Y-%m-%d", vec![]);
        let uptime_tool = make_tool("get_uptime", "uptime -p", vec![]);
        let agent = edgeloop::agent::Agent::new(backend, vec![date_tool, uptime_tool], &agent_cfg, &cache_cfg);

        let response = agent.run("What is today's date and the system uptime? Use both tools.", &[], None, "test").await;
        println!("[ollama parallel] {}", response);
        // The model should have called at least one tool and produced a response
        assert!(!response.is_empty(), "Response should not be empty");
        // Check history — if parallel worked, should have fewer iterations
        let history_len = agent.history_len().await;
        println!("[ollama parallel] history_len={}", history_len);
        assert!(history_len >= 3, "Should have at least user + assistant(tool) + user(result) + assistant(final)");
    }

    #[tokio::test]
    async fn test_ollama_single_tool_still_works_with_parallel_flag() {
        if !ollama_available() {
            eprintln!("Skipping: Ollama not running on :11434");
            return;
        }
        let agent_cfg = edgeloop::config::AgentConfig {
            system_prompt: "You are a helpful assistant.".into(),
            template: "chatml".into(),
            max_tokens: 4096, max_iterations: 8, max_retries: 2, temperature: 0.1,
            parallel_tools: true, stream_tokens: false,
        };
        let cache_cfg = edgeloop::config::CacheConfig { max_context: 4096, truncation_threshold: 0.8 };
        // Gemma 4 needs thinking=true for tool calls — without it, the model returns empty content
        let backend_cfg = backend_config_thinking("ollama", "http://localhost:11434", "gemma4:26b");
        let backend = Arc::new(edgeloop::backend::ollama::OllamaBackend::new(&backend_cfg));
        let calculator = make_tool("calculator", "python3 -c 'print(eval(\"{expression}\"))'", vec![("expression", "string", true)]);
        let agent = edgeloop::agent::Agent::new(backend, vec![calculator], &agent_cfg, &cache_cfg);

        let response = agent.run("What is 99 * 99? Use the calculator.", &[], None, "test").await;
        println!("[ollama single+parallel] {}", response);
        let clean = response.replace(",", "").replace(" ", "");
        assert!(clean.contains("9801"), "Expected '9801' in: {}", response);
    }

    // ─── llama-server parallel tool calls ───────────────────────────

    #[tokio::test]
    async fn test_llama_server_parallel_tool_calls() {
        if !llama_server_available() {
            eprintln!("Skipping: llama-server not running on :8090");
            return;
        }
        let agent_cfg = edgeloop::config::AgentConfig {
            system_prompt: "You are a helpful assistant. When asked for multiple pieces of info, call multiple tools at once using a JSON array.".into(),
            template: "chatml".into(),
            max_tokens: 4096, max_iterations: 8, max_retries: 2, temperature: 0.1,
            parallel_tools: true, stream_tokens: false,
        };
        let cache_cfg = edgeloop::config::CacheConfig { max_context: 4096, truncation_threshold: 0.8 };
        let backend_cfg = backend_config("llama-server", "http://localhost:8090", "");
        let backend = Arc::new(edgeloop::backend::llama_server::LlamaServerBackend::new(&backend_cfg));
        let date_tool = make_tool("get_date", "date +%Y-%m-%d", vec![]);
        let uptime_tool = make_tool("get_uptime", "uptime -p", vec![]);
        let agent = edgeloop::agent::Agent::new(backend, vec![date_tool, uptime_tool], &agent_cfg, &cache_cfg);

        let response = agent.run("What is today's date and the system uptime? Use both tools.", &[], None, "test").await;
        println!("[llama-server parallel] {}", response);
        assert!(!response.is_empty(), "Response should not be empty");
    }

    // ─── Gemma 4 template with llama-server ─────────────────────────

    #[tokio::test]
    async fn test_llama_server_gemma4_template() {
        if !llama_server_available() {
            eprintln!("Skipping: llama-server not running on :8090");
            return;
        }
        // Note: This uses a qwen model with the gemma4 template.
        // The template tokens won't match the model's training, so this mainly
        // verifies the template formatting doesn't crash and the model still responds.
        // For a real Gemma 4 test, load a Gemma 4 GGUF.
        let agent_cfg = edgeloop::config::AgentConfig {
            system_prompt: "You are a helpful assistant.".into(),
            template: "gemma4".into(),
            max_tokens: 4096, max_iterations: 8, max_retries: 2, temperature: 0.1,
            parallel_tools: false, stream_tokens: false,
        };
        let cache_cfg = edgeloop::config::CacheConfig { max_context: 4096, truncation_threshold: 0.8 };
        let backend_cfg = backend_config("llama-server", "http://localhost:8090", "");
        let backend = Arc::new(edgeloop::backend::llama_server::LlamaServerBackend::new(&backend_cfg));
        let agent = edgeloop::agent::Agent::new(backend, vec![], &agent_cfg, &cache_cfg);

        let response = agent.run("Say hello in one word.", &[], None, "test").await;
        println!("[llama-server gemma4-template] {}", response);
        // Template mismatch with model, but should still produce some output
        assert!(!response.is_empty(), "Response should not be empty");
    }

    // ─── Ollama + Gemma 4 tool calling ──────────────────────────────

    #[tokio::test]
    async fn test_ollama_gemma4_simple_chat() {
        if !ollama_available() {
            eprintln!("Skipping: Ollama not running on :11434");
            return;
        }
        let agent_cfg = edgeloop::config::AgentConfig {
            system_prompt: "You are a helpful assistant.".into(),
            template: "gemma4".into(),
            max_tokens: 4096, max_iterations: 8, max_retries: 2, temperature: 0.1,
            parallel_tools: false, stream_tokens: false,
        };
        let cache_cfg = edgeloop::config::CacheConfig { max_context: 4096, truncation_threshold: 0.8 };
        let backend_cfg = backend_config("ollama", "http://localhost:11434", "gemma4:26b");
        let backend = Arc::new(edgeloop::backend::ollama::OllamaBackend::new(&backend_cfg));
        let agent = edgeloop::agent::Agent::new(backend, vec![], &agent_cfg, &cache_cfg);

        let response = agent.run("What is 2+2? Answer with just the number.", &[], None, "test").await;
        println!("[ollama gemma4 simple] {}", response);
        assert!(response.contains("4"), "Expected '4' in: {}", response);
    }

    #[tokio::test]
    async fn test_ollama_gemma4_tool_call() {
        if !ollama_available() {
            eprintln!("Skipping: Ollama not running on :11434");
            return;
        }
        let agent_cfg = edgeloop::config::AgentConfig {
            system_prompt: "You are a helpful assistant.".into(),
            template: "gemma4".into(),
            max_tokens: 4096, max_iterations: 8, max_retries: 2, temperature: 0.1,
            parallel_tools: false, stream_tokens: false,
        };
        let cache_cfg = edgeloop::config::CacheConfig { max_context: 4096, truncation_threshold: 0.8 };
        // Gemma 4 needs thinking=true for tool calls
        let backend_cfg = backend_config_thinking("ollama", "http://localhost:11434", "gemma4:26b");
        let backend = Arc::new(edgeloop::backend::ollama::OllamaBackend::new(&backend_cfg));
        let calculator = make_tool("calculator", "python3 -c 'print(eval(\"{expression}\"))'", vec![("expression", "string", true)]);
        let agent = edgeloop::agent::Agent::new(backend, vec![calculator], &agent_cfg, &cache_cfg);

        let response = agent.run("What is 123 * 456? Use the calculator tool.", &[], None, "test").await;
        println!("[ollama gemma4 tool] {}", response);
        let clean = response.replace(",", "").replace(" ", "");
        assert!(clean.contains("56088"), "Expected '56088' in: {}", response);
    }

    #[tokio::test]
    async fn test_ollama_gemma4_parallel_tool_calls() {
        if !ollama_available() {
            eprintln!("Skipping: Ollama not running on :11434");
            return;
        }
        let agent_cfg = edgeloop::config::AgentConfig {
            system_prompt: "You are a helpful assistant. When asked for multiple pieces of info, call multiple tools at once using a JSON array.".into(),
            template: "gemma4".into(),
            max_tokens: 4096, max_iterations: 8, max_retries: 2, temperature: 0.1,
            parallel_tools: true, stream_tokens: false,
        };
        let cache_cfg = edgeloop::config::CacheConfig { max_context: 4096, truncation_threshold: 0.8 };
        let backend_cfg = backend_config_thinking("ollama", "http://localhost:11434", "gemma4:26b");
        let backend = Arc::new(edgeloop::backend::ollama::OllamaBackend::new(&backend_cfg));
        let date_tool = make_tool("get_date", "date +%Y-%m-%d", vec![]);
        let uptime_tool = make_tool("get_uptime", "uptime -p", vec![]);
        let agent = edgeloop::agent::Agent::new(backend, vec![date_tool, uptime_tool], &agent_cfg, &cache_cfg);

        let response = agent.run("What is today's date and the system uptime? Use both tools.", &[], None, "test").await;
        println!("[ollama gemma4 parallel] {}", response);
        assert!(!response.is_empty(), "Response should not be empty");
        let history_len = agent.history_len().await;
        println!("[ollama gemma4 parallel] history_len={}", history_len);
        assert!(history_len >= 3, "Should have at least 3 history entries");
    }
}
