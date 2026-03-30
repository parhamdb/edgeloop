//! Long chat tests — multi-turn conversations to verify context, truncation, tool chains.
//! Requires Ollama running with qwen2.5-coder:7b.

#[cfg(test)]
mod long_chat {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Instant;

    fn make_agent(tools: Vec<edgeloop::config::ToolDef>, max_tokens: usize) -> edgeloop::agent::Agent {
        let backend_cfg = edgeloop::config::BackendConfig {
            backend_type: "ollama".into(),
            endpoint: "http://localhost:11434".into(),
            model: "qwen2.5-coder:7b".into(),
            slot_id: None, n_keep: None, keep_alive: None, thinking: false, grammar: None, seed: None, num_ctx: None, cache_reuse: None, api_key_env: None,
        };
        let agent_cfg = edgeloop::config::AgentConfig {
            system_prompt: "You are a helpful assistant. Be concise.".into(),
            template: "chatml".into(),
            max_tokens,
            max_iterations: 8,
            max_retries: 2,
            temperature: 0.1,
        };
        let cache_cfg = edgeloop::config::CacheConfig { max_context: max_tokens, truncation_threshold: 0.8 };
        let backend = Arc::new(edgeloop::backend::ollama::OllamaBackend::new(&backend_cfg));
        edgeloop::agent::Agent::new(backend, tools, &agent_cfg, &cache_cfg)
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
            stdin: None, timeout: 10, workdir: None, env: HashMap::new(), parameters: params,
        }
    }

    /// 10-turn conversation — verify the agent doesn't degrade or crash
    #[tokio::test]
    async fn test_10_turn_conversation() {
        let agent = make_agent(vec![], 4096);
        let questions = [
            "What is the capital of France?",
            "What language do they speak there?",
            "Name 3 famous landmarks in that city.",
            "Which one is the tallest?",
            "When was it built?",
            "Who designed it?",
            "What other structures did that person design?",
            "Pick one and tell me about it.",
            "How tall is it?",
            "Compare it to the first landmark we discussed.",
        ];

        let mut total_time = std::time::Duration::ZERO;
        for (i, q) in questions.iter().enumerate() {
            let start = Instant::now();
            let response = agent.run(q).await;
            let elapsed = start.elapsed();
            total_time += elapsed;
            println!("Turn {}: {}ms — Q: {} A: {}", i + 1, elapsed.as_millis(), q, &response[..response.len().min(80)]);
            assert!(!response.is_empty(), "Empty response at turn {}", i + 1);
            assert!(!response.starts_with("Error"), "Error at turn {}: {}", i + 1, response);
        }
        println!("Total: {}ms, Avg: {}ms/turn", total_time.as_millis(), total_time.as_millis() / 10);
    }

    /// Sequential tool calls — verify tools work across multiple independent runs
    #[tokio::test]
    async fn test_sequential_tool_calls() {
        let agent = make_agent(vec![make_calculator()], 4096);
        let calculations = [
            ("15 + 27", "42"),
            ("100 * 3", "300"),
            ("999 - 1", "998"),
            ("2 ** 10", "1024"),
            ("144 / 12", "12"),
        ];

        for (expr, expected) in calculations {
            let start = Instant::now();
            let response = agent.run(&format!("Calculate {} using the calculator tool.", expr)).await;
            let elapsed = start.elapsed();
            let clean = response.replace(",", "");
            println!("Calc {}: {}ms — expected={} got={}", expr, elapsed.as_millis(), expected, &response[..response.len().min(80)]);
            assert!(clean.contains(expected), "Expected '{}' in: {}", expected, response);
        }
    }

    /// Verify context truncation doesn't crash — small context window
    #[tokio::test]
    async fn test_context_truncation() {
        // Very small context — forces truncation quickly
        let agent = make_agent(vec![], 1024);

        for i in 0..5 {
            let response = agent.run(&format!("Tell me fact #{} about the ocean. Keep it to one sentence.", i + 1)).await;
            println!("Turn {} (1024 ctx): {}", i + 1, &response[..response.len().min(100)]);
            assert!(!response.is_empty());
            assert!(!response.starts_with("Error"));
        }

        let summary = agent.cache.lock().await.summary();
        println!("Cache: {:?}", summary);
    }

    /// Mix of tool calls and plain responses
    #[tokio::test]
    async fn test_mixed_tool_and_chat() {
        let agent = make_agent(vec![make_calculator()], 4096);

        let r1 = agent.run("What is the capital of Germany?").await;
        println!("Chat: {}", &r1[..r1.len().min(60)]);
        assert!(r1.to_lowercase().contains("berlin"));

        let r2 = agent.run("Calculate 50 * 50 using the calculator.").await;
        let clean = r2.replace(",", "");
        println!("Tool: {}", &r2[..r2.len().min(60)]);
        assert!(clean.contains("2500"));

        let r3 = agent.run("What continent is Germany on?").await;
        println!("Chat: {}", &r3[..r3.len().min(60)]);
        assert!(r3.to_lowercase().contains("europ"));

        let r4 = agent.run("Calculate 2500 + 100 using the calculator.").await;
        let clean = r4.replace(",", "");
        println!("Tool: {}", &r4[..r4.len().min(60)]);
        assert!(clean.contains("2600"));
    }
}
