//! Performance benchmarks — measures latency with real Ollama

#[cfg(test)]
mod bench {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Instant;

    fn make_agent(model: &str, tools: Vec<edgeloop::config::ToolDef>) -> edgeloop::agent::Agent {
        let backend_cfg = edgeloop::config::BackendConfig {
            backend_type: "ollama".into(),
            endpoint: "http://localhost:11434".into(),
            model: model.into(),
            slot_id: None, n_keep: None, keep_alive: None, thinking: false, api_key_env: None,
        };
        let agent_cfg = edgeloop::config::AgentConfig {
            system_prompt: "You are helpful.".into(),
            template: "chatml".into(),
            max_tokens: 4096, max_iterations: 8, max_retries: 2, temperature: 0.1,
        };
        let cache_cfg = edgeloop::config::CacheConfig { max_context: 4096, truncation_threshold: 0.8 };
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

    #[tokio::test]
    async fn bench_simple_response() {
        let models = ["qwen3:0.6b", "qwen3:1.7b", "qwen2.5-coder:7b"];
        println!("\n=== Simple Response Benchmark ===");
        for model in models {
            let agent = make_agent(model, vec![]);
            let mut times = Vec::new();
            for _ in 0..3 {
                let start = Instant::now();
                let _ = agent.run("Say hello.").await;
                times.push(start.elapsed());
            }
            let avg_ms = times.iter().map(|t| t.as_millis()).sum::<u128>() / times.len() as u128;
            println!("  {:<25} avg={}ms (runs: {:?})", model, avg_ms,
                times.iter().map(|t| format!("{}ms", t.as_millis())).collect::<Vec<_>>());
        }
    }

    #[tokio::test]
    async fn bench_tool_roundtrip() {
        let models = ["qwen3:0.6b", "qwen3:1.7b", "qwen2.5-coder:7b"];
        println!("\n=== Tool Roundtrip Benchmark ===");
        for model in models {
            let agent = make_agent(model, vec![make_calculator()]);
            let start = Instant::now();
            let response = agent.run("What is 15+27? Use the calculator.").await;
            let elapsed = start.elapsed();
            let correct = response.contains("42");
            println!("  {:<25} {}ms correct={} resp={}", model, elapsed.as_millis(), correct, &response[..response.len().min(60)]);
        }
    }

    #[tokio::test]
    async fn bench_rapid_fire() {
        println!("\n=== Rapid Fire (5 requests, qwen3:0.6b) ===");
        let agent = make_agent("qwen3:0.6b", vec![make_calculator()]);
        let mut times = Vec::new();
        for i in 0..5 {
            let a = 10 + i * 7;
            let b = 20 + i * 3;
            let start = Instant::now();
            let response = agent.run(&format!("Calculate {}+{} using calculator.", a, b)).await;
            let elapsed = start.elapsed();
            let expected = (a + b).to_string();
            times.push(elapsed);
            println!("  Request {}: {}ms expected={} got={}", i+1, elapsed.as_millis(), expected, &response[..response.len().min(60)]);
        }
        let avg = times.iter().map(|t| t.as_millis()).sum::<u128>() / times.len() as u128;
        println!("  Avg: {}ms", avg);
    }
}
