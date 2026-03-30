//! Speculative decoding benchmark — compares llama-server with and without draft model.
//!
//! Run against an already-running llama-server:
//!   LLAMA_SERVER_PORT=8090 cargo test --test speculative_bench -- --nocapture
//!
//! Or use the shell script which manages both servers:
//!   ./scripts/bench_speculative.sh

#[cfg(test)]
mod speculative {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Instant;

    fn server_port() -> u16 {
        std::env::var("LLAMA_SERVER_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8090)
    }

    fn make_agent(
        tools: Vec<edgeloop::config::ToolDef>,
    ) -> edgeloop::agent::Agent {
        let port = server_port();
        let backend_cfg = edgeloop::config::BackendConfig {
            backend_type: "llama-server".into(),
            endpoint: format!("http://localhost:{}", port),
            model: String::new(),
            slot_id: Some(0),
            n_keep: None,
            keep_alive: None,
            thinking: false,
            grammar: None,
            seed: None,
            num_ctx: None,
            cache_reuse: None,
            api_key_env: None,
        };
        let agent_cfg = edgeloop::config::AgentConfig {
            system_prompt: "You are a helpful assistant. Use tools when asked.".into(),
            template: "chatml".into(),
            max_tokens: 4096,
            max_iterations: 8,
            max_retries: 2,
            temperature: 0.1,
        };
        let cache_cfg = edgeloop::config::CacheConfig {
            max_context: 4096,
            truncation_threshold: 0.8,
        };
        let backend = Arc::new(
            edgeloop::backend::llama_server::LlamaServerBackend::new(&backend_cfg),
        );
        edgeloop::agent::Agent::new(backend, tools, &agent_cfg, &cache_cfg)
    }

    fn make_calculator() -> edgeloop::config::ToolDef {
        let mut params = HashMap::new();
        params.insert(
            "expression".into(),
            edgeloop::config::ParamDef {
                param_type: "string".into(),
                required: true,
                default: None,
            },
        );
        edgeloop::config::ToolDef {
            name: "calculator".into(),
            description: "Evaluate a math expression".into(),
            command: "python3 -c 'print(eval(\"{expression}\"))'".into(),
            stdin: None,
            timeout: 10,
            workdir: None,
            env: HashMap::new(),
            parameters: params,
        }
    }

    /// Benchmark helper: run a prompt N times, return (avg_ms, individual_times)
    async fn bench_prompt(
        agent: &edgeloop::agent::Agent,
        prompt: &str,
        warmup: usize,
        runs: usize,
    ) -> (u128, Vec<u128>) {
        // Warmup
        for _ in 0..warmup {
            let _ = agent.run(prompt).await;
        }

        let mut times = Vec::with_capacity(runs);
        for _ in 0..runs {
            let start = Instant::now();
            let _ = agent.run(prompt).await;
            times.push(start.elapsed().as_millis());
        }

        let avg = times.iter().sum::<u128>() / times.len() as u128;
        (avg, times)
    }

    #[tokio::test]
    async fn bench_simple_response() {
        let port = server_port();
        println!(
            "\n=== Simple Response — llama-server :{} ===",
            port
        );

        let agent = make_agent(vec![]);

        let prompts = [
            ("hello", "Say hello in one sentence."),
            ("explain", "Explain what a hash table is in two sentences."),
            ("list", "List 5 sorting algorithms, one per line."),
        ];

        for (name, prompt) in &prompts {
            let (avg, times) = bench_prompt(&agent, prompt, 1, 3).await;
            println!(
                "  {:<12} avg={}ms  runs={:?}",
                name,
                avg,
                times.iter().map(|t| format!("{}ms", t)).collect::<Vec<_>>()
            );
        }
    }

    #[tokio::test]
    async fn bench_tool_roundtrip() {
        let port = server_port();
        println!(
            "\n=== Tool Roundtrip — llama-server :{} ===",
            port
        );

        let agent = make_agent(vec![make_calculator()]);

        let prompts = [
            ("simple_calc", "What is 15+27? Use the calculator."),
            ("multi_calc", "What is (123 * 456) + 789? Use the calculator."),
        ];

        for (name, prompt) in &prompts {
            let (avg, times) = bench_prompt(&agent, prompt, 1, 3).await;
            println!(
                "  {:<12} avg={}ms  runs={:?}",
                name,
                avg,
                times.iter().map(|t| format!("{}ms", t)).collect::<Vec<_>>()
            );
        }
    }

    #[tokio::test]
    async fn bench_code_generation() {
        let port = server_port();
        println!(
            "\n=== Code Generation — llama-server :{} ===",
            port
        );

        let agent = make_agent(vec![]);

        let prompts = [
            (
                "short_code",
                "Write a Python function that checks if a number is prime. Just the code.",
            ),
            (
                "medium_code",
                "Write a Python class for a binary search tree with insert and search methods. Just the code.",
            ),
        ];

        for (name, prompt) in &prompts {
            let start = Instant::now();
            let response = agent.run(prompt).await;
            let elapsed = start.elapsed().as_millis();
            let resp_len = response.len();
            println!("  {:<12} {}ms  response_len={}", name, elapsed, resp_len);
        }
    }

    #[tokio::test]
    async fn bench_multi_turn() {
        let port = server_port();
        println!(
            "\n=== Multi-turn Conversation — llama-server :{} ===",
            port
        );

        let agent = make_agent(vec![make_calculator()]);

        let turns = [
            "What is 10 * 20? Use the calculator.",
            "Now add 50 to that result. Use the calculator.",
            "What was the first result I asked about?",
        ];

        let mut total_ms = 0u128;
        for (i, prompt) in turns.iter().enumerate() {
            let start = Instant::now();
            let response = agent.run(prompt).await;
            let elapsed = start.elapsed().as_millis();
            total_ms += elapsed;
            println!(
                "  turn_{:<6} {}ms  resp={}",
                i + 1,
                elapsed,
                &response[..response.len().min(80)]
            );
        }
        println!("  total       {}ms", total_ms);
    }
}
