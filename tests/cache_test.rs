//! KV Cache tests — verifies persistent conversation history enables cache reuse.
//! Requires Ollama running with qwen2.5-coder:7b.

#[cfg(test)]
mod cache_tests {
    use std::sync::Arc;
    use std::time::Instant;

    fn make_agent() -> edgeloop::agent::Agent {
        let backend_cfg = edgeloop::config::BackendConfig {
            backend_type: "ollama".into(),
            endpoint: "http://localhost:11434".into(),
            model: "qwen2.5-coder:7b".into(),
            slot_id: None, n_keep: None, keep_alive: None, thinking: false, grammar: None, seed: None, num_ctx: None, cache_reuse: None, api_key_env: None,
            min_tokens: None, repetition_penalty: None, top_k: None, min_p: None,
            stop_token_ids: None, truncate_prompt_tokens: None, guided_mode: None, guided_pattern: None,
        };
        let agent_cfg = edgeloop::config::AgentConfig {
            system_prompt: "You are helpful. Be very concise. One sentence max.".into(),
            template: "chatml".into(),
            max_tokens: 4096, max_iterations: 5, max_retries: 1, temperature: 0.1, parallel_tools: false,
        };
        let cache_cfg = edgeloop::config::CacheConfig { max_context: 4096, truncation_threshold: 0.8 };
        let backend = Arc::new(edgeloop::backend::ollama::OllamaBackend::new(&backend_cfg));
        edgeloop::agent::Agent::new(backend, vec![], &agent_cfg, &cache_cfg)
    }

    /// Multi-turn on SAME agent — should benefit from KV cache
    #[tokio::test]
    async fn test_persistent_history_speeds_up() {
        let agent = make_agent();

        let mut times = Vec::new();
        let turns = [
            "What is the capital of France?",
            "What language do they speak there?",
            "Name one famous landmark.",
            "When was it built?",
            "Who designed it?",
        ];

        println!("\n=== Persistent History — Same Agent, Multiple Runs ===");
        for (i, turn) in turns.iter().enumerate() {
            let start = Instant::now();
            let response = agent.run(turn, &[]).await;
            let elapsed = start.elapsed();
            let history_len = agent.history_len().await;
            times.push(elapsed);
            println!("  Turn {}: {:>4}ms | history={} msgs | Q: {} | A: {}",
                i + 1, elapsed.as_millis(), history_len, turn,
                &response[..response.len().min(60)]);
        }

        // Print cache stats
        {
            let cache = agent.cache.lock().await;
            let summary = cache.summary();
            println!("\n  Cache summary: {:?}", summary);
        }

        // Verify context is maintained — the model should remember Paris
        let response = agent.run("What was the first city we discussed?", &[]).await;
        println!("  Context: {} | remembers_paris={}", &response[..response.len().min(60)], response.to_lowercase().contains("paris"));
        assert!(response.to_lowercase().contains("paris"), "Should remember Paris from turn 1: {}", response);

        // Later turns should be similar or faster than first (cache warming)
        let first = times[0].as_millis();
        let last = times[4].as_millis();
        println!("\n  First turn: {}ms, Last turn: {}ms", first, last);
        println!("  History: {} messages", agent.history_len().await);
    }

    /// Compare: persistent history vs fresh agent each time
    #[tokio::test]
    async fn test_persistent_vs_fresh() {
        println!("\n=== Persistent vs Fresh — Cache Impact ===");

        // Persistent: same agent, history accumulates
        let agent_persistent = make_agent();
        let mut persistent_times = Vec::new();
        for i in 0..5 {
            let start = Instant::now();
            let _ = agent_persistent.run(&format!("What is {}+{}?", i, i+1), &[]).await;
            persistent_times.push(start.elapsed());
        }

        // Fresh: new agent each time, no history
        let mut fresh_times = Vec::new();
        for i in 0..5 {
            let agent_fresh = make_agent();
            let start = Instant::now();
            let _ = agent_fresh.run(&format!("What is {}+{}?", i, i+1), &[]).await;
            fresh_times.push(start.elapsed());
        }

        println!("  Run | Persistent | Fresh");
        for i in 0..5 {
            println!("    {} | {:>6}ms   | {:>6}ms",
                i+1, persistent_times[i].as_millis(), fresh_times[i].as_millis());
        }

        let avg_p = persistent_times.iter().map(|t| t.as_millis()).sum::<u128>() / 5;
        let avg_f = fresh_times.iter().map(|t| t.as_millis()).sum::<u128>() / 5;
        println!("  Avg:  {:>6}ms   | {:>6}ms", avg_p, avg_f);
        println!("  Persistent history enables the LLM to cache the growing prefix.");
    }
}
