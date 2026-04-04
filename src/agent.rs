use std::sync::Arc;
use tokio::sync::Mutex;
use futures::StreamExt;
use tracing::{info, warn, debug};

use crate::backend::Backend;
use crate::cache::CacheManager;
use crate::config::{AgentConfig, CacheConfig, ToolDef};
use crate::message::Message;
use crate::repair;
use crate::tool;

const TOOL_CALL_FORMAT: &str = "When a task requires a tool, call it with ONLY a JSON object:\n{\"tool\": \"tool_name_here\", \"arguments\": {\"param_name\": \"value\"}}\n\nUse exact parameter names shown above. After getting tool results, respond with plain text.\nIf you can answer without tools, respond directly with plain text.";

const TOOL_CALL_FORMAT_PARALLEL: &str = "When a task requires a tool, call it with ONLY a JSON object:\n{\"tool\": \"tool_name_here\", \"arguments\": {\"param_name\": \"value\"}}\n\nTo call multiple tools at once, use a JSON array:\n[{\"tool\": \"tool1\", \"arguments\": {...}}, {\"tool\": \"tool2\", \"arguments\": {...}}]\n\nUse exact parameter names shown above. After getting tool results, respond with plain text.\nIf you can answer without tools, respond directly with plain text.";

struct ChatTemplate {
    system: &'static str,
    user: &'static str,
    assistant: &'static str,
    assistant_start: &'static str,
}

const CHATML: ChatTemplate = ChatTemplate {
    system: "<|im_start|>system\n{content}<|im_end|>\n",
    user: "<|im_start|>user\n{content}<|im_end|>\n",
    assistant: "<|im_start|>assistant\n{content}<|im_end|>\n",
    assistant_start: "<|im_start|>assistant\n",
};

const LLAMA3: ChatTemplate = ChatTemplate {
    system: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
    user: "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
    assistant: "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
    assistant_start: "<|start_header_id|>assistant<|end_header_id|>\n\n",
};

const MISTRAL: ChatTemplate = ChatTemplate {
    system: "[INST] {content}\n",
    user: "[INST] {content} [/INST]",
    assistant: "{content}</s>",
    assistant_start: "",
};

const GEMMA4: ChatTemplate = ChatTemplate {
    system: "<|turn>system\n{content}<turn|>\n",
    user: "<|turn>user\n{content}<turn|>\n",
    assistant: "<|turn>model\n{content}<turn|>\n",
    assistant_start: "<|turn>model\n",
};

fn get_template(name: &str) -> &'static ChatTemplate {
    match name { "llama3" => &LLAMA3, "mistral" => &MISTRAL, "gemma4" => &GEMMA4, _ => &CHATML }
}

pub struct Agent {
    backend: Arc<dyn Backend>,
    tools: Vec<ToolDef>,
    system_prompt: String,
    template: &'static ChatTemplate,
    max_iterations: usize,
    max_retries: usize,
    temperature: f64,
    parallel_tools: bool,
    stream_tokens: bool,
    pub cache: Mutex<CacheManager>,
    /// Persistent conversation history across run() calls.
    /// Each run appends user + assistant messages, maximizing KV cache reuse.
    conversation: Mutex<Vec<Message>>,
}

impl Agent {
    pub fn new(backend: Arc<dyn Backend>, tools: Vec<ToolDef>, agent_config: &AgentConfig, cache_config: &CacheConfig) -> Self {
        let system_prompt = build_system_prompt(&agent_config.system_prompt, &tools, agent_config.parallel_tools);
        let template = get_template(&agent_config.template);
        let mut cache = CacheManager::new(cache_config.max_context, cache_config.truncation_threshold);
        cache.system_prompt_tokens = system_prompt.len() / 4;
        Self {
            backend, tools, system_prompt, template,
            max_iterations: agent_config.max_iterations,
            max_retries: agent_config.max_retries,
            temperature: agent_config.temperature,
            parallel_tools: agent_config.parallel_tools,
            stream_tokens: agent_config.stream_tokens,
            cache: Mutex::new(cache),
            conversation: Mutex::new(Vec::new()),
        }
    }

    /// Run the agent loop. Conversation history persists across calls
    /// so the LLM backend can reuse its KV cache for the shared prefix.
    pub async fn run(&self, message: &str, images: &[crate::message::ImageAttachment], response_tx: Option<&tokio::sync::mpsc::Sender<crate::message::OutputEvent>>, session: &str) -> String {
        // Append user message to persistent conversation
        self.conversation.lock().await.push(
            Message::user_with_images(message, images.to_vec())
        );

        for iteration in 0..self.max_iterations {
            info!("Agent loop iteration {}", iteration + 1);

            // Truncate persistent history if needed
            {
                let cache = self.cache.lock().await;
                let mut conv = self.conversation.lock().await;
                if cache.needs_truncation() && conv.len() > 3 {
                    let target = cache.truncation_target();
                    let keep = (target / 50).max(3).min(conv.len());
                    let trimmed = conv.len() - keep - 1;
                    let first = conv[0].clone();
                    let recent: Vec<_> = conv[conv.len() - keep..].to_vec();
                    *conv = vec![first, Message::user(&format!("[{} earlier messages omitted]", trimmed))];
                    conv.extend(recent);
                    info!("Truncated {} messages", trimmed);
                }
            }

            let history = self.conversation.lock().await.clone();
            let prompt = self.format_prompt(&history);
            let mut messages = vec![Message::system(&self.system_prompt)];
            messages.extend(history);

            { self.cache.lock().await.update_history_tokens(prompt.len() / 4); }

            let max_tokens = { self.cache.lock().await.remaining_tokens().max(256) };

            // Stop sequences prevent over-generation — chat template EOS tokens
            let stop = vec!["<|im_end|>".to_string(), "<|eot_id|>".to_string(), "</s>".to_string(), "<turn|>".to_string()];

            let mut response_text = String::new();
            let mut stream = self.backend.stream_completion(&prompt, &messages, self.temperature, max_tokens, &stop);
            while let Some(result) = stream.next().await {
                match result {
                    Ok(token) => {
                        if self.stream_tokens {
                            if let Some(tx) = response_tx {
                                let _ = tx.try_send(crate::message::OutputEvent::Token {
                                    content: token.clone(),
                                    session: session.to_string(),
                                });
                            }
                        }
                        response_text.push_str(&token);
                    }
                    Err(e) => { warn!("Backend error: {}", e); return format!("Error: Backend failed: {}", e); }
                }
            }

            debug!("Raw LLM output: {}", &response_text[..response_text.len().min(200)]);

            if let Some(stats) = self.backend.last_cache_stats() {
                self.cache.lock().await.record(stats);
            }

            let tool_calls = if self.parallel_tools {
                repair::repair_tool_calls(&response_text, &self.tools)
            } else {
                repair::repair_tool_call(&response_text, &self.tools)
                    .into_iter().collect()
            };

            if tool_calls.is_empty() {
                let format_hint = if self.parallel_tools { TOOL_CALL_FORMAT_PARALLEL } else { TOOL_CALL_FORMAT };
                if repair::looks_like_broken_tool_call(&response_text) && iteration < self.max_retries {
                    warn!("Broken tool call, retrying");
                    let mut conv = self.conversation.lock().await;
                    conv.push(Message::assistant(&response_text));
                    conv.push(Message::user(&format!("Your response was not valid. Respond with a valid tool call JSON or plain text.\n\n{}", format_hint)));
                    continue;
                }
                // Final response — append assistant reply to conversation for next run
                self.conversation.lock().await.push(Message::assistant(&response_text));
                info!("Final response");
                self.cache.lock().await.log_summary();
                return response_text;
            }

            if tool_calls.len() == 1 {
                // Single tool call — execute directly (no JoinSet overhead)
                let tc = &tool_calls[0];
                info!("Executing tool: {}({:?})", tc.name, tc.arguments);

                if self.stream_tokens {
                    if let Some(tx) = response_tx {
                        let _ = tx.try_send(crate::message::OutputEvent::ToolCall {
                            tool: tc.name.clone(),
                            arguments: serde_json::to_value(&tc.arguments).unwrap_or_default(),
                            session: session.to_string(),
                        });
                    }
                }

                let tool_def = match self.tools.iter().find(|t| t.name == tc.name) {
                    Some(t) => t,
                    None => return format!("Error: Tool '{}' not found", tc.name),
                };

                let result = tool::execute_tool(tool_def, &tc.arguments).await;
                info!("Tool result: {}", &result[..result.len().min(100)]);

                if self.stream_tokens {
                    if let Some(tx) = response_tx {
                        let _ = tx.try_send(crate::message::OutputEvent::ToolResult {
                            tool: tc.name.clone(),
                            result: result.clone(),
                            session: session.to_string(),
                        });
                    }
                }

                let mut conv = self.conversation.lock().await;
                conv.push(Message::assistant(&response_text));
                conv.push(Message::user(&format!("Tool '{}' returned:\n{}", tc.name, result)));
            } else {
                // Multiple tool calls — execute in parallel
                info!("Executing {} tools in parallel", tool_calls.len());

                if self.stream_tokens {
                    for tc in &tool_calls {
                        if let Some(tx) = response_tx {
                            let _ = tx.try_send(crate::message::OutputEvent::ToolCall {
                                tool: tc.name.clone(),
                                arguments: serde_json::to_value(&tc.arguments).unwrap_or_default(),
                                session: session.to_string(),
                            });
                        }
                    }
                }

                let mut set = tokio::task::JoinSet::new();
                for tc in &tool_calls {
                    let tool_def = match self.tools.iter().find(|t| t.name == tc.name) {
                        Some(t) => t.clone(),
                        None => {
                            warn!("Tool '{}' not found, skipping", tc.name);
                            continue;
                        }
                    };
                    let args = tc.arguments.clone();
                    let name = tc.name.clone();
                    set.spawn(async move {
                        let result = tool::execute_tool(&tool_def, &args).await;
                        (name, result)
                    });
                }

                let mut results: Vec<(String, String)> = Vec::new();
                while let Some(res) = set.join_next().await {
                    match res {
                        Ok((name, output)) => {
                            info!("Tool '{}' result: {}", name, &output[..output.len().min(100)]);
                            if self.stream_tokens {
                                if let Some(tx) = response_tx {
                                    let _ = tx.try_send(crate::message::OutputEvent::ToolResult {
                                        tool: name.clone(),
                                        result: output.clone(),
                                        session: session.to_string(),
                                    });
                                }
                            }
                            results.push((name, output));
                        }
                        Err(e) => warn!("Tool task panicked: {}", e),
                    }
                }

                let mut conv = self.conversation.lock().await;
                conv.push(Message::assistant(&response_text));
                let batch = results.iter()
                    .map(|(name, output)| format!("Tool '{}' returned:\n{}", name, output))
                    .collect::<Vec<_>>()
                    .join("\n\n");
                conv.push(Message::user(&batch));
            }
        }

        warn!("Max iterations reached");
        self.cache.lock().await.log_summary();
        format!("Error: Maximum iterations ({}) reached", self.max_iterations)
    }

    /// Clear conversation history (start a new session).
    pub async fn clear_history(&self) {
        self.conversation.lock().await.clear();
    }

    /// Get current conversation length.
    pub async fn history_len(&self) -> usize {
        self.conversation.lock().await.len()
    }

    fn format_prompt(&self, history: &[Message]) -> String {
        let mut parts = vec![self.template.system.replace("{content}", &self.system_prompt)];
        for msg in history {
            let tmpl = match msg.role.as_str() {
                "user" => self.template.user,
                "assistant" => self.template.assistant,
                _ => continue,
            };
            let mut content = msg.content.clone();
            for img in &msg.images {
                if let Some(desc) = &img.description {
                    content.push_str(&format!("\n[Image: {}]", desc));
                } else {
                    content.push_str("\n[Image attached]");
                }
            }
            parts.push(tmpl.replace("{content}", &content));
        }
        parts.push(self.template.assistant_start.to_string());
        parts.join("")
    }
}

fn build_system_prompt(base: &str, tools: &[ToolDef], parallel_tools: bool) -> String {
    let mut parts = vec![base.to_string()];
    if !tools.is_empty() {
        parts.push("\n\nTools:".to_string());
        for t in tools { parts.push(tool::format_tool_schema(t)); }
        let format = if parallel_tools { TOOL_CALL_FORMAT_PARALLEL } else { TOOL_CALL_FORMAT };
        parts.push(format!("\n{}", format));
    }
    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::backend::mock::MockBackend;
    use crate::config::ParamDef;

    fn make_tool(name: &str) -> ToolDef {
        let mut params = HashMap::new();
        params.insert("path".to_string(), ParamDef { param_type: "string".to_string(), required: true, default: None });
        ToolDef { name: name.to_string(), description: "Read a file".to_string(), command: "echo 'contents of {path}'".to_string(), stdin: None, timeout: 5, workdir: None, env: HashMap::new(), parameters: params }
    }

    fn make_agent(responses: Vec<String>, tools: Vec<ToolDef>) -> Agent {
        let backend = Arc::new(MockBackend::new(responses));
        let agent_config = AgentConfig { system_prompt: "You are helpful.".into(), template: "chatml".into(), max_tokens: 4096, max_iterations: 10, max_retries: 2, temperature: 0.1, parallel_tools: false, stream_tokens: false };
        let cache_config = CacheConfig { max_context: 4096, truncation_threshold: 0.8 };
        Agent::new(backend, tools, &agent_config, &cache_config)
    }

    #[tokio::test]
    async fn test_simple_response() {
        let agent = make_agent(vec!["Hello there!".into()], vec![]);
        assert_eq!(agent.run("Hi", &[], None, "test").await, "Hello there!");
    }

    #[tokio::test]
    async fn test_tool_call() {
        let agent = make_agent(
            vec![r#"{"tool": "read_file", "arguments": {"path": "/tmp/test"}}"#.into(), "The file says: contents of /tmp/test".into()],
            vec![make_tool("read_file")],
        );
        let result = agent.run("Read /tmp/test", &[], None, "test").await;
        assert!(result.contains("contents of /tmp/test"));
    }

    #[tokio::test]
    async fn test_system_prompt_has_tools() {
        let agent = make_agent(vec![], vec![make_tool("read_file")]);
        assert!(agent.system_prompt.contains("read_file"));
        assert!(agent.system_prompt.contains("path:string"));
    }

    #[tokio::test]
    async fn test_prompt_is_chatml() {
        let agent = make_agent(vec!["ok".into()], vec![]);
        let prompt = agent.format_prompt(&[Message::user("hello")]);
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    fn make_parallel_agent(responses: Vec<String>, tools: Vec<ToolDef>) -> Agent {
        let backend = Arc::new(MockBackend::new(responses));
        let agent_config = AgentConfig { system_prompt: "You are helpful.".into(), template: "chatml".into(), max_tokens: 4096, max_iterations: 10, max_retries: 2, temperature: 0.1, parallel_tools: true, stream_tokens: false };
        let cache_config = CacheConfig { max_context: 4096, truncation_threshold: 0.8 };
        Agent::new(backend, tools, &agent_config, &cache_config)
    }

    fn make_echo_tool(name: &str) -> ToolDef {
        let mut params = HashMap::new();
        params.insert("text".to_string(), ParamDef { param_type: "string".to_string(), required: true, default: None });
        ToolDef { name: name.to_string(), description: format!("Echo for {}", name), command: "echo '{text}'".to_string(), stdin: None, timeout: 5, workdir: None, env: HashMap::new(), parameters: params }
    }

    #[tokio::test]
    async fn test_parallel_tool_calls() {
        let agent = make_parallel_agent(
            vec![
                r#"[{"tool": "tool_a", "arguments": {"text": "hello"}}, {"tool": "tool_b", "arguments": {"text": "world"}}]"#.into(),
                "Both tools returned results.".into(),
            ],
            vec![make_echo_tool("tool_a"), make_echo_tool("tool_b")],
        );
        let result = agent.run("Call both tools", &[], None, "test").await;
        assert_eq!(result, "Both tools returned results.");
        // Conversation should have: user, assistant (array), user (batch results), assistant (final)
        assert_eq!(agent.history_len().await, 4);
    }

    #[tokio::test]
    async fn test_parallel_system_prompt_has_array_format() {
        let agent = make_parallel_agent(vec![], vec![make_echo_tool("tool_a")]);
        assert!(agent.system_prompt.contains("multiple tools at once"));
    }

    #[tokio::test]
    async fn test_non_parallel_system_prompt_no_array_format() {
        let agent = make_agent(vec![], vec![make_tool("read_file")]);
        assert!(!agent.system_prompt.contains("multiple tools at once"));
    }

    #[tokio::test]
    async fn test_prompt_is_gemma4() {
        let backend = Arc::new(MockBackend::new(vec!["ok".into()]));
        let agent_config = AgentConfig { system_prompt: "You are helpful.".into(), template: "gemma4".into(), max_tokens: 4096, max_iterations: 10, max_retries: 2, temperature: 0.1, parallel_tools: false, stream_tokens: false };
        let cache_config = CacheConfig { max_context: 4096, truncation_threshold: 0.8 };
        let agent = Agent::new(backend, vec![], &agent_config, &cache_config);
        let prompt = agent.format_prompt(&[Message::user("hello")]);
        assert!(prompt.contains("<|turn>system\n"));
        assert!(prompt.contains("<|turn>user\nhello<turn|>"));
        assert!(prompt.ends_with("<|turn>model\n"));
    }

    #[tokio::test]
    async fn test_run_with_images_stores_them() {
        let agent = make_agent(vec!["I see a cat!".into()], vec![]);
        let images = vec![crate::message::ImageAttachment {
            b64: "abc".into(),
            description: Some("a cat".into()),
            mime_type: None,
        }];
        let result = agent.run("What do you see?", &images, None, "test").await;
        assert_eq!(result, "I see a cat!");
        let conv = agent.conversation.lock().await;
        assert_eq!(conv.len(), 2);
        assert_eq!(conv[0].images.len(), 1);
        assert_eq!(conv[0].images[0].b64, "abc");
    }

    #[tokio::test]
    async fn test_run_without_images_empty() {
        let agent = make_agent(vec!["Hello!".into()], vec![]);
        let result = agent.run("Hi", &[], None, "test").await;
        assert_eq!(result, "Hello!");
        let conv = agent.conversation.lock().await;
        assert!(conv[0].images.is_empty());
    }

    #[test]
    fn test_format_prompt_includes_image_descriptions() {
        let agent = make_agent(vec![], vec![]);
        let messages = vec![
            Message::user_with_images("what is this?", vec![
                crate::message::ImageAttachment { b64: "abc".into(), description: Some("a cat photo".into()), mime_type: None },
            ]),
        ];
        let prompt = agent.format_prompt(&messages);
        assert!(prompt.contains("a cat photo"), "Prompt should include image description: {}", prompt);
        assert!(prompt.contains("what is this?"), "Prompt should include message text");
        // Should NOT contain raw base64 in the prompt
        assert!(!prompt.contains("abc"), "Raw base64 should not appear in text prompt");
    }

    #[test]
    fn test_format_prompt_image_no_description() {
        let agent = make_agent(vec![], vec![]);
        let messages = vec![
            Message::user_with_images("look", vec![
                crate::message::ImageAttachment { b64: "xyz".into(), description: None, mime_type: None },
            ]),
        ];
        let prompt = agent.format_prompt(&messages);
        assert!(prompt.contains("[Image attached]"), "Should have placeholder: {}", prompt);
    }

    fn make_streaming_agent(responses: Vec<String>, tools: Vec<ToolDef>) -> Agent {
        let backend = Arc::new(MockBackend::new(responses));
        let agent_config = AgentConfig { system_prompt: "You are helpful.".into(), template: "chatml".into(), max_tokens: 4096, max_iterations: 10, max_retries: 2, temperature: 0.1, parallel_tools: false, stream_tokens: true };
        let cache_config = CacheConfig { max_context: 4096, truncation_threshold: 0.8 };
        Agent::new(backend, tools, &agent_config, &cache_config)
    }

    #[tokio::test]
    async fn test_stream_tokens_emits_token_events() {
        let agent = make_streaming_agent(vec!["Hello there!".into()], vec![]);
        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        let result = agent.run("Hi", &[], Some(&tx), "s1").await;
        drop(tx);

        let mut tokens = Vec::new();
        while let Ok(event) = rx.try_recv() {
            if let crate::message::OutputEvent::Token { content, session } = event {
                assert_eq!(session, "s1");
                tokens.push(content);
            }
        }
        assert!(!tokens.is_empty(), "Should have received at least one Token event");
        let concatenated: String = tokens.concat();
        assert_eq!(concatenated, result);
    }

    #[tokio::test]
    async fn test_stream_tokens_false_emits_no_token_events() {
        let agent = make_agent(vec!["Hello!".into()], vec![]);
        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        let _result = agent.run("Hi", &[], Some(&tx), "s1").await;
        drop(tx);

        let mut count = 0;
        while let Ok(_event) = rx.try_recv() {
            count += 1;
        }
        assert_eq!(count, 0, "No events should be emitted when stream_tokens is false");
    }

    #[tokio::test]
    async fn test_stream_tokens_emits_tool_call_and_result() {
        let agent = make_streaming_agent(
            vec![
                r#"{"tool": "read_file", "arguments": {"path": "/tmp/test"}}"#.into(),
                "The file contains: contents of /tmp/test".into(),
            ],
            vec![make_tool("read_file")],
        );
        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        let _result = agent.run("Read /tmp/test", &[], Some(&tx), "s1").await;
        drop(tx);

        let mut tool_calls = Vec::new();
        let mut tool_results = Vec::new();
        let mut tokens = Vec::new();
        while let Ok(event) = rx.try_recv() {
            match event {
                crate::message::OutputEvent::ToolCall { tool, .. } => tool_calls.push(tool),
                crate::message::OutputEvent::ToolResult { tool, .. } => tool_results.push(tool),
                crate::message::OutputEvent::Token { content, .. } => tokens.push(content),
                _ => {}
            }
        }
        assert!(tool_calls.iter().any(|t| t == "read_file"), "Should have ToolCall for read_file");
        assert!(tool_results.iter().any(|t| t == "read_file"), "Should have ToolResult for read_file");
        assert!(!tokens.is_empty(), "Should have Token events for final response");
    }
}
