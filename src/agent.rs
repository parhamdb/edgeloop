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

fn get_template(name: &str) -> &'static ChatTemplate {
    match name { "llama3" => &LLAMA3, "mistral" => &MISTRAL, _ => &CHATML }
}

pub struct Agent {
    backend: Arc<dyn Backend>,
    tools: Vec<ToolDef>,
    system_prompt: String,
    template: &'static ChatTemplate,
    max_iterations: usize,
    max_retries: usize,
    temperature: f64,
    pub cache: Mutex<CacheManager>,
    /// Persistent conversation history across run() calls.
    /// Each run appends user + assistant messages, maximizing KV cache reuse.
    conversation: Mutex<Vec<Message>>,
}

impl Agent {
    pub fn new(backend: Arc<dyn Backend>, tools: Vec<ToolDef>, agent_config: &AgentConfig, cache_config: &CacheConfig) -> Self {
        let system_prompt = build_system_prompt(&agent_config.system_prompt, &tools);
        let template = get_template(&agent_config.template);
        let mut cache = CacheManager::new(cache_config.max_context, cache_config.truncation_threshold);
        cache.system_prompt_tokens = system_prompt.len() / 4;
        Self {
            backend, tools, system_prompt, template,
            max_iterations: agent_config.max_iterations,
            max_retries: agent_config.max_retries,
            temperature: agent_config.temperature,
            cache: Mutex::new(cache),
            conversation: Mutex::new(Vec::new()),
        }
    }

    /// Run the agent loop. Conversation history persists across calls
    /// so the LLM backend can reuse its KV cache for the shared prefix.
    pub async fn run(&self, message: &str) -> String {
        // Append user message to persistent conversation
        self.conversation.lock().await.push(Message::user(message));

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
            let stop = vec!["<|im_end|>".to_string(), "<|eot_id|>".to_string(), "</s>".to_string()];

            let mut response_text = String::new();
            let mut stream = self.backend.stream_completion(&prompt, &messages, self.temperature, max_tokens, &stop);
            while let Some(result) = stream.next().await {
                match result {
                    Ok(token) => response_text.push_str(&token),
                    Err(e) => { warn!("Backend error: {}", e); return format!("Error: Backend failed: {}", e); }
                }
            }

            debug!("Raw LLM output: {}", &response_text[..response_text.len().min(200)]);

            if let Some(stats) = self.backend.last_cache_stats() {
                self.cache.lock().await.record(stats);
            }

            let tool_call = repair::repair_tool_call(&response_text, &self.tools);

            if tool_call.is_none() {
                if repair::looks_like_broken_tool_call(&response_text) && iteration < self.max_retries {
                    warn!("Broken tool call, retrying");
                    let mut conv = self.conversation.lock().await;
                    conv.push(Message::assistant(&response_text));
                    conv.push(Message::user(&format!("Your response was not valid. Respond with a valid tool call JSON or plain text.\n\n{}", TOOL_CALL_FORMAT)));
                    continue;
                }
                // Final response — append assistant reply to conversation for next run
                self.conversation.lock().await.push(Message::assistant(&response_text));
                info!("Final response");
                self.cache.lock().await.log_summary();
                return response_text;
            }

            let tc = tool_call.unwrap();
            info!("Executing tool: {}({:?})", tc.name, tc.arguments);

            let tool_def = match self.tools.iter().find(|t| t.name == tc.name) {
                Some(t) => t,
                None => return format!("Error: Tool '{}' not found", tc.name),
            };

            let result = tool::execute_tool(tool_def, &tc.arguments).await;
            info!("Tool result: {}", &result[..result.len().min(100)]);

            let mut conv = self.conversation.lock().await;
            conv.push(Message::assistant(&response_text));
            conv.push(Message::user(&format!("Tool '{}' returned:\n{}", tc.name, result)));
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
            parts.push(tmpl.replace("{content}", &msg.content));
        }
        parts.push(self.template.assistant_start.to_string());
        parts.join("")
    }
}

fn build_system_prompt(base: &str, tools: &[ToolDef]) -> String {
    let mut parts = vec![base.to_string()];
    if !tools.is_empty() {
        parts.push("\n\nTools:".to_string());
        for t in tools { parts.push(tool::format_tool_schema(t)); }
        parts.push(format!("\n{}", TOOL_CALL_FORMAT));
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
        let agent_config = AgentConfig { system_prompt: "You are helpful.".into(), template: "chatml".into(), max_tokens: 4096, max_iterations: 10, max_retries: 2, temperature: 0.1 };
        let cache_config = CacheConfig { max_context: 4096, truncation_threshold: 0.8 };
        Agent::new(backend, tools, &agent_config, &cache_config)
    }

    #[tokio::test]
    async fn test_simple_response() {
        let agent = make_agent(vec!["Hello there!".into()], vec![]);
        assert_eq!(agent.run("Hi").await, "Hello there!");
    }

    #[tokio::test]
    async fn test_tool_call() {
        let agent = make_agent(
            vec![r#"{"tool": "read_file", "arguments": {"path": "/tmp/test"}}"#.into(), "The file says: contents of /tmp/test".into()],
            vec![make_tool("read_file")],
        );
        let result = agent.run("Read /tmp/test").await;
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
}
