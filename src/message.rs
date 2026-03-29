use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: &str) -> Self { Self { role: "system".into(), content: content.into() } }
    pub fn user(content: &str) -> Self { Self { role: "user".into(), content: content.into() } }
    pub fn assistant(content: &str) -> Self { Self { role: "assistant".into(), content: content.into() } }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum OutputEvent {
    #[serde(rename = "token")]
    Token { content: String, session: String },
    #[serde(rename = "tool_call")]
    ToolCall { tool: String, arguments: Value, session: String },
    #[serde(rename = "tool_result")]
    ToolResult { tool: String, result: String, session: String },
    #[serde(rename = "done")]
    Done { content: String, session: String },
    #[serde(rename = "error")]
    Error { content: String, session: String },
}

#[derive(Debug, Deserialize)]
pub struct IncomingRequest {
    pub message: String,
    #[serde(default = "default_session")]
    pub session: String,
}

fn default_session() -> String { "default".to_string() }
