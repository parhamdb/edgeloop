use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageAttachment {
    pub b64: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<ImageAttachment>,
}

impl Message {
    pub fn system(content: &str) -> Self { Self { role: "system".into(), content: content.into(), images: vec![] } }
    pub fn user(content: &str) -> Self { Self { role: "user".into(), content: content.into(), images: vec![] } }
    pub fn assistant(content: &str) -> Self { Self { role: "assistant".into(), content: content.into(), images: vec![] } }
    pub fn user_with_images(content: &str, images: Vec<ImageAttachment>) -> Self {
        Self { role: "user".into(), content: content.into(), images }
    }
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
    #[serde(default)]
    pub images: Vec<ImageAttachment>,
}

fn default_session() -> String { "default".to_string() }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_attachment_serde() {
        let img = ImageAttachment {
            b64: "abc123".to_string(),
            description: Some("a cat".to_string()),
            mime_type: None,
        };
        let json = serde_json::to_string(&img).unwrap();
        assert!(json.contains("abc123"));
        assert!(json.contains("a cat"));
        let parsed: ImageAttachment = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.b64, "abc123");
        assert_eq!(parsed.description.as_deref(), Some("a cat"));
    }

    #[test]
    fn test_image_attachment_no_description() {
        let json = r#"{"b64":"data"}"#;
        let img: ImageAttachment = serde_json::from_str(json).unwrap();
        assert_eq!(img.b64, "data");
        assert!(img.description.is_none());
    }

    #[test]
    fn test_message_without_images_compat() {
        let json = r#"{"role":"user","content":"hello"}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content, "hello");
        assert!(msg.images.is_empty());
    }

    #[test]
    fn test_message_with_images() {
        let json = r#"{"role":"user","content":"look","images":[{"b64":"abc","description":"photo"}]}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.images.len(), 1);
        assert_eq!(msg.images[0].b64, "abc");
    }

    #[test]
    fn test_message_serializes_without_images_field() {
        let msg = Message::user("hello");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("images"), "Empty images should be omitted: {}", json);
    }

    #[test]
    fn test_incoming_request_with_images() {
        let json = r#"{"message":"hi","images":[{"b64":"img1"}]}"#;
        let req: IncomingRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.images.len(), 1);
        assert_eq!(req.images[0].b64, "img1");
    }

    #[test]
    fn test_incoming_request_without_images() {
        let json = r#"{"message":"hi"}"#;
        let req: IncomingRequest = serde_json::from_str(json).unwrap();
        assert!(req.images.is_empty());
    }
}
