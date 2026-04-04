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

/// Raw image from incoming JSON — one of b64, path, or url must be set.
/// Resolved into ImageAttachment before passing to the agent.
#[derive(Debug, Clone, Deserialize)]
pub struct RawImageAttachment {
    #[serde(default)]
    pub b64: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub mime_type: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct IncomingRequest {
    pub message: String,
    #[serde(default = "default_session")]
    pub session: String,
    #[serde(default)]
    pub images: Vec<RawImageAttachment>,
}

fn default_session() -> String { "default".to_string() }

/// Resolve raw image attachments: path → read file, url → HTTP GET, b64 → passthrough.
/// Skips images that fail to resolve (logs warning).
pub async fn resolve_images(raw: Vec<RawImageAttachment>) -> Vec<ImageAttachment> {
    let mut resolved = Vec::with_capacity(raw.len());
    for img in raw {
        match resolve_one(img).await {
            Ok(attachment) => resolved.push(attachment),
            Err(e) => tracing::warn!("Skipping unresolvable image: {}", e),
        }
    }
    resolved
}

async fn resolve_one(raw: RawImageAttachment) -> anyhow::Result<ImageAttachment> {
    use base64::Engine;

    // b64 takes priority if present
    if let Some(b64) = raw.b64 {
        return Ok(ImageAttachment {
            b64,
            description: raw.description,
            mime_type: raw.mime_type,
        });
    }

    // path: read file and base64 encode
    if let Some(ref path) = raw.path {
        let bytes = tokio::fs::read(path).await
            .map_err(|e| anyhow::anyhow!("Failed to read image file '{}': {}", path, e))?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let mime_type = raw.mime_type.or_else(|| mime_from_path(path));
        return Ok(ImageAttachment {
            b64,
            description: raw.description,
            mime_type,
        });
    }

    // url: HTTP GET and base64 encode
    if let Some(ref url) = raw.url {
        let response = reqwest::get(url).await
            .map_err(|e| anyhow::anyhow!("Failed to fetch image from '{}': {}", url, e))?;
        let content_type = response.headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        let bytes = response.bytes().await
            .map_err(|e| anyhow::anyhow!("Failed to read image bytes from '{}': {}", url, e))?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let mime_type = raw.mime_type.or(content_type);
        return Ok(ImageAttachment {
            b64,
            description: raw.description,
            mime_type,
        });
    }

    anyhow::bail!("Image has no b64, path, or url field")
}

fn mime_from_path(path: &str) -> Option<String> {
    let ext = path.rsplit('.').next()?.to_lowercase();
    match ext.as_str() {
        "jpg" | "jpeg" => Some("image/jpeg".into()),
        "png" => Some("image/png".into()),
        "webp" => Some("image/webp".into()),
        "gif" => Some("image/gif".into()),
        _ => None,
    }
}

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
        assert_eq!(req.images[0].b64.as_deref(), Some("img1"));
    }

    #[test]
    fn test_incoming_request_without_images() {
        let json = r#"{"message":"hi"}"#;
        let req: IncomingRequest = serde_json::from_str(json).unwrap();
        assert!(req.images.is_empty());
    }

    #[test]
    fn test_raw_image_attachment_with_path() {
        let json = r#"{"path":"/tmp/photo.png","description":"test"}"#;
        let img: RawImageAttachment = serde_json::from_str(json).unwrap();
        assert_eq!(img.path.as_deref(), Some("/tmp/photo.png"));
        assert_eq!(img.description.as_deref(), Some("test"));
        assert!(img.b64.is_none());
        assert!(img.url.is_none());
    }

    #[test]
    fn test_raw_image_attachment_with_url() {
        let json = r#"{"url":"https://example.com/img.jpg","mime_type":"image/jpeg"}"#;
        let img: RawImageAttachment = serde_json::from_str(json).unwrap();
        assert_eq!(img.url.as_deref(), Some("https://example.com/img.jpg"));
        assert_eq!(img.mime_type.as_deref(), Some("image/jpeg"));
        assert!(img.b64.is_none());
        assert!(img.path.is_none());
    }

    #[test]
    fn test_mime_from_path_known_extensions() {
        assert_eq!(mime_from_path("photo.jpg"), Some("image/jpeg".into()));
        assert_eq!(mime_from_path("photo.jpeg"), Some("image/jpeg".into()));
        assert_eq!(mime_from_path("icon.png"), Some("image/png".into()));
        assert_eq!(mime_from_path("anim.webp"), Some("image/webp".into()));
        assert_eq!(mime_from_path("anim.gif"), Some("image/gif".into()));
    }

    #[test]
    fn test_mime_from_path_unknown_extension() {
        assert_eq!(mime_from_path("file.bmp"), None);
        assert_eq!(mime_from_path("file"), None);
    }

    #[tokio::test]
    async fn test_resolve_images_b64_passthrough() {
        let raw = vec![RawImageAttachment {
            b64: Some("dGVzdA==".into()),
            path: None,
            url: None,
            description: Some("desc".into()),
            mime_type: Some("image/png".into()),
        }];
        let resolved = resolve_images(raw).await;
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].b64, "dGVzdA==");
        assert_eq!(resolved[0].description.as_deref(), Some("desc"));
        assert_eq!(resolved[0].mime_type.as_deref(), Some("image/png"));
    }

    #[tokio::test]
    async fn test_resolve_images_path() {
        use std::io::Write;
        let path = std::env::temp_dir().join("edgeloop_test_resolve_image.bin");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(b"imagedata").unwrap();
        }
        let path_str = path.to_str().unwrap().to_string();

        let raw = vec![RawImageAttachment {
            b64: None,
            path: Some(path_str),
            url: None,
            description: None,
            mime_type: None,
        }];
        let resolved = resolve_images(raw).await;
        let _ = std::fs::remove_file(&path);
        assert_eq!(resolved.len(), 1);
        use base64::Engine;
        let expected = base64::engine::general_purpose::STANDARD.encode(b"imagedata");
        assert_eq!(resolved[0].b64, expected);
    }

    #[tokio::test]
    async fn test_resolve_images_skips_on_missing_fields() {
        let raw = vec![RawImageAttachment {
            b64: None,
            path: None,
            url: None,
            description: None,
            mime_type: None,
        }];
        let resolved = resolve_images(raw).await;
        assert!(resolved.is_empty(), "Should skip image with no b64/path/url");
    }

    #[tokio::test]
    async fn test_resolve_images_path_infers_mime_type() {
        use std::io::Write;
        let path = std::env::temp_dir().join("edgeloop_test_resolve_image.png");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(b"pngdata").unwrap();
        }
        let path_str = path.to_str().unwrap().to_string();

        let raw = vec![RawImageAttachment {
            b64: None,
            path: Some(path_str),
            url: None,
            description: None,
            mime_type: None,
        }];
        let resolved = resolve_images(raw).await;
        let _ = std::fs::remove_file(&path);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].mime_type.as_deref(), Some("image/png"));
    }
}
