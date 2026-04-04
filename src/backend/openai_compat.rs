//! Shared wire types for OpenAI-compatible APIs (OpenAI, vLLM, llama-server /v1).

use serde::Serialize;
use crate::message::Message;

#[derive(Serialize)]
pub(crate) struct ApiMessage {
    pub role: String,
    pub content: ApiContent,
}

#[derive(Serialize)]
#[serde(untagged)]
pub(crate) enum ApiContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Serialize)]
#[serde(tag = "type")]
pub(crate) enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Serialize)]
pub(crate) struct ImageUrl {
    pub url: String,
}

impl ApiMessage {
    pub fn from_message(msg: &Message) -> Self {
        if msg.images.is_empty() {
            return Self {
                role: msg.role.clone(),
                content: ApiContent::Text(msg.content.clone()),
            };
        }

        let mut parts = Vec::new();
        for img in &msg.images {
            if let Some(desc) = &img.description {
                parts.push(ContentPart::Text { text: desc.clone() });
            }
            let mime = img.mime_type.as_deref().unwrap_or("image/jpeg");
            parts.push(ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: format!("data:{};base64,{}", mime, img.b64),
                },
            });
        }
        parts.push(ContentPart::Text { text: msg.content.clone() });

        Self {
            role: msg.role.clone(),
            content: ApiContent::Parts(parts),
        }
    }
}
