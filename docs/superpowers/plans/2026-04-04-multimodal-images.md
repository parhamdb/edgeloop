# Multimodal Image Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support base64-encoded images in incoming messages so VLM backends (Ollama, OpenAI, vLLM) can reason about them.

**Architecture:** Add `ImageAttachment` type and `images` field to `Message` and `IncomingRequest`. Each transport passes images through `TransportRequest`. Agent stores images on user messages in conversation history. Each backend converts to its own wire format: OpenAI/vLLM use content-array with `image_url`, Ollama uses its native `images` field, llama-server degrades gracefully (descriptions as text in the raw prompt).

**Tech Stack:** Rust, serde, existing backend HTTP clients

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/message.rs` | Modify | Add `ImageAttachment`, add `images` to `Message` and `IncomingRequest` |
| `src/transport/mod.rs` | Modify | Add `images` to `TransportRequest` |
| `src/transport/mqtt.rs` | Modify | Pass `req.images` to `TransportRequest` |
| `src/transport/websocket.rs` | Modify | Pass `req.images` to `TransportRequest` |
| `src/transport/socket.rs` | Modify | Pass `req.images` to `TransportRequest` |
| `src/transport/cli.rs` | Modify | Pass empty `images` vec to `TransportRequest` |
| `src/main.rs` | Modify | Pass `req.images` to `agent.run()` |
| `src/agent.rs` | Modify | `run()` accepts images, stores on user `Message`, `format_prompt()` appends image descriptions |
| `src/backend/openai.rs` | Modify | New `ApiMessage` wire type, convert `Message.images` to content-array format |
| `src/backend/vllm.rs` | Modify | Same content-array conversion as OpenAI |
| `src/backend/ollama.rs` | Modify | Add `images` field to `ChatMsg`, populate from `Message.images` |
| `src/backend/llama_server.rs` | No change | Uses raw prompt string; image descriptions included by `format_prompt()` |
| `src/backend/mod.rs` | Modify | Update `MockBackend` to accept new `Message` shape |

---

### Task 1: Add `ImageAttachment` type and update `Message`

**Files:**
- Modify: `src/message.rs:1-38`

- [ ] **Step 1: Write unit test for ImageAttachment serde**

Add to the bottom of `src/message.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_attachment_serde() {
        let img = ImageAttachment {
            b64: "abc123".to_string(),
            description: Some("a cat".to_string()),
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
        // Old-format JSON (no images field) still deserializes
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --bin edgeloop message::tests -- -v 2>&1 | head -30`
Expected: FAIL — `ImageAttachment` not defined, `images` field not found

- [ ] **Step 3: Implement ImageAttachment and update Message + IncomingRequest**

Replace the full content of `src/message.rs` with:

```rust
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageAttachment {
    pub b64: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --bin edgeloop message::tests -- -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/message.rs
git commit -m "feat: add ImageAttachment type and images field to Message and IncomingRequest"
```

---

### Task 2: Thread images through transport layer

**Files:**
- Modify: `src/transport/mod.rs:17-21`
- Modify: `src/transport/mqtt.rs:72-77`
- Modify: `src/transport/websocket.rs:80-85`
- Modify: `src/transport/socket.rs:38-43`
- Modify: `src/transport/cli.rs:42-47`

- [ ] **Step 1: Add `images` to `TransportRequest`**

In `src/transport/mod.rs`, change the struct:

```rust
pub struct TransportRequest {
    pub message: String,
    pub session: String,
    pub images: Vec<crate::message::ImageAttachment>,
    pub response_tx: mpsc::Sender<OutputEvent>,
}
```

- [ ] **Step 2: Fix all transport constructors to pass images**

In `src/transport/mqtt.rs`, change the `TransportRequest` construction (around line 73):

```rust
handler(TransportRequest {
    message: req.message,
    session: req.session,
    images: req.images,
    response_tx: tx,
});
```

In `src/transport/websocket.rs`, change the `TransportRequest` construction (around line 81):

```rust
handler(TransportRequest {
    message: req.message,
    session: req.session,
    images: req.images,
    response_tx: tx,
});
```

In `src/transport/socket.rs`, change the `TransportRequest` construction (around line 39):

```rust
handler(TransportRequest {
    message: req.message,
    session: req.session,
    images: req.images,
    response_tx: tx,
});
```

In `src/transport/cli.rs`, change the `TransportRequest` construction (around line 43):

```rust
handler(TransportRequest {
    message: trimmed,
    session: "cli".to_string(),
    images: vec![],
    response_tx: tx,
});
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check --features full 2>&1 | tail -5`
Expected: Compiler errors only from `src/main.rs` (where `req.images` isn't passed to agent yet) — transport layer should be clean.

- [ ] **Step 4: Commit**

```bash
git add src/transport/
git commit -m "feat: thread images through all transports via TransportRequest"
```

---

### Task 3: Update agent to accept and store images

**Files:**
- Modify: `src/agent.rs:88-113` (run method)
- Modify: `src/agent.rs:236-248` (format_prompt method)
- Modify: `src/main.rs:67-76` (handler closure)

- [ ] **Step 1: Write unit test for image-aware agent**

Add to the existing `mod tests` in `src/agent.rs`, after the last test:

```rust
    #[tokio::test]
    async fn test_run_with_images_stores_them() {
        let agent = make_agent(vec!["I see a cat!".into()], vec![]);
        let images = vec![crate::message::ImageAttachment {
            b64: "abc".into(),
            description: Some("a cat".into()),
        }];
        let result = agent.run("What do you see?", &images).await;
        assert_eq!(result, "I see a cat!");
        // Conversation should have user (with image) + assistant
        let conv = agent.conversation.lock().await;
        assert_eq!(conv.len(), 2);
        assert_eq!(conv[0].images.len(), 1);
        assert_eq!(conv[0].images[0].b64, "abc");
    }

    #[tokio::test]
    async fn test_run_without_images_empty() {
        let agent = make_agent(vec!["Hello!".into()], vec![]);
        let result = agent.run("Hi", &[]).await;
        assert_eq!(result, "Hello!");
        let conv = agent.conversation.lock().await;
        assert!(conv[0].images.is_empty());
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --bin edgeloop agent::tests::test_run_with_images -- -v 2>&1 | head -20`
Expected: FAIL — `run()` doesn't accept images parameter

- [ ] **Step 3: Update `Agent::run()` to accept images**

In `src/agent.rs`, change the `run` method signature and first line (line 90-92):

```rust
    pub async fn run(&self, message: &str, images: &[crate::message::ImageAttachment]) -> String {
        // Append user message to persistent conversation
        self.conversation.lock().await.push(
            Message::user_with_images(message, images.to_vec())
        );
```

- [ ] **Step 4: Update `format_prompt()` to include image descriptions**

In `src/agent.rs`, update the `format_prompt` method (line 236-248):

```rust
    fn format_prompt(&self, history: &[Message]) -> String {
        let mut parts = vec![self.template.system.replace("{content}", &self.system_prompt)];
        for msg in history {
            let tmpl = match msg.role.as_str() {
                "user" => self.template.user,
                "assistant" => self.template.assistant,
                _ => continue,
            };
            // For template-based prompts, append image descriptions as text
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
```

- [ ] **Step 5: Update all existing test calls to `run()`**

Every existing `agent.run("...")` call in `src/agent.rs` tests must become `agent.run("...", &[])`. There are 5 existing calls:

- `test_simple_response`: `agent.run("Hi", &[]).await`
- `test_tool_call`: `agent.run("Read /tmp/test", &[]).await`
- `test_parallel_tool_calls`: `agent.run("Call both tools", &[]).await`
- (The system prompt tests don't call `run()`)

- [ ] **Step 6: Update `main.rs` handler to pass images**

In `src/main.rs`, update the handler closure (line 67-76):

```rust
    let handler: transport::RequestHandler = Arc::new(move |req: transport::TransportRequest| {
        let agent = agent_clone.clone();
        tokio::spawn(async move {
            let result = agent.run(&req.message, &req.images).await;
            let _ = req.response_tx.send(message::OutputEvent::Done {
                content: result,
                session: req.session,
            }).await;
        });
    });
```

- [ ] **Step 7: Run all unit tests**

Run: `cargo test --bin edgeloop -- -v 2>&1 | tail -20`
Expected: All tests PASS (existing + 2 new)

- [ ] **Step 8: Commit**

```bash
git add src/agent.rs src/main.rs
git commit -m "feat: agent accepts and stores images on user messages"
```

---

### Task 4: OpenAI backend — multimodal wire format

**Files:**
- Modify: `src/backend/openai.rs:53-65` (ChatRequest and message types)
- Modify: `src/backend/openai.rs:98-115` (stream_completion)

- [ ] **Step 1: Write unit test for message conversion**

Add to `mod tests` at the bottom of `src/backend/openai.rs` (create the module — there isn't one currently):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::ImageAttachment;

    #[test]
    fn test_api_message_text_only() {
        let msg = Message::user("hello");
        let api_msg = ApiMessage::from_message(&msg);
        assert_eq!(api_msg.role, "user");
        // Text-only: content should be a string
        match &api_msg.content {
            ApiContent::Text(s) => assert_eq!(s, "hello"),
            ApiContent::Parts(_) => panic!("Expected text, got parts"),
        }
    }

    #[test]
    fn test_api_message_with_images() {
        let msg = Message::user_with_images("look at this", vec![
            ImageAttachment { b64: "abc123".into(), description: Some("a photo".into()) },
        ]);
        let api_msg = ApiMessage::from_message(&msg);
        match &api_msg.content {
            ApiContent::Parts(parts) => {
                // Should be: description text, image_url, main text
                assert_eq!(parts.len(), 3);
            }
            ApiContent::Text(_) => panic!("Expected parts, got text"),
        }
    }

    #[test]
    fn test_api_message_serializes_text_as_string() {
        let msg = Message::user("hello");
        let api_msg = ApiMessage::from_message(&msg);
        let json = serde_json::to_value(&api_msg).unwrap();
        assert_eq!(json["content"], "hello");
    }

    #[test]
    fn test_api_message_serializes_images_as_array() {
        let msg = Message::user_with_images("look", vec![
            ImageAttachment { b64: "abc".into(), description: None },
        ]);
        let api_msg = ApiMessage::from_message(&msg);
        let json = serde_json::to_value(&api_msg).unwrap();
        let content = &json["content"];
        assert!(content.is_array(), "Expected array, got: {}", content);
        assert_eq!(content.as_array().unwrap().len(), 2); // text + image
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
        assert!(content[1]["image_url"]["url"].as_str().unwrap().starts_with("data:image/jpeg;base64,"));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --bin edgeloop openai::tests -- -v 2>&1 | head -20`
Expected: FAIL — `ApiMessage` not defined

- [ ] **Step 3: Implement ApiMessage with content-array support**

Add these types after the existing `StreamOptions` struct (around line 70) in `src/backend/openai.rs`:

```rust
// --- Wire types for OpenAI-compatible API ---

#[derive(Serialize)]
struct ApiMessage {
    role: String,
    content: ApiContent,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ApiContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Serialize)]
struct ImageUrl {
    url: String,
}

impl ApiMessage {
    fn from_message(msg: &Message) -> Self {
        if msg.images.is_empty() {
            return Self {
                role: msg.role.clone(),
                content: ApiContent::Text(msg.content.clone()),
            };
        }

        let mut parts = Vec::new();

        // Prepend image descriptions + images before main text
        for img in &msg.images {
            if let Some(desc) = &img.description {
                parts.push(ContentPart::Text { text: desc.clone() });
            }
            parts.push(ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: format!("data:image/jpeg;base64,{}", img.b64),
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
```

- [ ] **Step 4: Update ChatRequest to use ApiMessage**

Change the `ChatRequest` struct's `messages` field:

```rust
#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ApiMessage>,
    stream: bool,
    temperature: f64,
    max_tokens: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
    stream_options: StreamOptions,
}
```

- [ ] **Step 5: Update stream_completion to convert messages**

In `stream_completion` (around line 107-115), change the body construction:

```rust
        let body = ChatRequest {
            model: self.model.clone(),
            messages: messages.iter().map(ApiMessage::from_message).collect(),
            stream: true,
            temperature,
            max_tokens,
            stop: stop.to_vec(),
            stream_options: StreamOptions { include_usage: true },
        };
```

- [ ] **Step 6: Run tests**

Run: `cargo test --bin edgeloop openai::tests -- -v`
Expected: All 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/backend/openai.rs
git commit -m "feat: OpenAI backend multimodal support via content-array format"
```

---

### Task 5: vLLM backend — multimodal wire format

**Files:**
- Modify: `src/backend/vllm.rs:163-179` (ChatRequest and message types)
- Modify: `src/backend/vllm.rs:225-246` (stream_completion)

The vLLM backend uses the same OpenAI-compatible format. The implementation is identical to the OpenAI backend's `ApiMessage`.

- [ ] **Step 1: Write unit test for message conversion**

Add to the existing `mod tests` in `src/backend/vllm.rs`:

```rust
    #[test]
    fn test_api_message_text_only() {
        let msg = Message::user("hello");
        let api_msg = ApiMessage::from_message(&msg);
        let json = serde_json::to_value(&api_msg).unwrap();
        assert_eq!(json["content"], "hello");
    }

    #[test]
    fn test_api_message_with_images() {
        let msg = Message::user_with_images("look", vec![
            crate::message::ImageAttachment { b64: "abc".into(), description: Some("photo".into()) },
        ]);
        let api_msg = ApiMessage::from_message(&msg);
        let json = serde_json::to_value(&api_msg).unwrap();
        let content = &json["content"];
        assert!(content.is_array());
        assert_eq!(content.as_array().unwrap().len(), 3); // desc text + image + main text
        assert_eq!(content[1]["type"], "image_url");
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --bin edgeloop vllm::tests::test_api_message -- -v 2>&1 | head -20`
Expected: FAIL — `ApiMessage` not defined

- [ ] **Step 3: Add ApiMessage types to vllm.rs**

Add the same wire types after the existing `StreamOptions` struct (around line 184) in `src/backend/vllm.rs`. These are identical to the OpenAI ones:

```rust
// --- Wire types for OpenAI-compatible API ---

#[derive(Serialize)]
struct ApiMessage {
    role: String,
    content: ApiContent,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ApiContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Serialize)]
struct ImageUrl {
    url: String,
}

impl ApiMessage {
    fn from_message(msg: &Message) -> Self {
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
            parts.push(ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: format!("data:image/jpeg;base64,{}", img.b64),
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
```

- [ ] **Step 4: Update ChatRequest to use ApiMessage**

Change the `ChatRequest` struct's `messages` field:

```rust
#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ApiMessage>,
    stream: bool,
    temperature: f64,
    max_tokens: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
    stream_options: StreamOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    extra_body: Option<Value>,
}
```

- [ ] **Step 5: Update stream_completion to convert messages**

In `stream_completion` (around line 234), change:

```rust
        let body = ChatRequest {
            model: self.model.clone(),
            messages: messages.iter().map(ApiMessage::from_message).collect(),
            stream: true,
            temperature,
            max_tokens,
            stop: stop.to_vec(),
            stream_options: StreamOptions { include_usage: true },
            seed: self.seed,
            extra_body: self.build_extra_body(),
        };
```

- [ ] **Step 6: Run tests**

Run: `cargo test --bin edgeloop vllm::tests -- -v`
Expected: All existing + 2 new tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/backend/vllm.rs
git commit -m "feat: vLLM backend multimodal support via content-array format"
```

---

### Task 6: Ollama backend — native images format

**Files:**
- Modify: `src/backend/ollama.rs:53-58` (ChatMsg)
- Modify: `src/backend/ollama.rs:98-103` (message mapping)

Ollama uses a different format: images go in a separate `images: ["base64..."]` field on the message object, not as content parts.

- [ ] **Step 1: Write unit test**

Add to the existing `mod tests` in `src/backend/ollama.rs`:

```rust
    #[test]
    fn test_chat_msg_with_images() {
        let msg = crate::message::Message::user_with_images("look", vec![
            crate::message::ImageAttachment { b64: "abc123".into(), description: Some("photo".into()) },
        ]);
        let chat_msg = ChatMsg::from_message(&msg);
        assert_eq!(chat_msg.content, "photo\nlook");
        assert_eq!(chat_msg.images.as_ref().unwrap().len(), 1);
        assert_eq!(chat_msg.images.as_ref().unwrap()[0], "abc123");
    }

    #[test]
    fn test_chat_msg_without_images() {
        let msg = crate::message::Message::user("hello");
        let chat_msg = ChatMsg::from_message(&msg);
        assert_eq!(chat_msg.content, "hello");
        assert!(chat_msg.images.is_none());
    }

    #[test]
    fn test_chat_msg_images_omitted_in_json() {
        let msg = crate::message::Message::user("hello");
        let chat_msg = ChatMsg::from_message(&msg);
        let json = serde_json::to_string(&chat_msg).unwrap();
        assert!(!json.contains("images"), "Should omit empty images: {}", json);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --bin edgeloop ollama::tests::test_chat_msg_with -- -v 2>&1 | head -20`
Expected: FAIL — `ChatMsg::from_message` not defined

- [ ] **Step 3: Update ChatMsg with images support**

Replace the `ChatMsg` struct and add a conversion method:

```rust
#[derive(Serialize)]
struct ChatMsg {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
}

impl ChatMsg {
    fn from_message(msg: &Message) -> Self {
        if msg.images.is_empty() {
            return Self {
                role: msg.role.clone(),
                content: msg.content.clone(),
                images: None,
            };
        }

        // Prepend image descriptions to content
        let mut content_parts = Vec::new();
        let mut b64_images = Vec::new();
        for img in &msg.images {
            if let Some(desc) = &img.description {
                content_parts.push(desc.clone());
            }
            b64_images.push(img.b64.clone());
        }
        content_parts.push(msg.content.clone());

        Self {
            role: msg.role.clone(),
            content: content_parts.join("\n"),
            images: Some(b64_images),
        }
    }
}
```

- [ ] **Step 4: Update message mapping in stream_completion**

In `stream_completion` (around line 99-103), change the message mapping:

```rust
            messages: messages.iter().map(ChatMsg::from_message).collect(),
```

- [ ] **Step 5: Run tests**

Run: `cargo test --bin edgeloop ollama::tests -- -v`
Expected: All existing + 3 new tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/backend/ollama.rs
git commit -m "feat: Ollama backend multimodal support via native images field"
```

---

### Task 7: Update MockBackend and integration tests

**Files:**
- Modify: `src/backend/mod.rs:48-73` (MockBackend)
- Modify: `tests/integration_test.rs:68-105` (run calls)

- [ ] **Step 1: Fix integration test `run()` calls**

In `tests/integration_test.rs`, update all `agent.run("...")` calls to `agent.run("...", &[])`:

- Line 69: `agent.run("What is 2+2? Answer with just the number.", &[]).await`
- Line 80: `agent.run("What is 123 * 456? Use the calculator tool.", &[]).await`
- Line 92: `agent.run("Read the file Cargo.toml and tell me the package name.", &[]).await`
- Line 103: `agent.run("What is the capital of Japan?", &[]).await`
- Line 114: `agent.run("Say hello.", &[]).await`
- Line 118: `agent.run("Say goodbye.", &[]).await`

- [ ] **Step 2: Run full unit test suite**

Run: `cargo test --bin edgeloop -- -v 2>&1 | tail -30`
Expected: All tests PASS

- [ ] **Step 3: Verify full build with all features**

Run: `cargo check --features full 2>&1 | tail -5`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add src/backend/mod.rs tests/integration_test.rs
git commit -m "test: update all run() calls for new images parameter"
```

---

### Task 8: Final verification and edge case tests

**Files:**
- Modify: `src/agent.rs` (add edge case test)

- [ ] **Step 1: Add test for format_prompt with image descriptions**

Add to `mod tests` in `src/agent.rs`:

```rust
    #[test]
    fn test_format_prompt_includes_image_descriptions() {
        let agent = make_agent(vec![], vec![]);
        let messages = vec![
            Message::user_with_images("what is this?", vec![
                crate::message::ImageAttachment { b64: "abc".into(), description: Some("a cat photo".into()) },
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
                crate::message::ImageAttachment { b64: "xyz".into(), description: None },
            ]),
        ];
        let prompt = agent.format_prompt(&messages);
        assert!(prompt.contains("[Image attached]"), "Should have placeholder: {}", prompt);
    }
```

- [ ] **Step 2: Run full test suite**

Run: `cargo test --bin edgeloop -- -v 2>&1 | tail -40`
Expected: All tests PASS

- [ ] **Step 3: Run cargo clippy**

Run: `cargo clippy --features full 2>&1 | tail -20`
Expected: No errors (warnings OK)

- [ ] **Step 4: Commit**

```bash
git add src/agent.rs
git commit -m "test: add edge case tests for image descriptions in prompt formatting"
```
