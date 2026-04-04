use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio::io::{AsyncBufReadExt, BufReader};
use anyhow::Result;

use crate::config::CliTransportConfig;
use crate::message::OutputEvent;
use crate::transport::{Transport, TransportRequest, RequestHandler};

pub struct CliTransport {
    prompt: String,
}

impl CliTransport {
    pub fn new(config: CliTransportConfig) -> Self {
        Self { prompt: config.prompt }
    }
}

#[async_trait]
impl Transport for CliTransport {
    fn name(&self) -> &str { "cli" }

    async fn serve(&self, handler: RequestHandler) -> Result<()> {
        let stdin = BufReader::new(tokio::io::stdin());
        let mut lines = stdin.lines();

        loop {
            eprint!("{}", self.prompt);
            let line: String = match lines.next_line().await? {
                Some(l) => l,
                None => break,
            };

            let trimmed = line.trim().to_string();
            if trimmed.is_empty() { continue; }
            if trimmed == "quit" || trimmed == "exit" {
                eprintln!("Bye!");
                break;
            }

            let (tx, mut rx) = mpsc::channel(64);
            handler(TransportRequest {
                message: trimmed,
                session: "cli".to_string(),
                images: vec![],
                response_tx: tx,
            });

            while let Some(event) = rx.recv().await {
                match event {
                    OutputEvent::Token { content, .. } => eprint!("{}", content),
                    OutputEvent::Done { content, .. } => { eprintln!("\n{}", content); break; }
                    OutputEvent::ToolCall { tool, .. } => eprintln!("\n[calling {}...]", tool),
                    OutputEvent::ToolResult { tool, result, .. } => eprintln!("[{} -> {}]", tool, &result[..result.len().min(80)]),
                    OutputEvent::Error { content, .. } => { eprintln!("\nError: {}", content); break; }
                }
            }
        }
        Ok(())
    }
}
