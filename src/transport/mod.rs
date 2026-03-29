use async_trait::async_trait;
use tokio::sync::mpsc;
use anyhow::Result;
use std::sync::Arc;

use crate::message::OutputEvent;

#[cfg(feature = "cli-transport")]
pub mod cli;
#[cfg(feature = "websocket")]
pub mod websocket;
#[cfg(feature = "mqtt")]
pub mod mqtt;
#[cfg(any(feature = "unix-socket", feature = "tcp-socket"))]
pub mod socket;

pub struct TransportRequest {
    pub message: String,
    pub session: String,
    pub response_tx: mpsc::Sender<OutputEvent>,
}

pub type RequestHandler = Arc<dyn Fn(TransportRequest) + Send + Sync>;

#[async_trait]
pub trait Transport: Send + Sync {
    async fn serve(&self, handler: RequestHandler) -> Result<()>;
    fn name(&self) -> &str;
}

pub fn create_transports(config: &crate::config::Config) -> Result<Vec<Box<dyn Transport>>> {
    let mut transports: Vec<Box<dyn Transport>> = Vec::new();
    for name in &config.transports {
        match name.as_str() {
            #[cfg(feature = "cli-transport")]
            "cli" => {
                let cfg = config.transport.cli.clone().unwrap_or(crate::config::CliTransportConfig { prompt: "you> ".into() });
                transports.push(Box::new(cli::CliTransport::new(cfg)));
            }
            other => { tracing::warn!("Transport '{}' not implemented or not compiled in", other); }
        }
    }
    Ok(transports)
}
