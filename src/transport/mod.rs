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
#[cfg(feature = "ros2")]
pub mod ros2;

pub struct TransportRequest {
    pub message: String,
    pub session: String,
    pub images: Vec<crate::message::ImageAttachment>,
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
            #[cfg(feature = "websocket")]
            "websocket" => {
                let cfg = config.transport.websocket.clone().unwrap_or(crate::config::WsTransportConfig {
                    host: "0.0.0.0".into(),
                    port: 8888,
                    path: "/agent".into(),
                });
                transports.push(Box::new(websocket::WebSocketTransport::new(cfg)));
            }
            #[cfg(feature = "mqtt")]
            "mqtt" => {
                if let Some(cfg) = config.transport.mqtt.clone() {
                    transports.push(Box::new(mqtt::MqttTransport::new(cfg)));
                } else {
                    tracing::warn!("Transport 'mqtt' requested but [transport.mqtt] config is missing");
                }
            }
            #[cfg(feature = "unix-socket")]
            "unix-socket" => {
                if let Some(cfg) = config.transport.unix.clone() {
                    transports.push(Box::new(socket::UnixSocketTransport::new(cfg)));
                } else {
                    tracing::warn!("Transport 'unix-socket' requested but [transport.unix] config is missing");
                }
            }
            #[cfg(feature = "tcp-socket")]
            "tcp-socket" => {
                if let Some(cfg) = config.transport.tcp.clone() {
                    transports.push(Box::new(socket::TcpSocketTransport::new(cfg)));
                } else {
                    tracing::warn!("Transport 'tcp-socket' requested but [transport.tcp] config is missing");
                }
            }
            #[cfg(feature = "ros2")]
            "ros2" => {
                if let Some(cfg) = config.transport.ros2.clone() {
                    transports.push(Box::new(ros2::Ros2Transport::new(cfg)));
                } else {
                    tracing::warn!("Transport 'ros2' requested but [transport.ros2] config is missing");
                }
            }
            other => { tracing::warn!("Transport '{}' not implemented or not compiled in", other); }
        }
    }
    Ok(transports)
}
