#![cfg(feature = "websocket")]

use async_trait::async_trait;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use anyhow::Result;
use futures::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message as WsMessage;

use crate::config::WsTransportConfig;
use crate::message::{IncomingRequest, OutputEvent};
use crate::transport::{Transport, TransportRequest, RequestHandler};

pub struct WebSocketTransport {
    config: WsTransportConfig,
}

impl WebSocketTransport {
    pub fn new(config: WsTransportConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl Transport for WebSocketTransport {
    fn name(&self) -> &str { "websocket" }

    async fn serve(&self, handler: RequestHandler) -> Result<()> {
        let addr = format!("{}:{}", self.config.host, self.config.port);
        let listener = TcpListener::bind(&addr).await?;
        tracing::info!("WebSocket transport listening on ws://{}{}", addr, self.config.path);

        loop {
            match listener.accept().await {
                Ok((stream, peer_addr)) => {
                    tracing::info!("WebSocket connection from {}", peer_addr);
                    let handler = handler.clone();
                    let path = self.config.path.clone();
                    tokio::spawn(async move {
                        if let Err(e) = handle_connection(stream, handler, path).await {
                            tracing::warn!("WebSocket connection error from {}: {}", peer_addr, e);
                        }
                    });
                }
                Err(e) => {
                    tracing::warn!("WebSocket accept error: {}", e);
                }
            }
        }
    }
}

async fn handle_connection(
    stream: tokio::net::TcpStream,
    handler: RequestHandler,
    _path: String,
) -> Result<()> {
    let ws_stream = tokio_tungstenite::accept_async(stream).await?;
    let (mut ws_sender, mut ws_receiver) = ws_stream.split();

    while let Some(msg) = ws_receiver.next().await {
        let msg = match msg {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("WebSocket receive error: {}", e);
                break;
            }
        };

        match msg {
            WsMessage::Text(text) => {
                let req: IncomingRequest = match serde_json::from_str(&text) {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::warn!("Failed to parse WebSocket message as IncomingRequest: {}", e);
                        continue;
                    }
                };

                let (tx, mut rx) = mpsc::channel::<OutputEvent>(64);
                handler(TransportRequest {
                    message: req.message,
                    session: req.session,
                    images: req.images,
                    response_tx: tx,
                });

                while let Some(event) = rx.recv().await {
                    let json = match serde_json::to_string(&event) {
                        Ok(j) => j,
                        Err(e) => {
                            tracing::warn!("Failed to serialize OutputEvent: {}", e);
                            continue;
                        }
                    };
                    if let Err(e) = ws_sender.send(WsMessage::Text(json.into())).await {
                        tracing::warn!("WebSocket send error: {}", e);
                        return Ok(());
                    }
                    if matches!(event, OutputEvent::Done { .. } | OutputEvent::Error { .. }) {
                        break;
                    }
                }
            }
            WsMessage::Close(_) => {
                tracing::info!("WebSocket client closed connection");
                break;
            }
            WsMessage::Ping(data) => {
                if let Err(e) = ws_sender.send(WsMessage::Pong(data)).await {
                    tracing::warn!("WebSocket pong error: {}", e);
                    break;
                }
            }
            _ => {}
        }
    }

    Ok(())
}
