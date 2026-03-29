#![cfg(any(feature = "unix-socket", feature = "tcp-socket"))]

use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;
use anyhow::Result;

use crate::message::{IncomingRequest, OutputEvent};
use crate::transport::{Transport, TransportRequest, RequestHandler};

async fn handle_stream<R, W>(
    reader: R,
    mut writer: W,
    handler: RequestHandler,
    peer: &str,
) -> Result<()>
where
    R: tokio::io::AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
{
    let buf_reader = BufReader::new(reader);
    let mut lines = buf_reader.lines();

    while let Some(line) = lines.next_line().await? {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let req: IncomingRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("Failed to parse line from {}: {}", peer, e);
                continue;
            }
        };

        let (tx, mut rx) = mpsc::channel::<OutputEvent>(64);
        handler(TransportRequest {
            message: req.message,
            session: req.session,
            response_tx: tx,
        });

        while let Some(event) = rx.recv().await {
            let mut json = match serde_json::to_string(&event) {
                Ok(j) => j,
                Err(e) => {
                    tracing::warn!("Failed to serialize OutputEvent: {}", e);
                    continue;
                }
            };
            json.push('\n');
            if let Err(e) = writer.write_all(json.as_bytes()).await {
                tracing::warn!("Write error to {}: {}", peer, e);
                return Ok(());
            }
            if matches!(event, OutputEvent::Done { .. } | OutputEvent::Error { .. }) {
                break;
            }
        }
    }

    Ok(())
}

// ── Unix socket ──────────────────────────────────────────────────────────────

#[cfg(feature = "unix-socket")]
use crate::config::UnixTransportConfig;

#[cfg(feature = "unix-socket")]
pub struct UnixSocketTransport {
    config: UnixTransportConfig,
}

#[cfg(feature = "unix-socket")]
impl UnixSocketTransport {
    pub fn new(config: UnixTransportConfig) -> Self {
        Self { config }
    }
}

#[cfg(feature = "unix-socket")]
#[async_trait]
impl Transport for UnixSocketTransport {
    fn name(&self) -> &str { "unix-socket" }

    async fn serve(&self, handler: RequestHandler) -> Result<()> {
        use tokio::net::UnixListener;

        // Remove stale socket file if it exists
        let path = std::path::Path::new(&self.config.path);
        if path.exists() {
            std::fs::remove_file(path)?;
        }

        let listener = UnixListener::bind(&self.config.path)?;
        tracing::info!("Unix socket transport listening on {}", self.config.path);

        loop {
            match listener.accept().await {
                Ok((stream, _addr)) => {
                    tracing::info!("Unix socket connection accepted");
                    let handler = handler.clone();
                    tokio::spawn(async move {
                        let (reader, writer) = tokio::io::split(stream);
                        if let Err(e) = handle_stream(reader, writer, handler, "unix").await {
                            tracing::warn!("Unix socket connection error: {}", e);
                        }
                    });
                }
                Err(e) => {
                    tracing::warn!("Unix socket accept error: {}", e);
                }
            }
        }
    }
}

// ── TCP socket ───────────────────────────────────────────────────────────────

#[cfg(feature = "tcp-socket")]
use crate::config::TcpTransportConfig;

#[cfg(feature = "tcp-socket")]
pub struct TcpSocketTransport {
    config: TcpTransportConfig,
}

#[cfg(feature = "tcp-socket")]
impl TcpSocketTransport {
    pub fn new(config: TcpTransportConfig) -> Self {
        Self { config }
    }
}

#[cfg(feature = "tcp-socket")]
#[async_trait]
impl Transport for TcpSocketTransport {
    fn name(&self) -> &str { "tcp-socket" }

    async fn serve(&self, handler: RequestHandler) -> Result<()> {
        use tokio::net::TcpListener;

        let addr = format!("{}:{}", self.config.host, self.config.port);
        let listener = TcpListener::bind(&addr).await?;
        tracing::info!("TCP socket transport listening on {}", addr);

        loop {
            match listener.accept().await {
                Ok((stream, peer_addr)) => {
                    tracing::info!("TCP connection from {}", peer_addr);
                    let handler = handler.clone();
                    tokio::spawn(async move {
                        let (reader, writer) = stream.into_split();
                        let peer = peer_addr.to_string();
                        if let Err(e) = handle_stream(reader, writer, handler, &peer).await {
                            tracing::warn!("TCP connection error from {}: {}", peer_addr, e);
                        }
                    });
                }
                Err(e) => {
                    tracing::warn!("TCP accept error: {}", e);
                }
            }
        }
    }
}
