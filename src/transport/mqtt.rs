#![cfg(feature = "mqtt")]

use async_trait::async_trait;
use tokio::sync::mpsc;
use anyhow::Result;
use rumqttc::{AsyncClient, Event, Incoming, MqttOptions, QoS};
use std::time::Duration;

use crate::config::MqttTransportConfig;
use crate::message::{IncomingRequest, OutputEvent};
use crate::transport::{Transport, TransportRequest, RequestHandler};

pub struct MqttTransport {
    config: MqttTransportConfig,
}

impl MqttTransport {
    pub fn new(config: MqttTransportConfig) -> Self {
        Self { config }
    }
}

fn qos_from_u8(q: u8) -> QoS {
    match q {
        0 => QoS::AtMostOnce,
        2 => QoS::ExactlyOnce,
        _ => QoS::AtLeastOnce,
    }
}

#[async_trait]
impl Transport for MqttTransport {
    fn name(&self) -> &str { "mqtt" }

    async fn serve(&self, handler: RequestHandler) -> Result<()> {
        // Parse broker URL: accept "host:port" or just "host" (default 1883)
        let (host, port) = parse_broker(&self.config.broker);

        let mut opts = MqttOptions::new(&self.config.client_id, host, port);
        opts.set_keep_alive(Duration::from_secs(30));
        opts.set_clean_session(true);

        let (client, mut eventloop) = AsyncClient::new(opts, 64);
        let qos = qos_from_u8(self.config.qos);

        client.subscribe(&self.config.topic_in, qos).await?;
        tracing::info!(
            "MQTT transport connected to {}, subscribed to {}",
            self.config.broker,
            self.config.topic_in
        );

        loop {
            match eventloop.poll().await {
                Ok(Event::Incoming(Incoming::Publish(publish))) => {
                    let payload = match std::str::from_utf8(&publish.payload) {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::warn!("MQTT message is not valid UTF-8: {}", e);
                            continue;
                        }
                    };

                    let req: IncomingRequest = match serde_json::from_str(payload) {
                        Ok(r) => r,
                        Err(e) => {
                            tracing::warn!("Failed to parse MQTT payload as IncomingRequest: {}", e);
                            continue;
                        }
                    };

                    let (tx, mut rx) = mpsc::channel::<OutputEvent>(64);
                    handler(TransportRequest {
                        message: req.message,
                        session: req.session,
                        response_tx: tx,
                    });

                    let client_clone = client.clone();
                    let topic_out = self.config.topic_out.clone();

                    tokio::spawn(async move {
                        while let Some(event) = rx.recv().await {
                            let json = match serde_json::to_string(&event) {
                                Ok(j) => j,
                                Err(e) => {
                                    tracing::warn!("Failed to serialize OutputEvent: {}", e);
                                    continue;
                                }
                            };
                            let is_terminal = matches!(
                                event,
                                OutputEvent::Done { .. } | OutputEvent::Error { .. }
                            );
                            if let Err(e) = client_clone
                                .publish(&topic_out, qos, false, json.as_bytes())
                                .await
                            {
                                tracing::warn!("MQTT publish error: {}", e);
                                break;
                            }
                            if is_terminal {
                                break;
                            }
                        }
                    });
                }
                Ok(Event::Incoming(Incoming::ConnAck(_))) => {
                    tracing::info!("MQTT connected");
                }
                Ok(Event::Incoming(Incoming::Disconnect)) => {
                    tracing::info!("MQTT broker disconnected, reconnecting...");
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!("MQTT eventloop error: {}, retrying in 5s", e);
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
            }
        }
    }
}

fn parse_broker(broker: &str) -> (String, u16) {
    // Strip scheme if present
    let stripped = broker
        .strip_prefix("mqtt://")
        .or_else(|| broker.strip_prefix("tcp://"))
        .unwrap_or(broker);

    if let Some(colon_pos) = stripped.rfind(':') {
        let host = stripped[..colon_pos].to_string();
        if let Ok(port) = stripped[colon_pos + 1..].parse::<u16>() {
            return (host, port);
        }
    }
    (stripped.to_string(), 1883)
}
