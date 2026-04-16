#![cfg(feature = "ros2")]

use async_trait::async_trait;
use tokio::sync::mpsc;
use anyhow::Result;
use std::pin::pin;

use crate::config::Ros2TransportConfig;
use crate::message::{IncomingRequest, OutputEvent};
use crate::transport::{Transport, TransportRequest, RequestHandler};

pub struct Ros2Transport {
    config: Ros2TransportConfig,
}

impl Ros2Transport {
    pub fn new(config: Ros2TransportConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl Transport for Ros2Transport {
    fn name(&self) -> &str { "ros2" }

    async fn serve(&self, handler: RequestHandler) -> Result<()> {
        use ros2_client::{Context, ContextOptions, MessageTypeName, Name, NodeName, NodeOptions};
        use ros2_client::rustdds::{
            QosPolicyBuilder,
            policy::{Deadline, Durability, History, Lifespan, Ownership, Reliability},
            Duration,
        };
        use ros2_interfaces_jazzy_serde::std_msgs;
        use futures::StreamExt;

        let depth = self.config.qos_depth;
        let sub_qos = QosPolicyBuilder::new()
            .durability(Durability::Volatile)
            .deadline(Deadline(Duration::INFINITE))
            .ownership(Ownership::Shared)
            .reliability(Reliability::Reliable { max_blocking_time: Duration::from_millis(100) })
            .history(History::KeepLast { depth: depth as i32 })
            .lifespan(Lifespan { duration: Duration::INFINITE })
            .build();
        let pub_qos = QosPolicyBuilder::new()
            .durability(Durability::Volatile)
            .deadline(Deadline(Duration::INFINITE))
            .ownership(Ownership::Shared)
            .reliability(Reliability::Reliable { max_blocking_time: Duration::from_millis(100) })
            .history(History::KeepLast { depth: depth as i32 })
            .lifespan(Lifespan { duration: Duration::INFINITE })
            .build();

        // Create DDS context and node
        let context = Context::with_options(
            ContextOptions::new().domain_id(self.config.domain_id),
        )?;
        let mut node = context.new_node(
            NodeName::new(&self.config.namespace, &self.config.node_name)
                .map_err(|e| anyhow::anyhow!("Invalid ROS2 node name: {:?}", e))?,
            NodeOptions::new().enable_rosout(true),
        )?;

        // Create input topic (subscribe)
        let topic_in = node.create_topic(
            &Name::new(&self.config.namespace, &self.config.topic_in)
                .map_err(|e| anyhow::anyhow!("Invalid ROS2 topic name: {:?}", e))?,
            MessageTypeName::new("std_msgs", "String"),
            &sub_qos,
        )?;
        let subscription = node.create_subscription::<std_msgs::msg::String>(
            &topic_in, None,
        )?;

        // Create output topic (publish)
        let topic_out = node.create_topic(
            &Name::new(&self.config.namespace, &self.config.topic_out)
                .map_err(|e| anyhow::anyhow!("Invalid ROS2 topic name: {:?}", e))?,
            MessageTypeName::new("std_msgs", "String"),
            &pub_qos,
        )?;
        let publisher = node.create_publisher::<std_msgs::msg::String>(
            &topic_out, None,
        )?;

        tracing::info!(
            "ROS2 transport started: node={}{}, domain_id={}, subscribe={}, publish={}",
            self.config.namespace, self.config.node_name,
            self.config.domain_id,
            self.config.topic_in, self.config.topic_out,
        );

        // Listen for incoming messages via async stream (pinned for Unpin requirement)
        let mut stream = pin!(subscription.async_stream());
        while let Some(result) = stream.next().await {
            let (msg, _info) = match result {
                Ok(pair) => pair,
                Err(e) => {
                    tracing::warn!("ROS2 receive error: {:?}", e);
                    continue;
                }
            };

            let payload = &msg.data;
            let req: IncomingRequest = match serde_json::from_str(payload) {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("Failed to parse ROS2 payload as IncomingRequest: {}", e);
                    continue;
                }
            };

            let images = crate::message::resolve_images(req.images).await;
            let (tx, mut rx) = mpsc::channel::<OutputEvent>(64);
            handler(TransportRequest {
                message: req.message,
                session: req.session,
                images,
                response_tx: tx,
            });

            // Publish responses synchronously — Publisher is not Send so we
            // drain the channel on the current task instead of spawning.
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
                let ros_msg = std_msgs::msg::String { data: json };
                if let Err(e) = publisher.publish(ros_msg) {
                    tracing::warn!("ROS2 publish error: {:?}", e);
                    break;
                }
                if is_terminal {
                    break;
                }
            }
        }

        Ok(())
    }
}
