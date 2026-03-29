mod agent;
mod backend;
mod cache;
mod config;
mod message;
mod repair;
mod tool;
mod transport;

use clap::Parser;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "edgeloop", about = "Minimal agentic framework for local LLMs")]
struct Cli {
    /// Path to config file
    #[arg(short, long, default_value = "edgeloop.toml")]
    config: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "edgeloop=info".into()),
        )
        .init();

    let cli = Cli::parse();
    tracing::info!("edgeloop v{}", env!("CARGO_PKG_VERSION"));

    // Load config
    let cfg = config::load_config(&cli.config)?;
    tracing::info!("Config loaded from {}", cli.config);

    // Load tools
    let base_dir = std::path::Path::new(&cli.config)
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_str()
        .unwrap_or(".");
    let tools = config::load_tool_packages(&cfg.tool_packages, base_dir)?;
    tracing::info!("Loaded {} tools", tools.len());

    // Create backend
    let backend: Arc<dyn backend::Backend> = Arc::from(backend::create_backend(&cfg.backend)?);

    // Create agent
    let agent = Arc::new(agent::Agent::new(
        backend,
        tools,
        &cfg.agent,
        &cfg.cache,
    ));

    // Create transports
    let transports = transport::create_transports(&cfg)?;
    if transports.is_empty() {
        anyhow::bail!("No transports configured. Add at least one to 'transports' in config.");
    }

    // Run first transport (Plan 3 will run multiple concurrently)
    let t = &transports[0];
    tracing::info!("Starting transport: {}", t.name());

    let agent_clone = agent.clone();
    let handler: transport::RequestHandler = Arc::new(move |req: transport::TransportRequest| {
        let agent = agent_clone.clone();
        tokio::spawn(async move {
            let result = agent.run(&req.message).await;
            let _ = req.response_tx.send(message::OutputEvent::Done {
                content: result,
                session: req.session,
            }).await;
        });
    });

    t.serve(handler).await?;

    Ok(())
}
