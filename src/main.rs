mod config;

use clap::Parser;

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
    tracing::info!("edgeloop starting, config: {}", cli.config);

    Ok(())
}
