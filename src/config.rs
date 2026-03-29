use std::collections::HashMap;
use serde::Deserialize;
use anyhow::Result;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    #[serde(default)]
    pub include: Vec<String>,
    pub agent: AgentConfig,
    pub backend: BackendConfig,
    #[serde(default)]
    pub cache: CacheConfig,
    #[serde(default)]
    pub transports: Vec<String>,
    #[serde(default)]
    pub transport: TransportConfigs,
    #[serde(default)]
    pub tool_packages: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AgentConfig {
    #[serde(default = "default_system_prompt")]
    pub system_prompt: String,
    #[serde(default = "default_template")]
    pub template: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
    #[serde(default = "default_max_retries")]
    pub max_retries: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct BackendConfig {
    #[serde(rename = "type")]
    pub backend_type: String,
    #[serde(default)]
    pub endpoint: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub slot_id: Option<usize>,
    #[serde(default)]
    pub thinking: bool,
    #[serde(default)]
    pub api_key_env: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct CacheConfig {
    #[serde(default = "default_max_context")]
    pub max_context: usize,
    #[serde(default = "default_truncation_threshold")]
    pub truncation_threshold: f64,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct TransportConfigs {
    #[serde(default)]
    pub cli: Option<CliTransportConfig>,
    #[serde(default)]
    pub websocket: Option<WsTransportConfig>,
    #[serde(default)]
    pub mqtt: Option<MqttTransportConfig>,
    #[serde(default)]
    pub unix: Option<UnixTransportConfig>,
    #[serde(default)]
    pub tcp: Option<TcpTransportConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CliTransportConfig {
    #[serde(default = "default_prompt")]
    pub prompt: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct WsTransportConfig {
    #[serde(default = "default_ws_host")]
    pub host: String,
    #[serde(default = "default_ws_port")]
    pub port: u16,
    #[serde(default = "default_ws_path")]
    pub path: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct MqttTransportConfig {
    pub broker: String,
    pub topic_in: String,
    pub topic_out: String,
    #[serde(default = "default_mqtt_client_id")]
    pub client_id: String,
    #[serde(default = "default_mqtt_qos")]
    pub qos: u8,
}

#[derive(Debug, Deserialize, Clone)]
pub struct UnixTransportConfig {
    pub path: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TcpTransportConfig {
    #[serde(default = "default_tcp_host")]
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ToolPackage {
    pub tools: Vec<ToolDef>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub command: String,
    #[serde(default)]
    pub stdin: Option<String>,
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    #[serde(default)]
    pub workdir: Option<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(default)]
    pub parameters: HashMap<String, ParamDef>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ParamDef {
    #[serde(rename = "type")]
    pub param_type: String,
    #[serde(default = "default_true")]
    pub required: bool,
    #[serde(default)]
    pub default: Option<String>,
}

fn default_system_prompt() -> String { "You are a helpful assistant.".into() }
fn default_template() -> String { "chatml".into() }
fn default_max_tokens() -> usize { 4096 }
fn default_max_iterations() -> usize { 10 }
fn default_max_retries() -> usize { 2 }
fn default_temperature() -> f64 { 0.7 }
fn default_max_context() -> usize { 4096 }
fn default_truncation_threshold() -> f64 { 0.8 }
fn default_prompt() -> String { "you> ".into() }
fn default_ws_host() -> String { "0.0.0.0".into() }
fn default_ws_port() -> u16 { 8888 }
fn default_ws_path() -> String { "/agent".into() }
fn default_mqtt_client_id() -> String { "edgeloop-01".into() }
fn default_mqtt_qos() -> u8 { 1 }
fn default_tcp_host() -> String { "127.0.0.1".into() }
fn default_timeout() -> u64 { 10 }
fn default_true() -> bool { true }

pub fn expand_env_vars(s: &str) -> String {
    let re = regex::Regex::new(r"\$\{([^}:]+)(?::-([^}]*))?\}").unwrap();
    re.replace_all(s, |caps: &regex::Captures| {
        let var = &caps[1];
        let default = caps.get(2).map(|m| m.as_str()).unwrap_or("");
        std::env::var(var).unwrap_or_else(|_| default.to_string())
    })
    .to_string()
}

pub fn load_config(path: &str) -> Result<Config> {
    let content = std::fs::read_to_string(path)?;
    let expanded = expand_env_vars(&content);
    let config: Config = toml::from_str(&expanded)?;

    if !config.include.is_empty() {
        let base_dir = std::path::Path::new(path).parent().unwrap_or(std::path::Path::new("."));
        let mut merged = expanded.clone();
        for inc_path in &config.include {
            let full = base_dir.join(inc_path);
            if full.exists() {
                let inc_content = std::fs::read_to_string(&full)?;
                let inc_expanded = expand_env_vars(&inc_content);
                merged.push('\n');
                merged.push_str(&inc_expanded);
            }
        }
        let merged_config: Config = toml::from_str(&merged)?;
        return Ok(merged_config);
    }

    Ok(config)
}

pub fn load_tool_packages(packages: &[String], base_dir: &str) -> Result<Vec<ToolDef>> {
    let mut tools = Vec::new();
    for pkg_path in packages {
        let full = std::path::Path::new(base_dir).join(pkg_path).join("tools.toml");
        let content = std::fs::read_to_string(&full)
            .map_err(|e| anyhow::anyhow!("Failed to load tool package {}: {}", full.display(), e))?;
        let pkg: ToolPackage = toml::from_str(&content)?;
        tools.extend(pkg.tools);
    }
    Ok(tools)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_config() {
        let toml_str = r#"
[agent]
system_prompt = "Hello"

[backend]
type = "ollama"
endpoint = "http://localhost:11434"
model = "qwen3:0.6b"
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.agent.system_prompt, "Hello");
        assert_eq!(config.backend.backend_type, "ollama");
        assert_eq!(config.agent.max_tokens, 4096);
    }

    #[test]
    fn test_env_var_expansion() {
        std::env::set_var("EDGELOOP_TEST_HOST", "myhost:1234");
        assert_eq!(expand_env_vars("http://${EDGELOOP_TEST_HOST}/api"), "http://myhost:1234/api");
        std::env::remove_var("EDGELOOP_TEST_HOST");
    }

    #[test]
    fn test_env_var_default() {
        std::env::remove_var("NONEXISTENT_VAR_XYZ");
        assert_eq!(expand_env_vars("${NONEXISTENT_VAR_XYZ:-fallback}"), "fallback");
    }

    #[test]
    fn test_parse_tool_package() {
        let toml_str = r#"
[[tools]]
name = "read_file"
description = "Read a file"
command = "cat {path}"

[tools.parameters]
path = { type = "string", required = true }
"#;
        let pkg: ToolPackage = toml::from_str(toml_str).unwrap();
        assert_eq!(pkg.tools.len(), 1);
        assert_eq!(pkg.tools[0].name, "read_file");
        assert_eq!(pkg.tools[0].parameters["path"].param_type, "string");
        assert!(pkg.tools[0].parameters["path"].required);
    }

    #[test]
    fn test_parse_full_config() {
        let toml_str = r#"
transports = ["cli", "websocket"]
tool_packages = ["tools/filesystem"]

[agent]
system_prompt = "You are helpful."
template = "chatml"
max_tokens = 2048
temperature = 0.3

[backend]
type = "llama-server"
endpoint = "http://localhost:8080"
slot_id = 1

[cache]
max_context = 2048
truncation_threshold = 0.75

[transport.cli]
prompt = "edge> "

[transport.websocket]
host = "0.0.0.0"
port = 9999
path = "/chat"
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.agent.template, "chatml");
        assert_eq!(config.agent.max_tokens, 2048);
        assert_eq!(config.backend.backend_type, "llama-server");
        assert_eq!(config.backend.slot_id, Some(1));
        assert_eq!(config.cache.truncation_threshold, 0.75);
        assert_eq!(config.transports, vec!["cli", "websocket"]);
        assert_eq!(config.transport.cli.as_ref().unwrap().prompt, "edge> ");
        assert_eq!(config.transport.websocket.as_ref().unwrap().port, 9999);
    }
}
