# edgeloop Rust Rewrite — Plan 1: Core

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the core edgeloop Rust binary — config loading, repair pipeline, tool executor, cache manager, and agent loop — testable with a mock backend before real LLM backends are added.

**Architecture:** Single async binary (tokio). Config from TOML files with env var expansion and includes. Tools are CLI commands defined in TOML. Agent loop is backend-agnostic — a Backend trait with a test mock lets us verify the full ReAct cycle without an LLM.

**Tech Stack:** Rust 2021, tokio, serde + serde_json, toml, clap, regex, tracing

**Note:** This plan removes all Python source code and replaces it with Rust. The Python edgeloop/ directory, tests/, and examples/ are deleted. README.md and CLAUDE.md are updated at the end.

---

### Task 0: Scaffold Rust project, remove Python

**Files:**
- Create: `Cargo.toml`
- Create: `src/main.rs`
- Create: `src/lib.rs`
- Delete: `edgeloop/` (Python package)
- Delete: `tests/` (Python tests)
- Delete: `examples/` (Python examples)
- Delete: `pyproject.toml`
- Keep: `README.md`, `CLAUDE.md`, `LICENSE`, `.gitignore`

- [ ] **Step 1: Remove Python source**

```bash
cd /home/parham/develop/src/parhamdb/localclaw
rm -rf edgeloop/ tests/ examples/ pyproject.toml
```

- [ ] **Step 2: Create Cargo.toml**

```toml
[package]
name = "edgeloop"
version = "0.1.0"
edition = "2021"
description = "Minimal agentic framework for local LLMs"
license = "MIT"

[features]
default = ["ollama", "llama-server", "cli-transport"]
ollama = []
llama-server = []
openai = []
cli-transport = []
websocket = ["dep:tokio-tungstenite"]
mqtt = ["dep:rumqttc"]
unix-socket = []
tcp-socket = []
full = ["ollama", "llama-server", "openai", "cli-transport", "websocket", "mqtt", "unix-socket", "tcp-socket"]

[dependencies]
tokio = { version = "1", features = ["rt-multi-thread", "macros", "net", "process", "io-util", "signal", "sync", "time"] }
reqwest = { version = "0.12", default-features = false, features = ["rustls-tls", "stream", "json"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
toml = "0.8"
clap = { version = "4", features = ["derive"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
regex = "1"
tokio-stream = "0.1"
async-trait = "0.1"
futures = "0.3"
anyhow = "1"

tokio-tungstenite = { version = "0.24", optional = true }
rumqttc = { version = "0.24", optional = true }

[dev-dependencies]
tokio-test = "0.4"

[profile.release]
opt-level = "z"
lto = true
codegen-units = 1
strip = true
panic = "abort"
```

- [ ] **Step 3: Create src/main.rs**

```rust
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
```

- [ ] **Step 4: Create src/lib.rs**

```rust
pub mod config;
pub mod repair;
pub mod cache;
pub mod tool;
pub mod message;
pub mod agent;
pub mod backend;
pub mod transport;
```

Create stub modules so it compiles:

`src/config.rs`:
```rust
// TODO: Task 1
```

`src/repair.rs`:
```rust
// TODO: Task 2
```

`src/cache.rs`:
```rust
// TODO: Task 3
```

`src/tool.rs`:
```rust
// TODO: Task 4
```

`src/message.rs`:
```rust
// TODO: Task 5
```

`src/agent.rs`:
```rust
// TODO: Task 6
```

`src/backend/mod.rs`:
```rust
// TODO: Plan 2
```

`src/transport/mod.rs`:
```rust
// TODO: Plan 3
```

- [ ] **Step 5: Verify it compiles**

```bash
cd /home/parham/develop/src/parhamdb/localclaw
cargo build 2>&1
```
Expected: compiles with warnings about unused TODO comments.

- [ ] **Step 6: Update .gitignore and commit**

Append to `.gitignore`:
```
target/
```

```bash
git add -A
git commit -m "feat: scaffold Rust project, remove Python source"
```

---

### Task 1: Config — TOML parsing, env var expansion, includes

**Files:**
- Create: `src/config.rs`
- Create: `edgeloop.toml` (example config)
- Create: `tools/filesystem/tools.toml` (example tool package)
- Test: inline `#[cfg(test)]` module

- [ ] **Step 1: Write failing tests**

In `src/config.rs`:
```rust
use std::collections::HashMap;
use serde::Deserialize;
use anyhow::Result;

// ── Structs ──

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

// ── Tool config ──

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

// ── Defaults ──

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

// ── Env var expansion ──

pub fn expand_env_vars(s: &str) -> String {
    let re = regex::Regex::new(r"\$\{([^}:]+)(?::-([^}]*))?\}").unwrap();
    re.replace_all(s, |caps: &regex::Captures| {
        let var = &caps[1];
        let default = caps.get(2).map(|m| m.as_str()).unwrap_or("");
        std::env::var(var).unwrap_or_else(|_| default.to_string())
    })
    .to_string()
}

// ── Loading ──

pub fn load_config(path: &str) -> Result<Config> {
    let content = std::fs::read_to_string(path)?;
    let expanded = expand_env_vars(&content);
    let config: Config = toml::from_str(&expanded)?;

    // Process includes
    if !config.include.is_empty() {
        let base_dir = std::path::Path::new(path).parent().unwrap_or(std::path::Path::new("."));
        let mut merged = expanded.clone();
        for inc_path in &config.include {
            let full = base_dir.join(inc_path);
            if full.exists() {
                let inc_content = std::fs::read_to_string(&full)?;
                let inc_expanded = expand_env_vars(&inc_content);
                // Simple merge: append include content (later values override via TOML re-parse)
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
    use std::io::Write;

    #[test]
    fn test_parse_minimal_config() {
        let toml = r#"
[agent]
system_prompt = "Hello"

[backend]
type = "ollama"
endpoint = "http://localhost:11434"
model = "qwen3:0.6b"
"#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.agent.system_prompt, "Hello");
        assert_eq!(config.backend.backend_type, "ollama");
        assert_eq!(config.agent.max_tokens, 4096); // default
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
        let toml = r#"
[[tools]]
name = "read_file"
description = "Read a file"
command = "cat {path}"

[tools.parameters]
path = { type = "string", required = true }
"#;
        let pkg: ToolPackage = toml::from_str(toml).unwrap();
        assert_eq!(pkg.tools.len(), 1);
        assert_eq!(pkg.tools[0].name, "read_file");
        assert_eq!(pkg.tools[0].parameters["path"].param_type, "string");
        assert!(pkg.tools[0].parameters["path"].required);
    }

    #[test]
    fn test_parse_full_config() {
        let toml = r#"
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

transports = ["cli", "websocket"]

[transport.cli]
prompt = "edge> "

[transport.websocket]
host = "0.0.0.0"
port = 9999
path = "/chat"

tool_packages = ["tools/filesystem"]
"#;
        let config: Config = toml::from_str(toml).unwrap();
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
```

- [ ] **Step 2: Run tests**

```bash
cargo test config::tests -- --nocapture
```
Expected: all 4 tests pass.

- [ ] **Step 3: Create example config files**

`edgeloop.toml`:
```toml
[agent]
system_prompt = "You are a helpful assistant."
template = "chatml"
max_tokens = 4096
max_iterations = 10
max_retries = 2
temperature = 0.7

[backend]
type = "ollama"
endpoint = "${OLLAMA_HOST:-http://localhost:11434}"
model = "qwen2.5-coder:7b"

transports = ["cli"]

[transport.cli]
prompt = "you> "

tool_packages = ["tools/filesystem"]
```

`tools/filesystem/tools.toml`:
```toml
[[tools]]
name = "read_file"
description = "Read a file from disk"
command = "cat {path}"
[tools.parameters]
path = { type = "string", required = true }

[[tools]]
name = "write_file"
description = "Write content to a file"
command = "tee {path}"
stdin = "{content}"
[tools.parameters]
path = { type = "string", required = true }
content = { type = "string", required = true }

[[tools]]
name = "list_dir"
description = "List files in a directory"
command = "ls -1 {path}"
[tools.parameters]
path = { type = "string", default = "." }
```

`tools/system/tools.toml`:
```toml
[[tools]]
name = "shell"
description = "Run a shell command and return output"
command = "sh -c '{command}'"
timeout = 30
[tools.parameters]
command = { type = "string", required = true }

[[tools]]
name = "find_files"
description = "Find files matching a pattern"
command = "find {path} -name '{pattern}'"
timeout = 15
[tools.parameters]
pattern = { type = "string", required = true }
path = { type = "string", default = "." }
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: add config module with TOML parsing, env vars, tool packages"
```

---

### Task 2: Repair pipeline — JSON extract, fix, fuzzy match, coerce

**Files:**
- Create: `src/repair.rs`

- [ ] **Step 1: Write tests and implementation**

`src/repair.rs`:
```rust
use serde_json::Value;
use std::collections::HashMap;
use tracing::debug;

use crate::config::ToolDef;

/// Result of repairing a tool call from LLM output.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub name: String,
    pub arguments: HashMap<String, Value>,
}

/// Full repair pipeline: extract JSON → fix syntax → parse → match tool → coerce args.
pub fn repair_tool_call(text: &str, tools: &[ToolDef]) -> Option<ToolCall> {
    let raw = extract_json(text)?;
    let repaired = repair_json(&raw);

    let parsed: Value = match serde_json::from_str(&repaired) {
        Ok(v) => v,
        Err(_) => {
            debug!("JSON repair failed for: {}", &raw[..raw.len().min(100)]);
            return None;
        }
    };

    let tool_name = parsed.get("tool")?.as_str()?;
    let arguments = parsed
        .get("arguments")
        .and_then(|v| v.as_object())
        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();

    let available: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    let matched = fuzzy_match_tool(tool_name, &available, 2)?;

    let tool_def = tools.iter().find(|t| t.name == matched)?;
    let coerced = coerce_arguments(&arguments, tool_def);

    Some(ToolCall {
        name: matched,
        arguments: coerced,
    })
}

/// Extract a JSON object from LLM output. Handles markdown fences, XML tags, raw JSON.
pub fn extract_json(text: &str) -> Option<String> {
    // Markdown fence
    let re_fence = regex::Regex::new(r"```(?:json)?\s*\n?([\s\S]*?)\n?```").unwrap();
    if let Some(caps) = re_fence.captures(text) {
        return Some(caps[1].trim().to_string());
    }

    // XML tags
    let re_xml = regex::Regex::new(r"<tool_call>([\s\S]*?)</tool_call>").unwrap();
    if let Some(caps) = re_xml.captures(text) {
        return Some(caps[1].trim().to_string());
    }

    // Raw brace matching
    let start = text.find('{')?;
    let bytes = text.as_bytes();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;

    for i in start..bytes.len() {
        let c = bytes[i] as char;
        if escape {
            escape = false;
            continue;
        }
        if c == '\\' {
            escape = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if c == '{' {
            depth += 1;
        } else if c == '}' {
            depth -= 1;
            if depth == 0 {
                return Some(text[start..=i].to_string());
            }
        }
    }

    if depth > 0 {
        return Some(text[start..].to_string());
    }

    None
}

/// Fix common JSON syntax issues from small models.
pub fn repair_json(text: &str) -> String {
    if serde_json::from_str::<Value>(text).is_ok() {
        return text.to_string();
    }

    let mut result = text.to_string();

    // Single quotes → double quotes
    let re_sq = regex::Regex::new(r"'([^']*)'").unwrap();
    result = re_sq.replace_all(&result, "\"$1\"").to_string();

    // Remove trailing commas
    let re_tc = regex::Regex::new(r",\s*([}\]])").unwrap();
    result = re_tc.replace_all(&result, "$1").to_string();

    // Close unmatched braces
    let open_b = result.matches('{').count() as i32 - result.matches('}').count() as i32;
    if open_b > 0 {
        result.push_str(&"}".repeat(open_b as usize));
    }
    let open_k = result.matches('[').count() as i32 - result.matches(']').count() as i32;
    if open_k > 0 {
        result.push_str(&"]".repeat(open_k as usize));
    }

    result
}

/// Levenshtein edit distance.
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a_len = a.len();
    let b_len = b.len();
    if a_len == 0 { return b_len; }
    if b_len == 0 { return a_len; }

    let mut prev: Vec<usize> = (0..=b_len).collect();
    let mut curr = vec![0; b_len + 1];

    for (i, ca) in a.chars().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.chars().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (curr[j] + 1).min(prev[j + 1] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[b_len]
}

/// Fuzzy match a tool name against available names.
pub fn fuzzy_match_tool(name: &str, available: &[&str], max_distance: usize) -> Option<String> {
    if available.contains(&name) {
        return Some(name.to_string());
    }

    let mut best: Option<&str> = None;
    let mut best_dist = max_distance + 1;

    for &candidate in available {
        let dist = levenshtein(name, candidate);
        if dist < best_dist {
            best_dist = dist;
            best = Some(candidate);
        }
    }

    if best_dist <= max_distance {
        best.map(|s| s.to_string())
    } else {
        None
    }
}

/// Coerce arguments to match tool schema types. Strips unknown fields, adds defaults.
pub fn coerce_arguments(args: &HashMap<String, Value>, tool: &ToolDef) -> HashMap<String, Value> {
    let mut result = HashMap::new();

    // Direct name matching
    for (key, param) in &tool.parameters {
        if let Some(val) = args.get(key) {
            result.insert(key.clone(), coerce_value(val, &param.param_type));
        } else if let Some(default) = &param.default {
            result.insert(key.clone(), Value::String(default.clone()));
        }
    }

    // Positional fallback for missing required params
    let missing_required: Vec<&String> = tool
        .parameters
        .iter()
        .filter(|(k, p)| p.required && !result.contains_key(*k))
        .map(|(k, _)| k)
        .collect();

    if !missing_required.is_empty() {
        let unmatched: Vec<&Value> = args
            .iter()
            .filter(|(k, _)| !tool.parameters.contains_key(k.as_str()))
            .map(|(_, v)| v)
            .collect();

        for (i, key) in missing_required.iter().enumerate() {
            if i < unmatched.len() {
                let param_type = &tool.parameters[*key].param_type;
                result.insert((*key).clone(), coerce_value(unmatched[i], param_type));
            }
        }
    }

    result
}

fn coerce_value(val: &Value, target_type: &str) -> Value {
    match target_type {
        "integer" => {
            if let Value::String(s) = val {
                s.parse::<i64>().map(|n| Value::Number(n.into())).unwrap_or_else(|_| val.clone())
            } else {
                val.clone()
            }
        }
        "number" => {
            if let Value::String(s) = val {
                s.parse::<f64>()
                    .ok()
                    .and_then(|f| serde_json::Number::from_f64(f))
                    .map(Value::Number)
                    .unwrap_or_else(|| val.clone())
            } else {
                val.clone()
            }
        }
        "boolean" => {
            if let Value::String(s) = val {
                Value::Bool(matches!(s.to_lowercase().as_str(), "true" | "1" | "yes"))
            } else {
                val.clone()
            }
        }
        _ => val.clone(),
    }
}

/// Heuristic: does this look like a broken tool call attempt?
pub fn looks_like_broken_tool_call(text: &str) -> bool {
    let indicators = ["\"tool\"", "'tool'", "tool_call", "arguments"];
    if indicators.iter().any(|ind| text.contains(ind)) {
        return true;
    }
    text.matches('{').count() >= 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ParamDef;

    fn make_tool(name: &str, params: Vec<(&str, &str, bool)>) -> ToolDef {
        let mut parameters = HashMap::new();
        for (pname, ptype, required) in params {
            parameters.insert(
                pname.to_string(),
                ParamDef {
                    param_type: ptype.to_string(),
                    required,
                    default: None,
                },
            );
        }
        ToolDef {
            name: name.to_string(),
            description: String::new(),
            command: String::new(),
            stdin: None,
            timeout: 10,
            workdir: None,
            env: HashMap::new(),
            parameters,
        }
    }

    #[test]
    fn test_extract_markdown_fence() {
        let text = "Sure!\n```json\n{\"tool\": \"read_file\"}\n```";
        assert_eq!(extract_json(text).unwrap(), "{\"tool\": \"read_file\"}");
    }

    #[test]
    fn test_extract_xml_tags() {
        let text = "<tool_call>{\"tool\": \"x\"}</tool_call>";
        assert_eq!(extract_json(text).unwrap(), "{\"tool\": \"x\"}");
    }

    #[test]
    fn test_extract_raw_json() {
        let text = "I will do that. {\"tool\": \"x\", \"arguments\": {}} Done.";
        let result = extract_json(text).unwrap();
        assert!(result.contains("\"tool\""));
    }

    #[test]
    fn test_extract_no_json() {
        assert!(extract_json("Just plain text.").is_none());
    }

    #[test]
    fn test_repair_trailing_comma() {
        let result = repair_json("{\"a\": 1,}");
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["a"], 1);
    }

    #[test]
    fn test_repair_single_quotes() {
        let result = repair_json("{'tool': 'read_file'}");
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["tool"], "read_file");
    }

    #[test]
    fn test_repair_unmatched_brace() {
        let result = repair_json("{\"tool\": \"x\"");
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["tool"], "x");
    }

    #[test]
    fn test_repair_valid_unchanged() {
        let original = "{\"tool\": \"x\"}";
        assert_eq!(repair_json(original), original);
    }

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein("abc", "abc"), 0);
    }

    #[test]
    fn test_levenshtein_one_off() {
        assert_eq!(levenshtein("red_file", "read_file"), 1);
    }

    #[test]
    fn test_fuzzy_exact() {
        assert_eq!(fuzzy_match_tool("read_file", &["read_file", "write_file"], 2), Some("read_file".into()));
    }

    #[test]
    fn test_fuzzy_close() {
        assert_eq!(fuzzy_match_tool("red_file", &["read_file", "write_file"], 2), Some("read_file".into()));
    }

    #[test]
    fn test_fuzzy_too_far() {
        assert_eq!(fuzzy_match_tool("xyz_abc", &["read_file", "write_file"], 2), None);
    }

    #[test]
    fn test_coerce_string_to_int() {
        let tool = make_tool("t", vec![("count", "integer", true)]);
        let mut args = HashMap::new();
        args.insert("count".into(), Value::String("42".into()));
        let result = coerce_arguments(&args, &tool);
        assert_eq!(result["count"], 42);
    }

    #[test]
    fn test_coerce_strip_extra() {
        let tool = make_tool("t", vec![("a", "string", true)]);
        let mut args = HashMap::new();
        args.insert("a".into(), Value::String("x".into()));
        args.insert("extra".into(), Value::String("y".into()));
        let result = coerce_arguments(&args, &tool);
        assert!(!result.contains_key("extra"));
    }

    #[test]
    fn test_full_pipeline() {
        let tools = vec![make_tool("read_file", vec![("path", "string", true)])];
        let text = "```json\n{'tool': 'red_file', 'arguments': {'path': '/tmp/x',}}\n```";
        let result = repair_tool_call(text, &tools).unwrap();
        assert_eq!(result.name, "read_file");
        assert_eq!(result.arguments["path"], "/tmp/x");
    }

    #[test]
    fn test_pipeline_no_tool() {
        let tools = vec![make_tool("read_file", vec![("path", "string", true)])];
        assert!(repair_tool_call("Just plain text.", &tools).is_none());
    }

    #[test]
    fn test_pipeline_hallucinated_tool() {
        let tools = vec![make_tool("read_file", vec![("path", "string", true)])];
        let text = "{\"tool\": \"delete_everything\", \"arguments\": {}}";
        assert!(repair_tool_call(text, &tools).is_none());
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test repair::tests -- --nocapture
```
Expected: all 16 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/repair.rs
git commit -m "feat: add repair pipeline — JSON extract, fix, fuzzy match, coercion"
```

---

### Task 3: Cache manager

**Files:**
- Create: `src/cache.rs`

- [ ] **Step 1: Write implementation with tests**

`src/cache.rs`:
```rust
use serde::Serialize;
use tracing::{debug, info};

#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_ms: f64,
    pub generation_ms: f64,
    pub cache_hit_tokens: usize,
}

impl CacheStats {
    pub fn cache_hit_ratio(&self) -> f64 {
        if self.prompt_tokens == 0 {
            return 0.0;
        }
        self.cache_hit_tokens as f64 / self.prompt_tokens as f64
    }
}

#[derive(Debug, Serialize)]
pub struct CacheSummary {
    pub total_requests: usize,
    pub total_prompt_tokens: usize,
    pub cache_hit_ratio: f64,
    pub current_context_tokens: usize,
    pub max_context_tokens: usize,
    pub last_prefill_ms: f64,
}

#[derive(Debug)]
pub struct CacheManager {
    pub max_context_tokens: usize,
    pub system_prompt_tokens: usize,
    pub truncation_threshold: f64,
    history_tokens: usize,
    total_requests: usize,
    total_cache_hits: usize,
    total_prompt_tokens: usize,
    last_stats: Option<CacheStats>,
}

impl CacheManager {
    pub fn new(max_context: usize, truncation_threshold: f64) -> Self {
        Self {
            max_context_tokens: max_context,
            system_prompt_tokens: 0,
            truncation_threshold,
            history_tokens: 0,
            total_requests: 0,
            total_cache_hits: 0,
            total_prompt_tokens: 0,
            last_stats: None,
        }
    }

    pub fn record(&mut self, stats: CacheStats) {
        self.total_requests += 1;
        self.total_cache_hits += stats.cache_hit_tokens;
        self.total_prompt_tokens += stats.prompt_tokens;
        debug!(
            "Cache: prefill={} tok ({:.0}ms), gen={} tok, hit={:.0}%",
            stats.prompt_tokens, stats.prefill_ms, stats.generated_tokens,
            stats.cache_hit_ratio() * 100.0,
        );
        self.last_stats = Some(stats);
    }

    pub fn update_history_tokens(&mut self, count: usize) {
        self.history_tokens = count;
    }

    pub fn total_tokens(&self) -> usize {
        self.system_prompt_tokens + self.history_tokens
    }

    pub fn remaining_tokens(&self) -> usize {
        self.max_context_tokens.saturating_sub(self.total_tokens())
    }

    pub fn needs_truncation(&self) -> bool {
        self.total_tokens() as f64 > self.max_context_tokens as f64 * self.truncation_threshold
    }

    pub fn truncation_target(&self) -> usize {
        let target = (self.max_context_tokens as f64 * 0.6) as usize;
        target.saturating_sub(self.system_prompt_tokens)
    }

    pub fn overall_cache_hit_ratio(&self) -> f64 {
        if self.total_prompt_tokens == 0 {
            return 0.0;
        }
        self.total_cache_hits as f64 / self.total_prompt_tokens as f64
    }

    pub fn summary(&self) -> CacheSummary {
        CacheSummary {
            total_requests: self.total_requests,
            total_prompt_tokens: self.total_prompt_tokens,
            cache_hit_ratio: self.overall_cache_hit_ratio(),
            current_context_tokens: self.total_tokens(),
            max_context_tokens: self.max_context_tokens,
            last_prefill_ms: self.last_stats.as_ref().map(|s| s.prefill_ms).unwrap_or(0.0),
        }
    }

    pub fn log_summary(&self) {
        let s = self.summary();
        info!(
            "Cache: {} requests, {:.0}% hit, {}/{} tokens, last prefill {:.0}ms",
            s.total_requests, s.cache_hit_ratio * 100.0,
            s.current_context_tokens, s.max_context_tokens, s.last_prefill_ms,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basics() {
        let mut cm = CacheManager::new(4096, 0.8);
        cm.system_prompt_tokens = 100;
        cm.update_history_tokens(200);
        assert_eq!(cm.total_tokens(), 300);
        assert_eq!(cm.remaining_tokens(), 3796);
        assert!(!cm.needs_truncation());
    }

    #[test]
    fn test_truncation_needed() {
        let mut cm = CacheManager::new(1000, 0.8);
        cm.system_prompt_tokens = 100;
        cm.update_history_tokens(750);
        assert!(cm.needs_truncation()); // 850 > 800 (80%)
    }

    #[test]
    fn test_record_stats() {
        let mut cm = CacheManager::new(4096, 0.8);
        cm.record(CacheStats {
            prompt_tokens: 100,
            generated_tokens: 20,
            prefill_ms: 15.0,
            generation_ms: 50.0,
            cache_hit_tokens: 80,
        });
        assert_eq!(cm.total_requests, 1);
        assert_eq!(cm.overall_cache_hit_ratio(), 0.8);
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test cache::tests -- --nocapture
```
Expected: 3 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/cache.rs
git commit -m "feat: add CacheManager with prefill tracking and truncation"
```

---

### Task 4: Tool executor — subprocess with arg substitution

**Files:**
- Create: `src/tool.rs`

- [ ] **Step 1: Write implementation with tests**

`src/tool.rs`:
```rust
use std::collections::HashMap;
use std::process::Stdio;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::{timeout, Duration};
use serde_json::Value;
use tracing::{info, warn};

use crate::config::ToolDef;

/// Substitute `{param}` placeholders in a command template.
pub fn substitute_args(template: &str, args: &HashMap<String, Value>) -> String {
    let mut result = template.to_string();
    for (key, val) in args {
        let val_str = match val {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        result = result.replace(&format!("{{{}}}", key), &val_str);
    }
    result
}

/// Execute a tool as a subprocess.
pub async fn execute_tool(tool: &ToolDef, args: &HashMap<String, Value>) -> String {
    let command = substitute_args(&tool.command, args);
    info!("Executing tool '{}': {}", tool.name, command);

    let has_stdin = tool.stdin.is_some();

    let mut cmd = Command::new("sh");
    cmd.arg("-c").arg(&command);
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    if has_stdin {
        cmd.stdin(Stdio::piped());
    }

    if let Some(ref workdir) = tool.workdir {
        cmd.current_dir(workdir);
    }

    for (k, v) in &tool.env {
        cmd.env(k, v);
    }

    let duration = Duration::from_secs(tool.timeout);

    let result = timeout(duration, async {
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => return format!("Error: failed to spawn command: {}", e),
        };

        // Write stdin if configured
        if let Some(ref stdin_template) = tool.stdin {
            let stdin_data = substitute_args(stdin_template, args);
            if let Some(mut stdin) = child.stdin.take() {
                let _ = stdin.write_all(stdin_data.as_bytes()).await;
                drop(stdin);
            }
        }

        match child.wait_with_output().await {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                if output.status.success() {
                    stdout.trim().to_string()
                } else {
                    format!("{}\n{}", stdout.trim(), stderr.trim()).trim().to_string()
                }
            }
            Err(e) => format!("Error: {}", e),
        }
    })
    .await;

    match result {
        Ok(output) => {
            info!("Tool '{}' result: {}", tool.name, &output[..output.len().min(100)]);
            output
        }
        Err(_) => {
            warn!("Tool '{}' timed out after {}s", tool.name, tool.timeout);
            format!("Error: Tool '{}' timed out after {}s", tool.name, tool.timeout)
        }
    }
}

/// Build compact tool schema string for the system prompt.
pub fn format_tool_schema(tool: &ToolDef) -> String {
    let mut args = Vec::new();
    let mut params: Vec<_> = tool.parameters.iter().collect();
    params.sort_by_key(|(k, _)| k.clone());

    for (name, param) in params {
        let mut arg = format!("{}:{}", name, param.param_type);
        if !param.required {
            if let Some(ref default) = param.default {
                arg.push_str(&format!("={}", default));
            } else {
                arg.push('?');
            }
        }
        args.push(arg);
    }

    format!("- {}({}) — {}", tool.name, args.join(", "), tool.description)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ParamDef;

    fn make_tool(name: &str, cmd: &str, params: Vec<(&str, &str, bool)>) -> ToolDef {
        let mut parameters = HashMap::new();
        for (pname, ptype, required) in params {
            parameters.insert(
                pname.to_string(),
                ParamDef {
                    param_type: ptype.to_string(),
                    required,
                    default: None,
                },
            );
        }
        ToolDef {
            name: name.to_string(),
            description: format!("Test tool {}", name),
            command: cmd.to_string(),
            stdin: None,
            timeout: 5,
            workdir: None,
            env: HashMap::new(),
            parameters,
        }
    }

    #[test]
    fn test_substitute_args() {
        let mut args = HashMap::new();
        args.insert("path".to_string(), Value::String("/tmp/test.txt".into()));
        assert_eq!(substitute_args("cat {path}", &args), "cat /tmp/test.txt");
    }

    #[test]
    fn test_substitute_multiple_args() {
        let mut args = HashMap::new();
        args.insert("a".to_string(), Value::Number(10.into()));
        args.insert("b".to_string(), Value::Number(20.into()));
        assert_eq!(substitute_args("echo {a} {b}", &args), "echo 10 20");
    }

    #[test]
    fn test_format_schema() {
        let tool = make_tool("read_file", "cat {path}", vec![("path", "string", true)]);
        let schema = format_tool_schema(&tool);
        assert!(schema.contains("read_file(path:string)"));
        assert!(schema.contains("Test tool read_file"));
    }

    #[tokio::test]
    async fn test_execute_echo() {
        let tool = make_tool("echo_test", "echo hello world", vec![]);
        let result = execute_tool(&tool, &HashMap::new()).await;
        assert_eq!(result, "hello world");
    }

    #[tokio::test]
    async fn test_execute_with_args() {
        let tool = make_tool("cat_test", "echo {content}", vec![("content", "string", true)]);
        let mut args = HashMap::new();
        args.insert("content".to_string(), Value::String("test output".into()));
        let result = execute_tool(&tool, &args).await;
        assert_eq!(result, "test output");
    }

    #[tokio::test]
    async fn test_execute_timeout() {
        let mut tool = make_tool("slow", "sleep 10", vec![]);
        tool.timeout = 1;
        let result = execute_tool(&tool, &HashMap::new()).await;
        assert!(result.contains("timed out"));
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test tool::tests -- --nocapture
```
Expected: 6 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/tool.rs
git commit -m "feat: add tool executor — subprocess with arg substitution and timeout"
```

---

### Task 5: Message types

**Files:**
- Create: `src/message.rs`

- [ ] **Step 1: Write types**

`src/message.rs`:
```rust
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: &str) -> Self {
        Self { role: "system".into(), content: content.into() }
    }
    pub fn user(content: &str) -> Self {
        Self { role: "user".into(), content: content.into() }
    }
    pub fn assistant(content: &str) -> Self {
        Self { role: "assistant".into(), content: content.into() }
    }
}

/// Events streamed back to transports.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum OutputEvent {
    #[serde(rename = "token")]
    Token { content: String, session: String },
    #[serde(rename = "tool_call")]
    ToolCall { tool: String, arguments: Value, session: String },
    #[serde(rename = "tool_result")]
    ToolResult { tool: String, result: String, session: String },
    #[serde(rename = "done")]
    Done { content: String, session: String },
    #[serde(rename = "error")]
    Error { content: String, session: String },
}

/// Incoming request from a transport.
#[derive(Debug, Deserialize)]
pub struct IncomingRequest {
    pub message: String,
    #[serde(default = "default_session")]
    pub session: String,
}

fn default_session() -> String {
    "default".to_string()
}
```

- [ ] **Step 2: Run build**

```bash
cargo build
```
Expected: compiles clean.

- [ ] **Step 3: Commit**

```bash
git add src/message.rs
git commit -m "feat: add Message, OutputEvent, IncomingRequest types"
```

---

### Task 6: Backend trait + mock

**Files:**
- Create: `src/backend/mod.rs`

- [ ] **Step 1: Write backend trait and mock**

`src/backend/mod.rs`:
```rust
use async_trait::async_trait;
use futures::stream::BoxStream;
use anyhow::Result;

use crate::cache::CacheStats;
use crate::message::Message;

#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "llama-server")]
pub mod llama_server;
#[cfg(feature = "openai")]
pub mod openai;

#[async_trait]
pub trait Backend: Send + Sync {
    /// Stream completion tokens.
    fn stream_completion(
        &self,
        prompt: &str,
        messages: &[Message],
        temperature: f64,
        max_tokens: usize,
        stop: &[String],
    ) -> BoxStream<'_, Result<String>>;

    /// Count tokens in text.
    async fn token_count(&self, text: &str) -> Result<usize>;

    /// Cache stats from the last completion.
    fn last_cache_stats(&self) -> Option<CacheStats>;
}

/// Create a backend from config. Feature-gated.
pub fn create_backend(config: &crate::config::BackendConfig) -> Result<Box<dyn Backend>> {
    match config.backend_type.as_str() {
        #[cfg(feature = "ollama")]
        "ollama" => Ok(Box::new(ollama::OllamaBackend::new(config))),
        #[cfg(feature = "llama-server")]
        "llama-server" => Ok(Box::new(llama_server::LlamaServerBackend::new(config))),
        #[cfg(feature = "openai")]
        "openai" => Ok(Box::new(openai::OpenAIBackend::new(config)?)),
        other => anyhow::bail!("Unknown or disabled backend type: '{}'. Check feature flags.", other),
    }
}

/// Mock backend for testing. Returns pre-configured responses.
#[cfg(test)]
pub mod mock {
    use super::*;
    use futures::stream;
    use std::sync::Mutex;

    pub struct MockBackend {
        responses: Mutex<Vec<String>>,
        last_stats: Mutex<Option<CacheStats>>,
    }

    impl MockBackend {
        pub fn new(responses: Vec<String>) -> Self {
            Self {
                responses: Mutex::new(responses),
                last_stats: Mutex::new(None),
            }
        }
    }

    #[async_trait]
    impl Backend for MockBackend {
        fn stream_completion(
            &self,
            _prompt: &str,
            _messages: &[Message],
            _temperature: f64,
            _max_tokens: usize,
            _stop: &[String],
        ) -> BoxStream<'_, Result<String>> {
            let mut responses = self.responses.lock().unwrap();
            let text = if responses.is_empty() {
                String::new()
            } else {
                responses.remove(0)
            };
            Box::pin(stream::once(async move { Ok(text) }))
        }

        async fn token_count(&self, text: &str) -> Result<usize> {
            Ok(text.len() / 4)
        }

        fn last_cache_stats(&self) -> Option<CacheStats> {
            self.last_stats.lock().unwrap().clone()
        }
    }
}
```

Create stub files for backends (will be filled in Plan 2):

`src/backend/ollama.rs`:
```rust
#![cfg(feature = "ollama")]

use async_trait::async_trait;
use futures::stream::BoxStream;
use anyhow::Result;
use crate::backend::Backend;
use crate::cache::CacheStats;
use crate::config::BackendConfig;
use crate::message::Message;

pub struct OllamaBackend;

impl OllamaBackend {
    pub fn new(_config: &BackendConfig) -> Self { Self }
}

#[async_trait]
impl Backend for OllamaBackend {
    fn stream_completion(&self, _prompt: &str, _messages: &[Message], _temperature: f64, _max_tokens: usize, _stop: &[String]) -> BoxStream<'_, Result<String>> {
        Box::pin(futures::stream::once(async { anyhow::bail!("OllamaBackend not yet implemented (Plan 2)") }))
    }
    async fn token_count(&self, text: &str) -> Result<usize> { Ok(text.len() / 4) }
    fn last_cache_stats(&self) -> Option<CacheStats> { None }
}
```

`src/backend/llama_server.rs`:
```rust
#![cfg(feature = "llama-server")]

use async_trait::async_trait;
use futures::stream::BoxStream;
use anyhow::Result;
use crate::backend::Backend;
use crate::cache::CacheStats;
use crate::config::BackendConfig;
use crate::message::Message;

pub struct LlamaServerBackend;

impl LlamaServerBackend {
    pub fn new(_config: &BackendConfig) -> Self { Self }
}

#[async_trait]
impl Backend for LlamaServerBackend {
    fn stream_completion(&self, _prompt: &str, _messages: &[Message], _temperature: f64, _max_tokens: usize, _stop: &[String]) -> BoxStream<'_, Result<String>> {
        Box::pin(futures::stream::once(async { anyhow::bail!("LlamaServerBackend not yet implemented (Plan 2)") }))
    }
    async fn token_count(&self, text: &str) -> Result<usize> { Ok(text.len() / 4) }
    fn last_cache_stats(&self) -> Option<CacheStats> { None }
}
```

`src/backend/openai.rs`:
```rust
#![cfg(feature = "openai")]

use async_trait::async_trait;
use futures::stream::BoxStream;
use anyhow::Result;
use crate::backend::Backend;
use crate::cache::CacheStats;
use crate::config::BackendConfig;
use crate::message::Message;

pub struct OpenAIBackend;

impl OpenAIBackend {
    pub fn new(_config: &BackendConfig) -> Result<Self> { Ok(Self) }
}

#[async_trait]
impl Backend for OpenAIBackend {
    fn stream_completion(&self, _prompt: &str, _messages: &[Message], _temperature: f64, _max_tokens: usize, _stop: &[String]) -> BoxStream<'_, Result<String>> {
        Box::pin(futures::stream::once(async { anyhow::bail!("OpenAIBackend not yet implemented (Plan 2)") }))
    }
    async fn token_count(&self, text: &str) -> Result<usize> { Ok(text.len() / 4) }
    fn last_cache_stats(&self) -> Option<CacheStats> { None }
}
```

- [ ] **Step 2: Run build**

```bash
cargo build
```
Expected: compiles clean.

- [ ] **Step 3: Commit**

```bash
git add src/backend/
git commit -m "feat: add Backend trait, mock backend, stub implementations"
```

---

### Task 7: Transport trait + stub

**Files:**
- Create: `src/transport/mod.rs`

- [ ] **Step 1: Write transport trait and stubs**

`src/transport/mod.rs`:
```rust
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

/// A request from a transport to the agent.
pub struct TransportRequest {
    pub message: String,
    pub session: String,
    pub response_tx: mpsc::Sender<OutputEvent>,
}

/// Handler function type that transports call with incoming requests.
pub type RequestHandler = Arc<dyn Fn(TransportRequest) + Send + Sync>;

#[async_trait]
pub trait Transport: Send + Sync {
    /// Start listening for incoming messages. Calls handler for each.
    async fn serve(&self, handler: RequestHandler) -> Result<()>;
    /// Name of this transport for logging.
    fn name(&self) -> &str;
}

/// Create transports from config. Feature-gated.
pub fn create_transports(config: &crate::config::Config) -> Result<Vec<Box<dyn Transport>>> {
    let mut transports: Vec<Box<dyn Transport>> = Vec::new();

    for name in &config.transports {
        match name.as_str() {
            #[cfg(feature = "cli-transport")]
            "cli" => {
                let cfg = config.transport.cli.clone().unwrap_or(crate::config::CliTransportConfig {
                    prompt: "you> ".into(),
                });
                transports.push(Box::new(cli::CliTransport::new(cfg)));
            }
            // WebSocket, MQTT, Unix, TCP — stubs for Plan 3
            other => {
                tracing::warn!("Transport '{}' not yet implemented or not compiled in", other);
            }
        }
    }

    Ok(transports)
}
```

`src/transport/cli.rs` (basic working implementation):
```rust
#![cfg(feature = "cli-transport")]

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio::io::{self, AsyncBufReadExt, BufReader};
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
        let stdin = BufReader::new(io::stdin());
        let mut lines = stdin.lines();

        loop {
            eprint!("{}", self.prompt);
            let line = match lines.next_line().await? {
                Some(l) => l,
                None => break, // EOF
            };

            let trimmed = line.trim().to_string();
            if trimmed.is_empty() { continue; }
            if trimmed == "quit" || trimmed == "exit" {
                eprintln!("Bye!");
                break;
            }

            let (tx, mut rx) = mpsc::channel(64);
            let request = TransportRequest {
                message: trimmed,
                session: "cli".to_string(),
                response_tx: tx,
            };

            handler(request);

            // Print events as they arrive
            while let Some(event) = rx.recv().await {
                match event {
                    OutputEvent::Token { content, .. } => eprint!("{}", content),
                    OutputEvent::Done { content, .. } => {
                        eprintln!("\n{}", content);
                        break;
                    }
                    OutputEvent::ToolCall { tool, .. } => {
                        eprintln!("\n[calling {}...]", tool);
                    }
                    OutputEvent::ToolResult { tool, result, .. } => {
                        eprintln!("[{} → {}]", tool, &result[..result.len().min(80)]);
                    }
                    OutputEvent::Error { content, .. } => {
                        eprintln!("\nError: {}", content);
                        break;
                    }
                }
            }
        }

        Ok(())
    }
}
```

Create stubs for Plan 3:

`src/transport/websocket.rs`:
```rust
#![cfg(feature = "websocket")]
// Implemented in Plan 3
```

`src/transport/mqtt.rs`:
```rust
#![cfg(feature = "mqtt")]
// Implemented in Plan 3
```

`src/transport/socket.rs`:
```rust
#![cfg(any(feature = "unix-socket", feature = "tcp-socket"))]
// Implemented in Plan 3
```

- [ ] **Step 2: Run build**

```bash
cargo build
```
Expected: compiles clean.

- [ ] **Step 3: Commit**

```bash
git add src/transport/
git commit -m "feat: add Transport trait, CLI transport, stub implementations"
```

---

### Task 8: Agent loop — ReAct cycle with chat templates

**Files:**
- Create: `src/agent.rs`

- [ ] **Step 1: Write agent with tests**

`src/agent.rs`:
```rust
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use futures::StreamExt;
use tracing::{info, warn, debug};

use crate::backend::Backend;
use crate::cache::CacheManager;
use crate::config::{AgentConfig, CacheConfig, ToolDef};
use crate::message::Message;
use crate::repair;
use crate::tool;

const TOOL_CALL_FORMAT: &str = r#"When a task requires a tool, call it with ONLY a JSON object:
{"tool": "tool_name_here", "arguments": {"param_name": "value"}}

Use exact parameter names shown above. After getting tool results, respond with plain text.
If you can answer without tools, respond directly with plain text."#;

struct ChatTemplate {
    system: &'static str,
    user: &'static str,
    assistant: &'static str,
    assistant_start: &'static str,
}

const CHATML: ChatTemplate = ChatTemplate {
    system: "<|im_start|>system\n{content}<|im_end|>\n",
    user: "<|im_start|>user\n{content}<|im_end|>\n",
    assistant: "<|im_start|>assistant\n{content}<|im_end|>\n",
    assistant_start: "<|im_start|>assistant\n",
};

const LLAMA3: ChatTemplate = ChatTemplate {
    system: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
    user: "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
    assistant: "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
    assistant_start: "<|start_header_id|>assistant<|end_header_id|>\n\n",
};

const MISTRAL: ChatTemplate = ChatTemplate {
    system: "[INST] {content}\n",
    user: "[INST] {content} [/INST]",
    assistant: "{content}</s>",
    assistant_start: "",
};

fn get_template(name: &str) -> &'static ChatTemplate {
    match name {
        "llama3" => &LLAMA3,
        "mistral" => &MISTRAL,
        _ => &CHATML,
    }
}

pub struct Agent {
    backend: Arc<dyn Backend>,
    tools: Vec<ToolDef>,
    system_prompt: String,
    template: &'static ChatTemplate,
    max_iterations: usize,
    max_retries: usize,
    temperature: f64,
    pub cache: Mutex<CacheManager>,
}

impl Agent {
    pub fn new(
        backend: Arc<dyn Backend>,
        tools: Vec<ToolDef>,
        agent_config: &AgentConfig,
        cache_config: &CacheConfig,
    ) -> Self {
        let system_prompt = build_system_prompt(&agent_config.system_prompt, &tools);
        let template = get_template(&agent_config.template);
        let mut cache = CacheManager::new(cache_config.max_context, cache_config.truncation_threshold);
        cache.system_prompt_tokens = system_prompt.len() / 4;

        Self {
            backend,
            tools,
            system_prompt,
            template,
            max_iterations: agent_config.max_iterations,
            max_retries: agent_config.max_retries,
            temperature: agent_config.temperature,
            cache: Mutex::new(cache),
        }
    }

    pub async fn run(&self, message: &str) -> String {
        let mut history = vec![Message::user(message)];

        for iteration in 0..self.max_iterations {
            info!("Agent loop iteration {}", iteration + 1);

            // Truncate if needed
            {
                let cache = self.cache.lock().await;
                if cache.needs_truncation() && history.len() > 3 {
                    let target = cache.truncation_target();
                    let keep = (target / 50).max(3).min(history.len());
                    let trimmed = history.len() - keep - 1;
                    let first = history[0].clone();
                    let recent: Vec<_> = history[history.len() - keep..].to_vec();
                    history = vec![first, Message::user(&format!("[{} earlier messages omitted]", trimmed))];
                    history.extend(recent);
                    info!("Truncated {} messages", trimmed);
                }
            }

            let prompt = self.format_prompt(&history);
            let mut messages = vec![Message::system(&self.system_prompt)];
            messages.extend(history.clone());

            // Update cache estimate
            {
                let mut cache = self.cache.lock().await;
                cache.update_history_tokens(prompt.len() / 4);
            }

            let max_tokens = {
                let cache = self.cache.lock().await;
                cache.remaining_tokens().max(256)
            };

            // Stream completion
            let mut response_text = String::new();
            let mut stream = self.backend.stream_completion(
                &prompt, &messages, self.temperature, max_tokens, &[],
            );

            while let Some(result) = stream.next().await {
                match result {
                    Ok(token) => response_text.push_str(&token),
                    Err(e) => {
                        warn!("Backend error: {}", e);
                        return format!("Error: Backend failed: {}", e);
                    }
                }
            }

            debug!("Raw LLM output: {}", &response_text[..response_text.len().min(200)]);

            // Record cache stats
            if let Some(stats) = self.backend.last_cache_stats() {
                self.cache.lock().await.record(stats);
            }

            // Try to parse as tool call
            let tool_call = repair::repair_tool_call(&response_text, &self.tools);

            if tool_call.is_none() {
                if repair::looks_like_broken_tool_call(&response_text) && iteration < self.max_retries {
                    warn!("Broken tool call, retrying");
                    history.push(Message::assistant(&response_text));
                    history.push(Message::user(&format!(
                        "Your response was not valid. Respond with a valid tool call JSON or plain text.\n\n{}",
                        TOOL_CALL_FORMAT
                    )));
                    continue;
                }

                info!("Final response");
                self.cache.lock().await.log_summary();
                return response_text;
            }

            // Execute tool
            let tc = tool_call.unwrap();
            info!("Executing tool: {}({:?})", tc.name, tc.arguments);

            let result = tool::execute_tool(
                self.tools.iter().find(|t| t.name == tc.name).unwrap(),
                &tc.arguments,
            )
            .await;

            info!("Tool result: {}", &result[..result.len().min(100)]);

            history.push(Message::assistant(&response_text));
            history.push(Message::user(&format!("Tool '{}' returned:\n{}", tc.name, result)));
        }

        warn!("Max iterations reached");
        self.cache.lock().await.log_summary();
        format!("Error: Maximum iterations ({}) reached", self.max_iterations)
    }

    fn format_prompt(&self, history: &[Message]) -> String {
        let mut parts = vec![self.template.system.replace("{content}", &self.system_prompt)];

        for msg in history {
            let tmpl = match msg.role.as_str() {
                "user" => self.template.user,
                "assistant" => self.template.assistant,
                _ => continue,
            };
            parts.push(tmpl.replace("{content}", &msg.content));
        }

        parts.push(self.template.assistant_start.to_string());
        parts.join("")
    }
}

fn build_system_prompt(base: &str, tools: &[ToolDef]) -> String {
    let mut parts = vec![base.to_string()];

    if !tools.is_empty() {
        parts.push("\n\nTools:".to_string());
        for t in tools {
            parts.push(tool::format_tool_schema(t));
        }
        parts.push(format!("\n{}", TOOL_CALL_FORMAT));
    }

    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::config::ParamDef;

    fn make_tool(name: &str) -> ToolDef {
        let mut params = HashMap::new();
        params.insert("path".to_string(), ParamDef {
            param_type: "string".to_string(),
            required: true,
            default: None,
        });
        ToolDef {
            name: name.to_string(),
            description: "Read a file".to_string(),
            command: format!("echo 'contents of {{path}}'"),
            stdin: None,
            timeout: 5,
            workdir: None,
            env: HashMap::new(),
            parameters: params,
        }
    }

    fn make_agent(responses: Vec<String>, tools: Vec<ToolDef>) -> Agent {
        let backend = Arc::new(MockBackend::new(responses));
        let agent_config = AgentConfig {
            system_prompt: "You are helpful.".into(),
            template: "chatml".into(),
            max_tokens: 4096,
            max_iterations: 10,
            max_retries: 2,
            temperature: 0.1,
        };
        let cache_config = CacheConfig {
            max_context: 4096,
            truncation_threshold: 0.8,
        };
        Agent::new(backend, tools, &agent_config, &cache_config)
    }

    #[tokio::test]
    async fn test_simple_response() {
        let agent = make_agent(vec!["Hello there!".into()], vec![]);
        let result = agent.run("Hi").await;
        assert_eq!(result, "Hello there!");
    }

    #[tokio::test]
    async fn test_tool_call() {
        let agent = make_agent(
            vec![
                r#"{"tool": "read_file", "arguments": {"path": "/tmp/test"}}"#.into(),
                "The file says: contents of /tmp/test".into(),
            ],
            vec![make_tool("read_file")],
        );
        let result = agent.run("Read /tmp/test").await;
        assert!(result.contains("contents of /tmp/test"));
    }

    #[tokio::test]
    async fn test_system_prompt_has_tools() {
        let agent = make_agent(vec![], vec![make_tool("read_file")]);
        assert!(agent.system_prompt.contains("read_file"));
        assert!(agent.system_prompt.contains("path:string"));
    }

    #[tokio::test]
    async fn test_prompt_is_chatml() {
        let agent = make_agent(vec!["ok".into()], vec![]);
        let prompt = agent.format_prompt(&[Message::user("hello")]);
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test agent::tests -- --nocapture
```
Expected: 4 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/agent.rs
git commit -m "feat: add Agent with ReAct loop, chat templates, tool execution"
```

---

### Task 9: Wire main.rs — load config, create agent, run transport

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Wire everything together**

`src/main.rs`:
```rust
mod config;
mod repair;
mod cache;
mod tool;
mod message;
mod agent;
mod backend;
mod transport;

use clap::Parser;
use std::sync::Arc;
use anyhow::Result;

#[derive(Parser)]
#[command(name = "edgeloop", about = "Minimal agentic framework for local LLMs")]
struct Cli {
    /// Path to config file
    #[arg(short, long, default_value = "edgeloop.toml")]
    config: String,
}

#[tokio::main]
async fn main() -> Result<()> {
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

    // Run first transport (for now — Plan 3 will run multiple concurrently)
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
```

- [ ] **Step 2: Delete src/lib.rs** (we're using main.rs with modules)

```bash
rm src/lib.rs
```

- [ ] **Step 3: Build and test**

```bash
cargo build
cargo test
```
Expected: compiles, all tests pass (~30 tests across config, repair, cache, tool, agent).

- [ ] **Step 4: Test the binary**

```bash
cargo run -- --config edgeloop.toml
```
Expected: starts, loads config, loads tools, prints "Starting transport: cli", shows prompt. Type "quit" to exit. (Backend call will fail since Ollama stub isn't implemented yet — that's Plan 2.)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: wire main.rs — config load, agent create, transport serve"
```

---

### Task 10: Run full test suite, update docs

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Run all tests**

```bash
cargo test -- --nocapture
```
Expected: ~30 tests pass. No failures.

- [ ] **Step 2: Build release binary and check size**

```bash
cargo build --release
ls -lh target/release/edgeloop
```

- [ ] **Step 3: Update README.md and CLAUDE.md**

Update both to reflect Rust project (remove Python references, update build commands, update project structure).

- [ ] **Step 4: Commit and push**

```bash
git add -A
git commit -m "docs: update README and CLAUDE.md for Rust rewrite"
git push
```
