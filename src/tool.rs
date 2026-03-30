use std::collections::HashMap;
use std::process::Stdio;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::{timeout, Duration};
use serde_json::Value;
use tracing::{info, warn};

use crate::config::ToolDef;

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

pub async fn execute_tool(tool: &ToolDef, args: &HashMap<String, Value>) -> String {
    let command = substitute_args(&tool.command, args);
    info!("Executing tool '{}': {}", tool.name, command);

    let has_stdin = tool.stdin.is_some();
    let mut cmd = Command::new("sh");
    cmd.arg("-c").arg(&command);
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    if has_stdin { cmd.stdin(Stdio::piped()); }
    if let Some(ref workdir) = tool.workdir { cmd.current_dir(workdir); }
    for (k, v) in &tool.env { cmd.env(k, v); }

    let duration = Duration::from_secs(tool.timeout);
    let result = timeout(duration, async {
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => return format!("Error: failed to spawn: {}", e),
        };
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
    }).await;

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

pub fn format_tool_schema(tool: &ToolDef) -> String {
    let mut args = Vec::new();
    let mut params: Vec<_> = tool.parameters.iter().collect();
    params.sort_by_key(|(k, _)| (*k).clone());
    for (name, param) in params {
        let mut arg = format!("{}:{}", name, param.param_type);
        if !param.required {
            if let Some(ref default) = param.default { arg.push_str(&format!("={}", default)); }
            else { arg.push('?'); }
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
            parameters.insert(pname.to_string(), ParamDef { param_type: ptype.to_string(), required, default: None });
        }
        ToolDef { name: name.to_string(), description: format!("Test {}", name), command: cmd.to_string(), stdin: None, timeout: 5, workdir: None, env: HashMap::new(), parameters }
    }

    #[test]
    fn test_substitute_args() {
        let mut args = HashMap::new();
        args.insert("path".to_string(), Value::String("/tmp/test.txt".into()));
        assert_eq!(substitute_args("cat {path}", &args), "cat /tmp/test.txt");
    }

    #[test]
    fn test_substitute_multiple() {
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
    }

    #[tokio::test]
    async fn test_execute_echo() {
        let tool = make_tool("echo_test", "echo hello world", vec![]);
        let result = execute_tool(&tool, &HashMap::new()).await;
        assert_eq!(result, "hello world");
    }

    #[tokio::test]
    async fn test_execute_with_args() {
        let tool = make_tool("t", "echo {content}", vec![("content", "string", true)]);
        let mut args = HashMap::new();
        args.insert("content".to_string(), Value::String("test output".into()));
        assert_eq!(execute_tool(&tool, &args).await, "test output");
    }

    #[tokio::test]
    async fn test_execute_timeout() {
        let mut tool = make_tool("slow", "sleep 10", vec![]);
        tool.timeout = 1;
        let result = execute_tool(&tool, &HashMap::new()).await;
        assert!(result.contains("timed out"));
    }
}
