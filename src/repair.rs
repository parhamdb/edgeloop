use serde_json::Value;
use std::collections::HashMap;
use tracing::debug;

use crate::config::ToolDef;

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub name: String,
    pub arguments: HashMap<String, Value>,
}

/// Parse multiple tool calls from LLM output. Returns empty vec if no valid tool calls found.
pub fn repair_tool_calls(text: &str, tools: &[ToolDef]) -> Vec<ToolCall> {
    let raw = match extract_json(text) {
        Some(r) => r,
        None => return vec![],
    };
    let repaired = repair_json(&raw);

    let parsed: Value = match serde_json::from_str(&repaired) {
        Ok(v) => v,
        Err(_) => {
            debug!("JSON repair failed for: {}", &raw[..raw.len().min(100)]);
            return vec![];
        }
    };

    let available: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();

    match &parsed {
        Value::Array(items) => items
            .iter()
            .filter_map(|item| parse_single_tool_call(item, tools, &available))
            .collect(),
        Value::Object(_) => parse_single_tool_call(&parsed, tools, &available)
            .into_iter()
            .collect(),
        _ => vec![],
    }
}

/// Parse a single tool call from LLM output. Backward-compatible wrapper.
pub fn repair_tool_call(text: &str, tools: &[ToolDef]) -> Option<ToolCall> {
    repair_tool_calls(text, tools).into_iter().next()
}

fn parse_single_tool_call(value: &Value, tools: &[ToolDef], available: &[&str]) -> Option<ToolCall> {
    let tool_name = value.get("tool")?.as_str()?;
    let arguments: HashMap<String, Value> = value
        .get("arguments")
        .and_then(|v| v.as_object())
        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();

    let matched = fuzzy_match_tool(tool_name, available, 2)?;
    let tool_def = tools.iter().find(|t| t.name == matched)?;
    let coerced = coerce_arguments(&arguments, tool_def);

    Some(ToolCall {
        name: matched,
        arguments: coerced,
    })
}

pub fn extract_json(text: &str) -> Option<String> {
    let re_fence = regex::Regex::new(r"```(?:json)?\s*\n?([\s\S]*?)\n?```").unwrap();
    if let Some(caps) = re_fence.captures(text) {
        return Some(caps[1].trim().to_string());
    }

    let re_xml = regex::Regex::new(r"<tool_call>([\s\S]*?)</tool_call>").unwrap();
    if let Some(caps) = re_xml.captures(text) {
        return Some(caps[1].trim().to_string());
    }

    // Find first '{' or '[', whichever comes first
    let first_brace = text.find('{');
    let first_bracket = text.find('[');

    let (start, open_ch, close_ch) = match (first_brace, first_bracket) {
        (Some(b), Some(k)) if k < b => (k, '[', ']'),
        (Some(b), _) => (b, '{', '}'),
        (None, Some(k)) => (k, '[', ']'),
        (None, None) => return None,
    };

    let bytes = text.as_bytes();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;

    for i in start..bytes.len() {
        let c = bytes[i] as char;
        if escape { escape = false; continue; }
        if c == '\\' { escape = true; continue; }
        if c == '"' { in_string = !in_string; continue; }
        if in_string { continue; }
        if c == open_ch { depth += 1; }
        else if c == close_ch {
            depth -= 1;
            if depth == 0 { return Some(text[start..=i].to_string()); }
        }
    }

    if depth > 0 { return Some(text[start..].to_string()); }
    None
}

pub fn repair_json(text: &str) -> String {
    if serde_json::from_str::<Value>(text).is_ok() {
        return text.to_string();
    }

    let mut result = text.to_string();

    let re_sq = regex::Regex::new(r"'([^']*)'").unwrap();
    result = re_sq.replace_all(&result, "\"$1\"").to_string();

    let re_tc = regex::Regex::new(r",\s*([}\]])").unwrap();
    result = re_tc.replace_all(&result, "$1").to_string();

    let open_b = result.matches('{').count() as i32 - result.matches('}').count() as i32;
    if open_b > 0 { result.push_str(&"}".repeat(open_b as usize)); }
    let open_k = result.matches('[').count() as i32 - result.matches(']').count() as i32;
    if open_k > 0 { result.push_str(&"]".repeat(open_k as usize)); }

    result
}

pub fn levenshtein(a: &str, b: &str) -> usize {
    let (a_len, b_len) = (a.len(), b.len());
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

pub fn fuzzy_match_tool(name: &str, available: &[&str], max_distance: usize) -> Option<String> {
    if available.contains(&name) { return Some(name.to_string()); }

    let mut best: Option<&str> = None;
    let mut best_dist = max_distance + 1;

    for &candidate in available {
        let dist = levenshtein(name, candidate);
        if dist < best_dist { best_dist = dist; best = Some(candidate); }
    }

    if best_dist <= max_distance { best.map(|s| s.to_string()) } else { None }
}

pub fn coerce_arguments(args: &HashMap<String, Value>, tool: &ToolDef) -> HashMap<String, Value> {
    let mut result = HashMap::new();

    for (key, param) in &tool.parameters {
        if let Some(val) = args.get(key) {
            result.insert(key.clone(), coerce_value(val, &param.param_type));
        } else if let Some(default) = &param.default {
            result.insert(key.clone(), Value::String(default.clone()));
        }
    }

    let missing_required: Vec<&String> = tool.parameters.iter()
        .filter(|(k, p)| p.required && !result.contains_key(*k))
        .map(|(k, _)| k).collect();

    if !missing_required.is_empty() {
        let unmatched: Vec<&Value> = args.iter()
            .filter(|(k, _)| !tool.parameters.contains_key(k.as_str()))
            .map(|(_, v)| v).collect();

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
            } else { val.clone() }
        }
        "number" => {
            if let Value::String(s) = val {
                s.parse::<f64>().ok()
                    .and_then(|f| serde_json::Number::from_f64(f))
                    .map(Value::Number)
                    .unwrap_or_else(|| val.clone())
            } else { val.clone() }
        }
        "boolean" => {
            if let Value::String(s) = val {
                Value::Bool(matches!(s.to_lowercase().as_str(), "true" | "1" | "yes"))
            } else { val.clone() }
        }
        _ => val.clone(),
    }
}

pub fn looks_like_broken_tool_call(text: &str) -> bool {
    let indicators = ["\"tool\"", "'tool'", "tool_call", "arguments"];
    if indicators.iter().any(|ind| text.contains(ind)) { return true; }
    text.matches('{').count() >= 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ParamDef;

    fn make_tool(name: &str, params: Vec<(&str, &str, bool)>) -> ToolDef {
        let mut parameters = HashMap::new();
        for (pname, ptype, required) in params {
            parameters.insert(pname.to_string(), ParamDef {
                param_type: ptype.to_string(), required, default: None,
            });
        }
        ToolDef {
            name: name.to_string(), description: String::new(), command: String::new(),
            stdin: None, timeout: 10, workdir: None, env: HashMap::new(), parameters,
        }
    }

    #[test] fn test_extract_markdown_fence() {
        let text = "Sure!\n```json\n{\"tool\": \"read_file\"}\n```";
        assert_eq!(extract_json(text).unwrap(), "{\"tool\": \"read_file\"}");
    }
    #[test] fn test_extract_xml_tags() {
        assert_eq!(extract_json("<tool_call>{\"tool\": \"x\"}</tool_call>").unwrap(), "{\"tool\": \"x\"}");
    }
    #[test] fn test_extract_raw_json() {
        let r = extract_json("I will do that. {\"tool\": \"x\", \"arguments\": {}} Done.").unwrap();
        assert!(r.contains("\"tool\""));
    }
    #[test] fn test_extract_no_json() { assert!(extract_json("Just plain text.").is_none()); }
    #[test] fn test_repair_trailing_comma() {
        let r = repair_json("{\"a\": 1,}");
        serde_json::from_str::<Value>(&r).unwrap();
    }
    #[test] fn test_repair_single_quotes() {
        let r = repair_json("{'tool': 'read_file'}");
        let p: Value = serde_json::from_str(&r).unwrap();
        assert_eq!(p["tool"], "read_file");
    }
    #[test] fn test_repair_unmatched_brace() {
        let r = repair_json("{\"tool\": \"x\"");
        serde_json::from_str::<Value>(&r).unwrap();
    }
    #[test] fn test_repair_valid_unchanged() { assert_eq!(repair_json("{\"tool\": \"x\"}"), "{\"tool\": \"x\"}"); }
    #[test] fn test_levenshtein_identical() { assert_eq!(levenshtein("abc", "abc"), 0); }
    #[test] fn test_levenshtein_one_off() { assert_eq!(levenshtein("red_file", "read_file"), 1); }
    #[test] fn test_fuzzy_exact() {
        assert_eq!(fuzzy_match_tool("read_file", &["read_file", "write_file"], 2), Some("read_file".into()));
    }
    #[test] fn test_fuzzy_close() {
        assert_eq!(fuzzy_match_tool("red_file", &["read_file", "write_file"], 2), Some("read_file".into()));
    }
    #[test] fn test_fuzzy_too_far() {
        assert_eq!(fuzzy_match_tool("xyz_abc", &["read_file", "write_file"], 2), None);
    }
    #[test] fn test_coerce_string_to_int() {
        let tool = make_tool("t", vec![("count", "integer", true)]);
        let mut args = HashMap::new();
        args.insert("count".into(), Value::String("42".into()));
        assert_eq!(coerce_arguments(&args, &tool)["count"], 42);
    }
    #[test] fn test_full_pipeline() {
        let tools = vec![make_tool("read_file", vec![("path", "string", true)])];
        let text = "```json\n{'tool': 'red_file', 'arguments': {'path': '/tmp/x',}}\n```";
        let r = repair_tool_call(text, &tools).unwrap();
        assert_eq!(r.name, "read_file");
        assert_eq!(r.arguments["path"], "/tmp/x");
    }
    #[test] fn test_pipeline_hallucinated() {
        let tools = vec![make_tool("read_file", vec![("path", "string", true)])];
        assert!(repair_tool_call("{\"tool\": \"delete_everything\", \"arguments\": {}}", &tools).is_none());
    }

    #[test] fn test_extract_json_array() {
        let text = r#"I'll call both: [{"tool": "a", "arguments": {}}, {"tool": "b", "arguments": {}}]"#;
        let extracted = extract_json(text).unwrap();
        let parsed: Value = serde_json::from_str(&extracted).unwrap();
        assert!(parsed.is_array());
        assert_eq!(parsed.as_array().unwrap().len(), 2);
    }

    #[test] fn test_extract_json_array_in_fence() {
        let text = "```json\n[{\"tool\": \"a\"}, {\"tool\": \"b\"}]\n```";
        let extracted = extract_json(text).unwrap();
        let parsed: Value = serde_json::from_str(&extracted).unwrap();
        assert!(parsed.is_array());
    }

    #[test] fn test_extract_json_brace_before_bracket() {
        // When { comes before [, should extract the object not the array
        let text = r#"{"tool": "a"} and then [1, 2, 3]"#;
        let extracted = extract_json(text).unwrap();
        assert!(extracted.starts_with('{'));
    }

    #[test] fn test_repair_tool_calls_array() {
        let tools = vec![
            make_tool("speak", vec![("text", "string", true)]),
            make_tool("get_time", vec![]),
        ];
        let text = r#"[{"tool": "speak", "arguments": {"text": "hi"}}, {"tool": "get_time", "arguments": {}}]"#;
        let calls = repair_tool_calls(text, &tools);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "speak");
        assert_eq!(calls[1].name, "get_time");
    }

    #[test] fn test_repair_tool_calls_single_still_works() {
        let tools = vec![make_tool("read_file", vec![("path", "string", true)])];
        let text = r#"{"tool": "read_file", "arguments": {"path": "/tmp/x"}}"#;
        let calls = repair_tool_calls(text, &tools);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
    }

    #[test] fn test_repair_tool_calls_mixed_valid_invalid() {
        let tools = vec![make_tool("speak", vec![("text", "string", true)])];
        let text = r#"[{"tool": "speak", "arguments": {"text": "hi"}}, {"tool": "nonexistent_tool", "arguments": {}}]"#;
        let calls = repair_tool_calls(text, &tools);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "speak");
    }

    #[test] fn test_repair_tool_calls_malformed_array() {
        let tools = vec![
            make_tool("speak", vec![("text", "string", true)]),
            make_tool("get_time", vec![]),
        ];
        // Trailing comma + single quotes
        let text = r#"[{'tool': 'speak', 'arguments': {'text': 'hi'}}, {'tool': 'get_time', 'arguments': {}},]"#;
        let calls = repair_tool_calls(text, &tools);
        assert_eq!(calls.len(), 2);
    }

    #[test] fn test_repair_tool_calls_empty_text() {
        let tools = vec![make_tool("read_file", vec![("path", "string", true)])];
        assert!(repair_tool_calls("Just plain text, no JSON.", &tools).is_empty());
    }
}
