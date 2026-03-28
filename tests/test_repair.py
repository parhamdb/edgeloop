import pytest
from edgeloop.repair import (
    extract_json,
    repair_json,
    fuzzy_match_tool,
    coerce_arguments,
    repair_tool_call,
)


class TestExtractJson:
    def test_markdown_fence(self):
        text = 'Sure!\n```json\n{"tool": "read_file", "arguments": {"path": "/tmp/x"}}\n```'
        assert extract_json(text) == '{"tool": "read_file", "arguments": {"path": "/tmp/x"}}'

    def test_xml_tags(self):
        text = '<tool_call>{"tool": "read_file", "arguments": {"path": "/tmp/x"}}</tool_call>'
        assert extract_json(text) == '{"tool": "read_file", "arguments": {"path": "/tmp/x"}}'

    def test_raw_json_in_text(self):
        text = 'I will read the file. {"tool": "read_file", "arguments": {"path": "/tmp/x"}} Done.'
        result = extract_json(text)
        assert '"tool"' in result
        assert '"read_file"' in result

    def test_no_json(self):
        assert extract_json("Just a plain response with no JSON.") is None

    def test_nested_braces(self):
        text = '{"tool": "x", "arguments": {"data": {"nested": true}}}'
        result = extract_json(text)
        assert '"nested"' in result


class TestRepairJson:
    def test_trailing_comma(self):
        assert repair_json('{"a": 1,}') == '{"a": 1}'

    def test_single_quotes(self):
        result = repair_json("{'tool': 'read_file'}")
        parsed = __import__("json").loads(result)
        assert parsed["tool"] == "read_file"

    def test_unmatched_brace(self):
        result = repair_json('{"tool": "read_file"')
        parsed = __import__("json").loads(result)
        assert parsed["tool"] == "read_file"

    def test_valid_json_unchanged(self):
        original = '{"tool": "x", "arguments": {}}'
        assert repair_json(original) == original


class TestFuzzyMatch:
    def test_exact_match(self):
        assert fuzzy_match_tool("read_file", ["read_file", "write_file"]) == "read_file"

    def test_close_match(self):
        assert fuzzy_match_tool("red_file", ["read_file", "write_file"]) == "read_file"

    def test_too_distant(self):
        assert fuzzy_match_tool("xyz_abc", ["read_file", "write_file"]) is None

    def test_single_char_typo(self):
        assert fuzzy_match_tool("reed_file", ["read_file", "write_file"]) == "read_file"


class TestCoerceArguments:
    def test_string_to_int(self):
        schema = {"properties": {"count": {"type": "integer"}}, "required": ["count"]}
        result = coerce_arguments({"count": "42"}, schema)
        assert result["count"] == 42

    def test_string_to_float(self):
        schema = {"properties": {"ratio": {"type": "number"}}, "required": ["ratio"]}
        result = coerce_arguments({"ratio": "3.14"}, schema)
        assert result["ratio"] == 3.14

    def test_string_to_bool(self):
        schema = {"properties": {"active": {"type": "boolean"}}, "required": ["active"]}
        result = coerce_arguments({"active": "true"}, schema)
        assert result["active"] is True

    def test_strip_extra_fields(self):
        schema = {"properties": {"a": {"type": "string"}}, "required": ["a"]}
        result = coerce_arguments({"a": "x", "extra": "y"}, schema)
        assert "extra" not in result

    def test_add_missing_defaults(self):
        schema = {
            "properties": {"a": {"type": "string"}, "b": {"type": "integer", "default": 5}},
            "required": ["a"],
        }
        result = coerce_arguments({"a": "x"}, schema)
        assert result["b"] == 5


class TestRepairToolCall:
    def test_full_pipeline_clean(self):
        tools = [_make_tool("read_file", {"path": {"type": "string"}})]
        text = '{"tool": "read_file", "arguments": {"path": "/tmp/x"}}'
        result = repair_tool_call(text, tools)
        assert result is not None
        assert result["name"] == "read_file"
        assert result["arguments"]["path"] == "/tmp/x"

    def test_full_pipeline_broken(self):
        tools = [_make_tool("read_file", {"path": {"type": "string"}})]
        text = "```json\n{'tool': 'red_file', 'arguments': {'path': '/tmp/x',}}\n```"
        result = repair_tool_call(text, tools)
        assert result is not None
        assert result["name"] == "read_file"
        assert result["arguments"]["path"] == "/tmp/x"

    def test_no_tool_call(self):
        tools = [_make_tool("read_file", {"path": {"type": "string"}})]
        result = repair_tool_call("I don't need any tools for this.", tools)
        assert result is None

    def test_hallucinated_tool_rejected(self):
        tools = [_make_tool("read_file", {"path": {"type": "string"}})]
        text = '{"tool": "delete_everything", "arguments": {}}'
        result = repair_tool_call(text, tools)
        assert result is None


def _make_tool(name, properties, required=None):
    """Helper to create a mock tool schema."""
    class FakeTool:
        pass
    ft = FakeTool()
    ft.__tool_schema__ = {
        "name": name,
        "description": "",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required or list(properties.keys()),
        },
    }
    return ft
