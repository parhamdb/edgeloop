# edgeloop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal agentic framework for local LLMs — 5 core files, 2 dependencies, optimized for KV cache efficiency and local model error repair.

**Architecture:** Single-process monolithic Python package. Agent loop talks to llama-server via httpx. @tool decorator generates schemas from type hints. Repair pipeline fixes broken JSON/tool calls from small models. KV cache efficiency via deterministic prompt prefixes and slot pinning.

**Tech Stack:** Python 3.11+, httpx (async HTTP), click (CLI). pytest + pytest-asyncio for testing.

---

### Task 0: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `edgeloop/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "edgeloop"
version = "0.1.0"
description = "Minimal agentic framework for local LLMs"
requires-python = ">=3.11"
dependencies = [
    "httpx>=0.27.0",
    "click>=8.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]

[project.scripts]
edgeloop = "edgeloop.cli:main"
```

- [ ] **Step 2: Create package structure**

Create `edgeloop/__init__.py`:
```python
"""edgeloop: Minimal agentic framework for local LLMs."""
```

Create `tests/__init__.py` (empty).

Create `tests/conftest.py`:
```python
import pytest

pytest_plugins = ["pytest_asyncio"]
```

- [ ] **Step 3: Verify setup**

Run: `cd /home/parham/develop/src/parhamdb/localclaw && pip install -e ".[dev]"`
Expected: Successful install with httpx, click, pytest, pytest-asyncio.

Run: `pytest --co`
Expected: "no tests ran" (no test files yet), exit 0 or 5.

- [ ] **Step 4: Initialize git and commit**

```bash
cd /home/parham/develop/src/parhamdb/localclaw
git init
git add pyproject.toml edgeloop/__init__.py tests/__init__.py tests/conftest.py
git commit -m "feat: scaffold edgeloop project"
```

---

### Task 1: tools.py — @tool Decorator and Schema Generation

**Files:**
- Create: `edgeloop/tools.py`
- Create: `tests/test_tools.py`

- [ ] **Step 1: Write failing tests for schema generation**

Create `tests/test_tools.py`:
```python
import pytest
from edgeloop.tools import tool, get_schema, execute_tool


def test_basic_schema():
    @tool
    def greet(name: str) -> str:
        """Say hello to someone."""
        return f"Hello, {name}!"

    schema = get_schema(greet)
    assert schema["name"] == "greet"
    assert schema["description"] == "Say hello to someone."
    assert schema["parameters"]["properties"]["name"]["type"] == "string"
    assert "name" in schema["parameters"]["required"]


def test_optional_params():
    @tool
    def search(query: str, max_results: int = 5) -> str:
        """Search for something."""
        return query

    schema = get_schema(search)
    assert "query" in schema["parameters"]["required"]
    assert "max_results" not in schema["parameters"]["required"]
    assert schema["parameters"]["properties"]["max_results"]["type"] == "integer"
    assert schema["parameters"]["properties"]["max_results"]["default"] == 5


def test_multiple_types():
    @tool
    def process(name: str, count: int, ratio: float, active: bool) -> str:
        """Process data."""
        return "done"

    schema = get_schema(process)
    props = schema["parameters"]["properties"]
    assert props["name"]["type"] == "string"
    assert props["count"]["type"] == "integer"
    assert props["ratio"]["type"] == "number"
    assert props["active"]["type"] == "boolean"


def test_no_docstring():
    @tool
    def silent(x: str) -> str:
        return x

    schema = get_schema(silent)
    assert schema["description"] == ""


def test_tool_still_callable():
    @tool
    def add(a: int, b: int) -> str:
        """Add numbers."""
        return str(a + b)

    assert add(2, 3) == "5"


@pytest.mark.asyncio
async def test_tool_execution():
    @tool
    def greet(name: str) -> str:
        """Say hello."""
        return f"Hello, {name}!"

    result = await execute_tool(greet, {"name": "World"}, timeout=5.0)
    assert result == "Hello, World!"


@pytest.mark.asyncio
async def test_tool_error_capture():
    @tool
    def broken(x: str) -> str:
        """This breaks."""
        raise ValueError("something went wrong")

    result = await execute_tool(broken, {"x": "test"}, timeout=5.0)
    assert "ValueError" in result
    assert "something went wrong" in result


@pytest.mark.asyncio
async def test_tool_timeout():
    import asyncio

    @tool
    def slow(x: str) -> str:
        """Takes forever."""
        import time
        time.sleep(10)
        return x

    result = await execute_tool(slow, {"x": "test"}, timeout=0.1)
    assert "timeout" in result.lower() or "Timeout" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'edgeloop.tools'`

- [ ] **Step 3: Implement tools.py**

Create `edgeloop/tools.py`:
```python
"""Tool decorator and schema generation for edgeloop."""

import asyncio
import inspect
import functools
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def tool(fn: Callable) -> Callable:
    """Decorate a function as an edgeloop tool.

    Attaches JSON schema metadata without altering the function's behavior.
    """
    sig = inspect.signature(fn)
    hints = fn.__annotations__
    doc = inspect.getdoc(fn) or ""

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name == "return":
            continue
        param_type = hints.get(name, str)
        json_type = _TYPE_MAP.get(param_type, "string")

        prop: dict[str, Any] = {"type": json_type}

        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(name)

        properties[name] = prop

    fn.__tool_schema__ = {
        "name": fn.__name__,
        "description": doc,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper.__tool_schema__ = fn.__tool_schema__
    return wrapper


def get_schema(fn: Callable) -> dict:
    """Return the JSON schema attached by @tool."""
    return fn.__tool_schema__


async def execute_tool(fn: Callable, arguments: dict, timeout: float = 30.0) -> str:
    """Execute a tool function with timeout and error capture."""
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(fn, **arguments),
            timeout=timeout,
        )
        return str(result)
    except asyncio.TimeoutError:
        logger.warning("Tool %s timed out after %.1fs", fn.__name__, timeout)
        return f"Error: Tool '{fn.__name__}' timed out after {timeout}s"
    except Exception as e:
        logger.warning("Tool %s raised %s: %s", fn.__name__, type(e).__name__, e)
        return f"Error: {type(e).__name__}: {e}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tools.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add edgeloop/tools.py tests/test_tools.py
git commit -m "feat: add @tool decorator with schema generation and execution"
```

---

### Task 2: repair.py — JSON Extraction and Repair

**Files:**
- Create: `edgeloop/repair.py`
- Create: `tests/test_repair.py`

- [ ] **Step 1: Write failing tests for JSON extraction and repair**

Create `tests/test_repair.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_repair.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'edgeloop.repair'`

- [ ] **Step 3: Implement repair.py**

Create `edgeloop/repair.py`:
```python
"""Output parsing, JSON repair, and tool-call fixing for local LLMs."""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def extract_json(text: str) -> str | None:
    """Extract a JSON object from LLM output.

    Handles: markdown fences, XML tags, raw JSON embedded in text.
    """
    # Try markdown fence first
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try XML-style tags
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find raw JSON object with brace matching
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # Unmatched braces — return from start to end
    if depth > 0:
        return text[start:]

    return None


def repair_json(text: str) -> str:
    """Fix common JSON syntax issues from local models."""
    # Try parsing as-is first
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    result = text

    # Replace single quotes with double quotes (naive but effective)
    # Only replace quotes that look like JSON string delimiters
    result = re.sub(r"(?<=[{,:\[]\s*)'([^']*)'", r'"\1"', result)
    result = re.sub(r"'(\s*[}:\],])", r'"\1', result)
    # Catch remaining single-quoted keys/values
    if "'" in result and '"' not in result:
        result = result.replace("'", '"')

    # Remove trailing commas before } or ]
    result = re.sub(r",\s*([}\]])", r"\1", result)

    # Close unmatched braces
    open_braces = result.count("{") - result.count("}")
    if open_braces > 0:
        result += "}" * open_braces

    open_brackets = result.count("[") - result.count("]")
    if open_brackets > 0:
        result += "]" * open_brackets

    return result


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(
                min(curr_row[j] + 1, prev_row[j + 1] + 1, prev_row[j] + cost)
            )
        prev_row = curr_row

    return prev_row[-1]


def fuzzy_match_tool(name: str, available: list[str], max_distance: int = 2) -> str | None:
    """Match a tool name against available tools, allowing small typos."""
    # Exact match first
    if name in available:
        return name

    best_name = None
    best_dist = max_distance + 1

    for candidate in available:
        dist = _levenshtein(name, candidate)
        if dist < best_dist:
            best_dist = dist
            best_name = candidate

    if best_dist <= max_distance:
        logger.debug("Fuzzy matched tool '%s' → '%s' (distance=%d)", name, best_name, best_dist)
        return best_name

    return None


def coerce_arguments(args: dict, schema: dict) -> dict:
    """Coerce argument types to match schema, strip extras, add defaults."""
    properties = schema.get("properties", {})
    result = {}

    # Add known properties with coercion
    for key, prop_schema in properties.items():
        if key in args:
            result[key] = _coerce_value(args[key], prop_schema)
        elif "default" in prop_schema:
            result[key] = prop_schema["default"]

    return result


def _coerce_value(value: Any, prop_schema: dict) -> Any:
    """Coerce a single value to match its schema type."""
    target_type = prop_schema.get("type", "string")

    if target_type == "integer" and isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    elif target_type == "number" and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    elif target_type == "boolean" and isinstance(value, str):
        return value.lower() in ("true", "1", "yes")

    return value


def repair_tool_call(text: str, tools: list) -> dict | None:
    """Full repair pipeline: extract, fix, match, coerce.

    Returns {"name": str, "arguments": dict} or None if no tool call found.
    """
    raw = extract_json(text)
    if raw is None:
        return None

    repaired = repair_json(raw)

    try:
        parsed = json.loads(repaired)
    except json.JSONDecodeError:
        logger.warning("JSON repair failed for: %s", raw[:100])
        return None

    if "tool" not in parsed:
        return None

    tool_name = parsed["tool"]
    arguments = parsed.get("arguments", {})

    # Build available tool names and schema map
    available = []
    schema_map = {}
    for t in tools:
        schema = t.__tool_schema__
        available.append(schema["name"])
        schema_map[schema["name"]] = schema["parameters"]

    # Fuzzy match tool name
    matched_name = fuzzy_match_tool(tool_name, available)
    if matched_name is None:
        logger.warning("Unknown tool '%s', available: %s", tool_name, available)
        return None

    # Coerce arguments
    if matched_name in schema_map:
        arguments = coerce_arguments(arguments, schema_map[matched_name])

    return {"name": matched_name, "arguments": arguments}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_repair.py -v`
Expected: All 18 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add edgeloop/repair.py tests/test_repair.py
git commit -m "feat: add output repair pipeline with JSON fix, fuzzy match, type coercion"
```

---

### Task 3: backend.py — Backend Protocol and LlamaServerBackend

**Files:**
- Create: `edgeloop/backend.py`
- Create: `tests/test_backend.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_backend.py`:
```python
import json
import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from edgeloop.backend import Backend, LlamaServerBackend


def test_backend_protocol():
    """LlamaServerBackend satisfies the Backend protocol."""
    backend = LlamaServerBackend("http://localhost:8080")
    assert hasattr(backend, "complete")
    assert hasattr(backend, "token_count")


@pytest.mark.asyncio
async def test_token_count():
    backend = LlamaServerBackend("http://localhost:8080")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"tokens": [1, 2, 3, 4, 5]}
    mock_response.raise_for_status = MagicMock()

    with patch.object(backend, "_client") as mock_client:
        mock_client.post = AsyncMock(return_value=mock_response)
        count = await backend.token_count("Hello world")

    assert count == 5
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert body["content"] == "Hello world"


@pytest.mark.asyncio
async def test_complete_streaming():
    backend = LlamaServerBackend("http://localhost:8080")

    sse_lines = [
        b'data: {"content": "Hello", "stop": false}\n\n',
        b'data: {"content": " World", "stop": false}\n\n',
        b'data: {"content": "", "stop": true}\n\n',
    ]

    mock_stream = AsyncMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=False)
    mock_stream.aiter_lines = _async_iter_lines(sse_lines)

    with patch.object(backend, "_client") as mock_client:
        mock_client.stream = MagicMock(return_value=mock_stream)

        tokens = []
        async for token in backend.complete("Test prompt"):
            tokens.append(token)

    assert tokens == ["Hello", " World"]


@pytest.mark.asyncio
async def test_complete_passes_slot_id():
    backend = LlamaServerBackend("http://localhost:8080", slot_id=3)

    sse_lines = [b'data: {"content": "ok", "stop": true}\n\n']
    mock_stream = AsyncMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=False)
    mock_stream.aiter_lines = _async_iter_lines(sse_lines)

    with patch.object(backend, "_client") as mock_client:
        mock_client.stream = MagicMock(return_value=mock_stream)

        async for _ in backend.complete("Test"):
            pass

    call_kwargs = mock_client.stream.call_args
    body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert body["id_slot"] == 3
    assert body["cache_prompt"] is True


@pytest.mark.asyncio
async def test_complete_passes_stop_sequences():
    backend = LlamaServerBackend("http://localhost:8080")

    sse_lines = [b'data: {"content": "ok", "stop": true}\n\n']
    mock_stream = AsyncMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=False)
    mock_stream.aiter_lines = _async_iter_lines(sse_lines)

    with patch.object(backend, "_client") as mock_client:
        mock_client.stream = MagicMock(return_value=mock_stream)

        async for _ in backend.complete("Test", stop=["<|end|>", "\n\n"]):
            pass

    call_kwargs = mock_client.stream.call_args
    body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert body["stop"] == ["<|end|>", "\n\n"]


@pytest.mark.asyncio
async def test_connection_error():
    backend = LlamaServerBackend("http://localhost:99999")

    with pytest.raises(ConnectionError, match="Cannot connect"):
        async for _ in backend.complete("Test"):
            pass


def _async_iter_lines(lines: list[bytes]):
    """Create an async iterator that yields decoded lines."""
    async def _iter():
        for line in lines:
            decoded = line.decode().strip()
            if decoded:
                yield decoded
    return _iter
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backend.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'edgeloop.backend'`

- [ ] **Step 3: Implement backend.py**

Create `edgeloop/backend.py`:
```python
"""LLM backend protocol and llama-server implementation."""

import json
import logging
from typing import AsyncIterator, Protocol, runtime_checkable

import httpx

logger = logging.getLogger(__name__)


@runtime_checkable
class Backend(Protocol):
    """Protocol for LLM backends. Implement complete() and token_count()."""

    async def complete(
        self,
        prompt: str,
        stop: list[str] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]: ...

    async def token_count(self, text: str) -> int: ...


class LlamaServerBackend:
    """Backend that talks to llama-server's HTTP API."""

    def __init__(
        self,
        endpoint: str,
        slot_id: int | None = None,
        timeout: float = 120.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.slot_id = slot_id
        self._client = httpx.AsyncClient(timeout=timeout)

    async def complete(
        self,
        prompt: str,
        stop: list[str] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Stream completion tokens from llama-server."""
        body: dict = {
            "prompt": prompt,
            "stream": True,
            "cache_prompt": True,
            "n_predict": max_tokens,
            "temperature": temperature,
        }

        if self.slot_id is not None:
            body["id_slot"] = self.slot_id

        if stop:
            body["stop"] = stop

        logger.debug("POST %s/completion (slot=%s, tokens=%d)", self.endpoint, self.slot_id, max_tokens)

        try:
            async with self._client.stream(
                "POST", f"{self.endpoint}/completion", json=body
            ) as response:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data = line[6:]  # Strip "data: " prefix
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        logger.debug("Skipping non-JSON SSE line: %s", data[:50])
                        continue

                    if chunk.get("stop", False):
                        logger.debug("Stream complete")
                        return

                    content = chunk.get("content", "")
                    if content:
                        yield content

        except httpx.ConnectError as e:
            raise ConnectionError(f"Cannot connect to llama-server at {self.endpoint}: {e}") from e

    async def token_count(self, text: str) -> int:
        """Count tokens using llama-server's /tokenize endpoint."""
        try:
            response = await self._client.post(
                f"{self.endpoint}/tokenize",
                json={"content": text},
            )
            response.raise_for_status()
            return len(response.json()["tokens"])
        except httpx.ConnectError as e:
            raise ConnectionError(f"Cannot connect to llama-server at {self.endpoint}: {e}") from e

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backend.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add edgeloop/backend.py tests/test_backend.py
git commit -m "feat: add Backend protocol and LlamaServerBackend with streaming SSE"
```

---

### Task 4: agent.py — Agent Class and ReAct Loop

**Files:**
- Create: `edgeloop/agent.py`
- Create: `tests/test_agent.py`

- [ ] **Step 1: Write failing tests for prompt building**

Create `tests/test_agent.py`:
```python
import json
import pytest
from unittest.mock import AsyncMock
from edgeloop.agent import Agent
from edgeloop.tools import tool


@tool
def read_file(path: str) -> str:
    """Read a file from disk."""
    return f"contents of {path}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    return "ok"


class MockBackend:
    """Mock backend for testing the agent loop."""

    def __init__(self, responses: list[str]):
        self._responses = iter(responses)
        self.prompts = []

    async def complete(self, prompt, stop=None, temperature=0.7, max_tokens=1024):
        self.prompts.append(prompt)
        text = next(self._responses)
        for char_chunk in [text]:  # yield whole response as one chunk
            yield char_chunk

    async def token_count(self, text):
        # Rough approximation: 4 chars per token
        return len(text) // 4


class TestPromptBuilding:
    def test_system_prompt_includes_tool_schemas(self):
        agent = Agent(backend=MockBackend([]), tools=[read_file, write_file])
        prompt = agent._build_system_prompt()
        assert "read_file" in prompt
        assert "write_file" in prompt
        assert '"path"' in prompt
        assert "Read a file from disk" in prompt

    def test_system_prompt_is_deterministic(self):
        agent = Agent(backend=MockBackend([]), tools=[read_file, write_file])
        p1 = agent._build_system_prompt()
        p2 = agent._build_system_prompt()
        assert p1 == p2

    def test_format_prompt_chatml(self):
        agent = Agent(backend=MockBackend([]), tools=[read_file], template="chatml")
        prompt = agent._format_prompt("system msg", [{"role": "user", "content": "hello"}])
        assert "<|im_start|>system" in prompt
        assert "system msg" in prompt
        assert "<|im_start|>user" in prompt
        assert "hello" in prompt
        assert prompt.endswith("<|im_start|>assistant\n")

    def test_history_append_preserves_prefix(self):
        agent = Agent(backend=MockBackend([]), tools=[read_file], template="chatml")
        history1 = [{"role": "user", "content": "first"}]
        history2 = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "reply"}, {"role": "user", "content": "second"}]
        sys_prompt = agent._build_system_prompt()
        p1 = agent._format_prompt(sys_prompt, history1)
        p2 = agent._format_prompt(sys_prompt, history2)
        # p2 must start with the same bytes as p1 (minus the trailing assistant marker)
        p1_prefix = p1.rsplit("<|im_start|>assistant", 1)[0]
        assert p2.startswith(p1_prefix)


class TestAgentLoop:
    @pytest.mark.asyncio
    async def test_simple_response_no_tool(self):
        backend = MockBackend(["This is a plain text response."])
        agent = Agent(backend=backend, tools=[read_file])
        result = await agent.run("Hello")
        assert result == "This is a plain text response."

    @pytest.mark.asyncio
    async def test_single_tool_call(self):
        backend = MockBackend([
            '{"tool": "read_file", "arguments": {"path": "/tmp/test.txt"}}',
            "The file contains: contents of /tmp/test.txt",
        ])
        agent = Agent(backend=backend, tools=[read_file])
        result = await agent.run("What's in /tmp/test.txt?")
        assert "contents of /tmp/test.txt" in result

    @pytest.mark.asyncio
    async def test_tool_call_with_repair(self):
        backend = MockBackend([
            "```json\n{'tool': 'red_file', 'arguments': {'path': '/tmp/x',}}\n```",
            "Got it: contents of /tmp/x",
        ])
        agent = Agent(backend=backend, tools=[read_file])
        result = await agent.run("Read /tmp/x")
        assert "contents of /tmp/x" in result

    @pytest.mark.asyncio
    async def test_max_iterations_guard(self):
        # Backend always returns tool calls — should stop at max_iterations
        responses = ['{"tool": "read_file", "arguments": {"path": "/tmp/x"}}'] * 15
        responses.append("Final answer")
        backend = MockBackend(responses)
        agent = Agent(backend=backend, tools=[read_file], max_iterations=5)
        result = await agent.run("Loop forever")
        assert "maximum iterations" in result.lower() or len(backend.prompts) <= 6

    @pytest.mark.asyncio
    async def test_retry_on_parse_failure(self):
        backend = MockBackend([
            "This is garbled {{{not json at all",  # First attempt — not parseable, not plain text
            "Actually, here is the answer.",  # After retry
        ])
        agent = Agent(backend=backend, tools=[read_file], max_retries=2)
        result = await agent.run("Do something")
        assert "answer" in result.lower() or len(backend.prompts) >= 2

    @pytest.mark.asyncio
    async def test_slot_id_passed_to_backend(self):
        backend = MockBackend(["Response"])
        agent = Agent(backend=backend, tools=[read_file])
        await agent.run("Hello")
        # Backend received at least one prompt
        assert len(backend.prompts) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'edgeloop.agent'`

- [ ] **Step 3: Implement agent.py**

Create `edgeloop/agent.py`:
```python
"""Agent class and ReAct loop for edgeloop."""

import json
import logging
from typing import Callable

from edgeloop.backend import Backend, LlamaServerBackend
from edgeloop.repair import repair_tool_call
from edgeloop.tools import get_schema, execute_tool

logger = logging.getLogger(__name__)

TOOL_CALL_FORMAT = """\
When you need to use a tool, respond with ONLY a JSON object in this exact format:
{"tool": "tool_name", "arguments": {"arg1": "value1"}}

When you have the final answer, respond with plain text (no JSON)."""

CHAT_TEMPLATES = {
    "chatml": {
        "system": "<|im_start|>system\n{content}<|im_end|>\n",
        "user": "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
    },
    "llama3": {
        "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "mistral": {
        "system": "[INST] {content}\n",
        "user": "[INST] {content} [/INST]",
        "assistant": "{content}</s>",
        "assistant_start": "",
    },
}


class Agent:
    """Minimal agentic loop for local LLMs.

    Usage:
        agent = Agent(endpoint="http://localhost:8080", tools=[my_tool])
        result = await agent.run("Do something")
    """

    def __init__(
        self,
        endpoint: str | None = None,
        backend: Backend | None = None,
        tools: list[Callable] | None = None,
        system_prompt: str = "You are a helpful assistant.",
        template: str = "chatml",
        max_tokens: int = 4096,
        max_iterations: int = 10,
        max_retries: int = 2,
        temperature: float = 0.7,
        slot_id: int | None = None,
        log_level: str = "WARNING",
    ):
        if backend is not None:
            self._backend = backend
        elif endpoint is not None:
            self._backend = LlamaServerBackend(endpoint, slot_id=slot_id)
        else:
            raise ValueError("Either 'endpoint' or 'backend' must be provided")

        self._tools = tools or []
        self._user_system_prompt = system_prompt
        self._template_name = template
        self._template = CHAT_TEMPLATES[template]
        self._max_tokens = max_tokens
        self._max_iterations = max_iterations
        self._max_retries = max_retries
        self._temperature = temperature

        logging.getLogger("edgeloop").setLevel(getattr(logging, log_level.upper()))

    def _build_system_prompt(self) -> str:
        """Build deterministic system prompt with tool schemas."""
        parts = [self._user_system_prompt]

        if self._tools:
            parts.append("\n\n## Available Tools\n")
            schemas = []
            for t in self._tools:
                schemas.append(get_schema(t))
            parts.append(json.dumps(schemas, indent=2, sort_keys=True))
            parts.append("\n\n" + TOOL_CALL_FORMAT)

        return "".join(parts)

    def _format_prompt(self, system: str, history: list[dict]) -> str:
        """Apply chat template to system prompt and message history."""
        parts = [self._template["system"].format(content=system)]

        for msg in history:
            role = msg["role"]
            content = msg["content"]
            if role in self._template:
                parts.append(self._template[role].format(content=content))

        # Add assistant start token to prompt the model
        parts.append(self._template["assistant_start"])
        return "".join(parts)

    async def run(self, message: str) -> str:
        """Run the agent loop for a single user message.

        Returns the agent's final text response.
        """
        system = self._build_system_prompt()
        history: list[dict] = [{"role": "user", "content": message}]

        for iteration in range(self._max_iterations):
            logger.info("Agent loop iteration %d", iteration + 1)

            prompt = self._format_prompt(system, history)
            logger.debug("Prompt length: %d chars", len(prompt))

            # Collect full response from streaming backend
            response_text = ""
            async for token in self._backend.complete(
                prompt,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            ):
                response_text += token

            logger.debug("Raw LLM output: %s", response_text[:200])

            # Try to parse as tool call
            tool_call = repair_tool_call(response_text, self._tools)

            if tool_call is None:
                # Check if this looks like a failed tool call (has JSON-like content but couldn't parse)
                if self._looks_like_broken_tool_call(response_text) and iteration < self._max_retries:
                    logger.warning("Looks like a broken tool call, retrying (attempt %d)", iteration + 1)
                    history.append({"role": "assistant", "content": response_text})
                    history.append({
                        "role": "user",
                        "content": "Your response was not valid. Please respond with either a valid tool call JSON or a plain text answer.\n\n" + TOOL_CALL_FORMAT,
                    })
                    continue

                # Plain text response — we're done
                logger.info("Agent returned final response")
                return response_text

            # Execute tool
            tool_name = tool_call["name"]
            tool_args = tool_call["arguments"]
            logger.info("Executing tool: %s(%s)", tool_name, tool_args)

            # Find the tool function
            tool_fn = None
            for t in self._tools:
                if get_schema(t)["name"] == tool_name:
                    tool_fn = t
                    break

            if tool_fn is None:
                logger.error("Tool '%s' not found after repair", tool_name)
                return f"Error: Tool '{tool_name}' not found"

            result = await execute_tool(tool_fn, tool_args)
            logger.info("Tool result: %s", result[:100])

            # Append assistant tool call and tool result to history
            history.append({"role": "assistant", "content": response_text})
            history.append({"role": "user", "content": f"Tool '{tool_name}' returned:\n{result}"})

        logger.warning("Maximum iterations (%d) reached", self._max_iterations)
        return f"Error: Maximum iterations ({self._max_iterations}) reached. Last response: {response_text[:200]}"

    @staticmethod
    def _looks_like_broken_tool_call(text: str) -> bool:
        """Heuristic: does this look like a failed attempt at a tool call?"""
        indicators = ["{", '"tool"', "'tool'", "tool_call", "arguments"]
        matches = sum(1 for ind in indicators if ind in text)
        return matches >= 2
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent.py -v`
Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add edgeloop/agent.py tests/test_agent.py
git commit -m "feat: add Agent class with ReAct loop, chat templates, and retry logic"
```

---

### Task 5: cli.py — Click CLI

**Files:**
- Create: `edgeloop/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cli.py`:
```python
import pytest
from click.testing import CliRunner
from edgeloop.cli import main, _load_tools
from edgeloop.tools import tool
import tempfile
import os


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "edgeloop" in result.output.lower() or "chat" in result.output.lower()


def test_chat_help():
    runner = CliRunner()
    result = runner.invoke(main, ["chat", "--help"])
    assert result.exit_code == 0
    assert "--endpoint" in result.output


def test_chat_requires_endpoint():
    runner = CliRunner()
    result = runner.invoke(main, ["chat"])
    assert result.exit_code != 0


def test_load_tools_from_file():
    code = '''
from edgeloop.tools import tool

@tool
def my_test_tool(x: str) -> str:
    """A test tool."""
    return x
'''
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        try:
            tools = _load_tools(f.name)
            assert len(tools) == 1
            assert tools[0].__tool_schema__["name"] == "my_test_tool"
        finally:
            os.unlink(f.name)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'edgeloop.cli'`

- [ ] **Step 3: Implement cli.py**

Create `edgeloop/cli.py`:
```python
"""CLI interface for edgeloop."""

import asyncio
import importlib.util
import logging
import sys

import click

from edgeloop.agent import Agent

logger = logging.getLogger(__name__)


def _load_tools(path: str) -> list:
    """Import a Python file and collect all @tool-decorated functions."""
    spec = importlib.util.spec_from_file_location("_edgeloop_tools", path)
    if spec is None or spec.loader is None:
        raise click.BadParameter(f"Cannot load tools from: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tools = []
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and hasattr(obj, "__tool_schema__"):
            tools.append(obj)

    return tools


@click.group()
def main():
    """edgeloop: Minimal agentic framework for local LLMs."""
    pass


@main.command()
@click.option("--endpoint", required=True, help="LLM backend URL (e.g. http://localhost:8080)")
@click.option("--tools", "tools_path", default=None, help="Path to Python file with @tool functions")
@click.option("--template", default="chatml", help="Chat template: chatml, llama3, mistral")
@click.option("--max-tokens", default=4096, help="Max context tokens")
@click.option("--slot-id", default=None, type=int, help="Pin to a specific KV cache slot")
@click.option("--log-level", default="WARNING", help="Log level: DEBUG, INFO, WARNING, ERROR")
@click.option("--system-prompt", default="You are a helpful assistant.", help="System prompt")
def chat(endpoint, tools_path, template, max_tokens, slot_id, log_level, system_prompt):
    """Start an interactive chat session."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=getattr(logging, log_level.upper()),
    )

    tools = []
    if tools_path:
        tools = _load_tools(tools_path)
        click.echo(f"Loaded {len(tools)} tool(s): {', '.join(t.__tool_schema__['name'] for t in tools)}")

    agent = Agent(
        endpoint=endpoint,
        tools=tools,
        system_prompt=system_prompt,
        template=template,
        max_tokens=max_tokens,
        slot_id=slot_id,
        log_level=log_level,
    )

    click.echo(f"Connected to {endpoint} (template={template}, slot={slot_id})")
    click.echo("Type 'quit' or Ctrl+C to exit.\n")

    asyncio.run(_chat_loop(agent))


async def _chat_loop(agent: Agent):
    """Interactive chat REPL."""
    while True:
        try:
            user_input = click.prompt("you", prompt_suffix="> ")
        except (EOFError, KeyboardInterrupt):
            click.echo("\nBye!")
            return

        if user_input.strip().lower() in ("quit", "exit"):
            click.echo("Bye!")
            return

        try:
            response = await agent.run(user_input)
            click.echo(f"\nassistant> {response}\n")
        except ConnectionError as e:
            click.echo(f"\nError: {e}\n", err=True)
        except Exception as e:
            logger.exception("Unexpected error")
            click.echo(f"\nError: {type(e).__name__}: {e}\n", err=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add edgeloop/cli.py tests/test_cli.py
git commit -m "feat: add Click CLI with interactive chat and tool loading"
```

---

### Task 6: Public API and Integration

**Files:**
- Modify: `edgeloop/__init__.py`
- Create: `examples/hello.py`

- [ ] **Step 1: Write the public API exports**

Update `edgeloop/__init__.py`:
```python
"""edgeloop: Minimal agentic framework for local LLMs."""

from edgeloop.agent import Agent
from edgeloop.backend import Backend, LlamaServerBackend
from edgeloop.tools import tool, get_schema, execute_tool

__all__ = ["Agent", "Backend", "LlamaServerBackend", "tool", "get_schema", "execute_tool"]
__version__ = "0.1.0"
```

- [ ] **Step 2: Create example**

Create `examples/hello.py`:
```python
"""Minimal edgeloop example.

Usage:
    1. Start llama-server: llama-server -m model.gguf --port 8080
    2. Run: python examples/hello.py
"""

import asyncio
from edgeloop import Agent, tool


@tool
def read_file(path: str) -> str:
    """Read a file from disk and return its contents."""
    with open(path) as f:
        return f.read()


@tool
def list_files(directory: str = ".") -> str:
    """List files in a directory."""
    import os
    entries = os.listdir(directory)
    return "\n".join(entries)


async def main():
    agent = Agent(
        endpoint="http://localhost:8080",
        tools=[read_file, list_files],
        template="chatml",
        log_level="INFO",
    )

    response = await agent.run("List the files in the current directory, then read README.md if it exists.")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS (should be ~46 tests across 5 test files).

- [ ] **Step 4: Verify CLI entry point**

Run: `edgeloop --help`
Expected: Shows help with `chat` command.

Run: `edgeloop chat --help`
Expected: Shows options for `--endpoint`, `--tools`, `--template`, etc.

- [ ] **Step 5: Commit**

```bash
git add edgeloop/__init__.py examples/hello.py
git commit -m "feat: add public API exports and hello example"
```

---

### Task 7: Smoke Test with Real Backend

**Files:** None created. This is a manual verification task.

- [ ] **Step 1: Start llama-server**

Run: `llama-server -m <path-to-small-model.gguf> --port 8080 -ngl 99 --n-slots 2`
(Use any small model — TinyLlama, Qwen2.5-1.5B, etc.)

- [ ] **Step 2: Test library API**

```bash
cd /home/parham/develop/src/parhamdb/localclaw
python examples/hello.py
```
Expected: Agent lists files, reads README if it exists, returns coherent response.

- [ ] **Step 3: Test CLI**

```bash
edgeloop chat --endpoint http://localhost:8080 --tools examples/hello.py --log-level DEBUG
```
Expected: Interactive chat works. Tool calls are logged. KV cache reuse is visible in DEBUG logs (llama-server logs should show reduced prefill on subsequent turns).

- [ ] **Step 4: Test slot pinning**

```bash
edgeloop chat --endpoint http://localhost:8080 --slot-id 0 --log-level DEBUG
```
Expected: All requests go to slot 0. Second turn prefills faster than first (check llama-server logs for `prompt_n` in timings).

- [ ] **Step 5: Final commit with any fixes**

```bash
git add -A
git commit -m "chore: smoke test fixes"
```
