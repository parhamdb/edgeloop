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
