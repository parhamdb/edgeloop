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
    code = """
from edgeloop.tools import tool

@tool
def my_test_tool(x: str) -> str:
    \"\"\"A test tool.\"\"\"
    return x
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        try:
            tools = _load_tools(f.name)
            assert len(tools) == 1
            assert tools[0].__tool_schema__["name"] == "my_test_tool"
        finally:
            os.unlink(f.name)
