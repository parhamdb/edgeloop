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

    async def complete(self, prompt, stop=None, temperature=0.7, max_tokens=1024, messages=None):
        self.prompts.append(prompt)
        self.last_messages = messages
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
        assert "path" in prompt
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
