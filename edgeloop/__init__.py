"""edgeloop: Minimal agentic framework for local LLMs."""

from edgeloop.agent import Agent
from edgeloop.backend import Backend, LlamaServerBackend
from edgeloop.tools import tool, get_schema, execute_tool

__all__ = ["Agent", "Backend", "LlamaServerBackend", "tool", "get_schema", "execute_tool"]
__version__ = "0.1.0"
