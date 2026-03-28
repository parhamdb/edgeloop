"""edgeloop: Minimal agentic framework for local LLMs."""

from edgeloop.agent import Agent
from edgeloop.backends import Backend, LlamaServerBackend, OllamaBackend
from edgeloop.cache import CacheManager, CacheStats
from edgeloop.connection import Connection
from edgeloop.tools import tool, get_schema, execute_tool

__all__ = [
    "Agent",
    "Backend", "LlamaServerBackend", "OllamaBackend",
    "CacheManager", "CacheStats",
    "Connection",
    "tool", "get_schema", "execute_tool",
]
__version__ = "0.1.0"
