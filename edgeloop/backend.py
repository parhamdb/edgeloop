"""Backward-compatible re-exports from edgeloop.backends package."""

from edgeloop.backends.protocol import Backend
from edgeloop.backends.ollama import OllamaBackend
from edgeloop.backends.llama_server import LlamaServerBackend

__all__ = ["Backend", "OllamaBackend", "LlamaServerBackend"]
