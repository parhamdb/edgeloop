"""LLM backend implementations for edgeloop.

Each backend talks to a different local LLM server. To add a new backend:
1. Create a new file in this package (e.g., vllm.py)
2. Implement the Backend protocol (complete + token_count methods)
3. Export it from this __init__.py

The Backend protocol is intentionally minimal — two methods:
- complete(): Stream tokens from the LLM
- token_count(): Count tokens in text (for context budget tracking)
"""

from edgeloop.backends.protocol import Backend
from edgeloop.backends.ollama import OllamaBackend
from edgeloop.backends.llama_server import LlamaServerBackend

__all__ = ["Backend", "OllamaBackend", "LlamaServerBackend"]
