"""Backend protocol — the contract all LLM backends must satisfy."""

from typing import AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    """Protocol for LLM backends.

    Any object with these two methods satisfies the protocol.
    No inheritance needed — just implement the methods.

    The 'messages' parameter enables KV cache reuse for backends
    that support structured chat (Ollama /api/chat). Backends that
    use raw prompts (llama-server) can ignore it.

    To implement a new backend:
        class MyBackend:
            async def complete(self, prompt, stop=None, temperature=0.7,
                             max_tokens=1024, messages=None):
                async for token in my_stream(...):
                    yield token

            async def token_count(self, text):
                return len(my_tokenizer.encode(text))
    """

    async def complete(
        self,
        prompt: str,
        stop: list[str] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        messages: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Stream completion tokens.

        Args:
            prompt: Raw text prompt (with chat template applied).
            stop: Stop sequences to end generation.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            messages: Optional structured messages for chat-based backends.
                      Format: [{"role": "system", "content": "..."}, ...]
                      Backends that support this should prefer it over prompt
                      for better KV cache reuse.
        """
        ...

    async def token_count(self, text: str) -> int:
        """Count tokens in text. Used for context budget tracking."""
        ...
