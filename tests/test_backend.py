import json
import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock, PropertyMock
from edgeloop.backend import Backend, LlamaServerBackend


def test_backend_protocol():
    """LlamaServerBackend satisfies the Backend protocol."""
    backend = LlamaServerBackend("http://localhost:8080")
    assert hasattr(backend, "complete")
    assert hasattr(backend, "token_count")


def _mock_backend_client(backend):
    """Patch the connection's client on a backend for testing."""
    mock_client = MagicMock()
    backend._conn._client = mock_client
    return mock_client


@pytest.mark.asyncio
async def test_token_count():
    backend = LlamaServerBackend("http://localhost:8080")
    mock_client = _mock_backend_client(backend)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"tokens": [1, 2, 3, 4, 5]}
    mock_response.raise_for_status = MagicMock()
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
    mock_client = _mock_backend_client(backend)

    sse_lines = [
        b'data: {"content": "Hello", "stop": false}\n\n',
        b'data: {"content": " World", "stop": false}\n\n',
        b'data: {"content": "", "stop": true}\n\n',
    ]

    mock_stream = AsyncMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=False)
    mock_stream.aiter_lines = _async_iter_lines(sse_lines)
    mock_client.stream = MagicMock(return_value=mock_stream)

    tokens = []
    async for token in backend.complete("Test prompt"):
        tokens.append(token)

    assert tokens == ["Hello", " World"]


@pytest.mark.asyncio
async def test_complete_passes_slot_id():
    backend = LlamaServerBackend("http://localhost:8080", slot_id=3)
    mock_client = _mock_backend_client(backend)

    sse_lines = [b'data: {"content": "ok", "stop": true}\n\n']
    mock_stream = AsyncMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=False)
    mock_stream.aiter_lines = _async_iter_lines(sse_lines)
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
    mock_client = _mock_backend_client(backend)

    sse_lines = [b'data: {"content": "ok", "stop": true}\n\n']
    mock_stream = AsyncMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=False)
    mock_stream.aiter_lines = _async_iter_lines(sse_lines)
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
