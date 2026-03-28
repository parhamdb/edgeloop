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
        return f"Error: Tool '{fn.__name__}' timeout after {timeout}s"
    except Exception as e:
        logger.warning("Tool %s raised %s: %s", fn.__name__, type(e).__name__, e)
        return f"Error: {type(e).__name__}: {e}"
