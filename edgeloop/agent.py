"""Agent class and ReAct loop for edgeloop."""

import json
import logging
from typing import Callable

from edgeloop.backend import Backend, LlamaServerBackend, OllamaBackend
from edgeloop.repair import repair_tool_call
from edgeloop.tools import get_schema, execute_tool

logger = logging.getLogger(__name__)

TOOL_CALL_FORMAT = """\
IMPORTANT: You MUST use the available tools to complete tasks. Do NOT guess or compute answers yourself.

To call a tool, respond with ONLY a JSON object in this exact format:
{"tool": "tool_name", "arguments": {"arg1": "value1"}}

When you have the final answer (after using tools), respond with plain text (no JSON).
Always use tools when they are relevant. Never skip a tool call to guess the answer."""

CHAT_TEMPLATES = {
    "chatml": {
        "system": "<|im_start|>system\n{content}<|im_end|>\n",
        "user": "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
    },
    "llama3": {
        "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "mistral": {
        "system": "[INST] {content}\n",
        "user": "[INST] {content} [/INST]",
        "assistant": "{content}</s>",
        "assistant_start": "",
    },
}


class Agent:
    """Minimal agentic loop for local LLMs.

    Usage:
        agent = Agent(endpoint="http://localhost:8080", tools=[my_tool])
        result = await agent.run("Do something")
    """

    def __init__(
        self,
        endpoint: str | None = None,
        backend: Backend | None = None,
        model: str | None = None,
        tools: list[Callable] | None = None,
        system_prompt: str = "You are a helpful assistant.",
        template: str = "chatml",
        max_tokens: int = 4096,
        max_iterations: int = 10,
        max_retries: int = 2,
        temperature: float = 0.7,
        slot_id: int | None = None,
        log_level: str = "WARNING",
    ):
        if backend is not None:
            self._backend = backend
        elif model is not None:
            # Ollama backend: model name specified
            self._backend = OllamaBackend(
                model=model,
                endpoint=endpoint or "http://localhost:11434",
            )
        elif endpoint is not None:
            self._backend = LlamaServerBackend(endpoint, slot_id=slot_id)
        else:
            raise ValueError("Either 'endpoint', 'backend', or 'model' must be provided")

        self._tools = tools or []
        self._user_system_prompt = system_prompt
        self._template_name = template
        self._template = CHAT_TEMPLATES[template]
        self._max_tokens = max_tokens
        self._max_iterations = max_iterations
        self._max_retries = max_retries
        self._temperature = temperature

        logging.getLogger("edgeloop").setLevel(getattr(logging, log_level.upper()))

    def _build_system_prompt(self) -> str:
        """Build deterministic system prompt with tool schemas."""
        parts = [self._user_system_prompt]

        if self._tools:
            parts.append("\n\n## Available Tools\n")
            schemas = []
            for t in self._tools:
                schemas.append(get_schema(t))
            parts.append(json.dumps(schemas, indent=2, sort_keys=True))
            parts.append("\n\n" + TOOL_CALL_FORMAT)

        return "".join(parts)

    def _format_prompt(self, system: str, history: list[dict]) -> str:
        """Apply chat template to system prompt and message history."""
        parts = [self._template["system"].format(content=system)]

        for msg in history:
            role = msg["role"]
            content = msg["content"]
            if role in self._template:
                parts.append(self._template[role].format(content=content))

        # Add assistant start token to prompt the model
        parts.append(self._template["assistant_start"])
        return "".join(parts)

    async def run(self, message: str) -> str:
        """Run the agent loop for a single user message.

        Returns the agent's final text response.
        """
        system = self._build_system_prompt()
        history: list[dict] = [{"role": "user", "content": message}]

        for iteration in range(self._max_iterations):
            logger.info("Agent loop iteration %d", iteration + 1)

            prompt = self._format_prompt(system, history)
            logger.debug("Prompt length: %d chars", len(prompt))

            # Collect full response from streaming backend
            response_text = ""
            async for token in self._backend.complete(
                prompt,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            ):
                response_text += token

            logger.debug("Raw LLM output: %s", response_text[:200])

            # Try to parse as tool call
            tool_call = repair_tool_call(response_text, self._tools)

            if tool_call is None:
                # Check if this looks like a failed tool call (has JSON-like content but couldn't parse)
                if self._looks_like_broken_tool_call(response_text) and iteration < self._max_retries:
                    logger.warning("Looks like a broken tool call, retrying (attempt %d)", iteration + 1)
                    history.append({"role": "assistant", "content": response_text})
                    history.append({
                        "role": "user",
                        "content": "Your response was not valid. Please respond with either a valid tool call JSON or a plain text answer.\n\n" + TOOL_CALL_FORMAT,
                    })
                    continue

                # Plain text response — we're done
                logger.info("Agent returned final response")
                return response_text

            # Execute tool
            tool_name = tool_call["name"]
            tool_args = tool_call["arguments"]
            logger.info("Executing tool: %s(%s)", tool_name, tool_args)

            # Find the tool function
            tool_fn = None
            for t in self._tools:
                if get_schema(t)["name"] == tool_name:
                    tool_fn = t
                    break

            if tool_fn is None:
                logger.error("Tool '%s' not found after repair", tool_name)
                return f"Error: Tool '{tool_name}' not found"

            result = await execute_tool(tool_fn, tool_args)
            logger.info("Tool result: %s", result[:100])

            # Append assistant tool call and tool result to history
            history.append({"role": "assistant", "content": response_text})
            history.append({"role": "user", "content": f"Tool '{tool_name}' returned:\n{result}"})

        logger.warning("Maximum iterations (%d) reached", self._max_iterations)
        return f"Error: Maximum iterations ({self._max_iterations}) reached. Last response: {response_text[:200]}"

    @staticmethod
    def _looks_like_broken_tool_call(text: str) -> bool:
        """Heuristic: does this look like a failed attempt at a tool call?"""
        indicators = ['"tool"', "'tool'", "tool_call", "arguments"]
        strong_match = any(ind in text for ind in indicators)
        has_brace = "{" in text
        # Strong match alone is enough; brace-only counts as suspicious too
        if strong_match:
            return True
        # Multiple braces with no successful parse = likely broken JSON
        if has_brace and text.count("{") >= 2:
            return True
        return False
