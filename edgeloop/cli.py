"""CLI interface for edgeloop."""

import asyncio
import importlib.util
import logging
import sys

import click

from edgeloop.agent import Agent

logger = logging.getLogger(__name__)


def _load_tools(path: str) -> list:
    """Import a Python file and collect all @tool-decorated functions."""
    spec = importlib.util.spec_from_file_location("_edgeloop_tools", path)
    if spec is None or spec.loader is None:
        raise click.BadParameter(f"Cannot load tools from: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tools = []
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and hasattr(obj, "__tool_schema__"):
            tools.append(obj)

    return tools


@click.group()
def main():
    """edgeloop: Minimal agentic framework for local LLMs."""
    pass


@main.command()
@click.option("--endpoint", required=True, help="LLM backend URL (e.g. http://localhost:8080)")
@click.option("--tools", "tools_path", default=None, help="Path to Python file with @tool functions")
@click.option("--template", default="chatml", help="Chat template: chatml, llama3, mistral")
@click.option("--max-tokens", default=4096, help="Max context tokens")
@click.option("--slot-id", default=None, type=int, help="Pin to a specific KV cache slot")
@click.option("--log-level", default="WARNING", help="Log level: DEBUG, INFO, WARNING, ERROR")
@click.option("--system-prompt", default="You are a helpful assistant.", help="System prompt")
def chat(endpoint, tools_path, template, max_tokens, slot_id, log_level, system_prompt):
    """Start an interactive chat session."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=getattr(logging, log_level.upper()),
    )

    tools = []
    if tools_path:
        tools = _load_tools(tools_path)
        click.echo(f"Loaded {len(tools)} tool(s): {', '.join(t.__tool_schema__['name'] for t in tools)}")

    agent = Agent(
        endpoint=endpoint,
        tools=tools,
        system_prompt=system_prompt,
        template=template,
        max_tokens=max_tokens,
        slot_id=slot_id,
        log_level=log_level,
    )

    click.echo(f"Connected to {endpoint} (template={template}, slot={slot_id})")
    click.echo("Type 'quit' or Ctrl+C to exit.\n")

    asyncio.run(_chat_loop(agent))


async def _chat_loop(agent: Agent):
    """Interactive chat REPL."""
    while True:
        try:
            user_input = click.prompt("you", prompt_suffix="> ")
        except (EOFError, KeyboardInterrupt):
            click.echo("\nBye!")
            return

        if user_input.strip().lower() in ("quit", "exit"):
            click.echo("Bye!")
            return

        try:
            response = await agent.run(user_input)
            click.echo(f"\nassistant> {response}\n")
        except ConnectionError as e:
            click.echo(f"\nError: {e}\n", err=True)
        except Exception as e:
            logger.exception("Unexpected error")
            click.echo(f"\nError: {type(e).__name__}: {e}\n", err=True)
