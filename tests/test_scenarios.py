"""Scenario tests for edgeloop — real-world usage patterns.

Tests diverse scenarios across multiple models and backends to find
edge cases, reliability issues, and performance characteristics.

Run: python tests/test_scenarios.py
"""

import asyncio
import os
import json
import tempfile
import time
from dataclasses import dataclass
from edgeloop import Agent, tool


# ═══════════════════════════════════════════════════════════════════
# Tools
# ═══════════════════════════════════════════════════════════════════

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression safely. Examples: '2+3', '10*5.5', '100/4'."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return f"Error: invalid characters in '{expression}'"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool
def read_file(path: str) -> str:
    """Read a file and return its contents."""
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates parent dirs if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return f"Wrote {len(content)} chars to {path}"


@tool
def list_dir(path: str = ".") -> str:
    """List files and folders in a directory."""
    try:
        entries = sorted(os.listdir(path))
        return "\n".join(entries[:30]) if entries else "(empty directory)"
    except Exception as e:
        return f"Error: {e}"


@tool
def search_text(path: str, query: str) -> str:
    """Search for a string in a file, return matching lines."""
    try:
        with open(path) as f:
            lines = f.readlines()
        matches = [f"L{i+1}: {l.rstrip()}" for i, l in enumerate(lines) if query.lower() in l.lower()]
        return "\n".join(matches[:10]) if matches else f"No matches for '{query}'"
    except Exception as e:
        return f"Error: {e}"


@tool
def get_json_field(path: str, field: str) -> str:
    """Read a JSON file and extract a specific field."""
    try:
        with open(path) as f:
            data = json.load(f)
        if field in data:
            return json.dumps(data[field])
        return f"Field '{field}' not found. Available: {', '.join(data.keys())}"
    except Exception as e:
        return f"Error: {e}"


@tool
def shell_command(command: str) -> str:
    """Run a shell command and return output. Only safe read-only commands."""
    import subprocess
    # Block dangerous commands
    blocked = ["rm", "del", "format", "mkfs", "dd", "kill", ">", ">>"]
    if any(b in command.lower() for b in blocked):
        return "Error: command blocked for safety"
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
        output = result.stdout + result.stderr
        return output[:1000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out"
    except Exception as e:
        return f"Error: {e}"


# ═══════════════════════════════════════════════════════════════════
# Backend configs
# ═══════════════════════════════════════════════════════════════════

BACKENDS = {
    "ollama-0.6b": lambda **kw: Agent(model="qwen3:0.6b", max_tokens=500, temperature=0.1, max_iterations=8, **kw),
    "ollama-1.7b": lambda **kw: Agent(model="qwen3:1.7b", max_tokens=500, temperature=0.1, max_iterations=8, **kw),
    "ollama-7b": lambda **kw: Agent(model="qwen2.5-coder:7b", max_tokens=500, temperature=0.1, max_iterations=8, **kw),
    "llama-1.5b": lambda **kw: Agent(endpoint="http://localhost:8082", max_tokens=500, temperature=0.1, max_iterations=8, template="chatml", **kw),
}


# ═══════════════════════════════════════════════════════════════════
# Scenario runner
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Result:
    backend: str
    scenario: str
    time_s: float
    passed: bool
    response: str
    note: str = ""


async def run_scenario(name, backend_name, make_agent, prompt, tools, check_fn):
    """Run a scenario and return the result."""
    agent = make_agent(tools=tools)
    start = time.time()
    try:
        response = await agent.run(prompt)
        elapsed = time.time() - start
        passed, note = check_fn(response)
        return Result(backend_name, name, elapsed, passed, response[:200], note)
    except Exception as e:
        elapsed = time.time() - start
        return Result(backend_name, name, elapsed, False, str(e)[:200], f"EXCEPTION: {type(e).__name__}")


# ═══════════════════════════════════════════════════════════════════
# Scenarios
# ═══════════════════════════════════════════════════════════════════

async def scenario_simple_math(backend_name, make_agent):
    """Can the model do basic arithmetic with calculator?"""
    return await run_scenario(
        "simple_math", backend_name, make_agent,
        "What is 847 + 253? Use the calculator tool.",
        [calculator],
        lambda r: ("1100" in r, ""),
    )


async def scenario_chained_math(backend_name, make_agent):
    """Can the model chain two calculator calls?"""
    return await run_scenario(
        "chained_math", backend_name, make_agent,
        "Calculate 25 * 4, then add 50 to the result. Use the calculator for each step.",
        [calculator],
        lambda r: ("150" in r, "25*4=100, +50=150"),
    )


async def scenario_file_create_and_read(backend_name, make_agent):
    """Can the model write a file then read it back?"""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.txt")
        result = await run_scenario(
            "file_roundtrip", backend_name, make_agent,
            f"Write the text 'edgeloop works!' to {filepath}, then read it back and tell me what it says.",
            [write_file, read_file],
            lambda r: ("edgeloop" in r.lower() or "works" in r.lower(), ""),
        )
        # Also check the file actually exists
        if os.path.exists(filepath):
            result.note += " [file created]"
        else:
            result.note += " [FILE NOT CREATED]"
            result.passed = False
        return result


async def scenario_read_json(backend_name, make_agent):
    """Can the model read and extract data from a JSON file?"""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "config.json")
        with open(filepath, "w") as f:
            json.dump({"name": "edgeloop", "version": "0.1.0", "author": "test"}, f)

        return await run_scenario(
            "read_json", backend_name, make_agent,
            f"Read the 'version' field from {filepath} using get_json_field.",
            [get_json_field],
            lambda r: ("0.1.0" in r, ""),
        )


async def scenario_search_in_file(backend_name, make_agent):
    """Can the model search for text in a file?"""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "log.txt")
        with open(filepath, "w") as f:
            f.write("2024-01-01 INFO: Server started\n")
            f.write("2024-01-01 ERROR: Connection refused to db\n")
            f.write("2024-01-01 INFO: Retry succeeded\n")
            f.write("2024-01-01 ERROR: Timeout on API call\n")

        return await run_scenario(
            "search_file", backend_name, make_agent,
            f"Search {filepath} for all lines containing 'ERROR' using search_text.",
            [search_text],
            lambda r: ("connection" in r.lower() or "timeout" in r.lower() or "error" in r.lower(), ""),
        )


async def scenario_list_and_read(backend_name, make_agent):
    """Can the model list a directory then read a specific file?"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "readme.txt"), "w") as f:
            f.write("This is the readme file for the project.")
        with open(os.path.join(tmpdir, "config.txt"), "w") as f:
            f.write("debug=true")
        with open(os.path.join(tmpdir, "data.csv"), "w") as f:
            f.write("a,b,c\n1,2,3\n")

        return await run_scenario(
            "list_and_read", backend_name, make_agent,
            f"List files in {tmpdir}, then read the readme.txt file.",
            [list_dir, read_file],
            lambda r: ("readme" in r.lower() or "project" in r.lower(), ""),
        )


async def scenario_error_recovery(backend_name, make_agent):
    """Can the model handle a tool error gracefully?"""
    return await run_scenario(
        "error_recovery", backend_name, make_agent,
        "Read the file /nonexistent/path/missing.txt and tell me what happened.",
        [read_file],
        lambda r: ("not found" in r.lower() or "error" in r.lower() or
                   "exist" in r.lower() or "does not" in r.lower() or
                   "cannot" in r.lower() or "no such" in r.lower(), ""),
    )


async def scenario_no_tool_needed(backend_name, make_agent):
    """Does the model answer directly when no tool is needed?"""
    return await run_scenario(
        "no_tool_needed", backend_name, make_agent,
        "What is the capital of France?",
        [calculator, read_file],  # tools available but not needed
        lambda r: ("paris" in r.lower(), ""),
    )


async def scenario_long_tool_output(backend_name, make_agent):
    """Can the model handle a tool that returns a lot of text?"""
    return await run_scenario(
        "long_output", backend_name, make_agent,
        "List the files in /usr/bin/.",
        [list_dir],
        lambda r: (len(r) > 20, "should describe the listing"),
    )


async def scenario_shell_command(backend_name, make_agent):
    """Can the model use shell to get system info?"""
    return await run_scenario(
        "shell_command", backend_name, make_agent,
        "Run 'uname -a' using the shell_command tool and tell me the OS.",
        [shell_command],
        lambda r: ("linux" in r.lower() or "arch" in r.lower(), ""),
    )


async def scenario_multi_tool_selection(backend_name, make_agent):
    """With many tools available, can the model pick the right one?"""
    return await run_scenario(
        "tool_selection", backend_name, make_agent,
        "What is 99 * 101? Use the appropriate tool.",
        [calculator, read_file, write_file, list_dir, search_text],
        lambda r: ("9999" in r, "should pick calculator"),
    )


async def scenario_conversation_context(backend_name, make_agent):
    """Does the model maintain context across tool results?"""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "numbers.txt")
        with open(filepath, "w") as f:
            f.write("42")

        return await run_scenario(
            "context_carry", backend_name, make_agent,
            f"Read the number from {filepath}, then multiply it by 10 using the calculator.",
            [read_file, calculator],
            lambda r: ("420" in r, "read 42, multiply by 10 = 420"),
        )


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

ALL_SCENARIOS = [
    scenario_simple_math,
    scenario_chained_math,
    scenario_file_create_and_read,
    scenario_read_json,
    scenario_search_in_file,
    scenario_list_and_read,
    scenario_error_recovery,
    scenario_no_tool_needed,
    scenario_long_tool_output,
    scenario_shell_command,
    scenario_multi_tool_selection,
    scenario_conversation_context,
]


async def main():
    print("=" * 90)
    print("EDGELOOP SCENARIO TESTS")
    print("=" * 90)
    print(f"Scenarios: {len(ALL_SCENARIOS)}")
    print(f"Backends:  {', '.join(BACKENDS.keys())}")
    print()

    results: list[Result] = []

    for scenario_fn in ALL_SCENARIOS:
        scenario_name = scenario_fn.__name__.replace("scenario_", "")
        print(f"--- {scenario_name}: {scenario_fn.__doc__} ---")

        for backend_name, make_fn in BACKENDS.items():
            result = await scenario_fn(backend_name, make_fn)
            results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{backend_name:14s}] {status} {result.time_s:.3f}s  {result.note}")
            if not result.passed:
                print(f"    Response: {result.response[:120]}")
        print()

    # Summary
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print()

    # Per-backend pass rate
    print(f"{'Backend':<16s} {'Pass':<6s} {'Fail':<6s} {'Rate':<8s} {'Avg Time':<10s}")
    print("-" * 50)
    for bn in BACKENDS:
        br = [r for r in results if r.backend == bn]
        passed = sum(1 for r in br if r.passed)
        failed = sum(1 for r in br if not r.passed)
        avg_t = sum(r.time_s for r in br) / len(br) if br else 0
        rate = f"{passed}/{passed+failed}"
        print(f"{bn:<16s} {passed:<6d} {failed:<6d} {rate:<8s} {avg_t:.3f}s")
    print()

    # Per-scenario pass rate
    print(f"{'Scenario':<22s}", end="")
    for bn in BACKENDS:
        print(f" {bn[:10]:>10s}", end="")
    print()
    print("-" * (22 + 11 * len(BACKENDS)))

    scenario_names = list(dict.fromkeys(r.scenario for r in results))
    for sn in scenario_names:
        print(f"{sn:<22s}", end="")
        for bn in BACKENDS:
            match = [r for r in results if r.scenario == sn and r.backend == bn]
            if match:
                r = match[0]
                status = f"{'OK' if r.passed else 'FAIL'} {r.time_s:.2f}s"
                print(f" {status:>10s}", end="")
            else:
                print(f" {'N/A':>10s}", end="")
        print()

    print()
    total_pass = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"Total: {total_pass}/{total} passed ({total_pass/total*100:.0f}%)")


if __name__ == "__main__":
    asyncio.run(main())
