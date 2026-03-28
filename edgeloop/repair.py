"""Output parsing, JSON repair, and tool-call fixing for local LLMs."""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def extract_json(text: str) -> str | None:
    """Extract a JSON object from LLM output.

    Handles: markdown fences, XML tags, raw JSON embedded in text.
    """
    # Try markdown fence first
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try XML-style tags
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find raw JSON object with brace matching
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # Unmatched braces — return from start to end
    if depth > 0:
        return text[start:]

    return None


def repair_json(text: str) -> str:
    """Fix common JSON syntax issues from local models."""
    # Try parsing as-is first
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    result = text

    # Replace single quotes with double quotes (naive but effective)
    # Replace all single-quoted strings: 'foo' -> "foo"
    result = re.sub(r"'([^']*)'", r'"\1"', result)
    # Fix any resulting doubled double-quotes from adjacent replacements
    result = re.sub(r'""', '"', result)

    # Remove trailing commas before } or ]
    result = re.sub(r",\s*([}\]])", r"\1", result)

    # Close unmatched braces
    open_braces = result.count("{") - result.count("}")
    if open_braces > 0:
        result += "}" * open_braces

    open_brackets = result.count("[") - result.count("]")
    if open_brackets > 0:
        result += "]" * open_brackets

    return result


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(
                min(curr_row[j] + 1, prev_row[j + 1] + 1, prev_row[j] + cost)
            )
        prev_row = curr_row

    return prev_row[-1]


def fuzzy_match_tool(name: str, available: list[str], max_distance: int = 2) -> str | None:
    """Match a tool name against available tools, allowing small typos."""
    # Exact match first
    if name in available:
        return name

    best_name = None
    best_dist = max_distance + 1

    for candidate in available:
        dist = _levenshtein(name, candidate)
        if dist < best_dist:
            best_dist = dist
            best_name = candidate

    if best_dist <= max_distance:
        logger.debug("Fuzzy matched tool '%s' → '%s' (distance=%d)", name, best_name, best_dist)
        return best_name

    return None


def coerce_arguments(args: dict, schema: dict) -> dict:
    """Coerce argument types to match schema, strip extras, add defaults.

    Handles common local model mistakes:
    - Wrong argument names (maps positionally if names don't match)
    - Wrong types (coerces str→int, str→float, str→bool)
    - Extra fields (stripped)
    - Missing optional fields (defaults added)
    """
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    result = {}

    # First try: direct name matching
    for key, prop_schema in properties.items():
        if key in args:
            result[key] = _coerce_value(args[key], prop_schema)
        elif "default" in prop_schema:
            result[key] = prop_schema["default"]

    # If we're missing required args but have unmatched values, try positional mapping
    missing_required = [k for k in required if k not in result]
    if missing_required:
        unmatched_values = [v for k, v in args.items() if k not in properties]
        if unmatched_values:
            prop_list = list(properties.keys())
            for i, key in enumerate(missing_required):
                if i < len(unmatched_values):
                    prop_schema = properties[key]
                    result[key] = _coerce_value(unmatched_values[i], prop_schema)
                    logger.debug("Positionally mapped arg %d → '%s'", i, key)

    return result


def _coerce_value(value: Any, prop_schema: dict) -> Any:
    """Coerce a single value to match its schema type."""
    target_type = prop_schema.get("type", "string")

    if target_type == "integer" and isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    elif target_type == "number" and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    elif target_type == "boolean" and isinstance(value, str):
        return value.lower() in ("true", "1", "yes")

    return value


def repair_tool_call(text: str, tools: list) -> dict | None:
    """Full repair pipeline: extract, fix, match, coerce.

    Returns {"name": str, "arguments": dict} or None if no tool call found.
    """
    raw = extract_json(text)
    if raw is None:
        return None

    repaired = repair_json(raw)

    try:
        parsed = json.loads(repaired)
    except json.JSONDecodeError:
        logger.warning("JSON repair failed for: %s", raw[:100])
        return None

    if "tool" not in parsed:
        return None

    tool_name = parsed["tool"]
    arguments = parsed.get("arguments", {})

    # Build available tool names and schema map
    available = []
    schema_map = {}
    for t in tools:
        schema = t.__tool_schema__
        available.append(schema["name"])
        schema_map[schema["name"]] = schema["parameters"]

    # Fuzzy match tool name
    matched_name = fuzzy_match_tool(tool_name, available)
    if matched_name is None:
        logger.warning("Unknown tool '%s', available: %s", tool_name, available)
        return None

    # Coerce arguments
    if matched_name in schema_map:
        arguments = coerce_arguments(arguments, schema_map[matched_name])

    return {"name": matched_name, "arguments": arguments}
