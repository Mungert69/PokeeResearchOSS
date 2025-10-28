"""Utilities for rendering Qwen-style chat prompts without relying on a tokenizer.

This replicates the Jinja chat template used by Qwen2 so that backends which
cannot call `tokenizer.apply_chat_template` (e.g. llama.cpp) can still produce
prompts in the format expected by the model.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

# Default fallback system prompt when none is provided.
_DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)


def _stringify_tool_call(tool_call: Dict[str, Any]) -> str:
    """Convert a tool call dict (possibly nested under `function`) to JSON."""
    if "function" in tool_call and isinstance(tool_call["function"], dict):
        tool_call = tool_call["function"]

    name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})
    return json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)


def render_qwen_chat(
    messages: Iterable[Dict[str, Any]],
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = False,
) -> str:
    """Render chat messages to the raw string prompt expected by Qwen models.

    Args:
        messages: Sequence of OpenAI-style chat messages.
        tools: Optional list of tool definitions (OpenAI function schema).
        add_generation_prompt: If true, append the assistant prefix to prompt
            the model for the next response.

    Returns:
        A string formatted per the official Qwen2 chat template.
    """
    messages = list(messages)
    parts: List[str] = []

    if tools:
        system_content = (
            messages[0]["content"]
            if messages
            and messages[0].get("role") == "system"
            and messages[0].get("content")
            else _DEFAULT_SYSTEM_PROMPT
        )
        parts.append("<|im_start|>system\n")
        parts.append(system_content)
        parts.append(
            "\n\n# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>"
        )
        for tool in tools:
            parts.append("\n")
            parts.append(json.dumps(tool, ensure_ascii=False))
        parts.append(
            "\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments within "
            "<tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call><|im_end|>\n"
        )
        start_index = 1 if messages and messages[0].get("role") == "system" else 0
    else:
        if messages and messages[0].get("role") == "system":
            parts.append(
                f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
            )
            start_index = 1
        else:
            parts.append(
                f"<|im_start|>system\n{_DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
            )
            start_index = 0

    for idx in range(start_index, len(messages)):
        message = messages[idx]
        role = message.get("role", "")
        content = message.get("content", "")

        if role in {"user", "system"} or (
            role == "assistant" and not message.get("tool_calls")
        ):
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        elif role == "assistant":
            parts.append("<|im_start|>assistant")
            if content:
                parts.append("\n")
                parts.append(content)
            for tool_call in message.get("tool_calls", []):
                parts.append("\n<tool_call>\n")
                parts.append(_stringify_tool_call(tool_call))
                parts.append("\n</tool_call>")
            parts.append("<|im_end|>\n")

        elif role == "tool":
            previous_is_tool = idx > 0 and messages[idx - 1].get("role") == "tool"
            next_is_tool = idx + 1 < len(messages) and messages[idx + 1].get("role") == "tool"

            if not previous_is_tool:
                parts.append("<|im_start|>user")
            parts.append("\n<tool_response>\n")
            parts.append(content)
            parts.append("\n</tool_response>")
            if not next_is_tool:
                parts.append("<|im_end|>\n")

    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")

    return "".join(parts)


__all__ = ["render_qwen_chat"]
