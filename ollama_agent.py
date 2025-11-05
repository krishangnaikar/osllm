"""
Wire the Ubuntu MCP-like tools into an Ollama model via tool/function calling.

Prereqs
  - Ollama running locally (default http://localhost:11434)
  - An Ollama model that supports tool calling well (e.g. `llama3.1` family)
  - Python deps: `pip install ollama psutil pydantic`
  - Also install the deps from `mcp_tools_server.py` if you import from it.

Usage
  python ollama_agent.py  # then type questions interactively

Notes
  - This agent exposes SAFE, mostly read-only tools.
  - Mutating/admin ops are intentionally excluded.
  - You can swap `MODEL_NAME` as you like (ensure it supports tools).
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

import ollama  # pip install ollama

# --- Import the tool implementations from the Ubuntu server file ---
# If the file is in the same dir and named mcp_tools_server.py, this works:
from mcp_tools_server import (
    get_top,            # keep for schema/name only
    list_listeners,
    list_kmods,
    kmod_info,
    disk_smart,
    find_large,
    dns_healthcheck,
    trace_syscalls,

    get_top_impl,
)


MODEL_NAME = "llama3.1"

# --- Tool registry: map tool name -> callable and JSON schema ---
# Ollama accepts OpenAI-style tool schemas (functions with JSON Schema inputs).

TOOLS: Dict[str, Dict[str, Any]] = {
    "get_top": {
        "callable": get_top_impl,
        "schema": {
            "type": "object",
            "properties": {
                "by": {"type": "string", "enum": ["cpu", "mem"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["by"],
        },
        "description": "Return top processes by CPU or memory.",
    },
    "list_listeners": {
        "callable": list_listeners,
        "schema": {
            "type": "object",
            "properties": {
                "ports": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional filter for local ports",
                }
            },
        },
        "description": "List inet connections (TCP/UDP) with owning process.",
    },
    "list_kmods": {
        "callable": list_kmods,
        "schema": {"type": "object", "properties": {}},
        "description": "List loaded kernel modules (lsmod).",
    },
    "kmod_info": {
        "callable": kmod_info,
        "schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        "description": "Return modinfo fields for a kernel module by name.",
    },
    "disk_smart": {
        "callable": disk_smart,
        "schema": {
            "type": "object",
            "properties": {
                "devices": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list like /dev/sda, /dev/nvme0n1",
                }
            },
        },
        "description": "Query SMART health via smartctl (-H -A).",
    },
    "find_large": {
        "callable": find_large,
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "min_mb": {"type": "integer", "minimum": 1, "default": 100},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
            },
            "required": ["path"],
        },
        "description": "Scan path recursively and return large files (>= min_mb).",
    },
    "dns_healthcheck": {
        "callable": dns_healthcheck,
        "schema": {
            "type": "object",
            "properties": {
                "domains": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["domains"],
        },
        "description": "Resolve domains and measure latency.",
    },
    "trace_syscalls": {
        "callable": trace_syscalls,
        "schema": {
            "type": "object",
            "properties": {
                "pid": {"type": "integer"},
                "command": {"type": "string"},
                "duration_s": {"type": "integer", "minimum": 1, "maximum": 300, "default": 10},
            },
            "oneOf": [
                {"required": ["pid"]},
                {"required": ["command"]},
            ],
        },
        "description": "Summarize syscalls for a PID or command using strace -c (sudo).",
    },
}

OLLAMA_TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": meta["description"],
            "parameters": meta["schema"],
        },
    }
    for name, meta in TOOLS.items()
]


def call_model(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Call Ollama chat completions with our tool definitions."""
    resp = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        tools=OLLAMA_TOOLS_SPEC,
    )
    return resp


def handle_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Execute requested tools and return messages to append to the chat."""
    followups: List[Dict[str, Any]] = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name")
        args_json = fn.get("arguments") or "{}"
        try:
            args = json.loads(args_json) if isinstance(args_json, str) else args_json
        except json.JSONDecodeError:
            args = {}
        if name not in TOOLS:
            result = {"error": f"Unknown tool: {name}"}
        else:
            try:
                result = TOOLS[name]["callable"](**args)
            except Exception as e:
                result = {"error": str(e)}
        # Per OpenAI-style semantics, reply with a tool message containing the result
        followups.append(
            {
                "role": "tool",
                "tool_call_id": tc.get("id"),
                "name": name,
                "content": json.dumps(
                    result, default=lambda o: o.dict() if hasattr(o, "dict") else o
                ),
            }
        )
    return followups


def chat_loop():
    print(f"Using model: {MODEL_NAME}. Type 'exit' to quit.\n")
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You have system-inspection TOOLS. If a user asks anything about CPU, memory, "
                "processes, network, disks, or syscalls, you MUST call an appropriate tool "
                "instead of answering from prior knowledge. If the user asks about CPU-heavy "
                "processes, ALWAYS call get_top(by='cpu'). Return concise, structured results."
            ),
        }
    ]

    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        messages.append({"role": "user", "content": user})

        # 1st call
        resp = call_model(messages)
        messages.append(resp["message"])  # assistant draft (may include tool calls)

        # If there are tool calls, execute them and send results back
        if "tool_calls" in resp["message"] and resp["message"]["tool_calls"]:
            tool_results = handle_tool_calls(resp["message"]["tool_calls"])  # type: ignore
            messages.extend(tool_results)
            # 2nd call for the final answer using tool outputs as context
            resp2 = call_model(messages)
            messages.append(resp2["message"])  # final assistant message
            print(f"Assistant: {resp2['message']['content']}\n")
        else:
            print(f"Assistant: {resp['message']['content']}\n")


if __name__ == "__main__":
    try:
        chat_loop()
    except KeyboardInterrupt:
        print("\nGoodbye!")
