#!/usr/bin/env python3
"""
OpenAI Responses API -> GLM Chat Completions API Proxy

Converts the new Responses API format to the traditional Chat Completions format
so that Codex can work with GLM (智谱 AI) models.

Powered by aiohttp for high-performance async I/O with connection pooling.
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
import time
import uuid

import aiohttp
from aiohttp import web

# Configuration
GLM_API_BASE = os.environ.get("GLM_API_BASE", "https://open.bigmodel.cn/api/coding/paas/v4")
GLM_API_KEY = os.environ.get("GLM_API_KEY", "")
try:
    PROXY_PORT = int(os.environ.get("PROXY_PORT", 18765))
except ValueError:
    print("Error: PROXY_PORT must be a number", file=sys.stderr)
    sys.exit(1)
MAX_CONTENT_SIZE = 1_000_000  # 1MB limit for accumulated stream content

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("codex-glm-proxy")

# Model name mapping: OpenAI -> GLM
MODEL_MAPPING = {
    "glm-5.1": "glm-5.1",
    "glm-5": "glm-5",
    "glm-5-turbo": "glm-5-turbo",
    "gpt-4": "glm-5.1",
    "glm-4-flash": "glm-5.1",
    "gpt-4-turbo": "glm-5.1",
    "gpt-4o": "glm-5.1",
    "gpt-4o-mini": "glm-5.1",
    "gpt-3.5-turbo": "glm-5.1",
    "gpt-5.2-codex": "glm-5.1",
    "gpt-5.3-codex": "glm-5.1",
    "gpt-5.4-mini": "glm-5.1",
    "gpt-5.4-codex": "glm-5.1",
    "gpt-5.5": "glm-5.1",
}


# ---------------------------------------------------------------------------
# Reasoning Content Store
# ---------------------------------------------------------------------------

_reasoning_store: dict[str, str] = {}


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _store_reasoning(text: str, reasoning: str, tool_call_ids: list[str] | None = None):
    if not reasoning:
        return
    h = _content_hash(text)
    _reasoning_store[h] = reasoning
    if tool_call_ids:
        for tc_id in tool_call_ids:
            _reasoning_store[f"tc_{tc_id}"] = reasoning
    log.info("Stored reasoning: hash=%s, len=%d", h[:12], len(reasoning))


def _lookup_reasoning(text: str) -> str:
    h = _content_hash(text)
    return _reasoning_store.get(h, "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(content) -> str:
    """Extract text from various content formats (string, list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type") in ("input_text", "text", "output_text"):
                parts.append(p.get("text", ""))
        return "\n".join(parts)
    return ""


# ---------------------------------------------------------------------------
# Conversion: Responses API -> Chat Completions API
# ---------------------------------------------------------------------------

def convert_responses_to_chat(body: dict) -> dict:
    chat_body = {}

    model = body.get("model", "glm-5.1")
    chat_body["model"] = MODEL_MAPPING.get(model, "glm-5.1")

    messages = []

    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})

    inp = body.get("input")
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        _convert_input_list(inp, messages)
    elif isinstance(inp, dict):
        if "messages" in inp:
            for msg in inp["messages"]:
                role = msg.get("role", "user")
                if role == "developer":
                    role = "system"
                messages.append({"role": role, "content": msg.get("content", "")})
        elif "content" in inp:
            messages.append({"role": "user", "content": inp["content"]})

    chat_body["messages"] = messages

    for key in ("temperature", "top_p", "max_tokens",
                "stream", "frequency_penalty", "presence_penalty", "stop",
                "parallel_tool_calls"):
        if key in body:
            chat_body[key] = body[key]

    # Map max_output_tokens → max_completion_tokens
    if "max_output_tokens" in body:
        chat_body["max_completion_tokens"] = body["max_output_tokens"]
    elif "max_completion_tokens" in body:
        chat_body["max_completion_tokens"] = body["max_completion_tokens"]

    # Add stream_options for usage reporting
    if body.get("stream"):
        chat_body["stream_options"] = {"include_usage": True}

    if "tools" in body:
        tool_ctx = _build_tool_context(body["tools"])
        chat_tools = _convert_tools(body["tools"], tool_ctx)
        if chat_tools:
            chat_body["tools"] = chat_tools
        # Store tool context on the body for later response-side use
        body["_tool_ctx"] = tool_ctx

    if "tool_choice" in body:
        chat_body["tool_choice"] = body["tool_choice"]

    # Map reasoning.effort → reasoning_effort
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort"):
        effort = reasoning["effort"]
        effort_map = {
            "none": "none", "auto": "auto", "minimal": "low",
            "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh",
        }
        chat_body["reasoning_effort"] = effort_map.get(effort, "auto")

    # Passthrough user field
    if "user" in body:
        chat_body["user"] = body["user"]

    return chat_body


def _remove_orphan_tool_messages(messages: list):
    """Remove {role: tool} messages that have no preceding assistant.tool_calls."""
    valid_ids: set[str] | None = None
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.get("role") == "assistant":
            tcs = m.get("tool_calls", [])
            valid_ids = {tc["id"] for tc in tcs if tc.get("id")} if tcs else None
            i += 1
        elif m.get("role") == "tool":
            if valid_ids and m.get("tool_call_id") and m["tool_call_id"] in valid_ids:
                i += 1
            else:
                log.warning("Dropping orphan tool message: tool_call_id=%s", m.get("tool_call_id"))
                messages.pop(i)
        else:
            valid_ids = None
            i += 1


def _ensure_tool_calls_have_outputs(messages: list):
    """Ensure every assistant message with tool_calls has a matching tool output."""
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            i += 1
            continue
        seen = set()
        j = i + 1
        while j < len(messages) and messages[j].get("role") == "tool":
            tcid = messages[j].get("tool_call_id")
            if tcid:
                seen.add(tcid)
            j += 1
        missing = [tc["id"] for tc in m["tool_calls"] if tc.get("id") and tc["id"] not in seen]
        if missing:
            placeholders = [
                {"role": "tool", "tool_call_id": tid,
                 "content": "[tool output missing — no function_call_output was provided for this call_id]"}
                for tid in missing
            ]
            for k, ph in enumerate(placeholders):
                messages.insert(j + k, ph)
            i = j + len(placeholders)
        else:
            i += 1


def _convert_input_list(items: list, messages: list):
    """Convert Responses API input items to Chat Completions messages.

    Key challenge: In Responses API, an assistant turn with both text and tool_calls
    appears as separate items (message + function_call), but Chat Completions needs
    them merged into a single assistant message with tool_calls.

    Codex pattern: function_call → function_call → message(assistant) →
    function_call_output → function_call_output. The assistant text message must be
    merged with the preceding tool_calls into one assistant message.
    """
    pending_assistant = None
    pending_tc: list[dict] = []
    pending_reasoning: list[str] = []

    def flush_assistant():
        nonlocal pending_assistant
        if pending_assistant:
            if pending_reasoning:
                pending_assistant["reasoning_content"] = "\n".join(pending_reasoning)
                pending_reasoning.clear()
            messages.append(pending_assistant)
            pending_assistant = None

    def flush_tool_calls():
        nonlocal pending_tc
        if not pending_tc:
            return
        rc = ""
        if pending_reasoning:
            rc = "\n".join(pending_reasoning)
            pending_reasoning.clear()
        else:
            for tc in pending_tc:
                rc = _reasoning_store.get(f"tc_{tc['id']}", "")
                if rc:
                    break
        msg = {"role": "assistant", "content": None, "tool_calls": list(pending_tc)}
        if rc:
            msg["reasoning_content"] = rc
        messages.append(msg)
        pending_tc = []

    for item in items:
        if isinstance(item, str):
            flush_assistant()
            flush_tool_calls()
            if pending_reasoning:
                pending_reasoning.clear()
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue

        item_type = item.get("type", "")

        if item_type == "function_call":
            pending_tc.append({
                "id": item.get("call_id", item.get("id", f"call_{uuid.uuid4().hex[:8]}")),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}"),
                },
            })

        elif item_type == "function_call_output":
            # Merge pending_assistant + pending_tc into one assistant message
            if pending_assistant:
                if pending_tc:
                    pending_assistant["tool_calls"] = list(pending_tc)
                    pending_tc = []
                if pending_reasoning:
                    pending_assistant["reasoning_content"] = "\n".join(pending_reasoning)
                    pending_reasoning.clear()
                messages.append(pending_assistant)
                pending_assistant = None
            else:
                flush_tool_calls()

            tool_output = item.get("output", "")
            if isinstance(tool_output, list):
                tool_output = _extract_text(tool_output)
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": tool_output,
            })

        else:
            # message, reasoning, custom_tool_call, custom_tool_call_output, text, or unknown
            flush_assistant()

            item_type = item.get("type", "")

            # Accumulate reasoning — merge into next assistant message (ccx pattern)
            if item_type == "reasoning":
                reasoning_text = ""
                enc = item.get("encrypted_content")
                if isinstance(enc, str) and enc:
                    reasoning_text = enc
                else:
                    summary = item.get("summary", [])
                    if isinstance(summary, list):
                        reasoning_text = "\n".join(
                            s.get("text", "") for s in summary
                            if isinstance(s, dict) and s.get("type") == "summary_text"
                        )
                if reasoning_text:
                    pending_reasoning.append(reasoning_text)
                continue

            # Support custom_tool_call → assistant with tool_calls
            if item_type == "custom_tool_call":
                flush_tool_calls()
                call_id = item.get("call_id", item.get("id", ""))
                name = item.get("name", "")
                input_text = item.get("input", item.get("arguments", ""))
                if name:
                    tc = {
                        "id": call_id or name,
                        "type": "function",
                        "function": {"name": name, "arguments": input_text},
                    }
                    messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                continue

            # Support custom_tool_call_output → tool message
            if item_type == "custom_tool_call_output":
                flush_tool_calls()
                call_id = item.get("call_id", item.get("id", ""))
                output = item.get("output", "")
                if isinstance(output, list):
                    output = _extract_text(output)
                messages.append({"role": "tool", "tool_call_id": call_id, "content": str(output)})
                continue

            # Handle items without explicit type but with role (legacy compatibility)
            role = item.get("role", "")
            if not item_type and role:
                item_type = "message"

            # Support "text" type (old format)
            if item_type == "text":
                flush_tool_calls()
                role = role or "assistant"
                content = item.get("content", "")
                if isinstance(content, list):
                    content = _extract_text(content)
                if role == "developer":
                    role = "system"
                msg = {"role": role}
                if content is not None:
                    msg["content"] = content
                if msg.get("content") is not None:
                    messages.append(msg)
                continue

            if item_type != "message":
                flush_tool_calls()
                continue

            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            content = item.get("content", "")
            content = _extract_text(content)

            msg = {"role": role}
            if content is not None:
                msg["content"] = content

            # For assistant messages, hold for possible merge with pending tool_calls
            # (Codex pattern: function_call → message(assistant) → function_call_output)
            if role == "assistant":
                stored_rc = _lookup_reasoning(content or "")
                if stored_rc:
                    msg["reasoning_content"] = stored_rc
                elif pending_reasoning:
                    msg["reasoning_content"] = "\n".join(pending_reasoning)
                    pending_reasoning.clear()
                msg.setdefault("content", None)
                # DON'T flush tool_calls — function_call_output will merge them
                pending_assistant = msg
                continue

            # Non-assistant messages: flush tool_calls first
            flush_tool_calls()
            if pending_reasoning:
                pending_reasoning.clear()
            if msg.get("content") is not None:
                messages.append(msg)

    # Flush any remaining
    if pending_assistant and pending_tc:
        pending_assistant["tool_calls"] = list(pending_tc)
        if pending_reasoning:
            pending_assistant["reasoning_content"] = "\n".join(pending_reasoning)
            pending_reasoning.clear()
        messages.append(pending_assistant)
    else:
        flush_assistant()
        flush_tool_calls()

    # Remove orphan tool messages and ensure all tool_calls have outputs
    _remove_orphan_tool_messages(messages)
    _ensure_tool_calls_have_outputs(messages)


def _normalize_tool_params(params: dict | None) -> dict:
    """Ensure tool parameters have type, properties, and required fields."""
    if not params or not isinstance(params, dict):
        params = {}
    params.setdefault("type", "object")
    params.setdefault("properties", {})
    params.setdefault("required", [])
    return params


# ---------------------------------------------------------------------------
# Codex Custom Tool Compatibility
# ---------------------------------------------------------------------------

# local_shell builtin → standard shell function tool
_LOCAL_SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "shell",
        "description": "Execute a shell command on the local machine.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Argv array, e.g. [\"ls\", \"-la\"].",
                },
                "workdir": {"type": "string", "description": "Working directory."},
                "timeout_ms": {"type": "number", "description": "Timeout in milliseconds."},
            },
            "required": ["command"],
        },
    },
}


def _is_apply_patch_tool(tool: dict) -> bool:
    """Check if a tool is Codex's apply_patch custom tool."""
    if tool.get("type") != "custom":
        return False
    name = tool.get("name", "")
    if name == "apply_patch":
        return True
    fmt = tool.get("format", {})
    if isinstance(fmt, dict):
        grammar = fmt.get("definition", "")
        if isinstance(grammar, str) and "begin_patch" in grammar and "end_patch" in grammar:
            return True
    return False


def _apply_patch_proxy_tools(name: str, description: str = "") -> list[dict]:
    """Generate 5 proxy function tools for apply_patch."""
    base_desc = description or "Edit files using structured patch operations."
    tools = []
    for action, desc, schema in [
        ("_add_file", "Create a new file.",
         {"type": "object", "properties": {
             "path": {"type": "string", "description": "Target file path."},
             "content": {"type": "string", "description": "Full file content."}},
          "required": ["path", "content"]}),
        ("_delete_file", "Delete a file.",
         {"type": "object", "properties": {
             "path": {"type": "string", "description": "Target file path."}},
          "required": ["path"]}),
        ("_update_file", "Edit an existing file with hunks.",
         {"type": "object", "properties": {
             "path": {"type": "string", "description": "Target file path."},
             "hunks": {"type": "array", "items": {
                 "type": "object", "properties": {
                     "lines": {"type": "array", "items": {
                         "type": "object", "properties": {
                             "op": {"type": "string", "enum": ["context", "add", "remove"]},
                             "text": {"type": "string"}},
                          "required": ["op", "text"]}}},
                  "required": ["lines"]}}},
          "required": ["path", "hunks"]}),
        ("_replace_file", "Replace a file entirely.",
         {"type": "object", "properties": {
             "path": {"type": "string", "description": "Target file path."},
             "content": {"type": "string", "description": "Full replacement content."}},
          "required": ["path", "content"]}),
        ("_batch", "Edit multiple files in one operation.",
         {"type": "object", "properties": {
             "operations": {"type": "array", "items": {
                 "type": "object", "properties": {
                     "type": {"type": "string", "enum": ["add_file", "delete_file", "update_file", "replace_file"]},
                     "path": {"type": "string"},
                     "content": {"type": "string"},
                     "hunks": {"type": "array"}},
                  "required": ["type", "path"]}}},
          "required": ["operations"]}),
    ]:
        tools.append({
            "type": "function",
            "function": {
                "name": f"{name}{action}",
                "description": f"{base_desc} (proxy: {action[1:]})" if base_desc else desc,
                "parameters": schema,
            },
        })
    return tools


def _reconstruct_apply_patch_input(name: str, action: str, raw_args: str) -> str:
    """Reconstruct apply_patch grammar text from proxy tool arguments."""
    try:
        args = json.loads(raw_args)
    except (json.JSONDecodeError, TypeError):
        return raw_args

    parts = ["*** Begin Patch\n"]

    if action == "add_file":
        parts.append(f"*** Add File: {args.get('path', '')}\n")
        content = args.get("content", "")
        for line in content.rstrip("\n").split("\n"):
            parts.append(f"+{line}\n")
    elif action == "delete_file":
        parts.append(f"*** Delete File: {args.get('path', '')}\n")
    elif action == "update_file":
        parts.append(f"*** Update File: {args.get('path', '')}\n")
        for hunk in args.get("hunks", []):
            parts.append("@@\n")
            for line in hunk.get("lines", []):
                op = line.get("op", "context")
                text = line.get("text", "")
                prefix = {"context": " ", "add": "+", "remove": "-"}.get(op, " ")
                parts.append(f"{prefix}{text}\n")
    elif action == "replace_file":
        parts.append(f"*** Delete File: {args.get('path', '')}\n")
        parts.append(f"*** Add File: {args.get('path', '')}\n")
        content = args.get("content", "")
        for line in content.rstrip("\n").split("\n"):
            parts.append(f"+{line}\n")
    elif action == "batch":
        for op in args.get("operations", []):
            op_type = op.get("type", "")
            path = op.get("path", "")
            if op_type == "add_file":
                parts.append(f"*** Add File: {path}\n")
                for line in op.get("content", "").rstrip("\n").split("\n"):
                    parts.append(f"+{line}\n")
            elif op_type == "delete_file":
                parts.append(f"*** Delete File: {path}\n")
            elif op_type == "update_file":
                parts.append(f"*** Update File: {path}\n")
                for hunk in op.get("hunks", []):
                    parts.append("@@\n")
                    for line in hunk.get("lines", []):
                        o = line.get("op", "context")
                        prefix = {"context": " ", "add": "+", "remove": "-"}.get(o, " ")
                        parts.append(f"{prefix}{line.get('text', '')}\n")
            elif op_type == "replace_file":
                parts.append(f"*** Delete File: {path}\n")
                parts.append(f"*** Add File: {path}\n")
                for line in op.get("content", "").rstrip("\n").split("\n"):
                    parts.append(f"+{line}\n")
    else:
        return raw_args

    parts.append("*** End Patch")
    return "".join(parts)


def _is_apply_patch_proxy(name: str) -> tuple[bool, str]:
    """Check if function name is an apply_patch proxy. Returns (is_proxy, action)."""
    for suffix in ("_add_file", "_delete_file", "_update_file", "_replace_file", "_batch"):
        if name.endswith(suffix):
            return True, suffix[1:]  # strip leading _
    return False, ""


def _generic_custom_tool(name: str, description: str = "") -> dict:
    """Convert a custom tool to a generic function tool with {input: string}."""
    desc = description or f"Custom tool: {name}. Put the tool input text here."
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Raw input for this tool."},
                },
                "required": ["input"],
            },
        },
    }


def _flatten_namespace_tool(namespace: str, name: str) -> str:
    """Flatten namespace + tool name into a single function name."""
    if not namespace:
        return name
    sep = "" if namespace.endswith("__") or name.startswith("__") else "__"
    return f"{namespace}{sep}{name}"


class _ToolContext:
    """Track tool metadata for request/response conversion."""

    def __init__(self):
        self.apply_patch_names: set[str] = set()  # original apply_patch tool names
        self.custom_tools: dict[str, str] = {}     # proxy_name → original_name
        self.namespace_tools: dict[str, tuple[str, str]] = {}  # flat_name → (namespace, name)

    def is_custom_proxy(self, name: str) -> bool:
        return name in self.custom_tools

    def original_name(self, name: str) -> str:
        return self.custom_tools.get(name, name)

    def unflatten_namespace(self, name: str) -> tuple[str, str]:
        """Return (name, namespace) for a flat function name."""
        if name in self.namespace_tools:
            return self.namespace_tools[name]
        return name, ""

    def reconstruct_input(self, name: str, raw_args: str) -> str:
        """Reconstruct custom tool input from proxy arguments."""
        if name in self.apply_patch_names:
            is_proxy, action = _is_apply_patch_proxy(name)
            if is_proxy:
                return _reconstruct_apply_patch_input(name, action, raw_args)
            # Direct apply_patch call
            try:
                args = json.loads(raw_args)
                if isinstance(args, dict) and "input" in args:
                    return args["input"]
            except (json.JSONDecodeError, TypeError):
                pass
            return raw_args
        # Generic custom tool
        try:
            args = json.loads(raw_args)
            if isinstance(args, dict) and "input" in args:
                return args["input"]
        except (json.JSONDecodeError, TypeError):
            pass
        return raw_args


def _build_tool_context(tools: list) -> _ToolContext:
    """Build tool context from Responses API tools list."""
    ctx = _ToolContext()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type", "")
        name = tool.get("name", "")

        if tool_type == "custom":
            if _is_apply_patch_tool(tool):
                ctx.apply_patch_names.add(name)
                for suffix in ("_add_file", "_delete_file", "_update_file", "_replace_file", "_batch"):
                    proxy_name = f"{name}{suffix}"
                    ctx.custom_tools[proxy_name] = name
                ctx.custom_tools[name] = name
            else:
                ctx.custom_tools[name] = name
        elif tool_type == "namespace":
            namespace = name
            children = tool.get("tools", [])
            for child in children:
                if isinstance(child, dict) and child.get("type") == "function":
                    child_name = child.get("name", "")
                    if child_name:
                        flat = _flatten_namespace_tool(namespace, child_name)
                        ctx.namespace_tools[flat] = (child_name, namespace)
    return ctx


def _convert_tools(tools: list, ctx: _ToolContext | None = None) -> list[dict] | None:
    """Convert Responses API tools to Chat Completions tools format."""
    ctx = ctx or _ToolContext()
    out = []
    seen_apply_patch: set[str] = set()

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type", "")

        # GLM web_search tool
        if tool_type == "web_search":
            out.append({"type": "web_search", "web_search": {"enable": True, "search_result": True}})
            continue

        # Skip server-side-only tools not supported by GLM
        if tool_type in ("code_interpreter", "file_search", "computer_use",
                         "image_generation", "computer_use_preview", "web_search_preview"):
            continue

        # Standard function tool
        if tool_type == "function":
            if "function" in tool:
                func_def = tool["function"]
                func = {"name": func_def.get("name", "")}
                if func_def.get("description"):
                    func["description"] = func_def["description"]
                func["parameters"] = _normalize_tool_params(func_def.get("parameters"))
                if isinstance(func_def.get("strict"), bool):
                    func["strict"] = func_def["strict"]
                out.append({"type": "function", "function": func})
            else:
                func = {"name": tool.get("name", "")}
                if tool.get("description"):
                    func["description"] = tool["description"]
                func["parameters"] = _normalize_tool_params(tool.get("parameters"))
                if isinstance(tool.get("strict"), bool):
                    func["strict"] = tool["strict"]
                out.append({"type": "function", "function": func})
            continue

        # local_shell → shell function tool
        if tool_type == "local_shell":
            out.append(_LOCAL_SHELL_TOOL)
            continue

        # custom tool (apply_patch → 5 proxies, else generic)
        if tool_type == "custom":
            name = tool.get("name", "")
            description = tool.get("description", "")
            if _is_apply_patch_tool(tool) and name not in seen_apply_patch:
                seen_apply_patch.add(name)
                out.extend(_apply_patch_proxy_tools(name, description))
            else:
                out.append(_generic_custom_tool(name, description))
            continue

        # namespace → flatten child function tools
        if tool_type == "namespace":
            namespace = tool.get("name", "")
            children = tool.get("tools", [])
            for child in children:
                if not isinstance(child, dict):
                    continue
                if child.get("type") == "function":
                    child_name = child.get("name", "")
                    if not child_name:
                        continue
                    flat = _flatten_namespace_tool(namespace, child_name)
                    func = {"name": flat}
                    child_desc = child.get("description", "")
                    ns_desc = tool.get("description", "")
                    combined = f"{ns_desc}\n\n{child_desc}".strip() if ns_desc and child_desc else child_desc or ns_desc
                    if combined:
                        func["description"] = combined
                    func["parameters"] = _normalize_tool_params(child.get("parameters"))
                    out.append({"type": "function", "function": func})
            continue

        # Unknown tool types that have a function key — pass through
        if "function" in tool:
            out.append(tool)

    return out or None


# ---------------------------------------------------------------------------
# Conversion: Chat Completions -> Responses API (non-streaming)
# ---------------------------------------------------------------------------

def _normalize_usage(usage: dict | None) -> dict:
    """Ensure usage has all fields Codex expects."""
    if not usage:
        usage = {}
    return {
        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
        "output_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
        "total_tokens": usage.get("total_tokens", 0),
        "input_tokens_details": {"cached_tokens": usage.get("input_tokens_details", {}).get("cached_tokens", 0)},
        "output_tokens_details": {"reasoning_tokens": usage.get("output_tokens_details", {}).get("reasoning_tokens", 0)},
    }


def convert_chat_to_responses(response_body: dict, body: dict | None = None) -> dict:
    outputs = []
    for choice in response_body.get("choices", []):
        msg = choice.get("message", {})

        # Store reasoning_content for multi-turn restoration
        reasoning = msg.get("reasoning_content", "")
        content_text = msg.get("content", "") or ""

        # Reasoning output item (with encrypted_content for round-trip)
        if reasoning:
            outputs.append({
                "type": "reasoning",
                "id": f"rs_{response_body.get('id', '')}",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": reasoning}],
                "encrypted_content": reasoning,
            })

        content = []
        if content_text:
            content.append({"type": "output_text", "text": content_text, "annotations": []})

        msg_id = f"msg_{response_body.get('id', '')}"
        outputs.append({
            "type": "message",
            "id": msg_id,
            "status": "completed",
            "role": msg.get("role", "assistant"),
            "content": content if content else [],
        })

        # Tool calls as separate output items
        tc_ids = []
        tool_ctx: _ToolContext | None = body.get("_tool_ctx") if isinstance(body, dict) else None
        for tc in msg.get("tool_calls", []):
            tc_id = tc.get("id", "")
            tc_ids.append(tc_id)
            tc_name = tc.get("function", {}).get("name", "")
            tc_args = tc.get("function", {}).get("arguments", "{}")

            # Reconstruct custom tool input if applicable
            if tool_ctx and tool_ctx.is_custom_proxy(tc_name):
                reconstructed = tool_ctx.reconstruct_input(tc_name, tc_args)
                original = tool_ctx.original_name(tc_name)
                outputs.append({
                    "type": "custom_tool_call",
                    "id": f"ctc_{tc_id}",
                    "call_id": tc_id,
                    "name": original,
                    "input": reconstructed,
                    "status": "completed",
                })
                continue

            # Unflatten namespace tools
            display_name = tc_name
            namespace = ""
            if tool_ctx:
                display_name, namespace = tool_ctx.unflatten_namespace(tc_name)

            item = {
                "type": "function_call",
                "id": f"fc_{tc_id}",
                "call_id": tc_id,
                "name": display_name,
                "arguments": tc_args,
                "status": "completed",
            }
            if namespace:
                item["namespace"] = namespace
            outputs.append(item)

        # Store reasoning for multi-turn
        _store_reasoning(content_text, reasoning, tc_ids)

    resp_id = response_body.get("id", "")
    if not resp_id.startswith("resp_"):
        resp_id = f"resp_{resp_id}"

    body = body or {}

    # Determine status from finish_reason
    finish_reason = ""
    for choice in response_body.get("choices", []):
        fr = choice.get("finish_reason", "")
        if fr:
            finish_reason = fr
            break

    status = "completed"
    incomplete_details = None
    if finish_reason == "length":
        status = "incomplete"
        incomplete_details = {"reason": "max_output_tokens"}

    result = {
        "id": resp_id,
        "object": "response",
        "created_at": response_body.get("created", 0),
        "model": response_body.get("model", ""),
        "output": outputs,
        "status": status,
        "background": False,
        "error": None,
        "incomplete_details": incomplete_details,
        "usage": _normalize_usage(response_body.get("usage")),
        "parallel_tool_calls": body.get("parallel_tool_calls", True),
        "truncation": "disabled",
        "tool_choice": body.get("tool_choice", "auto"),
        "text": body.get("text", {"format": {"type": "text"}}),
        "reasoning": {"effort": "high", "summary": "auto"} if reasoning else body.get("reasoning", {"effort": None, "summary": None}),
    }

    # Echo back request-level fields
    for key in ("instructions", "tools", "metadata", "previous_response_id",
                 "temperature", "top_p", "max_output_tokens"):
        val = body.get(key)
        if val is not None:
            result[key] = val

    return result


# ---------------------------------------------------------------------------
# Stream Converter: Chat Completions SSE -> Responses API SSE
# ---------------------------------------------------------------------------

class StreamConverter:
    """Stateful converter for one streaming request.

    Call process_line() for each raw SSE line from upstream.
    Returns a list of bytes objects to write downstream.
    """

    def __init__(self, body: dict | None = None):
        self.seq = 0
        self.item_id = None
        self.reasoning_item_id = None
        self.response_id = None
        self.created_at = None
        self.model = None
        self.full_content = ""
        self.full_reasoning = ""
        self.tool_calls: dict[int, dict] = {}
        self._lifecycle_emitted = False
        self._finished = False
        self._text_emitted = False
        self._reasoning_active = False
        self._text_active = False
        self._reasoning_part_added = False
        self._usage = None
        self._finish_reason = None
        self._body = body or {}
        self._tool_ctx: _ToolContext | None = (body or {}).get("_tool_ctx")

    def _next_seq(self):
        s = self.seq
        self.seq += 1
        return s

    def process_line(self, line: bytes) -> list[bytes]:
        if not line.startswith(b"data: "):
            return [line + b"\n"]

        data = line[6:].strip()
        if data == b"[DONE]":
            return self._on_done()

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            return [line + b"\n"]

        results = []

        if not self._lifecycle_emitted:
            self._lifecycle_emitted = True
            self.response_id = chunk.get("id", "")
            if not self.response_id.startswith("resp_"):
                self.response_id = f"resp_{self.response_id}"
            self.created_at = chunk.get("created", int(time.time()))
            self.model = chunk.get("model", "")
            self.item_id = f"msg_{self.response_id}"
            self.reasoning_item_id = f"rs_{self.response_id}"
            # Emit lifecycle events immediately at stream start
            results.extend(self._emit_lifecycle_events())

        # Capture usage from upstream (usually in last chunk)
        if "usage" in chunk and chunk["usage"]:
            self._usage = chunk["usage"]

        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            content = delta.get("content", "")
            finish_reason = choice.get("finish_reason")

            # Capture reasoning_content from GLM thinking mode
            reasoning = delta.get("reasoning_content")
            if reasoning:
                self.full_reasoning += reasoning
                results.extend(self._on_reasoning_delta(reasoning))

            if content:
                # Close reasoning block if active before opening text
                if self._reasoning_active:
                    results.extend(self._close_reasoning_block())

                if not self._text_active:
                    self._text_active = True
                    results.extend(self._open_text_block())

                self.full_content += content
                if len(self.full_content) > MAX_CONTENT_SIZE:
                    self.full_content = self.full_content[-MAX_CONTENT_SIZE:]
                results.append(self._sse("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "sequence_number": self._next_seq(),
                    "output_index": self._text_output_index(),
                    "content_index": 0,
                    "item_id": self.item_id,
                    "delta": content,
                }))

            if "tool_calls" in delta:
                # Close reasoning block if active
                if self._reasoning_active:
                    results.extend(self._close_reasoning_block())
                # Close text block if active
                if self._text_active:
                    results.extend(self._close_text_block())

                results.extend(self._process_tool_calls(delta["tool_calls"]))

            if finish_reason and finish_reason != "null":
                self._finish_reason = finish_reason
                if not self._finished:
                    self._finished = True
                    results.extend(self._on_finish())

        return results

    # --- private helpers ---

    def _text_output_index(self) -> int:
        """Output index for the text message item (1 if reasoning was emitted, else 0)."""
        return 1 if self._reasoning_part_added else 0

    def _tool_output_index(self, tc_index: int) -> int:
        """Output index for a tool call item."""
        base = 0
        if self._reasoning_part_added:
            base += 1
        if self._text_active or self.item_id:
            base += 1
        return base + tc_index

    @staticmethod
    def _sse(event_type: str, data: dict) -> bytes:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()

    def _build_envelope(self, status: str = "in_progress", output=None, usage=None) -> dict:
        resp_obj = {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at or 0,
            "model": self.model or "",
            "status": status,
            "output": output or [],
            "background": False,
            "error": None,
        }
        for key in ("instructions", "tools", "tool_choice", "parallel_tool_calls",
                     "reasoning", "text", "metadata", "previous_response_id",
                     "temperature", "top_p", "max_output_tokens"):
            val = self._body.get(key)
            if val is not None:
                resp_obj[key] = val
        resp_obj.setdefault("parallel_tool_calls", True)
        resp_obj.setdefault("truncation", "disabled")
        resp_obj.setdefault("tool_choice", "auto")
        resp_obj.setdefault("text", {"format": {"type": "text"}})
        # GLM-5.x always thinks — tell Codex to render the thinking section
        resp_obj.setdefault("reasoning", {"effort": "high", "summary": "auto"})
        if usage is not None:
            resp_obj["usage"] = usage
        # incomplete_details
        if status == "incomplete":
            resp_obj["incomplete_details"] = {"reason": "max_output_tokens"}
        else:
            resp_obj["incomplete_details"] = None
        return resp_obj

    def _emit_lifecycle_events(self) -> list[bytes]:
        """Emit response.created + response.in_progress."""
        resp_obj = self._build_envelope("in_progress")
        return [
            self._sse("response.created", {
                "type": "response.created",
                "sequence_number": self._next_seq(),
                "response": resp_obj,
            }),
            self._sse("response.in_progress", {
                "type": "response.in_progress",
                "sequence_number": self._next_seq(),
                "response": resp_obj,
            }),
        ]

    def _on_reasoning_delta(self, reasoning_text: str) -> list[bytes]:
        """Handle reasoning_content delta: open reasoning block if needed, emit delta."""
        results = []
        if not self._reasoning_active:
            self._reasoning_active = True
            output_index = 0
            # output_item.added for reasoning
            results.append(self._sse("response.output_item.added", {
                "type": "response.output_item.added",
                "sequence_number": self._next_seq(),
                "output_index": output_index,
                "item": {
                    "id": self.reasoning_item_id,
                    "type": "reasoning",
                    "status": "in_progress",
                    "summary": [],
                    "encrypted_content": None,
                },
            }))
            # reasoning_summary_part.added
            results.append(self._sse("response.reasoning_summary_part.added", {
                "type": "response.reasoning_summary_part.added",
                "sequence_number": self._next_seq(),
                "item_id": self.reasoning_item_id,
                "output_index": output_index,
                "summary_index": 0,
                "part": {"type": "summary_text", "text": ""},
            }))
            self._reasoning_part_added = True

        output_index = 0
        results.append(self._sse("response.reasoning_summary_text.delta", {
            "type": "response.reasoning_summary_text.delta",
            "sequence_number": self._next_seq(),
            "item_id": self.reasoning_item_id,
            "output_index": output_index,
            "summary_index": 0,
            "delta": reasoning_text,
        }))
        return results

    def _close_reasoning_block(self) -> list[bytes]:
        """Close the reasoning block: emit done events."""
        if not self._reasoning_active:
            return []
        self._reasoning_active = False
        output_index = 0
        full = self.full_reasoning

        results = []
        # reasoning_summary_text.done
        results.append(self._sse("response.reasoning_summary_text.done", {
            "type": "response.reasoning_summary_text.done",
            "sequence_number": self._next_seq(),
            "item_id": self.reasoning_item_id,
            "output_index": output_index,
            "summary_index": 0,
            "text": full,
        }))
        # reasoning_summary_part.done
        results.append(self._sse("response.reasoning_summary_part.done", {
            "type": "response.reasoning_summary_part.done",
            "sequence_number": self._next_seq(),
            "item_id": self.reasoning_item_id,
            "output_index": output_index,
            "summary_index": 0,
            "part": {"type": "summary_text", "text": full},
        }))
        # output_item.done for reasoning
        results.append(self._sse("response.output_item.done", {
            "type": "response.output_item.done",
            "sequence_number": self._next_seq(),
            "output_index": output_index,
            "item": {
                "id": self.reasoning_item_id,
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": full}],
                "encrypted_content": full,
            },
        }))
        return results

    def _open_text_block(self) -> list[bytes]:
        """Open the text message block: output_item.added + content_part.added."""
        results = []
        output_index = self._text_output_index()
        self.item_id = f"msg_{self.response_id}_{output_index}"

        results.append(self._sse("response.output_item.added", {
            "type": "response.output_item.added",
            "sequence_number": self._next_seq(),
            "output_index": output_index,
            "item": {
                "type": "message",
                "id": self.item_id,
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        }))
        results.append(self._sse("response.content_part.added", {
            "type": "response.content_part.added",
            "sequence_number": self._next_seq(),
            "output_index": output_index,
            "content_index": 0,
            "item_id": self.item_id,
            "part": {"type": "output_text", "text": "", "annotations": []},
        }))
        return results

    def _close_text_block(self) -> list[bytes]:
        """Close the text message block."""
        if not self._text_active:
            return []
        self._text_active = False
        self._text_emitted = True
        output_index = self._text_output_index()
        results = []

        results.append(self._sse("response.output_text.done", {
            "type": "response.output_text.done",
            "sequence_number": self._next_seq(),
            "output_index": output_index,
            "content_index": 0,
            "item_id": self.item_id,
            "text": self.full_content,
        }))
        results.append(self._sse("response.content_part.done", {
            "type": "response.content_part.done",
            "sequence_number": self._next_seq(),
            "output_index": output_index,
            "content_index": 0,
            "item_id": self.item_id,
            "part": {"type": "output_text", "text": self.full_content, "annotations": []},
        }))
        results.append(self._sse("response.output_item.done", {
            "type": "response.output_item.done",
            "sequence_number": self._next_seq(),
            "output_index": output_index,
            "item": {
                "type": "message",
                "id": self.item_id,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": self.full_content, "annotations": []}],
            },
        }))
        return results

    def _process_tool_calls(self, tool_calls: list) -> list[bytes]:
        results = []
        for tc in tool_calls:
            tc_index = tc.get("index", 0)
            tc_func = tc.get("function", {})
            tc_id = tc.get("id", "")
            tc_name = tc_func.get("name", "")
            tc_args = tc_func.get("arguments", "")

            if tc_index not in self.tool_calls:
                self.tool_calls[tc_index] = {"id": tc_id, "name": tc_name, "arguments": ""}
                output_index = self._tool_output_index(tc_index)
                results.append(self._sse("response.output_item.added", {
                    "type": "response.output_item.added",
                    "sequence_number": self._next_seq(),
                    "output_index": output_index,
                    "item": {
                        "type": "function_call",
                        "id": f"fc_{tc_id}",
                        "call_id": tc_id,
                        "name": tc_name,
                        "arguments": "",
                        "status": "in_progress",
                    },
                }))
            else:
                if tc_id:
                    self.tool_calls[tc_index]["id"] = tc_id
                if tc_name:
                    self.tool_calls[tc_index]["name"] = tc_name

            if tc_args:
                self.tool_calls[tc_index]["arguments"] += tc_args
                output_index = self._tool_output_index(tc_index)
                tc_id_resolved = self.tool_calls[tc_index]["id"]
                results.append(self._sse("response.function_call_arguments.delta", {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": self._next_seq(),
                    "output_index": output_index,
                    "item_id": f"fc_{tc_id_resolved}",
                    "call_id": tc_id_resolved,
                    "delta": tc_args,
                }))
        return results

    def _on_finish(self) -> list[bytes]:
        results = []
        if self._reasoning_active:
            results.extend(self._close_reasoning_block())
        if self._text_active:
            results.extend(self._close_text_block())
        for tc_index in sorted(self.tool_calls.keys()):
            results.extend(self._build_tool_call_done_events(tc_index, self.tool_calls[tc_index]))
        return results

    def _build_tool_call_done_events(self, tc_index: int, tc_data: dict) -> list[bytes]:
        """Build done events for a single tool call, handling custom/namespace remapping."""
        results = []
        tc_id = tc_data["id"]
        tc_name = tc_data["name"]
        tc_args = tc_data["arguments"]
        output_index = self._tool_output_index(tc_index)

        if self._tool_ctx and self._tool_ctx.is_custom_proxy(tc_name):
            reconstructed = self._tool_ctx.reconstruct_input(tc_name, tc_args)
            original = self._tool_ctx.original_name(tc_name)
            item_id = f"ctc_{tc_id}"
            # custom_tool_call_input.delta
            results.append(self._sse("response.custom_tool_call_input.delta", {
                "type": "response.custom_tool_call_input.delta",
                "sequence_number": self._next_seq(),
                "output_index": output_index,
                "item_id": item_id,
                "call_id": tc_id,
                "delta": reconstructed,
            }))
            # output_item.done for custom_tool_call
            results.append(self._sse("response.output_item.done", {
                "type": "response.output_item.done",
                "sequence_number": self._next_seq(),
                "output_index": output_index,
                "item": {
                    "id": item_id,
                    "type": "custom_tool_call",
                    "status": "completed",
                    "call_id": tc_id,
                    "name": original,
                    "input": reconstructed,
                },
            }))
        else:
            display_name = tc_name
            namespace = ""
            if self._tool_ctx:
                display_name, namespace = self._tool_ctx.unflatten_namespace(tc_name)
            # function_call_arguments.done
            results.append(self._sse("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "sequence_number": self._next_seq(),
                "output_index": output_index,
                "item_id": f"fc_{tc_id}",
                "call_id": tc_id,
                "arguments": tc_args,
            }))
            item = {
                "type": "function_call",
                "id": f"fc_{tc_id}",
                "call_id": tc_id,
                "name": display_name,
                "arguments": tc_args,
                "status": "completed",
            }
            if namespace:
                item["namespace"] = namespace
            results.append(self._sse("response.output_item.done", {
                "type": "response.output_item.done",
                "sequence_number": self._next_seq(),
                "output_index": output_index,
                "item": item,
            }))
        return results

    def _build_tool_call_output_item(self, tc_data: dict) -> dict:
        """Build output item dict for response.completed's output array."""
        tc_id = tc_data["id"]
        tc_name = tc_data["name"]
        tc_args = tc_data["arguments"]

        if self._tool_ctx and self._tool_ctx.is_custom_proxy(tc_name):
            reconstructed = self._tool_ctx.reconstruct_input(tc_name, tc_args)
            original = self._tool_ctx.original_name(tc_name)
            return {
                "type": "custom_tool_call",
                "id": f"ctc_{tc_id}",
                "call_id": tc_id,
                "name": original,
                "input": reconstructed,
                "status": "completed",
            }

        display_name = tc_name
        namespace = ""
        if self._tool_ctx:
            display_name, namespace = self._tool_ctx.unflatten_namespace(tc_name)
        item = {
            "type": "function_call",
            "id": f"fc_{tc_id}",
            "call_id": tc_id,
            "name": display_name,
            "arguments": tc_args,
            "status": "completed",
        }
        if namespace:
            item["namespace"] = namespace
        return item

    def _on_done(self) -> list[bytes]:
        results = []

        # Close any remaining open blocks
        if self._reasoning_active:
            results.extend(self._close_reasoning_block())

        # If no text block was opened, open a minimal one
        if not self._text_active and not self._text_emitted and not self.tool_calls:
            self._text_active = True
            results.extend(self._open_text_block())
            self._text_active = False

        if self._text_active:
            results.extend(self._close_text_block())

        # Close any tool call blocks not yet closed
        if not self._finished:
            for tc_index in sorted(self.tool_calls.keys()):
                results.extend(self._build_tool_call_done_events(tc_index, self.tool_calls[tc_index]))

        # Build final output array for response.completed
        outputs = []
        if self.full_reasoning or self._reasoning_part_added:
            outputs.append({
                "id": self.reasoning_item_id,
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": self.full_reasoning}],
                "encrypted_content": self.full_reasoning,
            })
        if self.item_id:
            outputs.append({
                "type": "message",
                "id": self.item_id,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": self.full_content, "annotations": []}],
            })
        for tc_index in sorted(self.tool_calls.keys()):
            outputs.append(self._build_tool_call_output_item(self.tool_calls[tc_index]))

        # Store reasoning for multi-turn restoration
        tc_ids = [self.tool_calls[i]["id"] for i in sorted(self.tool_calls.keys())] if self.tool_calls else None
        _store_reasoning(self.full_content, self.full_reasoning, tc_ids)

        # Determine status
        status = "completed"
        if self._finish_reason == "length":
            status = "incomplete"

        if self.response_id:
            resp_obj = self._build_envelope(status, outputs, _normalize_usage(self._usage))
            # If reasoning was generated, set proper effort so Codex displays it
            if self.full_reasoning:
                resp_obj["reasoning"] = {"effort": "high", "summary": "auto"}
            results.append(self._sse("response.completed", {
                "type": "response.completed",
                "sequence_number": self._next_seq(),
                "response": resp_obj,
            }))

        results.append(b"data: [DONE]\n\n")
        return results


# ---------------------------------------------------------------------------
# HTTP Handlers
# ---------------------------------------------------------------------------

async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def handle_responses(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return web.json_response({"error": "invalid_json"}, status=400)

    chat_body = convert_responses_to_chat(body)
    is_stream = body.get("stream", False)

    log.info("Request: model=%s stream=%s tools=%d reasoning=%s",
             body.get("model"), is_stream, len(body.get("tools", [])), body.get("reasoning"))
    # Save request for debugging
    with open(os.path.join(os.path.dirname(__file__), "last_request.json"), "w") as f:
        json.dump(body, f, ensure_ascii=False, indent=2, default=str)
    log.debug("Converted body: %s", json.dumps(chat_body, ensure_ascii=False)[:2000])

    session: aiohttp.ClientSession = request.app["session"]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GLM_API_KEY}",
        "Accept": "text/event-stream" if is_stream else "application/json",
    }

    url = f"{GLM_API_BASE}/chat/completions"
    log.info("Forwarding to: %s (stream=%s)", url, is_stream)

    try:
        async with session.post(url, json=chat_body, headers=headers) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                log.error("GLM API error: %d - %s", resp.status, error_body[:500])
                return web.json_response(
                    {"error": {"message": error_body, "type": "upstream_error", "code": resp.status}},
                    status=resp.status,
                )

            if is_stream:
                return await _stream_to_client(request, resp, body)

            response_body = await resp.json()
            log.debug("GLM response: %s", json.dumps(response_body, ensure_ascii=False)[:2000])
            converted = convert_chat_to_responses(response_body, body=body)
            return web.json_response(converted)

    except asyncio.TimeoutError:
        log.error("Upstream request timeout")
        return web.json_response({"error": "upstream_timeout"}, status=504)
    except aiohttp.ClientError as e:
        log.error("Upstream connection error: %s", e)
        return web.json_response({"error": str(e)}, status=502)


async def _stream_to_client(request: web.Request, upstream: aiohttp.ClientResponse,
                            req_body: dict) -> web.StreamResponse:
    downstream = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await downstream.prepare(request)

    converter = StreamConverter(body=req_body)
    chunk_count = 0
    completed = False

    try:
        while True:
            line = await upstream.content.readline()
            if not line:
                break
            if not line.strip():
                continue
            events = converter.process_line(line)
            for event_bytes in events:
                try:
                    await downstream.write(event_bytes)
                except (ConnectionResetError, ConnectionError):
                    log.warning("Client disconnected during streaming")
                    return downstream
            # Check if converter sent its own [DONE]
            if events and events[-1] == b"data: [DONE]\n\n":
                completed = True
                break
            chunk_count += 1
    except Exception as e:
        log.error("Streaming error: %s", e)

    # Synthesize response.completed if upstream stream ended abnormally
    if not completed:
        try:
            synth = converter._on_done()
            for event_bytes in synth:
                try:
                    await downstream.write(event_bytes)
                except (ConnectionResetError, ConnectionError):
                    break
        except Exception as e:
            log.error("Failed to synthesize completion: %s", e)

    log.info("Streaming complete: %d chunks, model=%s", chunk_count, converter.model)
    return downstream


async def handle_forward(request: web.Request) -> web.Response:
    session: aiohttp.ClientSession = request.app["session"]
    path = request.path
    if path.startswith("/v4/"):
        path = path[3:]
    elif path.startswith("/v1/"):
        path = path[3:]

    url = f"{GLM_API_BASE}{path}"
    headers = {
        "Content-Type": request.content_type or "application/json",
        "Authorization": f"Bearer {GLM_API_KEY}",
    }

    body = await request.read()

    try:
        async with session.request(request.method, url, data=body, headers=headers) as resp:
            content_type = resp.headers.get("Content-Type", "application/json")

            if "text/event-stream" in content_type:
                downstream = web.StreamResponse(
                    status=resp.status,
                    headers={"Content-Type": content_type},
                )
                await downstream.prepare(request)
                async for chunk in resp.content.iter_any():
                    await downstream.write(chunk)
                return downstream

            resp_body = await resp.read()
            return web.Response(
                body=resp_body,
                status=resp.status,
                content_type=content_type,
            )
    except asyncio.TimeoutError:
        return web.json_response({"error": "upstream_timeout"}, status=504)
    except aiohttp.ClientError as e:
        log.error("Forward error: %s", e)
        return web.json_response({"error": str(e)}, status=502)


# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------

async def on_startup(app: web.Application):
    connector = aiohttp.TCPConnector(
        limit=100,
        limit_per_host=20,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )
    timeout = aiohttp.ClientTimeout(total=120, connect=10, sock_read=60)
    app["session"] = aiohttp.ClientSession(connector=connector, timeout=timeout)
    log.info("Connection pool initialized")


async def on_shutdown(app: web.Application):
    log.info("Shutting down, closing connections...")
    await app["session"].close()
    log.info("Done")


def create_app() -> web.Application:
    app = web.Application(client_max_size=10 * 1024 * 1024)  # 10MB

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    app.router.add_get("/health", handle_health)
    app.router.add_post("/v4/responses", handle_responses)
    app.router.add_post("/v4/chat/completions", handle_forward)
    app.router.add_get("/v4/models", handle_forward)
    app.router.add_get("/v1/models", handle_forward)
    app.router.add_post("/v1/chat/completions", handle_forward)
    app.router.add_post("/{path:.*}", handle_forward)

    return app


def main():
    if not GLM_API_KEY:
        log.error("GLM_API_KEY environment variable is required")
        sys.exit(1)

    app = create_app()
    log.info("Codex-GLM proxy starting on port %d", PROXY_PORT)
    log.info("GLM API base: %s", GLM_API_BASE)
    web.run_app(app, host="0.0.0.0", port=PROXY_PORT, print=None, access_log=None)


if __name__ == "__main__":
    main()
