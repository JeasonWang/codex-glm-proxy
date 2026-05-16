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
    "gpt-4": "glm-4",
    "gpt-4-turbo": "glm-4",
    "gpt-4o": "glm-5",
    "gpt-4o-mini": "glm-4-flash",
    "gpt-3.5-turbo": "glm-4-flash",
    "gpt-5.2-codex": "glm-5",
    "gpt-5.3-codex": "glm-5",
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
                "stream", "frequency_penalty", "presence_penalty", "stop"):
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
        chat_tools = _convert_tools(body["tools"])
        if chat_tools:
            chat_body["tools"] = chat_tools

    if "tool_choice" in body:
        chat_body["tool_choice"] = body["tool_choice"]

    if "reasoning" in body:
        chat_body["reasoning"] = body["reasoning"]

    return chat_body


def _convert_input_list(items: list, messages: list):
    """Convert Responses API input items to Chat Completions messages.

    Key challenge: In Responses API, an assistant turn with both text and tool_calls
    appears as separate items (message + function_call), but Chat Completions needs
    them merged into a single assistant message with tool_calls.
    """
    pending_assistant = None
    pending_tc: list[dict] = []

    def flush_assistant():
        nonlocal pending_assistant
        if pending_assistant:
            messages.append(pending_assistant)
            pending_assistant = None

    def flush_tool_calls():
        nonlocal pending_tc
        if not pending_tc:
            return
        # Look up stored reasoning by tool_call IDs
        rc = ""
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
            # message or unknown type — flush any pending tool calls first
            flush_assistant()
            flush_tool_calls()

            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            content = item.get("content", "")
            content = _extract_text(content)

            msg = {"role": role}
            if content is not None:
                msg["content"] = content

            # For assistant messages, restore reasoning_content and hold for possible merge
            if role == "assistant":
                stored_rc = _lookup_reasoning(content or "")
                if stored_rc:
                    msg["reasoning_content"] = stored_rc
                msg.setdefault("content", None)
                pending_assistant = msg
                continue

            if msg.get("content") is not None:
                messages.append(msg)

    # Flush any remaining
    if pending_assistant and pending_tc:
        pending_assistant["tool_calls"] = list(pending_tc)
        messages.append(pending_assistant)
    else:
        flush_assistant()
        flush_tool_calls()


def _convert_tools(tools: list) -> list[dict] | None:
    """Convert Responses API tools to Chat Completions tools format."""
    out = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type", "")
        if tool_type in ("web_search", "code_interpreter", "file_search", "computer_use"):
            continue
        if tool_type == "function":
            if "function" in tool:
                out.append(tool)
            else:
                func = {"name": tool.get("name", "")}
                if tool.get("description"):
                    func["description"] = tool["description"]
                if tool.get("parameters"):
                    func["parameters"] = tool["parameters"]
                out.append({"type": "function", "function": func})
        elif "function" in tool:
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
        for tc in msg.get("tool_calls", []):
            tc_id = tc.get("id", "")
            tc_ids.append(tc_id)
            outputs.append({
                "type": "function_call",
                "id": f"fc_{tc_id}",
                "call_id": tc_id,
                "name": tc.get("function", {}).get("name", ""),
                "arguments": tc.get("function", {}).get("arguments", "{}"),
                "status": "completed",
            })

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
        "reasoning": body.get("reasoning", {"effort": None, "summary": None}),
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
        resp_obj.setdefault("reasoning", {"effort": None, "summary": None})
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
        # Close reasoning block
        if self._reasoning_active:
            results.extend(self._close_reasoning_block())
        # Close text block
        if self._text_active:
            results.extend(self._close_text_block())
        # Close tool call blocks
        for tc_index in sorted(self.tool_calls.keys()):
            tc_data = self.tool_calls[tc_index]
            tc_id = tc_data["id"]
            output_index = self._tool_output_index(tc_index)
            results.append(self._sse("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "sequence_number": self._next_seq(),
                "output_index": output_index,
                "item_id": f"fc_{tc_id}",
                "call_id": tc_id,
                "arguments": tc_data["arguments"],
            }))
            results.append(self._sse("response.output_item.done", {
                "type": "response.output_item.done",
                "sequence_number": self._next_seq(),
                "output_index": output_index,
                "item": {
                    "type": "function_call",
                    "id": f"fc_{tc_id}",
                    "call_id": tc_id,
                    "name": tc_data["name"],
                    "arguments": tc_data["arguments"],
                    "status": "completed",
                },
            }))
        return results

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
                tc_data = self.tool_calls[tc_index]
                tc_id = tc_data["id"]
                output_index = self._tool_output_index(tc_index)
                results.append(self._sse("response.function_call_arguments.done", {
                    "type": "response.function_call_arguments.done",
                    "sequence_number": self._next_seq(),
                    "output_index": output_index,
                    "item_id": f"fc_{tc_id}",
                    "call_id": tc_id,
                    "arguments": tc_data["arguments"],
                }))
                results.append(self._sse("response.output_item.done", {
                    "type": "response.output_item.done",
                    "sequence_number": self._next_seq(),
                    "output_index": output_index,
                    "item": {
                        "type": "function_call",
                        "id": f"fc_{tc_id}",
                        "call_id": tc_id,
                        "name": tc_data["name"],
                        "arguments": tc_data["arguments"],
                        "status": "completed",
                    },
                }))

        # Build final output array for response.completed
        outputs = []
        # Reasoning item
        if self.full_reasoning or self._reasoning_part_added:
            outputs.append({
                "id": self.reasoning_item_id,
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": self.full_reasoning}],
                "encrypted_content": self.full_reasoning,
            })
        # Message item
        if self.item_id:
            outputs.append({
                "type": "message",
                "id": self.item_id,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": self.full_content, "annotations": []}],
            })
        # Function call items
        for tc_index in sorted(self.tool_calls.keys()):
            tc_data = self.tool_calls[tc_index]
            outputs.append({
                "type": "function_call",
                "id": f"fc_{tc_data['id']}",
                "call_id": tc_data["id"],
                "name": tc_data["name"],
                "arguments": tc_data["arguments"],
                "status": "completed",
            })

        # Store reasoning for multi-turn restoration
        tc_ids = [self.tool_calls[i]["id"] for i in sorted(self.tool_calls.keys())] if self.tool_calls else None
        _store_reasoning(self.full_content, self.full_reasoning, tc_ids)

        # Determine status
        status = "completed"
        if self._finish_reason == "length":
            status = "incomplete"

        if self.response_id:
            resp_obj = self._build_envelope(status, outputs, _normalize_usage(self._usage))
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

    log.info("Request: model=%s stream=%s tools=%d",
             body.get("model"), is_stream, len(body.get("tools", [])))
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
            chunk_count += 1
    except Exception as e:
        log.error("Streaming error: %s", e)

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
