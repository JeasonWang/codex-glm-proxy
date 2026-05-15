#!/usr/bin/env python3
"""
OpenAI Responses API -> GLM Chat Completions API Proxy

Converts the new Responses API format to the traditional Chat Completions format
so that Codex can work with GLM (智谱 AI) models.

Powered by aiohttp for high-performance async I/O with connection pooling.
"""

import asyncio
import json
import logging
import os
import sys

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
# Conversion: Responses API -> Chat Completions API
# ---------------------------------------------------------------------------

def convert_responses_to_chat(body: dict) -> dict:
    chat_body = {}

    model = body.get("model", "glm-5.1")
    chat_body["model"] = MODEL_MAPPING.get(model, "glm-5.1")

    messages = []

    if "instructions" in body and body["instructions"]:
        messages.append({"role": "system", "content": body["instructions"]})

    if "input" in body:
        inp = body["input"]
        if isinstance(inp, str):
            messages.append({"role": "user", "content": inp})
        elif isinstance(inp, list):
            for item in inp:
                if not isinstance(item, dict) or "type" not in item:
                    continue
                if item["type"] == "message":
                    role = item.get("role", "user")
                    if role == "developer":
                        role = "system"
                    content = item.get("content", [])
                    if isinstance(content, list):
                        text_parts = []
                        for c in content:
                            if isinstance(c, dict):
                                if c.get("type") == "input_text":
                                    text_parts.append(c.get("text", ""))
                        if text_parts:
                            messages.append({"role": role, "content": " ".join(text_parts)})
                    elif isinstance(content, str):
                        messages.append({"role": role, "content": content})

                elif item["type"] == "function_call":
                    call_id = item.get("call_id", item.get("id", ""))
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", "{}"),
                            }
                        }]
                    })

                elif item["type"] == "function_call_output":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": item.get("output", ""),
                    })

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

    for key in ("temperature", "top_p", "max_tokens", "stream", "frequency_penalty",
                "presence_penalty", "stop"):
        if key in body:
            chat_body[key] = body[key]

    if "tools" in body:
        chat_tools = []
        for tool in body["tools"]:
            if not isinstance(tool, dict):
                continue
            tool_type = tool.get("type", "")
            if tool_type in ("web_search", "code_interpreter", "file_search", "computer_use"):
                log.info("Skipping unsupported tool type: %s", tool_type)
                continue
            if tool_type == "function":
                if "function" in tool:
                    chat_tools.append(tool)
                else:
                    chat_tool = {"type": "function", "function": {}}
                    if "name" in tool:
                        chat_tool["function"]["name"] = tool["name"]
                    if "description" in tool:
                        chat_tool["function"]["description"] = tool["description"]
                    if "parameters" in tool:
                        chat_tool["function"]["parameters"] = tool["parameters"]
                    chat_tools.append(chat_tool)
            elif "function" in tool:
                chat_tools.append(tool)
        if chat_tools:
            chat_body["tools"] = chat_tools
            log.info("Converted tools: %d (from %d)", len(chat_tools), len(body["tools"]))

    if "tool_choice" in body:
        chat_body["tool_choice"] = body["tool_choice"]

    if "reasoning" in body:
        chat_body["reasoning"] = body["reasoning"]

    return chat_body


# ---------------------------------------------------------------------------
# Conversion: Chat Completions -> Responses API (non-streaming)
# ---------------------------------------------------------------------------

def convert_chat_to_responses(response_body: dict) -> dict:
    outputs = []
    for choice in response_body.get("choices", []):
        msg = choice.get("message", {})

        # Message output with text content only
        content = []
        if msg.get("content"):
            content.append({"type": "output_text", "text": msg["content"]})

        outputs.append({
            "type": "message",
            "id": f"msg_{response_body.get('id', '')}",
            "status": "completed",
            "role": msg.get("role", "assistant"),
            "content": content,
        })

        # Tool calls as separate output items (matches streaming behavior)
        for tc in msg.get("tool_calls", []):
            outputs.append({
                "type": "function_call",
                "id": f"fc_{tc.get('id', '')}",
                "call_id": tc.get("id", ""),
                "name": tc.get("function", {}).get("name", ""),
                "arguments": tc.get("function", {}).get("arguments", "{}"),
                "status": "completed",
            })

    return {
        "id": response_body.get("id", ""),
        "object": "response",
        "created": response_body.get("created", 0),
        "model": response_body.get("model", ""),
        "output": outputs,
        "usage": response_body.get("usage", {}),
        "status": "completed",
    }


# ---------------------------------------------------------------------------
# Stream Converter: Chat Completions SSE -> Responses API SSE
# ---------------------------------------------------------------------------

class StreamConverter:
    """Stateful converter for one streaming request.

    Call process_line() for each raw SSE line from upstream.
    Returns a list of bytes objects to write downstream.
    """

    def __init__(self):
        self.seq = 0
        self.item_id = None
        self.response_id = None
        self.created_at = None
        self.model = None
        self.full_content = ""
        self.tool_calls = {}
        self._initialized = False
        self._finished = False

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

        if not self._initialized:
            self._initialized = True
            self.response_id = chunk.get("id", "")
            if not self.response_id.startswith("resp_"):
                self.response_id = f"resp_{self.response_id}"
            self.created_at = chunk.get("created", 0)
            self.model = chunk.get("model", "")
            self.item_id = f"msg_{self.response_id}"
            results.extend(self._emit_init_events())

        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            content = delta.get("content", "")
            finish_reason = choice.get("finish_reason")

            if content:
                self.full_content += content
                if len(self.full_content) > MAX_CONTENT_SIZE:
                    self.full_content = self.full_content[-MAX_CONTENT_SIZE:]
                results.append(self._sse("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "sequence_number": self._next_seq(),
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": self.item_id,
                    "delta": content,
                    "logprobs": [],
                }))

            if "tool_calls" in delta:
                results.extend(self._process_tool_calls(delta["tool_calls"]))

            if finish_reason and not self._finished:
                self._finished = True
                results.extend(self._on_finish())

        return results

    # --- private helpers ---

    @staticmethod
    def _sse(event_type: str, data: dict) -> bytes:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()

    def _emit_init_events(self) -> list[bytes]:
        return [
            self._sse("response.created", {
                "type": "response.created",
                "sequence_number": self._next_seq(),
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "created_at": self.created_at,
                    "model": self.model,
                    "output": [],
                    "status": "in_progress",
                },
            }),
            self._sse("response.output_item.added", {
                "type": "response.output_item.added",
                "sequence_number": self._next_seq(),
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": self.item_id,
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            }),
            self._sse("response.content_part.added", {
                "type": "response.content_part.added",
                "sequence_number": self._next_seq(),
                "output_index": 0,
                "content_index": 0,
                "item_id": self.item_id,
                "content_part": {"type": "output_text", "text": ""},
            }),
        ]

    def _process_tool_calls(self, tool_calls: list) -> list[bytes]:
        results = []
        for tc in tool_calls:
            tc_index = tc.get("index", 0)
            tc_id = tc.get("id", "")
            tc_func = tc.get("function", {})
            tc_name = tc_func.get("name", "")
            tc_args = tc_func.get("arguments", "")

            if tc_index not in self.tool_calls:
                self.tool_calls[tc_index] = {"id": tc_id, "name": tc_name, "arguments": ""}
                results.append(self._sse("response.output_item.added", {
                    "type": "response.output_item.added",
                    "sequence_number": self._next_seq(),
                    "output_index": tc_index + 1,
                    "item": {
                        "type": "function_call",
                        "id": f"fc_{tc_id}",
                        "call_id": tc_id,
                        "name": tc_name,
                        "arguments": "",
                        "status": "in_progress",
                    },
                }))

            if tc_args:
                self.tool_calls[tc_index]["arguments"] += tc_args
                results.append(self._sse("response.function_call_arguments.delta", {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": self._next_seq(),
                    "output_index": tc_index + 1,
                    "item_id": f"fc_{tc_id}",
                    "delta": tc_args,
                    "call_id": tc_id,
                }))
        return results

    def _on_finish(self) -> list[bytes]:
        results = []

        # tool call done events
        for tc_index, tc_data in self.tool_calls.items():
            tc_id = tc_data["id"]
            results.append(self._sse("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "sequence_number": self._next_seq(),
                "output_index": tc_index + 1,
                "item_id": f"fc_{tc_id}",
                "arguments": tc_data["arguments"],
                "call_id": tc_id,
            }))
            results.append(self._sse("response.output_item.done", {
                "type": "response.output_item.done",
                "sequence_number": self._next_seq(),
                "output_index": tc_index + 1,
                "item": {
                    "type": "function_call",
                    "id": f"fc_{tc_id}",
                    "call_id": tc_id,
                    "name": tc_data["name"],
                    "arguments": tc_data["arguments"],
                    "status": "completed",
                },
            }))

        # text + message done events (always emit, even with empty content)
        text = self.full_content
        if text:
            results.append(self._sse("response.output_text.done", {
                "type": "response.output_text.done",
                "sequence_number": self._next_seq(),
                "output_index": 0,
                "content_index": 0,
                "item_id": self.item_id,
                "text": text,
            }))
            results.append(self._sse("response.content_part.done", {
                "type": "response.content_part.done",
                "sequence_number": self._next_seq(),
                "output_index": 0,
                "content_index": 0,
                "item_id": self.item_id,
                "content_part": {"type": "output_text", "text": text},
            }))

        results.append(self._sse("response.output_item.done", {
            "type": "response.output_item.done",
            "sequence_number": self._next_seq(),
            "output_index": 0,
            "item": {
                "type": "message",
                "id": self.item_id,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            },
        }))

        return results

    def _on_done(self) -> list[bytes]:
        results = []
        outputs = []

        # Always include message output
        if self.item_id:
            outputs.append({
                "type": "message",
                "id": self.item_id,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": self.full_content}],
            })

        for tc_index, tc_data in self.tool_calls.items():
            outputs.append({
                "type": "function_call",
                "id": f"fc_{tc_data['id']}",
                "call_id": tc_data["id"],
                "name": tc_data["name"],
                "arguments": tc_data["arguments"],
                "status": "completed",
            })

        if self.response_id:
            results.append(self._sse("response.completed", {
                "type": "response.completed",
                "sequence_number": self._next_seq(),
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "created_at": self.created_at or 0,
                    "model": self.model or "",
                    "output": outputs,
                    "status": "completed",
                },
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
                return await _stream_to_client(request, resp)

            response_body = await resp.json()
            log.debug("GLM response: %s", json.dumps(response_body, ensure_ascii=False)[:2000])
            converted = convert_chat_to_responses(response_body)
            return web.json_response(converted)

    except asyncio.TimeoutError:
        log.error("Upstream request timeout")
        return web.json_response({"error": "upstream_timeout"}, status=504)
    except aiohttp.ClientError as e:
        log.error("Upstream connection error: %s", e)
        return web.json_response({"error": str(e)}, status=502)


async def _stream_to_client(request: web.Request, upstream: aiohttp.ClientResponse) -> web.StreamResponse:
    downstream = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await downstream.prepare(request)

    converter = StreamConverter()
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
