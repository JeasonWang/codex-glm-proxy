#!/usr/bin/env python3
"""
OpenAI Responses API -> GLM Chat Completions API Proxy

Converts the new Responses API format to the traditional Chat Completions format
so that Codex can work with GLM (智谱 AI) models.
"""

import json
import http.server
import socketserver
import http.client
import urllib.request
import urllib.error
import urllib.parse
import os
import sys
import logging

# Unbuffered stdout for logging
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# Configuration
GLM_API_BASE = os.environ.get("GLM_API_BASE", "https://open.bigmodel.cn/api/coding/paas/v4")
GLM_API_KEY = os.environ.get("GLM_API_KEY", "")
PROXY_PORT = int(os.environ.get("PROXY_PORT", 18765))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("codex-glm-proxy")


def convert_responses_to_chat(body: dict) -> dict:
    """Convert Responses API format to Chat Completions API format."""
    chat_body = {}

    # Model mapping
    model = body.get("model", "gpt-4")
    # Map OpenAI model names to GLM equivalents if needed
    model_mapping = {
	"glm-5": "glm-5",
        "gpt-4": "glm-4",
        "gpt-4-turbo": "glm-4",
        "gpt-4o": "glm-5",  # Use glm-5 for best coding performance
        "gpt-4o-mini": "glm-4-flash",
        "gpt-3.5-turbo": "glm-4-flash",
        "gpt-5.2-codex": "glm-5",
        "gpt-5.3-codex": "glm-5",
    }
    chat_body["model"] = model_mapping.get(model, "glm-5")  # Default to glm-5

    messages = []

    # Convert instructions to system message
    if "instructions" in body and body["instructions"]:
        messages.append({"role": "system", "content": body["instructions"]})

    # Convert input to messages
    if "input" in body:
        inp = body["input"]
        if isinstance(inp, str):
            messages.append({"role": "user", "content": inp})
        elif isinstance(inp, list):
            # Responses API format: list of message objects
            for item in inp:
                if isinstance(item, dict) and "type" in item:
                    if item["type"] == "message":
                        role = item.get("role", "user")
                        # Map "developer" to "system" for GLM compatibility
                        if role == "developer":
                            role = "system"
                        
                        content = item.get("content", [])
                        if isinstance(content, list):
                            # Extract text from content blocks
                            text_parts = []
                            for c in content:
                                if isinstance(c, dict):
                                    if c.get("type") == "input_text":
                                        text_parts.append(c.get("text", ""))
                                    elif c.get("type") == "input_image":
                                        # Skip images for now, or handle differently
                                        pass
                            if text_parts:
                                messages.append({"role": role, "content": " ".join(text_parts)})
                        elif isinstance(content, str):
                            messages.append({"role": role, "content": content})
                    
                    elif item["type"] == "function_call":
                        # This is a historical tool call from the model
                        # Convert to assistant message with tool_calls
                        call_id = item.get("call_id", item.get("id", ""))
                        name = item.get("name", "")
                        arguments = item.get("arguments", "{}")
                        
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": arguments
                                }
                            }]
                        })
                    
                    elif item["type"] == "function_call_output":
                        # This is the result of a tool call
                        # Convert to tool message
                        call_id = item.get("call_id", "")
                        output = item.get("output", "")
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": output
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

    # Pass through other fields
    for key in ["temperature", "top_p", "max_tokens", "stream", "frequency_penalty", "presence_penalty", "stop"]:
        if key in body:
            chat_body[key] = body[key]

    # Handle tools - convert Responses API format to Chat Completions format
    if "tools" in body:
        chat_tools = []
        for tool in body["tools"]:
            if isinstance(tool, dict):
                tool_type = tool.get("type", "")
                # Skip tools that GLM doesn't support
                if tool_type in ["web_search", "code_interpreter", "file_search", "computer_use"]:
                    log.info(f"Skipping unsupported tool type: {tool_type}")
                    continue
                    
                # Responses API uses different tool format
                if tool_type == "function":
                    # Already in chat format
                    if "function" in tool:
                        chat_tools.append(tool)
                    # Responses format - function definition is at top level
                    else:
                        chat_tool = {"type": "function", "function": {}}
                        if "name" in tool:
                            chat_tool["function"]["name"] = tool["name"]
                        if "description" in tool:
                            chat_tool["function"]["description"] = tool["description"]
                        if "parameters" in tool:
                            chat_tool["function"]["parameters"] = tool["parameters"]
                        chat_tools.append(chat_tool)
                else:
                    # Unknown format, try to pass through but only if function is present
                    if "function" in tool:
                        chat_tools.append(tool)
        if chat_tools:
            chat_body["tools"] = chat_tools
            log.info(f"Converted tools: {len(chat_tools)} tools (filtered from {len(body['tools'])})")

    if "tool_choice" in body:
        chat_body["tool_choice"] = body["tool_choice"]

    # Handle reasoning/extended thinking
    if "reasoning" in body:
        # GLM may not support this, but pass it through
        chat_body["reasoning"] = body["reasoning"]

    return chat_body


def convert_chat_to_responses(response_body: dict, is_stream: bool) -> dict:
    """Convert Chat Completions response back to Responses format."""
    if is_stream:
        # For streaming, the format is similar but with different event types
        return response_body

    # Responses API format:
    # {
    #   "id": "resp_xxx",
    #   "object": "response",
    #   "output": [
    #     {
    #       "type": "message",
    #       "id": "msg_xxx",
    #       "status": "completed",
    #       "role": "assistant",
    #       "content": [
    #         {"type": "output_text", "text": "..."}
    #       ]
    #     }
    #   ],
    #   "usage": {...}
    # }
    
    outputs = []
    if "choices" in response_body:
        for choice in response_body["choices"]:
            msg = choice.get("message", {})
            content_text = msg.get("content", "")
            
            # Build content array
            content = []
            if content_text:
                content.append({
                    "type": "output_text",
                    "text": content_text
                })
            
            # Handle tool calls
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    content.append({
                        "type": "tool_call",
                        "id": tc.get("id", ""),
                        "call_id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", "{}")
                    })
            
            output_item = {
                "type": "message",
                "id": f"msg_{response_body.get('id', '')}",
                "status": "completed",
                "role": msg.get("role", "assistant"),
                "content": content,
            }
            
            outputs.append(output_item)
    
    responses_body = {
        "id": response_body.get("id", ""),
        "object": "response",
        "created": response_body.get("created", 0),
        "model": response_body.get("model", ""),
        "output": outputs,
        "usage": response_body.get("usage", {}),
        "status": "completed",
    }

    return responses_body


def convert_stream_line(line: bytes) -> bytes:
    """Convert a single SSE line from Chat to Responses format."""
    if not line.startswith(b"data: "):
        return line

    data = line[6:].strip()
    if data == b"[DONE]":
        return b"data: [DONE]\n\n"

    try:
        chunk = json.loads(data)

        # Transform the chunk format
        response_chunk = {
            "id": chunk.get("id", ""),
            "object": "response.chunk",
            "created": chunk.get("created", 0),
            "model": chunk.get("model", ""),
            "output": []
        }

        if "choices" in chunk:
            for choice in chunk["choices"]:
                delta = choice.get("delta", {})
                response_chunk["output"].append({
                    "index": choice.get("index", 0),
                    "delta": delta,
                    "finish_reason": choice.get("finish_reason"),
                })

        return f"data: {json.dumps(response_chunk)}\n\n".encode()
    except json.JSONDecodeError:
        return line


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Thread-per-request HTTP server."""
    daemon_threads = True
    allow_reuse_address = True


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        log.info(format, *args)

    def do_GET(self):
        """Handle health checks."""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        elif self.path == "/v4/models" or self.path == "/v1/models":
            self.forward_request("GET")
        else:
            self.send_response(404)
            self.send_header("Connection", "close")
            self.end_headers()

    def do_POST(self):
        """Handle POST requests - main proxy logic."""
        if self.path.endswith("/responses"):
            self.handle_responses()
        elif self.path.endswith("/chat/completions"):
            self.forward_request("POST")
        else:
            self.forward_request("POST")

    def handle_responses(self):
        """Convert Responses API to Chat Completions and proxy."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body_data = self.rfile.read(content_length)
            body = json.loads(body_data)

            log.info(f"Raw request body: {json.dumps(body, ensure_ascii=False, indent=2)[:10000]}")

            # Convert to Chat Completions format
            chat_body = convert_responses_to_chat(body)
            is_stream = body.get("stream", False)

            log.info(f"Stream mode: {is_stream}")
            log.info(f"Converted chat_body: {json.dumps(chat_body, ensure_ascii=False, indent=2)[:2000]}")

            # Forward to GLM
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GLM_API_KEY}",
                "Accept": "text/event-stream" if is_stream else "application/json",
            }
            
            # Use http.client for proper streaming support
            url_parts = urllib.parse.urlparse(GLM_API_BASE)
            conn = http.client.HTTPSConnection(url_parts.netloc, timeout=120)
            
            log.info(f"Forwarding to GLM: {GLM_API_BASE}/chat/completions (stream={is_stream})")
            
            try:
                conn.request("POST", f"{url_parts.path}/chat/completions", 
                            body=json.dumps(chat_body).encode(), headers=headers)
                glm_resp = conn.getresponse()
                
                if is_stream:
                    self.stream_response(glm_resp)
                else:
                    response_body = json.loads(glm_resp.read())
                    log.info(f"GLM response: {json.dumps(response_body, ensure_ascii=False)[:2000]}")
                    converted = convert_chat_to_responses(response_body, False)
                    log.info(f"Converted response: {json.dumps(converted, ensure_ascii=False)[:2000]}")

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(json.dumps(converted).encode())
            finally:
                conn.close()

        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            log.error(f"GLM API error: {e.code} - {error_body}")
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(error_body.encode())

        except Exception as e:
            log.error(f"Proxy error: {e}")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def stream_response(self, glm_response):
        """Handle streaming SSE response from GLM and convert to Responses format."""
        # Reset state for this request
        self.sequence_number = 0
        self.item_id = None
        self.response_id = None
        self.created_at = None
        self.model = None
        self.full_content = ""
        self.content_part_id = None
        self.tool_calls = {}  # Track tool calls by index
        self.current_tool_index = 0
        
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        
        log.info("Starting streaming response...")
        chunk_count = 0

        try:
            # Read line by line from GLM response
            buffer = b""
            while True:
                chunk = glm_response.read(1)
                if not chunk:
                    break
                    
                buffer += chunk
                if chunk == b"\n":
                    line = buffer.strip()
                    buffer = b""
                    
                    if not line:
                        continue
                    
                    # Convert the line
                    converted_lines = self.convert_stream_line(line)
                    for converted in converted_lines:
                        self.wfile.write(converted)
                        self.wfile.flush()
                        chunk_count += 1
        
            log.info(f"Streaming complete, sent {chunk_count} chunks")
            
        except Exception as e:
            log.error(f"Streaming error: {e}")

    def convert_stream_line(self, line: bytes) -> list:
        """Convert a single SSE line from Chat Completions to Responses format.
        
        Returns a list of SSE lines to send.
        """
        results = []
        
        if not line.startswith(b"data: "):
            return [line + b"\n"]

        data = line[6:].strip()
        if data == b"[DONE]":
            # Build output array for completed event
            outputs = []
            
            # Add message output if there was content
            if self.full_content and self.item_id:
                outputs.append({
                    "type": "message",
                    "id": self.item_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": self.full_content
                    }]
                })
            
            # Add function_call outputs
            for tc_index, tc_data in self.tool_calls.items():
                outputs.append({
                    "type": "function_call",
                    "id": f"fc_{tc_data['id']}",
                    "call_id": tc_data["id"],
                    "name": tc_data["name"],
                    "arguments": tc_data["arguments"],
                    "status": "completed"
                })
            
            # Send response.completed event before DONE
            if self.response_id:
                completed_event = {
                    "type": "response.completed",
                    "sequence_number": self.sequence_number,
                    "response": {
                        "id": self.response_id,
                        "object": "response",
                        "created_at": self.created_at or 0,
                        "model": self.model or "",
                        "output": outputs,
                        "status": "completed"
                    }
                }
                self.sequence_number += 1
                results.append(f"event: response.completed\ndata: {json.dumps(completed_event)}\n\n".encode())
            
            results.append(b"data: [DONE]\n\n")
            return results

        try:
            chunk = json.loads(data)
            
            # Store response metadata from first chunk
            if not self.item_id:
                self.response_id = chunk.get("id", "")
                # Ensure ID format matches OpenAI's format
                if not self.response_id.startswith("resp_"):
                    self.response_id = f"resp_{self.response_id}"
                self.created_at = chunk.get("created", 0)
                self.model = chunk.get("model", "")
                self.item_id = f"msg_{self.response_id}"
                self.content_part_id = f"cp_{self.response_id}"
                
                # Send response.created event
                created_event = {
                    "type": "response.created",
                    "sequence_number": self.sequence_number,
                    "response": {
                        "id": self.response_id,
                        "object": "response",
                        "created_at": self.created_at,
                        "model": self.model,
                        "output": [],
                        "status": "in_progress"
                    }
                }
                self.sequence_number += 1
                results.append(f"event: response.created\ndata: {json.dumps(created_event)}\n\n".encode())
                
                # Send output_item.added event
                item_added_event = {
                    "type": "response.output_item.added",
                    "sequence_number": self.sequence_number,
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "id": self.item_id,
                        "status": "in_progress",
                        "role": "assistant",
                        "content": []
                    }
                }
                self.sequence_number += 1
                results.append(f"event: response.output_item.added\ndata: {json.dumps(item_added_event)}\n\n".encode())
                
                # Send content_part.added event
                content_part_event = {
                    "type": "response.content_part.added",
                    "sequence_number": self.sequence_number,
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": self.item_id,
                    "content_part": {
                        "type": "output_text",
                        "text": ""
                    }
                }
                self.sequence_number += 1
                results.append(f"event: response.content_part.added\ndata: {json.dumps(content_part_event)}\n\n".encode())

            if "choices" in chunk:
                for choice in chunk["choices"]:
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    finish_reason = choice.get("finish_reason")
                    
                    if content:
                        # Send response.output_text.delta event
                        self.full_content += content
                        delta_event = {
                            "type": "response.output_text.delta",
                            "sequence_number": self.sequence_number,
                            "output_index": 0,
                            "content_index": 0,
                            "item_id": self.item_id,
                            "delta": content,
                            "logprobs": []  # Required field
                        }
                        self.sequence_number += 1
                        results.append(f"event: response.output_text.delta\ndata: {json.dumps(delta_event)}\n\n".encode())
                    
                    # Handle tool calls in delta
                    if "tool_calls" in delta:
                        for tc in delta["tool_calls"]:
                            tc_index = tc.get("index", 0)
                            tc_id = tc.get("id", "")
                            tc_function = tc.get("function", {})
                            tc_name = tc_function.get("name", "")
                            tc_args = tc_function.get("arguments", "")
                            
                            # If this is a new tool call, send output_item.added event
                            if tc_index not in self.tool_calls:
                                self.tool_calls[tc_index] = {
                                    "id": tc_id,
                                    "name": tc_name,
                                    "arguments": ""
                                }
                                
                                # Send function_call item added event
                                tool_item_event = {
                                    "type": "response.output_item.added",
                                    "sequence_number": self.sequence_number,
                                    "output_index": tc_index + 1,  # After text output
                                    "item": {
                                        "type": "function_call",
                                        "id": f"fc_{tc_id}",
                                        "call_id": tc_id,
                                        "name": tc_name,
                                        "arguments": "",
                                        "status": "in_progress"
                                    }
                                }
                                self.sequence_number += 1
                                results.append(f"event: response.output_item.added\ndata: {json.dumps(tool_item_event)}\n\n".encode())
                            
                            # Send function_call_arguments.delta event
                            if tc_args:
                                self.tool_calls[tc_index]["arguments"] += tc_args
                                tool_delta_event = {
                                    "type": "response.function_call_arguments.delta",
                                    "sequence_number": self.sequence_number,
                                    "output_index": tc_index + 1,
                                    "item_id": f"fc_{tc_id}",
                                    "delta": tc_args,
                                    "call_id": tc_id
                                }
                                self.sequence_number += 1
                                results.append(f"event: response.function_call_arguments.delta\ndata: {json.dumps(tool_delta_event)}\n\n".encode())
                    
                    if finish_reason:
                        # If there are tool calls, send done events for them
                        if finish_reason == "tool_calls" and self.tool_calls:
                            for tc_index, tc_data in self.tool_calls.items():
                                tc_id = tc_data["id"]
                                tc_name = tc_data["name"]
                                tc_args = tc_data["arguments"]
                                
                                # Send function_call_arguments.done event
                                tool_done_event = {
                                    "type": "response.function_call_arguments.done",
                                    "sequence_number": self.sequence_number,
                                    "output_index": tc_index + 1,
                                    "item_id": f"fc_{tc_id}",
                                    "arguments": tc_args,
                                    "call_id": tc_id
                                }
                                self.sequence_number += 1
                                results.append(f"event: response.function_call_arguments.done\ndata: {json.dumps(tool_done_event)}\n\n".encode())
                                
                                # Send output_item.done for function_call
                                tool_item_done = {
                                    "type": "response.output_item.done",
                                    "sequence_number": self.sequence_number,
                                    "output_index": tc_index + 1,
                                    "item": {
                                        "type": "function_call",
                                        "id": f"fc_{tc_id}",
                                        "call_id": tc_id,
                                        "name": tc_name,
                                        "arguments": tc_args,
                                        "status": "completed"
                                    }
                                }
                                self.sequence_number += 1
                                results.append(f"event: response.output_item.done\ndata: {json.dumps(tool_item_done)}\n\n".encode())
                        
                        # Send output_text.done event (if there was text content)
                        if self.full_content:
                            done_event = {
                                "type": "response.output_text.done",
                                "sequence_number": self.sequence_number,
                                "output_index": 0,
                                "content_index": 0,
                                "item_id": self.item_id,
                                "text": self.full_content
                            }
                            self.sequence_number += 1
                            results.append(f"event: response.output_text.done\ndata: {json.dumps(done_event)}\n\n".encode())
                            
                            # Send content_part.done event
                            content_done_event = {
                                "type": "response.content_part.done",
                                "sequence_number": self.sequence_number,
                                "output_index": 0,
                                "content_index": 0,
                                "item_id": self.item_id,
                                "content_part": {
                                    "type": "output_text",
                                    "text": self.full_content
                                }
                            }
                            self.sequence_number += 1
                            results.append(f"event: response.content_part.done\ndata: {json.dumps(content_done_event)}\n\n".encode())
                        
                        # Send output_item.done event for message
                        if self.full_content:
                            item_done_event = {
                                "type": "response.output_item.done",
                                "sequence_number": self.sequence_number,
                                "output_index": 0,
                                "item": {
                                    "type": "message",
                                    "id": self.item_id,
                                    "status": "completed",
                                    "role": "assistant",
                                    "content": [{
                                        "type": "output_text",
                                        "text": self.full_content
                                    }]
                                }
                            }
                            self.sequence_number += 1
                            results.append(f"event: response.output_item.done\ndata: {json.dumps(item_done_event)}\n\n".encode())

            return results
            
        except json.JSONDecodeError as e:
            log.error(f"Failed to parse chunk: {e}, line: {line}")
            return [line + b"\n"]

    def forward_request(self, method):
        """Forward request directly without conversion."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else None

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GLM_API_KEY}",
            }

            path = self.path
            if path.startswith("/v4/"):
                path = path[3:]  # Remove /v4 prefix for GLM

            req = urllib.request.Request(
                f"{GLM_API_BASE}{path}",
                data=body,
                headers=headers,
                method=method,
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                response_body = resp.read()
                self.send_response(200)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.end_headers()
                self.wfile.write(response_body)

        except Exception as e:
            log.error(f"Forward error: {e}")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())


def main():
    if not GLM_API_KEY:
        log.error("GLM_API_KEY environment variable is required")
        sys.exit(1)

    with ThreadingHTTPServer(("", PROXY_PORT), ProxyHandler) as httpd:
        log.info(f"Codex-GLM proxy running on port {PROXY_PORT}")
        log.info(f"GLM API base: {GLM_API_BASE}")
        log.info("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            log.info("Shutting down...")


if __name__ == "__main__":
    main()
