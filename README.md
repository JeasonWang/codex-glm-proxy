# Codex GLM Proxy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**English** | [中文](README_CN.md)

Enable **OpenAI Codex CLI** to work with **GLM (智谱 AI)** models by running a local proxy that converts OpenAI Responses API format to GLM Chat Completions format.

## ✨ Features

- ✅ **Full Codex Compatibility** - Works seamlessly with OpenAI Codex CLI
- ✅ **Streaming Support** - Real-time streaming responses
- ✅ **Tool Calling** - Supports `apply_patch`, `exec`, and other Codex tools
- ✅ **Multi-turn Conversations** - Maintains conversation context
- ✅ **Automatic Model Mapping** - Maps OpenAI model names to GLM equivalents
- ✅ **Easy Setup** - Single Python file, no complex dependencies

## 📊 Use Cases

### 🎯 Perfect For

- **Chinese Developers** - Use GLM (智谱 AI) instead of OpenAI GPT-4
- **Cost-Conscious Teams** - GLM offers competitive pricing for Chinese users
- **Local Development** - Run Codex with domestic LLM providers
- **Privacy-Focused Projects** - Keep code within China's borders
- **Learning & Education** - Access powerful coding AI without OpenAI account

### 💡 Common Scenarios

| Scenario | Example Command |
|----------|----------------|
| **Quick Prototyping** | `codex exec "Create a REST API with authentication"` |
| **Code Generation** | `codex exec "Generate TypeScript interfaces from JSON schema"` |
| **Bug Fixing** | `codex exec "Fix the null pointer exception in UserService.java"` |
| **Testing** | `codex exec "Add unit tests for the calculator module"` |
| **Documentation** | `codex exec "Generate API documentation from code comments"` |
| **Refactoring** | `codex exec "Refactor this class to use dependency injection"` |

## 🔄 Architecture & Data Flow

### System Architecture

```
┌─────────────────┐
│   Codex CLI     │  Sends: Responses API Request
│   (User Side)   │  Receives: Responses API Response
└────────┬────────┘
         │ Responses API Format
         ▼
┌─────────────────────────────────────────┐
│   Codex GLM Proxy (localhost:18765)    │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Request Converter                 │ │
│  │  - Responses API → Chat Completions│ │
│  │  - Tool call history handling      │ │
│  │  - Model name mapping              │ │
│  └────────────────────────────────────┘ │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Response Converter                │ │
│  │  - Chat Completions → Responses API│ │
│  │  - Tool call streaming             │ │
│  │  - Event sequencing                │ │
│  └────────────────────────────────────┘ │
└────────┬────────────────────────────────┘
         │ Chat Completions API Format
         ▼
┌─────────────────┐
│   GLM API       │  Sends: Chat Completions Request
│   (智谱 AI)     │  Receives: Chat Completions Response
└─────────────────┘
```

### Request Flow

```mermaid
sequenceDiagram
    participant U as User
    participant C as Codex CLI
    participant P as GLM Proxy
    participant G as GLM API

    U->>C: codex exec "Create hello.py"
    C->>P: Responses API Request<br/>(Responses API format)
    Note over P: Convert to Chat Completions API
    P->>G: Chat Completions Request<br/>(GLM format)
    G-->>P: Chat Completions Response<br/>(Streaming)
    Note over P: Convert back to Responses API
    P-->>C: Responses API Response<br/>(Streaming events)
    C-->>U: Display results & apply changes
```

### Key Conversion Points

**1. Request Conversion (Codex → GLM)**
```json
// OpenAI Responses API (from Codex)
{
  "model": "gpt-4o",
  "input": [
    {"type": "message", "role": "user", "content": "..."},
    {"type": "function_call", "call_id": "...", "name": "exec"}
  ]
}

        ⬇️ Proxy Converts ⬇️

// GLM Chat Completions (to GLM)
{
  "model": "glm-4-plus",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "tool_calls": [...]}
  ],
  "tools": [...]
}
```

**2. Response Conversion (GLM → Codex)**
```
GLM Chat Completions     →  Responses API Event
────────────────────         ────────────────────────
delta.content            →  response.output_text.delta
delta.tool_calls         →  response.function_call_arguments.delta
finish_reason            →  response.completed
```

**Note:** Codex CLI sends and receives **Responses API format**, while GLM API uses **Chat Completions format**. The proxy handles the bidirectional conversion.

### Tool Call Flow

```mermaid
graph LR
    A[Codex Request] --> B[Proxy]
    B --> C{Tool Call?}
    C -->|Yes| D[Execute Tool]
    D --> E[Return Result]
    E --> B
    C -->|No| F[Stream to GLM]
    F --> G[Get Response]
    G --> H[Convert & Return]
    H --> A
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- GLM API key ([Get one here](https://open.bigmodel.cn/))
- [OpenAI Codex CLI](https://github.com/openai/codex) installed

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/JichinX/codex-glm-proxy.git
   cd codex-glm-proxy
   ```

2. **Set your GLM API key**
   ```bash
   export GLM_API_KEY="your_glm_api_key_here"
   ```

3. **Start the proxy**
   ```bash
   python3 proxy.py
   # 或使用便捷脚本
   ./scripts/start.sh
   ```

   Proxy will run on `http://localhost:18765`

4. **Configure Codex CLI**
   
   Create or update `~/.codex/config.toml`:
   ```toml
   model_provider = "glm-proxy"
   model = "gpt-4o"
   
   [model_providers.glm-proxy]
   name = "GLM via Proxy"
   base_url = "http://localhost:18765/v4"
   wire_api = "responses"
   ```

5. **Test it!**
   ```bash
   mkdir test-codex && cd test-codex && git init
   codex exec "Create a Python hello world program" --full-auto
   ```

## 📋 Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `GLM_API_KEY` | *(required)* | Your GLM API key |
| `GLM_API_BASE` | `https://open.bigmodel.cn/api/coding/paas/v4` | GLM API endpoint |
| `PROXY_PORT` | `18765` | Local proxy port |

## 🗺️ Model Mapping

The proxy automatically maps OpenAI model names to GLM equivalents:

| OpenAI Model | GLM Model | Notes |
|--------------|-----------|-------|
| `gpt-4` | `glm-4` | Standard GPT-4 |
| `gpt-4-turbo` | `glm-4` | GPT-4 Turbo |
| `gpt-4o` | `glm-4-plus` | **Recommended** for best coding |
| `gpt-4o-mini` | `glm-4-flash` | Faster, cheaper |
| `gpt-3.5-turbo` | `glm-4-flash` | Legacy support |
| `gpt-5.x-codex` | `glm-5` | Future Codex models |

**Recommendation:** Use `model = "gpt-4o"` in your Codex config for best results.

## 🔧 Management

```bash
# Start proxy (background)
./scripts/start.sh

# Check if running
curl http://localhost:18765/health

# View logs
tail -f /tmp/codex-glm-proxy.log

# Stop proxy
./scripts/stop.sh
```

## 🛠️ How It Works

The proxy sits between Codex CLI and GLM API, converting:

```
Codex CLI → Responses API → Proxy → Chat Completions API → GLM
```

### Request Conversion
- Converts OpenAI Responses API format to GLM Chat Completions
- Handles tool call history and function results
- Filters unsupported tools

### Response Conversion
- Streams Chat Completions responses back to Responses API format
- Maintains proper event sequencing
- Handles tool call streaming

## 📝 Example Usage

```bash
# Simple task
codex exec "Create a Python function to calculate Fibonacci" --full-auto

# More complex project
codex exec "Build a REST API with FastAPI for todo management" --full-auto

# With tests
codex exec "Create a calculator module with unit tests" --full-auto
```

## 🐛 Troubleshooting

### "Streaming complete, sent 0 chunks"
**Cause:** Model name not properly mapped  
**Solution:** Ensure you're using a known model like `gpt-4o` in config

### Codex loops / repeats actions
**Cause:** Tool call history not properly handled  
**Solution:** Update to latest version of proxy

### 502 Bad Gateway
**Cause:** Proxy crashed  
**Solution:** Check logs at `/tmp/codex-glm-proxy.log` and restart

### Connection refused
**Cause:** Proxy not running  
**Solution:** Start proxy with `./scripts/start.sh`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Codex](https://github.com/openai/codex) - The amazing coding agent
- [智谱 AI GLM](https://open.bigmodel.cn/) - Powerful Chinese LLM
- Inspired by the need to use Codex with local/alternative LLM providers

## 📊 Project Status

✅ **Production Ready** - Fully tested with all Codex features

| Feature | Status |
|---------|--------|
| Text conversations | ✅ Working |
| Model mapping | ✅ Working |
| Streaming responses | ✅ Working |
| Tool calling | ✅ Working |
| Multi-turn conversations | ✅ Working |
| Tool call history | ✅ Working |
| Tool call results | ✅ Working |

---

**Made with ❤️ by the community, for the community**

**Star ⭐ this repo if you find it useful!**
