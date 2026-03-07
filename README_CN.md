# Codex GLM 代理

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[English](README.md) | **中文**

让 **OpenAI Codex CLI** 能够使用 **GLM（智谱 AI）** 模型，通过本地代理将 OpenAI Responses API 格式转换为 GLM Chat Completions 格式。

## ✨ 特性

- ✅ **完整 Codex 兼容** - 与 OpenAI Codex CLI 无缝协作
- ✅ **流式响应支持** - 实时流式响应
- ✅ **工具调用** - 支持 `apply_patch`、`exec` 等 Codex 工具
- ✅ **多轮对话** - 保持对话上下文
- ✅ **自动模型映射** - 自动将 OpenAI 模型名映射到 GLM 对应版本
- ✅ **简单配置** - 单个 Python 文件，无需复杂依赖

## 🚀 快速开始

### 前置要求

- Python 3.8+
- GLM API 密钥（[在这里获取](https://open.bigmodel.cn/)）
- 已安装 [OpenAI Codex CLI](https://github.com/openai/codex)

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/JichinX/codex-glm-proxy.git
   cd codex-glm-proxy
   ```

2. **设置 GLM API 密钥**
   ```bash
   export GLM_API_KEY="你的_GLM_API_密钥"
   ```

3. **启动代理**
   ```bash
   python3 proxy.py
   # 或使用便捷脚本
   ./scripts/start.sh
   ```

   代理将运行在 `http://localhost:18765`

4. **配置 Codex CLI**
   
   创建或更新 `~/.codex/config.toml`：
   ```toml
   model_provider = "glm-proxy"
   model = "gpt-4o"
   
   [model_providers.glm-proxy]
   name = "GLM via Proxy"
   base_url = "http://localhost:18765/v4"
   wire_api = "responses"
   ```

5. **测试**
   ```bash
   mkdir test-codex && cd test-codex && git init
   codex exec "创建一个 Python hello world 程序" --full-auto
   ```

## 📋 配置说明

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `GLM_API_KEY` | *(必需)* | 你的 GLM API 密钥 |
| `GLM_API_BASE` | `https://open.bigmodel.cn/api/coding/paas/v4` | GLM API 端点 |
| `PROXY_PORT` | `18765` | 本地代理端口 |

## 🗺️ 模型映射

代理自动将 OpenAI 模型名映射到 GLM 对应版本：

| OpenAI 模型 | GLM 模型 | 说明 |
|------------|----------|------|
| `gpt-4` | `glm-4` | 标准 GPT-4 |
| `gpt-4-turbo` | `glm-4` | GPT-4 Turbo |
| `gpt-4o` | `glm-4-plus` | **推荐**，最佳编码体验 |
| `gpt-4o-mini` | `glm-4-flash` | 更快、更便宜 |
| `gpt-3.5-turbo` | `glm-4-flash` | 旧版支持 |
| `gpt-5.x-codex` | `glm-5` | 未来 Codex 模型 |

**建议：** 在 Codex 配置中使用 `model = "gpt-4o"` 以获得最佳效果。

## 🔧 管理命令

```bash
# 启动代理（后台运行）
./scripts/start.sh

# 检查是否运行
curl http://localhost:18765/health

# 查看日志
tail -f /tmp/codex-glm-proxy.log

# 停止代理
./scripts/stop.sh
```

## 🛠️ 工作原理

代理位于 Codex CLI 和 GLM API 之间，进行格式转换：

```
Codex CLI → Responses API → 代理 → Chat Completions API → GLM
```

### 请求转换
- 将 OpenAI Responses API 格式转换为 GLM Chat Completions
- 处理工具调用历史和函数结果
- 过滤不支持的工具

### 响应转换
- 将 Chat Completions 响应流式转换回 Responses API 格式
- 维护正确的事件顺序
- 处理工具调用流

## 📝 使用示例

```bash
# 简单任务
codex exec "创建一个计算斐波那契数列的 Python 函数" --full-auto

# 更复杂的项目
codex exec "用 FastAPI 构建一个待办事项管理的 REST API" --full-auto

# 包含测试
codex exec "创建一个计算器模块并编写单元测试" --full-auto
```

## 🐛 故障排除

### "Streaming complete, sent 0 chunks"
**原因：** 模型名未正确映射  
**解决：** 确保配置中使用已知模型如 `gpt-4o`

### Codex 循环/重复操作
**原因：** 工具调用历史未正确处理  
**解决：** 更新到最新版本的代理

### 502 Bad Gateway
**原因：** 代理崩溃  
**解决：** 检查日志 `/tmp/codex-glm-proxy.log` 并重启

### Connection refused
**原因：** 代理未运行  
**解决：** 使用 `./scripts/start.sh` 启动代理

## 🤝 贡献

欢迎贡献！请随时提交 Pull Request。

## 📄 许可证

本项目采用 MIT 许可证 - 详情见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [OpenAI Codex](https://github.com/openai/codex) - 强大的编程助手
- [智谱 AI GLM](https://open.bigmodel.cn/) - 强大的国产大模型
- 灵感来源于使用 Codex 与本地/替代 LLM 提供商的需求

## 📊 项目状态

✅ **生产就绪** - 已全面测试所有 Codex 功能

| 功能 | 状态 |
|------|------|
| 文本对话 | ✅ 正常 |
| 模型映射 | ✅ 正常 |
| 流式响应 | ✅ 正常 |
| 工具调用 | ✅ 正常 |
| 多轮对话 | ✅ 正常 |
| 工具调用历史 | ✅ 正常 |
| 工具调用结果 | ✅ 正常 |

## 💬 社区

欢迎加入讨论：
- 提交 Issue 反馈问题
- 提交 PR 贡献代码
- Star ⭐ 本项目支持开发

---

**用 ❤️ 打造，服务社区**

**觉得有用请点个 Star ⭐**
