# ReasoningAgent

ReasoningAgent 是一个基于 LangGraph 的多源证据研究智能体。它通过 MCP 子进程检索网页、GitHub 与学术论文，结合本地 FAISS 知识库，由 Researcher/Critic 多轮检查证据并生成回答。项目同时提供中文 Gradio 多轮研究界面和 CLI。

> 发布提醒：仓库目前未选择开源许可证。公开发布前请根据实际需要添加 MIT、Apache-2.0 或其他许可证。

## 项目结构

```text
app/                     # Gradio Web 入口
research_agent/
  agents/                # Researcher 与 Critic
  llm/                   # 本地 Qwen 适配器
  rag/                   # LangChain FAISS 知识库与语义相关度
  retrieval/             # MCP 检索客户端
  workflow/              # LangGraph 状态、节点与图
mcp_servers/             # Web、GitHub、Paper MCP stdio 服务
scripts/                 # CLI 与索引构建入口
data/knowledge/          # 用户自行准备的本地知识文件
```

## 安装

建议使用 Python 3.10 或 3.11。项目不会自动下载 Qwen 或 BGE-M3 模型。

如果使用 CUDA，请先按照 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)安装与机器 CUDA 环境匹配的 PyTorch，再安装其余依赖：

```bash
pip install -r requirements.txt
```

## 配置

复制 `.env.example` 并填写本机路径。代码不会自动加载 `.env`，请通过 shell 导出变量；例如 Linux/macOS：

```bash
cp .env.example .env
set -a
source .env
set +a
```

必须设置：

- `LLM_MODEL_PATH`：本地 Qwen2.5 Instruct 模型目录。
- `EMBEDDING_MODEL_PATH`：本地 BGE-M3 模型目录。

可选设置：

- `WEB_PROXY`：Web/MCP 客户端代理；留空时直接联网。
- `GITHUB_TOKEN`：提高 GitHub API 限额；匿名访问也可使用。
- `SEMANTIC_SCHOLAR_API_KEY`：Semantic Scholar API Key；留空时论文检索使用 OpenAlex，并可对英文查询回退到 arXiv。
- `OPENALEX_MAILTO`：可选的 OpenAlex 联系邮箱。

## 本地知识库

将 `.txt`、`.md` 或 `.pdf` 文件放入 `data/knowledge/`。除 `example.md` 外，该目录内容默认不会提交到 Git。然后构建本地索引：

```bash
python -m scripts.build_index
```

索引写入 `data/index/`，同样不会提交到 GitHub。更换嵌入模型后需要重新构建。

## 运行

从项目根目录启动 CLI：

```bash
python -m scripts.run_cli
```

启动 Gradio Web UI：

```bash
python -m app.web_app
```

默认监听 `0.0.0.0:7860`。远程服务器可按需自行配置安全组、反向代理或 SSH 端口转发，例如：

```bash
ssh -L 7860:localhost:7860 user@your-server
```
