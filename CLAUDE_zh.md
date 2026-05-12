# CLAUDE_zh.md

本文件为中文版本，为 Claude Code 提供关于本仓库的指导说明。

## 📋 项目概览

这是一个基于 **LangGraph** 构建的智能体框架，专注于充电（充电宝/充电桩）行业的协议研究与本地知识查询。它采用多智能体协作与状态机架构，实现自动化的网络调研与本地知识检索。

## 🏗️ 核心架构

```
├── agents/               # 智能体实现
│   ├── base_agent.py     # 基础智能体（LLM + 工具绑定）
│   ├── researcher_agent.py  # 研究智能体（侧重工具调用，温度=0.1）
│   └── report_agent.py       # 报告撰写智能体
├── core/                 # 核心框架
│   ├── graph.py          # 工作流构建器
│   ├── nodes.py          # 节点定义与路由
│   └── state.py          # 智能体状态定义
├── tools/                # 自定义工具
│   ├── rag_tool.py       # FAISS 本地知识检索
│   └── search_tool.py    # Tavily 网页搜索
├── scripts/              # 工具脚本
│   └── ingest_data.py    # PDF 摄入与向量化
├── data/                 # 输入 PDF 数据
├── faiss_db/             # 向量数据库
└── main.py               # Streamlit Web 界面
```

## 🚀 开发流程

### 1. 环境准备
```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量
在根目录创建 `.env` 文件：
```bash
OPENAI_API_KEY="sk-xxxx"
OPENAI_BASE_URL="https://api.openai.com/v1"
TAVILY_API_KEY="tvly-xxxx"
```

### 3. 构建本地知识库（可选）
将 PDF 文档放入 `data/` 目录：
```bash
python scripts/ingest_data.py
```

### 4. 启动应用
```bash
streamlit run main.py
```

## 🔍 关键设计模式

### 多智能体状态机
- **研究智能体**：处理深层协议分析，温度严格设为 0.1
- **撰写智能体**：生成最终报告
- **路由逻辑**：使用 `should_continue()` 决定工具执行或撰写阶段
- **工具绑定**：工具通过 `llm.bind_tools()` 动态绑定到研究智能体

### 检索策略
1. **优先级**：`local_knowledge_search`（FAISS RAG）→ `tavily_web_search`（公开数据）
2. **交叉验证**：必须多源验证
3. **熔断机制**：最多 10 次搜索后强制进入撰写阶段

### 内存与状态
- **会话隔离**：通过 `st.session_state.thread_id` 实现
- **检查点**：Redis 或 InMemory 用于 LangGraph 持久化
- **状态管理**：`AgentState` 管理消息、搜索计数与草稿报告

## ⚙️ 工具配置

### 必需环境变量
| 变量 | 用途 | 示例 |
|------|------|------|
| `OPENAI_API_KEY` | OpenAI API 认证 | `sk-xxx` |
| `TAVILY_API_KEY` | Tavily 搜索服务密钥 | `tvly-xxx` |

### FAISS 数据库
- 路径：`./faiss_db`
- 通过 `scripts/ingest_data.py` 初始化
- 使用 OpenAI 嵌入（可替换）

## 🐛 调试建议

1. **智能体失败**：检查控制台输出的具体工具调用错误
2. **RAG 未工作**：确认 `faiss_db` 在运行摄入脚本后存在
3. **搜索错误**：确保 `TAVILY_API_KEY` 已设置
4. **Streamlit 问题**：若界面未更新，请清除浏览器缓存

## 📝 注意事项

- 研究智能体使用 **temperature=0.1** 以保证精确性
- 工具调用在异步节点内为 **同步执行**
- 工作流使用 **自定义路由** 而非内置 `tools_condition`
- Redis 连接地址在 `graph.py` 第 40 行硬编码