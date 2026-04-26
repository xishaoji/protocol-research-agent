# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 📋 Project Overview

This is a **LangGraph-based AI Agent framework** for automated web research and local knowledge queries focused on the charging pile (electric vehicle charging) industry. It uses multi-agent collaboration with a state machine architecture.

## 🏗️ Core Architecture

```
├── agents/              # Agent implementations
│   ├── base_agent.py    # Base agent with LLM + tool binding
│   ├── researcher_agent.py  # Researcher node (tool-heavy, temperature=0.1)
│   └── report_agent.py      # Writer node (report generation)
├── core/                # Framework structure
│   ├── graph.py         # LangGraph workflow builder
│   ├── nodes.py         # Node definitions and routing
│   └── state.py         # Agent state definition
├── tools/               # Custom tools
│   ├── rag_tool.py      # FAISS RAG search (local/private data)
│   └── search_tool.py   # Tavily web search (public data)
├── scripts/             # Utility scripts
│   └── ingest_data.py   # PDF ingestion & vectorization
├── data/                # Input PDFs for RAG
├── faiss_db/            # Vector database
└── main.py              # Streamlit web interface
```

## 🚀 Development Workflow

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create `.env` file:
```bash
OPENAI_API_KEY="sk-xxxx"
OPENAI_BASE_URL="https://api.openai.com/v1"
TAVILY_API_KEY="tvly-xxxx"
```

### 3. Build Local Knowledge Base (Optional)
Add PDF documents to `data/` directory:
```bash
python scripts/ingest_data.py
```

### 4. Run the Application
```bash
streamlit run main.py
```

## 🔍 Key Design Patterns

### Multi-Agent State Machine
- **Researcher Agent**: Handles deep protocol analysis with strict temperature (0.1)
- **Writer Agent**: Generates final reports
- **Router Logic**: Uses `should_continue()` to decide between tool execution vs. reporting
- **Tool Binding**: Tools dynamically bound to researcher's LLM via `llm.bind_tools()`

### Search Strategy
1. **Priority**: `local_knowledge_search` (FAISS RAG) → `tavily_web_search` (public data)
2. **Cross-validation**: Must verify with multiple sources
3. **Circuit Breaker**: Max 10 searches, then force report generation

### Memory & State
- **Session isolation**: Thread ID via `st.session_state.thread_id`
- **Checkpointer**: Redis or InMemory for LangGraph persistence
- **State**: `AgentState` manages messages, search counts, and draft reports

## ⚙️ Tool Configuration

### Required Environment Variables
| Variable | Purpose | Example |
|----------|---------|---------|
| `OPENAI_API_KEY` | Authentication for OpenAI API | `sk-xxx` |
| `TAVILY_API_KEY` | Tavily search service key | `tvly-xxx` |

### FAISS Database
- Path: `./faiss_db`
- Initialized via `scripts/ingest_data.py`
- Uses OpenAI embeddings (can be changed)

## 🐛 Debugging Tips

1. **Agent failures**: Check console output for specific tool call errors
2. **RAG not working**: Verify `faiss_db` exists after running ingest script
3. **Search errors**: Ensure `TAVILY_API_KEY` is set in environment
4. **Streamlit issues**: Clear browser cache if UI doesn't update

## 📖 Important Notes

- The researcher agent uses **temperature=0.1** for precision
- Tool calls are **synchronous** within async nodes
- The graph uses **custom routing** instead of `tools_condition`
- Redis connection URL is hardcoded (line 40 in `graph.py`)