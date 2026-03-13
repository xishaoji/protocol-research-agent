import streamlit as st
import asyncio
import uuid
from core.graph import build_research_agent
from langchain_core.messages import HumanMessage, AIMessage

# --- 页面配置 ---
st.set_page_config(page_title="AI Research Copilot", page_icon="🧠", layout="wide")
st.title("🧠 深度研究 AI 代理系统 (Startup 版)")

# --- 初始化 Session State ---
# 赋予每个访客独立的 Thread ID，实现记忆隔离
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    
if "messages" not in st.session_state:
    st.session_state.messages = []

# 全局单例实例化 Graph
@st.cache_resource
def get_agent():
    return build_research_agent()

agent_app = get_agent()

# --- 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 控制面板")
    st.info(f"当前 Session ID:\n`{st.session_state.thread_id[:8]}...`")
    if st.button("🗑️ 清空上下文并重启"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# --- 渲染历史对话 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 异步处理核心逻辑 ---
async def process_agent_stream(user_query: str):
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    initial_input = {"messages": [HumanMessage(content=user_query)]}
    
    # 占位符：用于显示 Agent 的思考过程
    status_container = st.empty()
    final_response = ""
    
    with status_container.container():
        with st.status("Agent 正在深度思考与执行...", expanded=True) as status:
            async for event in agent_app.astream(initial_input, config=config, stream_mode="values"):
                if "messages" in event:
                    last_msg = event["messages"][-1]
                    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                        st.write(f"🛠️ **调用工具**: `{last_msg.tool_calls[0]['name']}`")
                    elif isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                        final_response = last_msg.content
                        st.write("✍️ **正在起草报告...**")
            status.update(label="任务完成", state="complete", expanded=False)
            
    return final_response

# --- 对话输入框 ---
if prompt := st.chat_input("输入你的研究课题，例如：'对比 2024 年 Q1 各大厂大模型的多模态能力'"):
    # 1. 显式用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 执行 Agent 逻辑
    with st.chat_message("assistant"):
        # Streamlit 默认运行在同步环境中，需要用 asyncio.run 驱动 LangGraph
        response = asyncio.run(process_agent_stream(prompt))
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})