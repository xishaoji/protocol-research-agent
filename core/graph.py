from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition # 引入官方现成的工具处理模块
from langgraph.checkpoint.sqlite import SqliteSaver # 引入 SQLite 持久化模块
import sqlite3


from core.state import AgentState
from core.nodes import ResearchNodes, AVAILABLE_TOOLS

def build_research_agent():
    nodes = ResearchNodes()
    workflow = StateGraph(AgentState)
    
    # 1. 添加普通节点
    workflow.add_node("researcher", nodes.researcher_node)
    
    # 2. 添加工具执行节点 (LangGraph 提供的现成 Node，会自动执行 LLM 请求的工具)
    tool_node = ToolNode(AVAILABLE_TOOLS)
    workflow.add_node("tools", tool_node)

    # 3. 设置入口
    workflow.set_entry_point("researcher")

    # 4. 核心路由逻辑：LLM 决定下一步去哪
    # tools_condition 会自动检查大模型的输出：
    # - 如果模型输出了 tool_calls，它就路由到 "tools" 节点
    # - 如果模型直接输出了文本，它就路由到 END (或你定义的下一个节点如 "write")
    workflow.add_conditional_edges(
        "researcher",
        tools_condition, 
        {
            "tools": "tools", # 去执行工具
            END: END          # 思考完毕，结束（或连向 writer 节点）
        }
    )
    
    # 5. 工具执行完后，必须把结果交回给 researcher 继续思考
    workflow.add_edge("tools", "researcher")

    # memory = MemorySaver()
    # 连接到本地的 SQLite 数据库文件
    conn = sqlite3.connect("agent_memory.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    return workflow.compile(checkpointer=memory)