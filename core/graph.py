from langgraph.graph import StateGraph, END, START
from typing import Literal
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from utils.logger import setup_logger

from core.state import AgentState
from core.nodes import GraphNodes, AVAILABLE_TOOLS

logger = setup_logger("Graph")


def should_continue(state: AgentState) -> Literal["tools", "report"]:
    """Route decision: tools if LLM requested them, otherwise report."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    if isinstance(last_message, SystemMessage):
        logger.info("收到系统消息，强制进入写作阶段")
    return "report"


def build_research_agent():
    nodes = GraphNodes(model_name="deepseek-chat")

    workflow = StateGraph(AgentState)

    workflow.add_node("researcher", nodes.researcher_node)
    workflow.add_node("report", nodes.writer_node)

    tool_node = ToolNode(AVAILABLE_TOOLS)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "researcher")
    workflow.add_conditional_edges(
        "researcher",
        should_continue,
        {
            "tools": "tools",
            "report": "report"
        }
    )

    workflow.add_edge("tools", "researcher")
    workflow.add_edge("report", END)

    memory = InMemorySaver()
    return workflow.compile(checkpointer=memory)