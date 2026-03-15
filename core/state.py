from typing import Annotated, List, Union, Dict
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import operator

class AgentState(TypedDict):
    # 使用 add_messages 自动管理对话历史
    messages: Annotated[List[BaseMessage], add_messages]
    search_count: int
    # 结构化字段
    query: str
    research_targets: List[str]  # 分解后的研究子任务
    search_results: Annotated[List[Dict], operator.add] # 原始资料库
    report_draft: str
    is_satisfactory: bool
    revision_count: int