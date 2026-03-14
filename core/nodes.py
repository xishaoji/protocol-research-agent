from core.state import AgentState
from agents.researcher_agent import ResearcherAgent
from agents.report_agent import ReportAgent
from tools.search_tool import search_web
from tools.rag_tool import search_internal_docs
from langchain_core.messages import HumanMessage

# 组合可用工具
AVAILABLE_TOOLS = [
    search_web, 
    search_internal_docs
]

class GraphNodes:
    def __init__(self, model_name="deepseek-reasoner"):
        # 在节点初始化时，实例化各个 Agent
        self.researcher = ResearcherAgent(model_name=model_name, available_tools=AVAILABLE_TOOLS)
        self.writer = ReportAgent(model_name=model_name)

    async def researcher_node(self, state: AgentState):
        """调度研究员 Agent"""
        if state.get("search_count", 0) >= 5:  # 搜索次数超过阈值，强制进入写作阶段
            return {"messages": [HumanMessage(content="信息已足够，请直接总结。")]}
        print("▶️ [Node] Researcher 正在执行...")
        response = await self.researcher.ainvoke(state["messages"])
        return {"search_count": 1, "messages": [response]}

    async def writer_node(self, state: AgentState):
        """调度主笔 Agent"""
        print("▶️ [Node] Writer 正在撰写报告...")
        response = await self.writer.ainvoke(state["messages"])
        return {"messages": [response], "report_draft": response.content}
