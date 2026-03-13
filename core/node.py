from core.state import AgentState
from agents.researcher import ResearcherAgent
from agents.writer import ReportAgent
from tools.search_tool import research_tools_instance
from tools.rag_tool import rag_tools_instance

# 组合可用工具
AVAILABLE_TOOLS = [
    research_tools_instance.search_web, 
    rag_tools_instance.search_internal_docs
]

class GraphNodes:
    def __init__(self):
        # 在节点初始化时，实例化各个 Agent
        self.researcher = ResearcherAgent(AVAILABLE_TOOLS)
        self.writer = ReportAgent()

    async def researcher_node(self, state: AgentState):
        """调度研究员 Agent"""
        print("▶️ [Node] Researcher 正在执行...")
        response = await self.researcher.ainvoke(state["messages"])
        return {"messages": [response]}

    async def writer_node(self, state: AgentState):
        """调度主笔 Agent"""
        print("▶️ [Node] Writer 正在撰写报告...")
        response = await self.writer.ainvoke(state["messages"])
        return {"messages": [response], "report_draft": response.content}
