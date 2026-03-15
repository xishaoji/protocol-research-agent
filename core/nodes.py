from core.state import AgentState
from agents.researcher_agent import ResearcherAgent
from agents.report_agent import ReportAgent
from tools.search_tool import search_web
from tools.rag_tool import search_internal_docs
from langchain_core.messages import SystemMessage

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
        if state.get("search_count", 0) >= 10:  # 搜索次数超过阈值，强制进入写作阶段
            return {"search_count":0, "messages": [SystemMessage(
                content=(
                "[系统强制中断]：检索次数已达上限，必须停止搜索。 "
                "当前资料极有可能是不完整的。 "
                "请基于上方已有的上下文进行回答。如果上下文中找不到用户询问的具体协议相关信息，"
                "请直接明确回复：'本地知识库中未检索到该协议的相关资料'。"
                "严禁尝试调用任何工具！严禁输出类似 <｜DSML｜ 这样的系统标签！"
                )
                )]} # 超过后把次数清0，并让研究员输出一个提示信息，告诉它直接总结就好，不要再调用工具了。
        print("▶️ [Node] Researcher 正在执行...")
        response = await self.researcher.ainvoke(state["messages"])
        return {"search_count": state.get("search_count", 0) + 1, "messages": [response]}

    async def writer_node(self, state: AgentState):
        """调度主笔 Agent"""
        print("▶️ [Node] Writer 正在撰写报告...")
        response = await self.writer.ainvoke(state["messages"])
        return {"messages": [response], "report_draft": response.content}
