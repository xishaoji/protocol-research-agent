from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from core.state import AgentState

# 引入我们写好的两个工具
from tools.search_tool import research_tools_instance
from tools.rag_tool import rag_tools_instance

# 将工具打包成列表
AVAILABLE_TOOLS = [
    research_tools_instance.search_web, 
    rag_tools_instance.search_internal_docs
]

class ResearchNodes:
    def __init__(self, model_name="gpt-4o"):
        # 1. 初始化模型
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        # 2. 核心魔法：将工具说明书“注入”给大模型
        self.llm_with_tools = self.llm.bind_tools(AVAILABLE_TOOLS)

    async def researcher_node(self, state: AgentState):
        """
        智能体大脑：根据当前状态，决定是直接回答，还是调用工具（公网搜索 or RAG）
        """
        # 给 Agent 一个清晰的系统提示词，告诉它如何使用工具
        system_msg = SystemMessage(content="""
        你是一个顶级的 AI 研究员。
        你可以使用 tavily_web_search 获取最新互联网信息，或者使用 local_knowledge_search 获取内部私有文档。
        请仔细分析用户的问题，按需调用工具。如果资料充足，请进行总结。
        """)
        
        # 将系统提示词与历史对话合并
        messages = [system_msg] + state["messages"]
        
        # 调用绑定了工具的 LLM
        response = await self.llm_with_tools.ainvoke(messages)
        
        # 返回新的消息追加到状态中
        return {"messages": [response]}
        
    # (analyzer_node 和 writer_node 保持之前的逻辑不变)