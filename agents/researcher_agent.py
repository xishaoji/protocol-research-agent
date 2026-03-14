from agents.base_agent import BaseAgent
from langchain_core.messages import AIMessage


class ResearcherAgent(BaseAgent):
    def __init__(self, available_tools: list, model_name: str = "unsloth/Qwen2.5-1.5B-Instruct"):
        # 研究员需要严谨，temperature 设为 0.1
        super().__init__(role_name="Senior_Researcher", temperature=0.1, model_name=model_name)
        
        # 将工具绑定到该 Agent 的专属大模型上
        self.llm_with_tools = self.llm.bind_tools(available_tools)
        
        # 独立维护极其复杂的 System Prompt
        self.system_prompt = """
        你是一名顶级的 AI 行业数据分析师与研究员。
        你的任务是根据用户的主题，利用手头的工具进行深度调研。
        
        【工具使用原则】：
        1. 优先使用 local_knowledge_search 检索内部私有数据。
        2. 如果内部数据不足，再使用 tavily_web_search 获取最新公网信息。
        3. 必须交叉验证数据来源，拒绝盲目相信单一网页。
        4. 如果你认为不需要使用工具了，请直接输出【RESEARCH_COMPLETE】，并把所有结论和数据整理在一起。
        
        【工作流】：
        - 分析问题 -> 调用工具 -> 获取结果 -> 总结提炼
        - 如果发现搜集的信息已经足以回答问题，请输出：【RESEARCH_COMPLETE】
        """
        self.prompt_template = self.get_prompt_template(self.system_prompt)
        
        # 将 Prompt 和 LLM 组合成 Runnable
        self.chain = self.prompt_template | self.llm_with_tools

    # async def ainvoke(self, state_messages):
    #     """对外的异步调用接口"""
    #     return await self.chain.ainvoke({"messages": state_messages})
    async def ainvoke(self, state_messages):
        """对外的异步调用接口"""
        try:
            response = await self.chain.ainvoke({"messages": state_messages})
            print(f"✅ [ResearcherAgent] 执行成功: {response.content[:100]}...")
            return response
        except Exception as e:
            print(f"❌ [ResearcherAgent] 执行失败: {str(e)}")
            return AIMessage(content="【ERROR】研究员执行失败，请稍后再试。")

