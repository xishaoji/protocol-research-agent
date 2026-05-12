from agents.base_agent import BaseAgent
from langchain_core.messages import AIMessage
from utils.logger import setup_logger

logger = setup_logger("ResearcherAgent")


class ResearcherAgent(BaseAgent):
    def __init__(self, available_tools: list, model_name: str = "unsloth/Qwen2.5-1.5B-Instruct"):
        super().__init__(role_name="Senior_Researcher", temperature=0.1, model_name=model_name)

        self.llm_with_tools = self.llm.bind_tools(available_tools)

        self.system_prompt = """
        你是一名顶级的充电桩行业通信协议开发和研究专家。
        你的任务是根据用户的主题，利用手头的工具进行深度分析协议内容，解答用户疑问。

        【工具使用原则】：
        1. 优先使用 local_knowledge_search 检索内部私有数据。
        2. 如果内部数据不足，再使用 tavily_web_search 获取最新公网信息。
        3. 必须交叉验证数据来源，拒绝盲目相信单一网页。
        4. 不要过度依赖工具，始终保持批判性思维，结合已有知识进行分析。
        5. 不要太依赖历史消息，每次都要根据当前问题重新评估需要哪些工具和信息。

        【工作流】：
        - 分析问题 -> 调用工具 -> 获取结果 -> 总结提炼
        - 如果发现搜集的信息已经足以回答问题，请输出：【RESEARCH_COMPLETE】
        """
        self.prompt_template = self.get_prompt_template(self.system_prompt)
        self.chain = self.prompt_template | self.llm_with_tools

    async def ainvoke(self, state_messages):
        try:
            response = await self.chain.ainvoke({"messages": state_messages})
            content_preview = response.content[:100] if response.content else "(tool call)"
            logger.info("执行成功: %s...", content_preview)
            return response
        except Exception as e:
            logger.error("执行失败: %s", e)
            return AIMessage(content="【ERROR】研究员执行失败，请稍后再试。")

