from agents.base import BaseAgent

class WriterAgent(BaseAgent):
    def __init__(self):
        # 创作者需要一定的发散性，temperature 设为 0.5
        super().__init__(role_name="Lead_Writer", temperature=0.5)
        
        self.system_prompt = """
        你是一名科学期刊的专业主笔。
        你的任务是将 Research 团队搜集到的所有零碎信息，融合成一篇极具洞察力的长篇报告。
        
        【排版要求】：
        1. 必须使用 Markdown 格式。
        2. 必须包含：执行摘要、核心发现、数据支撑、未来展望。
        3. 语言风格：专业、克制、富有逻辑。
        """
        self.prompt_template = self.get_prompt_template(self.system_prompt)
        self.chain = self.prompt_template | self.llm
        
    async def ainvoke(self, state_messages):
        return await self.chain.ainvoke({"messages": state_messages})
