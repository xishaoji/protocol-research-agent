import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class BaseAgent:
    def __init__(self, role_name: str, model_name: str = "unsloth/Qwen2.5-1.5B-Instruct", temperature: float = 0.2):
        self.role_name = role_name
        # 生产级项目必须配置 max_retries 处理网络波动
        self.llm = ChatOpenAI(
            model=model_name, 
            # base_url="https://xishaoji-my-llmtest.hf.space/v1", 
            # api_key="EMPTY",
            temperature=temperature,
            max_retries=3,
            timeout=60
        )
        print(f"🤖 [Agent Init] 角色 '{self.role_name}' 已就绪 (Model: {model_name})")

    def get_prompt_template(self, system_message: str):
        """
        统一的 Prompt 模板生成器，强制包含上下文历史
        """
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
        ])
