import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from utils.logger import setup_logger

logger = setup_logger("BaseAgent")


class BaseAgent:
    def __init__(self, role_name: str, model_name: str = "unsloth/Qwen2.5-1.5B-Instruct", temperature: float = 0.2):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        if not openai_api_key:
            raise ValueError("缺少 OPENAI_API_KEY 环境变量")
        if not openai_base_url:
            raise ValueError("缺少 OPENAI_BASE_URL 环境变量")

        self.role_name = role_name
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_retries=3,
            timeout=60
        )
        logger.info("角色 '%s' 已就绪 (Model: %s)", self.role_name, model_name)

    def get_prompt_template(self, system_message: str):
        """
        统一的 Prompt 模板生成器，强制包含上下文历史
        """
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ])
