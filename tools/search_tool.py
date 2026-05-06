import os
import json
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from utils.logger import setup_logger

logger = setup_logger("Tavily")

search_client = None
_initialized = False


def _get_search_client():
    global search_client, _initialized
    if _initialized:
        return search_client

    _initialized = True
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("缺少 TAVILY_API_KEY 环境变量")

    search_client = TavilySearch(
        max_results=5,
        search_depth="advanced",
        include_answer=True
    )
    return search_client


@tool("tavily_web_search")
def search_web(query: str) -> str:
    """
    核心搜索工具。当需要从互联网获取最新信息、新闻、数据或学术资料时调用。
    输入应该是一个清晰、具体的搜索关键词。
    """
    try:
        client = _get_search_client()
        logger.info("正在搜索: %s", query)

        results = client.invoke({"query": query})

        formatted_results = []
        for item in results:
            formatted_results.append({
                "title": item.get("title", "未知标题"),
                "url": item.get("url", ""),
                "content": item.get("content", "")[:800]
            })

        return json.dumps(formatted_results, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"搜索服务暂时不可用，原因: {str(e)}。请尝试使用其他工具或修改搜索策略。"})
