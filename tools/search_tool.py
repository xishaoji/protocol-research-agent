import os
import json
from typing import List, Dict, Any
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

# ==========================================
# 1. 全局初始化区 (模块导入时只会执行一次)
# ==========================================

# 确保环境变量中已配置 TAVILY_API_KEY
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("⚠️ 缺少 TAVILY_API_KEY 环境变量")

# 初始化 Tavily 搜索客户端
search_client = TavilySearch(
    max_results=5,
    search_depth="advanced", # advanced 会获取更长的 snippet
    include_answer=True      # 让 Tavily 自身也做一次初步总结
)

# ==========================================
# 2. 工具定义
# ==========================================

@tool("tavily_web_search")
def search_web(query: str) -> str:
    """
    核心搜索工具。当需要从互联网获取最新信息、新闻、数据或学术资料时调用。
    输入应该是一个清晰、具体的搜索关键词。
    """
    try:
        print(f"🌍 [Tavily] 正在搜索: {query}")
        
        # 直接使用全局的 search_client
        results = search_client.invoke({"query": query})
        
        # 清洗和格式化数据，确保丢给 LLM 的 Context 具有极高的信息密度
        formatted_results = []
        for item in results:
            formatted_results.append({
                "title": item.get("title", "未知标题"),
                "url": item.get("url", ""),
                "content": item.get("content", "")[:800] # 截断超长文本防止 Token 爆炸
            })
        
        return json.dumps(formatted_results, ensure_ascii=False)
        
    except Exception as e:
        # 自愈机制：搜索失败时返回明确的错误，让 Agent 决定是否重试或换词
        return json.dumps({"error": f"搜索服务暂时不可用，原因: {str(e)}。请尝试使用其他工具或修改搜索策略。"})


# ==========================================
# 3. 导出供 Graph 使用
# ==========================================
# 把定义好的工具放进列表里
tools = [search_web] 

# 在你的 agent 文件中，你只需要这样做：
# from research_tools import tools
# workflow.add_node("tools", ToolNode(tools))