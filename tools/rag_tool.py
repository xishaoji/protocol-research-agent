import os
import json
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# ==========================================
# 1. 全局初始化区 (模块导入时只会执行一次)
# ==========================================
# 数据库路径需要和离线脚本保持一致
FAISS_PATH = "./faiss_db"

# 初始化 Embeddings 模型
embeddings = OpenAIEmbeddings(
    base_url="https://xishaoji-qwen3-embedding.hf.space/v1", 
    api_key="EMPTY",
    model="gpt-3.5-turbo"
)

# 启动时检查数据库是否存在
if os.path.exists(FAISS_PATH):
    vector_db = FAISS.load_local(
        folder_path=FAISS_PATH, 
        embeddings=embeddings,  # ✅ 修正了原代码里的拼写错误 (原为 embeddinings)
        allow_dangerous_deserialization=True
    )
    print("🗄️ [RAG] 本地私有知识库 (FAISS) 已成功挂载！")
else:
    vector_db = None
    print("⚠️ [RAG] 未检测到本地数据库。请先运行 scripts/ingest_data.py")


# ==========================================
# 2. 工具定义
# ==========================================
@tool("local_knowledge_search")
def search_internal_docs(query: str) -> str:
    """
    核心 RAG 工具。当用户的问题涉及“内部数据”、“本地文档”、“私有项目”或你无法在公网上找到的信息时，必须调用此工具。
    输入应该是一个针对内部文档的语义搜索词。
    """
    # 直接使用全局变量 vector_db
    if not vector_db:
        return json.dumps({"error": "本地知识库尚未初始化，请联系管理员。"})

    try:
        print(f"🕵️ [RAG] 正在深入检索知识库: {query}")
        
        # 执行相似度检索，获取最相关的 4 个数据块
        results = vector_db.similarity_search_with_score(query, k=4)
        
        formatted_results = []
        for doc, score in results:
            # 可以根据 score 过滤掉相关度太低的结果 (FAISS 默认情况下的 score 是 L2 距离，越小越相关)
            formatted_results.append({
                "source": doc.metadata.get("source", "未知来源"),
                "page": doc.metadata.get("page", -1),
                "content": doc.page_content
            })
            
        return json.dumps(formatted_results, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"error": f"本地知识库读取失败: {str(e)}"})

# ==========================================
# 3. 导出供 Graph 使用
# ==========================================
# 将工具放入列表，方便在其他文件中导入
tools = [search_internal_docs]

# 在其他文件中使用：
# from your_rag_file import tools as rag_tools