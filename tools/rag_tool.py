import os
import json
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from utils.logger import setup_logger

logger = setup_logger("RAG")

FAISS_PATH = "./faiss_db"

vector_db = None
_initialized = False


def _init_vector_db():
    global vector_db, _initialized
    if _initialized:
        return vector_db

    _initialized = True
    if not os.path.exists(FAISS_PATH):
        logger.warning("未检测到本地数据库。请先运行 scripts/ingest_data.py")
        return None

    embeddings = OpenAIEmbeddings(
        base_url="https://xishaoji-qwen3-embedding.hf.space/v1",
        api_key="EMPTY",
        model="qwen3-embedding"
    )
    vector_db = FAISS.load_local(
        folder_path=FAISS_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    logger.info("本地私有知识库 (FAISS) 已成功挂载！")
    return vector_db


@tool("local_knowledge_search")
def search_internal_docs(query: str) -> str:
    """
    核心 RAG 工具。当用户的问题涉及"内部数据"、"本地文档"、"私有项目"或你无法在公网上找到的信息时，必须调用此工具。
    输入应该是一个针对内部文档的语义搜索词。
    """
    global vector_db
    if vector_db is None:
        vector_db = _init_vector_db()
    if not vector_db:
        return json.dumps({"error": "本地知识库尚未初始化，请联系管理员。"})

    try:
        logger.info("正在深入检索知识库: %s", query)
        results = vector_db.similarity_search_with_score(query, k=4)

        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "source": doc.metadata.get("source", "未知来源"),
                "page": doc.metadata.get("page", -1),
                "content": doc.page_content
            })

        return json.dumps(formatted_results, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"本地知识库读取失败: {str(e)}"})
