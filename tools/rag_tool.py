import os
import json
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 数据库路径需要和离线脚本保持一致
CHROMA_PATH = "./chroma_db"

class PrivateKnowledgeTools:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 启动时检查数据库是否存在
        if os.path.exists(CHROMA_PATH):
            self.vector_db = Chroma(
                persist_directory=CHROMA_PATH, 
                embedding_function=self.embeddings
            )
            print("🗄️ [RAG] 本地私有知识库 (Chroma) 已成功挂载！")
        else:
            self.vector_db = None
            print("⚠️ [RAG] 未检测到本地数据库。请先运行 scripts/ingest_data.py")

    @tool("local_knowledge_search")
    def search_internal_docs(self, query: str) -> str:
        """
        核心 RAG 工具。当用户的问题涉及“内部数据”、“本地文档”、“私有项目”或你无法在公网上找到的信息时，必须调用此工具。
        输入应该是一个针对内部文档的语义搜索词。
        """
        if not self.vector_db:
            return json.dumps({"error": "本地知识库尚未初始化，请联系管理员。"})

        try:
            print(f"🕵️ [RAG] 正在深入检索知识库: {query}")
            
            # 执行相似度检索，获取最相关的 4 个数据块
            results = self.vector_db.similarity_search_with_score(query, k=4)
            
            formatted_results = []
            for doc, score in results:
                # 可以根据 score 过滤掉相关度太低的结果 (Chroma 的 score 是距离，越小越相关)
                formatted_results.append({
                    "source": doc.metadata.get("source", "未知来源"),
                    "page": doc.metadata.get("page", -1),
                    "content": doc.page_content
                })
                
            return json.dumps(formatted_results, ensure_ascii=False)
            
        except Exception as e:
            return json.dumps({"error": f"本地知识库读取失败: {str(e)}"})

rag_tools_instance = PrivateKnowledgeTools()