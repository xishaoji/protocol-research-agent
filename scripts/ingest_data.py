import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 加载环境变量 (确保有 OPENAI_API_KEY)
load_dotenv()

# 配置路径
DATA_DIR = "./data"
CHROMA_PATH = "./chroma_db"

def clear_database():
    """清理旧的数据库，防止数据重复"""
    if os.path.exists(CHROMA_PATH):
        print("🗑️ 正在清理旧的向量数据库...")
        shutil.rmtree(CHROMA_PATH)

def generate_data_store():
    """执行完整的数据清洗与向量化管道"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 已创建 {DATA_DIR} 文件夹，请放入 PDF 文档后重试。")
        return

    # 1. 加载文档
    print(f"📄 正在从 {DATA_DIR} 加载 PDF 文档...")
    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()
    
    if not documents:
        print("⚠️ 未找到任何文档，请确保 data/ 目录下有 PDF 文件。")
        return
        
    print(f"✅ 成功加载 {len(documents)} 页文档。")

    # 2. 文本分块 (Chunking)
    # 这是 RAG 的核心技术点：块太大容易丢失细节，块太小会丧失上下文
    print("✂️ 正在进行文本切分...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # 每个文本块的最大字符数
        chunk_overlap=100,    # 块与块之间的重叠，保持语义连贯
        length_function=len,
        add_start_index=True, # 记录该块在原文档中的位置
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ 文档已被切分为 {len(chunks)} 个数据块。")

    # 3. 向量化与持久化存储 (Embedding & Vector DB)
    print("🧠 正在调用大模型生成向量，并存入 Chroma 数据库...")
    # 默认使用 text-embedding-3-small，性价比极高
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 存入本地磁盘
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH
    )
    print(f"🎉 知识库构建完成！已持久化存储至 {CHROMA_PATH} 目录。")

if __name__ == "__main__":
    clear_database()
    generate_data_store()