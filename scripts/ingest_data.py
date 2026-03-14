import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma   windows 环境下有点问题，改用FAISS
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


import time


# 配置路径
DATA_DIR = "./data"
# CHROMA_PATH = "./chroma_db"
FAISS_PATH = "./faiss_db"

def clear_database():
    """清理旧的数据库，防止数据重复"""
    if os.path.exists(FAISS_PATH):
        print("🗑️ 正在清理旧的向量数据库...")
        shutil.rmtree(FAISS_PATH)

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
    print("🧠 正在调用大模型生成向量，并存入 FAISS 数据库...")
    embeddings = OpenAIEmbeddings(
        base_url="https://xishaoji-qwen3-embedding.hf.space/v1", 
        api_key="EMPTY",
        model="gpt-3.5-turbo"
    )
    
# #------------------------ 先单独测试一下 HF Space 的接口，确保它在当前环境下能正常
#     print("正在单独测试 HF Space 接口...")
#     try:
#         # 模拟单次请求，查看是否报错
#         test_vec = embeddings.embed_query("这是一段测试文本")
#         print(f"✅ 单次请求成功！返回了 {len(test_vec)} 维度的向量。")
        
#         # 模拟批量请求（取前 5 个 chunk），测试接口并发能力
#         print("正在测试批量请求...")
#         test_batch = [doc.page_content for doc in chunks[:5]]
#         test_vecs = embeddings.embed_documents(test_batch)
#         print(f"✅ 批量请求成功！成功向量化 {len(test_vecs)} 条数据。")
        
#     except Exception as e:
#         print(f"❌ 接口调用失败，捕获到真实报错: {e}")
#         # 强制退出，避免执行到后面 Chroma 导致静默崩溃
#         exit()
# # ----------------------


    # 不能并发调用接口了，改成逐条调用，虽然慢但能保证成功率
    # # 存入本地磁盘
    # db = Chroma.from_documents(
    #     chunks, 
    #     embedding=embeddings, 
    #     persist_directory=CHROMA_PATH
    # )

    # 按批次向量化并存储，虽然慢但能保证成功率
    print("🧠 正在连接本地 FAISS 数据库...")

    # 第一步：先初始化一个空的 FAISS 客户端，绑定目录和 Embedding 模型
    db = FAISS.from_documents(
        documents=chunks[0:5], # 先测试前 5 条，确保接口和数据库都正常
        embedding=embeddings
    )

    # 第二步：设置安全批次大小（既然测试 5 个没问题，我们就用 5）
    BATCH_SIZE = 5
    total_chunks = len(chunks)

    print(f"📦 开始分批向数据库写入向量，每批 {BATCH_SIZE} 条...")

    # 第三步：利用 for 循环分批写入
    for i in range(5, total_chunks, BATCH_SIZE):
        batch_chunks = chunks[i : i + BATCH_SIZE]
        print(f"   ⏳ 正在处理第 {i+1} 到 {min(i+BATCH_SIZE, total_chunks)} 个数据块...")
        
        try:
            # 使用 add_documents 追加写入
            db.add_documents(batch_chunks)
        except Exception as e:
            print(f"❌ 在写入第 {i+1} 批次时接口报错了: {e}")
            print("建议：如果报错，请尝试将 BATCH_SIZE 改为 2 或 1 再试。")
            exit() # 遇到错误安全退出，避免静默崩溃
        time.sleep(1) 
    db.save_local("./faiss_db") # FAISS 的持久化方法
    print(f"🎉 知识库构建完成！已成功持久化存储至 {FAISS_PATH} 目录。")

if __name__ == "__main__":
    clear_database()
    generate_data_store()