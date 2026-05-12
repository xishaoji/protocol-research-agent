import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import time
from utils.logger import setup_logger

logger = setup_logger("Ingest")

DATA_DIR = "./data"
FAISS_PATH = "./faiss_db"


def clear_database():
    if os.path.exists(FAISS_PATH):
        logger.info("正在清理旧的向量数据库...")
        shutil.rmtree(FAISS_PATH)


def generate_data_store():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info("已创建 %s 文件夹，请放入 PDF 文档后重试。", DATA_DIR)
        return

    logger.info("正在从 %s 加载 PDF 文档...", DATA_DIR)
    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()

    if not documents:
        logger.warning("未找到任何文档，请确保 data/ 目录下有 PDF 文件。")
        return

    logger.info("成功加载 %d 页文档。", len(documents))

    logger.info("正在进行文本切分...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info("文档已被切分为 %d 个数据块。", len(chunks))

    logger.info("正在调用 Embedding 模型生成向量...")
    embeddings = OpenAIEmbeddings(
        base_url="https://xishaoji-qwen3-embedding.hf.space/v1",
        api_key="EMPTY",
        model="qwen3-embedding"
    )

    logger.info("正在连接本地 FAISS 数据库...")

    db = FAISS.from_documents(
        documents=chunks[0:5],
        embedding=embeddings
    )

    BATCH_SIZE = 5
    total_chunks = len(chunks)

    logger.info("开始分批向数据库写入向量，每批 %d 条...", BATCH_SIZE)

    for i in range(5, total_chunks, BATCH_SIZE):
        batch_chunks = chunks[i : i + BATCH_SIZE]
        logger.info("正在处理第 %d 到 %d 个数据块...", i + 1, min(i + BATCH_SIZE, total_chunks))

        try:
            db.add_documents(batch_chunks)
        except Exception as e:
            logger.error("在写入第 %d 批次时接口报错: %s", i + 1, e)
            raise
        time.sleep(1)

    db.save_local(FAISS_PATH)
    logger.info("知识库构建完成！已成功持久化存储至 %s 目录。", FAISS_PATH)


if __name__ == "__main__":
    clear_database()
    generate_data_store()