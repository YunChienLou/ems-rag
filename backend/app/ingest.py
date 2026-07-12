import os
import pickle
from pypdf import PdfReader 
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 修正：用最明確的方式拆開導入，確保 Pylance 絕對能解析
from google import genai
from google.genai import types

# 基礎配置
PDF_PATH = "/app/ntpc_ems_protocol.pdf"
BM25_PKL_PATH = "/app/app/bm25_model.pkl"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
COLLECTION_NAME = "ntpc_ems_protocols"

def split_text_by_overlap(text: str, chunk_size: int = 600, overlap: int = 150):
    """
    手寫滑動視窗切片演算法 (Sliding Window Chunking)
    確保救護禁忌症不因斷句而遺失
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

def build_knowledge_base():
    """
    資料清洗與演算法索引建立核心邏輯
    """
    if not os.path.exists(PDF_PATH):
        print(f"❌ 找不到救護準則 PDF：{PDF_PATH}，請確認檔案位置。")
        return

    print("📖 1. 正在讀取新北救護技術準則 PDF...")
    # 修正：使用原生 pypdf PdfReader 讀取檔案
    reader = PdfReader(PDF_PATH)
    
    raw_chunks = []
    metadata_list = []
    
    print("✂️ 2. 執行滑動視窗切片演算法...")
    # 修正：逐頁提取文字
    for idx, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if not page_text:
            continue
        page_num = idx + 1
        
        # 對每頁文本進行切片
        page_chunks = split_text_by_overlap(page_text, chunk_size=600, overlap=150)
        
        for c_idx, chunk in enumerate(page_chunks):
            if len(chunk.strip()) < 10:  # 過濾空白或無意義的雜訊
                continue
            raw_chunks.append(chunk)
            metadata_list.append({
                "page": page_num,
                "chunk_index": c_idx,
                "text": chunk
            })

    print(f"✅ 文本切片完成，共生成 {len(raw_chunks)} 個片段。")

    # ==========================================
    # 演算法一：建構傳統統計特徵 BM25 詞頻索引
    # ==========================================
    print("🧮 3. 正在訓練 BM25 Okapi 傳統檢索模型...")
    tokenized_corpus = [list(chunk) for chunk in raw_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(BM25_PKL_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "metadata": metadata_list, "raw_chunks": raw_chunks}, f)
    print("💾 BM25 模型已本地持久化成功。")

    # ==========================================
    # 演算法二：計算語意向量並注入分散式向量資料庫 Qdrant
    # ==========================================
    print("🤖 4. 正在透過 Gemini API 計算語意向量 (Dense Embeddings)...")
    client = genai.Client()
    
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=6333)
    
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    
    points = []
    for i, chunk in enumerate(raw_chunks):
        # 使用 2026 年最穩定的嵌入模型呼叫方式
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk,
            config=types.EmbedContentConfig(
                output_dimensionality=768  # <-- 強制截斷為 768 維度，完美相容 Qdrant 配置！
            )
        )
        # 新版 SDK 的結構解析
        embedding = response.embeddings[0].values
        
        points.append(PointStruct(
            id=i,
            vector=embedding,
            payload=metadata_list[i]
        ))
        
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"🚀 語意向量已成功注入 Qdrant 資料庫！知識庫構建完成。")

if __name__ == "__main__":
    build_knowledge_base()