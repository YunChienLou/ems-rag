import os
import pickle
import time
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from google import genai
from google.genai import types

# 基礎配置
PDF_PATH = "/app/ntpc_ems_protocol.pdf"
CACHE_TXT_PATH = "/app/app/cleaned_text.txt"  # <-- 乾淨文本快取路徑
BM25_PKL_PATH = "/app/app/bm25_model.pkl"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
COLLECTION_NAME = "ntpc_ems_protocols"

def split_by_markdown_headers(full_text: str):
    """依據 Markdown 的三級標題 (###) 進行語意區塊精準切片"""
    chunks = []
    parts = full_text.split("### ")
    
    if parts[0].strip():
        chunks.append(parts[0].strip())
        
    for part in parts[1:]:
        if part.strip():
            chunks.append("### " + part.strip())
    return chunks

def build_knowledge_base():
    cleaned_full_text = ""

    # ==========================================
    # 核心快取防禦：有快取就直接讀，不再呼叫 AI
    # ==========================================
    if os.path.exists(CACHE_TXT_PATH):
        print(f"📦 【偵測到本地快取】直接讀取快取文本，跳過 AI 轉譯！")
        with open(CACHE_TXT_PATH, "r", encoding="utf-8") as f:
            cleaned_full_text = f.read()
    else:
        print("📥 1. 找不到本地快取，開始將多模態 PDF 上傳至 Google AI...")
        if not os.path.exists(PDF_PATH):
            print(f"❌ 找不到救護準則 PDF：{PDF_PATH}")
            return

        client = genai.Client()
        ems_file = client.files.upload(file=PDF_PATH)
        print(f"✅ 上傳成功！檔案識別碼: {ems_file.name}")

        print("⏳ 2. 等待雲端多模態預處理...")
        while ems_file.state.name == "PROCESSING":
            time.sleep(2)
            ems_file = client.files.get(name=ems_file.name)
        
        if ems_file.state.name == "FAILED":
            print("❌ Google 雲端處理 PDF 失敗。")
            return

        print("🤖 3. 呼叫 gemini-3.5-flash 執行 AI 結構化轉譯 (僅需執行這一次)...")
        parsing_prompt = (
            "你是一個高精度的醫療文獻 OCR 與文字轉譯機器人。\n"
            "請將整份文件完整轉譯為乾淨的繁體中文 Markdown 格式。\n"
            "必須保留所有章節標題（### 一、 、### 二、 等）、步驟與藥物劑量，不要有任何遺漏與摘要。"
        )

        response = client.models.generate_content(
            model="models/gemini-3.5-flash",
            contents=[ems_file, parsing_prompt]
        )
        cleaned_full_text = response.text

        # 將成果寫入本地快取，保護錢包與時間
        with open(CACHE_TXT_PATH, "w", encoding="utf-8") as f:
            f.write(cleaned_full_text)
        print(f"💾 轉譯成果已寫入快取：{CACHE_TXT_PATH}")

        # 清理雲端
        client.files.delete(name=ems_file.name)

    # ==========================================
    # 4. 後續切片與向量寫入（這段跑起來只要 1~2 秒）
    # ==========================================
    print("📝 4. 啟動 Markdown 語意區塊精準切片...")
    raw_chunks = split_by_markdown_headers(cleaned_full_text)
    
    metadata_list = []
    for c_idx, chunk in enumerate(raw_chunks):
        metadata_list.append({
            "page": 1,  # 標題切片不綁死硬性分頁
            "chunk_index": c_idx,
            "text": chunk
        })

    print(f"✂️ 5. 語意切片完成，共生成 {len(raw_chunks)} 個完整技術區塊。")

    print("🧮 6. 正在訓練 BM25 Okapi 傳統檢索模型...")
    tokenized_corpus = [list(chunk) for chunk in raw_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_PKL_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "metadata": metadata_list, "raw_chunks": raw_chunks}, f)

    print("🤖 7. 正在透過 gemini-embedding-001 計算向量並同步至 Qdrant...")
    client = genai.Client()  # 確保計算向量時有 client
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=6333)
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    
    points = []
    for i, chunk in enumerate(raw_chunks):
        res = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        embedding = res.embeddings[0].values
        points.append(PointStruct(id=i, vector=embedding, payload=metadata_list[i]))
        
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"🚀 RAG 知識庫全面構建完成！")

if __name__ == "__main__":
    build_knowledge_base()