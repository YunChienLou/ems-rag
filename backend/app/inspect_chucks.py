import os
import pickle
from qdrant_client import QdrantClient

BM25_PKL_PATH = "/app/app/bm25_model.pkl"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
COLLECTION_NAME = "ntpc_ems_protocols"

def inspect_my_chunks():
    print("📊 ===== RAG 切片質量定量分析 =====")
    
    # 1. 讀取本地 BM25 暫存檔（裡面有完整的 metadata 和原始文字）
    if not os.path.exists(BM25_PKL_PATH):
        print("❌ 找不到本地快取檔案，請先確認 ingest.py 是否成功執行。")
        return
        
    with open(BM25_PKL_PATH, "rb") as f:
        data = pickle.load(f)
        metadata_list = data["metadata"]
        raw_chunks = data["raw_chunks"]
    
    total_chunks = len(raw_chunks)
    chunk_lengths = [len(c) for c in raw_chunks]
    avg_length = sum(chunk_lengths) / total_chunks
    
    print(f"總切片數量 (Total Chunks): {total_chunks} 個")
    print(f"平均切片字數 (Avg Chunk Length): {avg_length:.1f} 字")
    print(f"最大切片字數 (Max Chunk Length): {max(chunk_lengths)} 字")
    print(f"最小切片字數 (Min Chunk Length): {min(chunk_lengths)} 字")
    
    print("\n🔍 ===== Qdrant 資料庫實體定性抽查 =====")
    # 2. 連線到 Qdrant 撈出真實寫入的 Payload
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=6333)
        # 隨便撈出前 2 筆點資料 (Points)
        results, _ = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=2,
            with_payload=True,
            with_vectors=False
        )
        
        for point in results:
            p_id = point.id
            payload = point.payload
            print(f"\n📍 【Qdrant Chunk ID: {p_id}】(對應 PDF 第 {payload['page']} 頁, 第 {payload['chunk_index']} 個切片)")
            print("-" * 50)
            print(payload['text'])
            print("-" * 50)
            
    except Exception as e:
        print(f"連線 Qdrant 失敗: {e}")

if __name__ == "__main__":
    inspect_my_chunks()