import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse  # <-- 引入正規 JSONResponse 處理編碼
from pydantic import BaseModel
from qdrant_client import QdrantClient
from google import genai
from google.genai import types

app = FastAPI(title="NTPC EMS RAG Engine", version="2.1.0")

BM25_PKL_PATH = "/app/app/bm25_model.pkl"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
COLLECTION_NAME = "ntpc_ems_protocols"

try:
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=6333)
    ai_client = genai.Client()
except Exception as e:
    print(f"⚠️ 基礎組件初始化警告: {e}")

class QueryRequest(BaseModel):
    question: str

def reciprocal_rank_fusion(bm25_results, qdrant_results, k=60, top_n=3):
    rrf_scores = {}
    metadata_map = {}
    
    # 如果兩邊都空空如也，直接回傳空陣列，防範 0.01666 的死魚翻身
    if not bm25_results and not qdrant_results:
        return []
        
    for rank, item in enumerate(bm25_results):
        doc_id = item["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        metadata_map[doc_id] = item["metadata"]
        
    for rank, item in enumerate(qdrant_results):
        doc_id = item.id
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        metadata_map[doc_id] = item.payload

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    final_context = []
    for doc_id, score in sorted_docs:
        final_context.append({
            "page": metadata_map[doc_id].get("page", 1),
            "text": metadata_map[doc_id].get("text", ""),
            "score": score
        })
    return final_context

@app.post("/api/search")
async def hybrid_search(request: QueryRequest):
    if not os.path.exists(BM25_PKL_PATH):
        raise HTTPException(status_code=500, detail="BM25 索引未構建，請先執行 ingest.py")
        
    query = request.question
    
    with open(BM25_PKL_PATH, "rb") as f:
        data = pickle.load(f)
        bm25 = data["bm25"]
        metadata_list = data["metadata"]
        
    tokenized_query = list(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # 防禦性過濾：分數必須大於 0 才是真正有關鍵字匹配到
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:5]
    bm25_results = []
    for idx in top_bm25_idx:
        if bm25_scores[idx] > 0.1:  # 稍微提高關鍵字門檻
            bm25_results.append({"id": int(idx), "metadata": metadata_list[idx]})

    # 語意向量檢索
    try:
        res = ai_client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=query,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        query_vector = res.embeddings[0].values
        
        # 只撈取相似度大於一定水準的向量結果 (Qdrant 預設可以用 score 篩選，這裡交給 RRF)
        qdrant_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=5
        )
        # 防禦：過濾掉低於 0.6 相似度的不相干雜訊
        qdrant_results = [r for r in qdrant_results if r.score > 0.6]
    except Exception as e:
        qdrant_results = []
        print(f"⚠️ Qdrant 檢索失敗: {e}")

    merged_context = reciprocal_rank_fusion(bm25_results, qdrant_results, top_n=3)
    return {"query": query, "contexts": merged_context}

@app.post("/api/chat")
async def rag_chat(request: QueryRequest):
    search_data = await hybrid_search(request)
    contexts = search_data["contexts"]
    
    # 核心防禦：如果混合檢索完全撈不到任何有用的知識片段，直接觸發 Fallback 
    if not contexts:
        return JSONResponse(
            content={
                "answer": "目前救護手冊中查無此項處置規範，或目前的知識庫未完整涵蓋此技術。",
                "references": []
            },
            headers={"Content-Type": "application/json; charset=utf-8"}
        )
        
    context_str = ""
    for idx, ctx in enumerate(contexts):
        context_str += f"[參考準則片段 {idx+1}]:\n{ctx['text']}\n\n"
        
    system_prompt = (
        "你是一位專精於 EMS 的救護專家大腦。\n"
        "請嚴格根據下方提供的新北市消防局救護單項技術操作原則與處置規範回答問題。\n"
        "在回答的段落結尾，必須精確註明參考自救護手冊的第幾頁。"
    )
    
    user_prompt = f"【問題】：{request.question}\n\n【新北救護準則 Context】:\n{context_str}"
    
    try:
        response = ai_client.models.generate_content(
            model="models/gemini-3.5-flash",
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=f"{system_prompt}\n\n{user_prompt}")])]
        )
        
        # 強制使用帶有 charset=utf-8 的正規 JSONResponse，消滅 PowerShell 亂碼
        return JSONResponse(
            content={
                "answer": response.text,
                "references": contexts
            },
            headers={"Content-Type": "application/json; charset=utf-8"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API 呼叫失敗: {e}")