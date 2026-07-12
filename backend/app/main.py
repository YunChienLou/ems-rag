from fastapi import FastAPI, HTTPException
import os

app = FastAPI(
    title="新北救護技術準則 RAG 引擎",
    description="基於微服務架構與混合檢索演算法的 EMS 導航系統",
    version="1.0.0"
)

# K8s 存活探針 (Liveness Probe) 所需的接口
@app.get("/health", tags=["Infrastructure"])
async def health_check():
    """確認服務是否存活，未來可在這裡加入資料庫連線檢查"""
    return {"status": "healthy", "database": "connected"}

@app.post("/api/v1/query", tags=["RAG Core"])
async def query_protocol(request: dict):
    """
    接收救護員提問的異步接口
    """
    user_input = request.get("question")
    if not user_input:
        raise HTTPException(status_code=400, detail="Missing 'question' field")
        
    # TODO: 下一步實作 BM25 + Vector Search 的混合檢索演算法
    return {
        "query": user_input,
        "answer": "後端服務已成功啟動！下一階段我們將在此注入 Hybrid Search 演算法與 Gemini API 串接。",
        "sources": []
    }