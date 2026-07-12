import os
from google import genai

def discover_my_available_models():
    print("🔍 ===== 正在透過 API 查詢你帳戶當前 100% 可用的正規模型名稱 =====")
    
    # 初始化客戶端
    client = genai.Client()
    
    try:
        # 呼叫正規的 list 接口
        models = client.models.list()
        
        print("\n【你當前可以使用的模型真實名稱清單】")
        print("=" * 60)
        
        for m in models:
            # 每個 Model 物件絕對都有 name 屬性，直接印出字串
            print(f"🔹 {m.name}")
            
        print("=" * 60)
        print("✅ 查詢結束，請從上方選取一個填入 ingest.py。")
        
    except Exception as e:
        print(f"❌ 查詢失敗，錯誤訊息：{e}")

if __name__ == "__main__":
    discover_my_available_models()