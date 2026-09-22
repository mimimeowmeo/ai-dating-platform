"""AI 推薦回覆（reply suggestions）功能的 AI 端實作。

這個套件只做「運算」：產生推薦、向量化、對話切片、AI 話題區段、聊天室摘要、
風格卡萃取。依 ADR 0002，這裡**不連資料庫**；需要的資料都由 NestJS 放在 HTTP
請求或 BullMQ job 裡傳進來，算完的結果再交回 NestJS 寫入。

模組分工：
- schemas.py     ：API 請求／回應與模型輸出的 Pydantic 格式（對外契約）
- textutil.py    ：文字工具（字數、token 估計、emoji、簡轉繁、相似度…）
- style.py       ：風格數值統計、混合目標、風格距離、B 的反應熱度
- chunking.py    ：對話切片
- topics.py      ：AI 開啟的話題區段偵測
- safety.py      ：推薦內容的安全規則
- prompts.py     ：各任務的 prompt 與版本號
- llm.py         ：建立 Gemini／Ollama 模型、執行 agent、錯誤轉換
- embeddings.py  ：Gemini 向量化
- suggest.py     ：產生推薦的主流程
- extraction.py  ：風格卡萃取與聊天室摘要
- service.py     ：把以上元件組起來，給 API 與 worker 共用
- api.py         ：FastAPI 內部路由
- worker.py      ：BullMQ 背景 worker（queue `ai-jobs` → `ai-results`）

完整規格見 docs/ai/REPLY-SUGGESTIONS-SPEC.md。
"""

import os

# Pydantic AI 第一次執行 agent 時會印出推廣 Logfire 的橫幅；關掉以免污染服務日誌。
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
