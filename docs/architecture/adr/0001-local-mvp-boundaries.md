# ADR 0001：Phase 1–7 本機 MVP 邊界

狀態：本機實作決策，正式產品政策待確認。

使用者要求一次完成 Phase 1–7，原文件逐項停止的流程不適用於本次工作。

- Next.js UI、NestJS API、Prisma PostgreSQL、FastAPI 與 Python BullMQ worker 分離。
- 興趣、活動、食物採固定白名單與 PostgreSQL 字串陣列；目前不需要管理後台。日後需要多語目錄、動態分類時再正規化，與原 ERD 概念表有所簡化。
- 本機最低年齡 18 歲；城市中心座標用於雙向距離偏好，公開回應不包含生日與座標。
- 未驗證使用者可完成本機探索／配對／聊天；驗證標記只接受真人驗證服務的成功結果。正式產品若需強制驗證方可交友，需另行定義。
- 未選定模型、活體流程、身分參照來源或判定門檻；因此 Phase 5 只有完整服務串接與錯誤流程，尚不能宣稱真實辨識完成。
- 自拍只在記憶體處理，不存物件儲存，不記錄影像、token 或密碼。驗證紀錄只存狀態、模型版本與結果欄位。
- 本期前端聊天使用 HTTP 建立訊息、Socket.IO 接收、typing、presence、read；API 同時支援 Socket.IO 傳送並共用授權與去重服務。
- 使用單一 API instance。Redis 處理 rate limit 與 BullMQ；多 instance Socket.IO Redis adapter、跨 instance presence 與推播屬後續擴充。
- 封鎖與結束配對不刪除歷史訊息，但停止成員存取和收發。解除封鎖不自動恢復配對。
- 本機容器只綁定 loopback，不代表已具備正式部署、TLS、備份、內容審核或法遵驗收。

MinIO 映像使用官方文件列出的 Quay 來源：<https://min.io/docs/minio/container/index.html>。pnpm 建置腳本依官方 allowBuilds 設定逐項啟用：<https://pnpm.io/settings/build#allowbuilds>。
