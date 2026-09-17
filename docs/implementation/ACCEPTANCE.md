# Phase 1–7 本機驗收紀錄

日期：2026-09-16。驗收對象為本機 Docker 開發環境，非正式上線。

## 已驗證

- 既有 PostgreSQL 16 資料卷沿用，pgvector 0.8.6 已啟用；初始 migration 成功套用。
- PostgreSQL、Redis、MinIO、AI、AI worker、API、Web、Nginx 共 8 個服務均可建置／啟動並回報健康。
- 前端 Next.js production build、API TypeScript build 與全專案 typecheck 通過。
- ESLint 通過，程式碼經 Prettier 排版。
- 真實 API 整合測試通過：
  - 建立三個帳號、密碼錯誤與 email 重複處理。
  - 私有欄位不出現在探索回應，拒絕 mass assignment。
  - 雙向年齡偏好與距離篩選、排除自己。
  - JPEG 實際上傳／讀取、拒絕 SVG、拒絕刪除他人照片。
  - 無模型時自拍為 unavailable，帳號不會標記 verified。
  - 並行互讚只建立一組配對，重複互動不重複建立。
  - 第三者無法讀取聊天室或加入 Socket room。
  - Socket 即時傳訊、HTTP 重送去重、內容衝突拒絕、typing、已讀、通知。
  - 封鎖後既有 Socket 與 HTTP 都無法繼續收發；解除封鎖不恢復配對。
  - refresh token rotation、重放拒絕、登出後 access token 失效。
- Node.js BullMQ producer → Python worker 通過：真 Redis 消費、結果回傳、影像欄位去敏。
- AI 容器內 24 項測試全部通過：token、影像驗證、provider 契約與失敗處理、worker 狀態等。
- 正式執行依賴安全掃描：0 高風險、0 嚴重；2 個 MinIO SDK 間接依賴的中度公告，見 SECURITY.md。

## 瀏覽器驗證

Playwright 涵蓋桌面／手機首頁、登入保護、註冊、個人檔案、偏好儲存、refresh cookie 恢復登入與頁面水平溢位；另以兩個獨立桌面瀏覽器帳號實測互讚、配對與即時聊天。最終結果：**5 項通過、0 項失敗、1 項依測試規劃略過**（手機雙人聊天重複案例；雙人流程已在桌面測試通過）。額外驗證重新整理後仍顯示已讀，並以瀏覽器時鐘前進 12 分鐘驗證自動 refresh 後仍可即時聊天。

測試中已修正：註冊後導向與登入 effect 衝突；偏好尚未載入時可輸入，導致回應覆蓋剛輸入的值。所有測試帳號皆以本次 run ID 清理；API 測試同時清理照片物件。

## 尚未完成／不宣稱通過的項目

1. **Phase 5 真實驗證能力**：尚無選定模型、活體採集流程、身分參照、判定政策或正式資料政策。只有流程與 adapter 完成；無模型時正確顯示未驗證。
2. GitHub CI 僅建立工作流程檔；尚無 Git repository／remote，因此沒有遠端綠燈紀錄。
3. 尚未正式部署：TLS、備份、郵件驗證、密碼重設、檢舉／審核、帳號刪除與法遵政策未納入本機驗收。
4. API 為單一 instance；分散式 Socket.IO adapter 與跨 instance presence 尚未實作。
5. Phase 8–10 未實作；本期探索使用可檢查的偏好篩選，不聲稱已有 AI 推薦模型。
6. 原 ERD 的標籤關聯表目前簡化為固定白名單與字串陣列；UI 使用專案內共用樣式／元件，尚未引入完整 shadcn/ui 元件目錄。

## 交付狀態

網站保持運行於 http://localhost:8080。測試帳號與照片均已清理；未建立示範資料、未部署至公開網路。既有 PostgreSQL volume 保留，沒有執行 reset 或 down -v。

驗收命令：`pnpm lint`、`pnpm typecheck`、Docker 內 API／Web production build、`pnpm test`（Nginx 入口）、`pnpm test:e2e`、AI 容器 unittest、`pnpm audit --prod --audit-level high`、`docker compose ps --all`。全部完成對應檢查；遠端 GitHub CI 尚未執行。

## 2026-09-18 缺陷修正

修正 19 項在瀏覽器與 API 實測中重現的缺陷：多分頁登入與即時訊息、限流的來源 IP 判斷、個人檔案表單與照片、手機聊天空白畫面、照片排序、訊息分頁游標、線上狀態、驗證狀態與錯誤訊息／狀態碼。行為細節見[實作 API 契約](PHASE-1-7-CONTRACT.md)與 [SECURITY.md](../../SECURITY.md)。

- 資料庫新增 `sessions.previous_token_hash`、`sessions.rotated_at`（migration `20260918000000_session_refresh_overlap`）。refresh 改為在同一筆 session 上換 token，並保留 30 秒重疊期，其他分頁的 access token 與 Socket 連線不再被輪替中斷。
- API 設定 `trust proxy = 1`，限流改以反向代理轉送的真實來源 IP 計算，不再全站共用額度；前端只有 refresh 回 401／403 才視為登出。
- 回歸測試：`apps/api/test/integration.test.mjs` 新增「修正回歸」測試，`apps/web/tests/site.spec.ts` 新增多分頁、個人檔案表單與手機對話測試。以修正前的程式碼執行同一批測試會失敗（同時 refresh 其中一個回 401、未建檔仍可上傳），修正後全部通過。

驗收命令與結果：`pnpm lint`、`pnpm typecheck`、`pnpm build`、`pnpm test`（5 項）、`pnpm test:e2e`（7 通過、3 依測試規劃略過）、AI 容器 24 項 unittest、`pnpm audit --prod --audit-level high`（0 高風險／嚴重、2 中度為既有 MinIO SDK 公告）全部通過。另外以 Playwright 重跑修正前重現問題的 16 個情境，全部通過；限流以另一個容器（不同來源 IP，並偽造 X-Forwarded-For）驗證彼此不再互相影響。測試帳號與照片已清除，資料卷保留。
