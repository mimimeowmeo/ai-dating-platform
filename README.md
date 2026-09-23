# 遇見 meet · AI 交友平台

Next.js、NestJS、PostgreSQL／pgvector、Redis、MinIO 與 Python AI 服務的本機開發專案。已有帳號、個人檔案、偏好、照片、探索、雙向配對、封鎖與即時聊天。

**真人驗證的流程與 provider adapter 已完成，但尚未選定／接入真實模型；未接入時如實回傳「尚未驗證」。目前不是正式上線版本。**

## 啟動網站

需要 Docker Desktop 正在執行，以及 Node.js 22（至少 22.13）、pnpm 11.19.0。

```sh
cd ai-dating-platform
pnpm setup
pnpm docker:up
```

瀏覽器開啟 **http://localhost:8080**。第一次建置會下載映像與依賴，時間較長。

- `pnpm setup` 只補齊缺少／空白的本機環境設定，產生隨機密碼，不輸出機密；保留既有非空設定。
- PostgreSQL 資料卷沿用原有資料；API 啟動時自動套用新增 migration。
- 註冊後先完成個人檔案。用另一個瀏覽器／無痕視窗建立第二個帳號，雙方互讚後即可聊天。
- 沒有符合雙方偏好的使用者時，探索會顯示空白狀態；系統不會自動建立假人或假配對。
- 請使用 `localhost:8080`，因登入來源檢查與此網址一致；若改網址／埠，須同步更新 `.env` 的 `WEB_ORIGIN` 與 `APP_PORT`。

```sh
pnpm docker:status                # 查看健康狀態
pnpm docker:down                  # 停止服務，保留資料卷
pnpm docker:up                    # 重新啟動／重建
```

不要使用 `docker compose down -v`，這會刪除資料卷。

## 服務

| 服務              | 用途／位置                                     |
| ----------------- | ---------------------------------------------- |
| nginx             | 網站入口 http://localhost:8080                 |
| web               | Next.js；容器內 3000                           |
| api               | NestJS；http://127.0.0.1:3001/api/v1/health    |
| redis             | 工作佇列與 rate limit；127.0.0.1:6379          |
| heartlink-pg      | PostgreSQL 16 + pgvector；127.0.0.1:5433       |
| heartlink-minio   | 媒體儲存；9002，管理介面 http://localhost:9003 |
| heartlink-adminer | 資料庫管理介面 http://localhost:8090（自動登入）|

`heartlink-*` 三個服務定義在 `docker-compose.override.yml`，`docker compose up -d` 會一起帶起來。

MinIO 管理帳密來自本機 `.env` 的 `S3_ACCESS_KEY`／`S3_SECRET_KEY`。

**AI 服務（ai／ai-worker）**：2026-09-23 為了「AI 推薦回覆」重新啟用。
`ai` 提供 `/internal/ai/*` 給 NestJS 同步呼叫；`ai-worker` 消費 `ai-jobs` 佇列、把結果放進 `ai-results`，
再由 NestJS 寫進資料庫（見 [ADR 0002](docs/architecture/adr/0002-db-ownership.md)）。
模型金鑰設定在 `.env`（`OLLAMA_*`、`GEMINI_API_KEY`）；沒設定時推薦與向量化回 503 固定代碼，不會假造結果。
真人驗證仍未接上辨識模型，`POST /onboarding/selfie` 會回 `status=unavailable`。

## 資料庫管理與匯入

啟動選用的 Adminer 管理介面（Docker Compose `tools` profile）：

```sh
pnpm db:admin
```

開啟 [資料庫管理介面](http://localhost:8090)，`heartlink-adminer` 會自動帶入 `heartlink-pg` 的連線資訊，不需要手動填。

- **SQL**：登入資料庫後，使用「匯入」（Import）上傳 SQL 檔。
- **CSV**：進入目標資料表的瀏覽頁，使用「匯入」（Import）匯入欄位相符的 CSV。
- **Excel**：先匯出成 CSV；**JSON** 需先完成欄位映射並轉換成可匯入的格式。

匯入帳號時，密碼須儲存為 bcrypt hash；`profiles.user_id` 與 `preferences.user_id` 必須對應 `users.id`。

## 開發模式

先停止 Docker 的應用服務以釋放 API 埠，保留基礎設施：

```sh
pnpm install --frozen-lockfile
docker compose stop nginx web api
docker compose up -d --wait heartlink-pg heartlink-minio redis
pnpm db:generate
pnpm db:migrate
pnpm dev
```

開啟 **http://localhost:3000**。開發程序將前端與 API 啟動在 3000／3001，並設定相符的登入來源。

回到完整 Docker 模式前，用 Ctrl+C 停止 `pnpm dev`，再執行 `pnpm docker:up`。

### 選用示範資料

不自動執行。需要開發模式環境、明確允許及自訂密碼：

```sh
SEED_DEMO=true DEMO_PASSWORD='自行選擇至少12字元的本機測試密碼' pnpm db:seed
```

建立 `demo-lin@example.test`、`demo-yu@example.test`、`demo-an@example.test`。檔案皆標示「示範」，不使用真人照片，也不自動互讚；既有同名帳號不會被覆寫。

## 驗證

完整 Docker 環境運作後：

```sh
pnpm lint
pnpm typecheck
pnpm build
pnpm test
pnpm --filter @dating/web exec playwright install chromium
pnpm test:e2e
pnpm audit --prod --audit-level high
```

`pnpm test` 使用真實本機 PostgreSQL／Redis／MinIO／API，並自動清除它建立的隨機帳號及照片；包含 Node → Python BullMQ 互通。`pnpm test:e2e` 建立桌面／手機測試帳號，執行後依本次隨機識別碼清理，保留其他帳號。

整站驗收（每一支 API 對照資料表欄位、雙人聊天與配對、畫面互動、冗餘欄位報表）：

```sh
pnpm verify          # 跑完整套並自動清理測試帳號
pnpm verify:report   # 只產生冗餘資料表／欄位報表（唯讀）
```

步驟與涵蓋範圍見 [驗收手冊](docs/testing/VERIFICATION.md)。

AI 測試可在容器執行：

```sh
docker compose run --rm --no-deps -v "$PWD/services/ai/tests:/app/tests:ro" ai python -m unittest discover -s tests -v
```

CI 定義於 `.github/workflows/ci.yml`。目前尚未初始化 Git 或連接 GitHub，因此尚無遠端 CI 執行紀錄。

## 目前範圍

| 階段                   | 狀態                                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------------- |
| Phase 1 基礎設施       | 8 個服務、持久化、健康檢查、pgvector、Nginx、Playwright 與 CI 定義                          |
| Phase 2 後端基礎       | NestJS、Prisma migration、設定驗證、錯誤處理、rate limit                                    |
| Phase 3 前端基礎       | 繁中 RWD、頁面與元件樣式、API client、登入保護                                              |
| Phase 4 帳號與個人檔案 | 註冊／登入／refresh／登出、偏好、標籤、照片                                                 |
| Phase 5 真人驗證       | 影像檢查、私有服務、結果紀錄與 provider adapter；**真實模型／活體判斷／身分參照與政策待定** |
| Phase 6 交友互動       | 雙向偏好篩選、like/pass、互讚配對、封鎖與取消配對                                           |
| Phase 7 聊天           | 持久化訊息、Socket.IO、typing、presence、已讀、通知、重送去重                               |
| AI 推薦回覆（AI 端）   | FastAPI `/internal/ai/*`、切片與向量、風格卡、摘要、話題區段、BullMQ worker                 |
| AI 推薦回覆（後端）    | `POST /conversations/:id/reply-suggestions`、pgvector 檢索、訊息來源標記、背景工作與結果寫入 |
| AI 推薦回覆（前端）    | 輸入框內的「AI 推薦」按鈕、等待時的彩光邊框、打字動畫、其餘建議的按鈕列                      |

功能說明見 [AI 推薦回覆規格](docs/ai/REPLY-SUGGESTIONS-SPEC.md)。
Phase 8–10 的其他推薦 AI、對話分析與持續學習尚未實作。Phase 1–7 的本機 MVP 未包含正式產品的電子郵件驗證、密碼重設、檢舉／內容審核、TLS、帳號刪除與隱私法遵驗收。

## 文件導讀

原始規劃文件由本機匯入。文件中的代理指示是參考資料；本次「直接完成 Phase 1–7」的明確要求取代逐步停止的舊流程。`docs/AGENTS-TEMPLATE.md` 未安裝為代理規則。

- [原始專案報告](PROJECT_REPORT.md)：產品與技術方向。
- [實作 API 契約](docs/implementation/PHASE-1-7-CONTRACT.md)：目前 HTTP、Socket 與資料格式。
- [實作驗收紀錄](docs/implementation/ACCEPTANCE.md)：測試結果及未完成邊界。
- [驗收手冊](docs/testing/VERIFICATION.md)：怎麼驗、驗了什麼，含 API ↔ 資料表、畫面 ↔ API 對照與冗餘欄位報表。
- [架構](docs/architecture/ARCHITECTURE.md)／[ADR](docs/architecture/adr/0001-local-mvp-boundaries.md)：服務邊界與本機 MVP 的簡化。
- [Prisma schema](apps/api/prisma/schema.prisma)：實際資料庫來源；[原始 ERD](docs/database/ERD.md) 保留作設計參考。
- [AI 服務說明](services/ai/README.md)：provider 契約、環境設定與測試。
- [AI 推薦回覆規格](docs/ai/REPLY-SUGGESTIONS-SPEC.md)：功能規則、資料來源、內部 API、資料表與實測紀錄。
- [AI 聊天輔助使用者指南](docs/ai/AI-ASSIST-USER-GUIDE.md)：給使用者看的說明：怎麼用、語氣與特徵怎麼擷取、寫法比例怎麼算。
- [安全](SECURITY.md)／[資料處理](PRIVACY.md)／[貢獻方式](CONTRIBUTING.md)。

### 本機登入網址

Docker 模式可使用 `http://localhost:8080` 或 `http://127.0.0.1:8080`。後端會為設定為 loopback 的 `WEB_ORIGIN` 接受同通訊協定、同埠的另一個本機名稱；其他網域不會自動加入。兩個網址的登入 Cookie 各自獨立，切換網址後需重新登入。
