# ADR 0002：資料庫只由 NestJS 存取，AI 服務不碰資料庫

狀態：已採用（2026-09-22）。

## 背景

「AI 推薦回覆」需要 pgvector 儲存對話片段與風格特徵的向量，也需要讀取大量聊天紀錄做背景萃取。原本的 `ARCHITECTURE.md` 允許 AI Worker 直接寫入 PostgreSQL；規劃過程中考慮過讓 Python 用 psycopg＋pgvector-python 直接讀寫資料庫。

## 決策

- **只有 NestJS（Prisma）存取資料庫**，包含 pgvector 的寫入與相似度搜尋。
- **Python AI 服務是無狀態的運算服務**：不持有資料庫帳密，只接收資料、呼叫模型、回傳結果。
- 同步請求：NestJS 把整理好的資料放進 HTTP 請求交給 AI 服務。
- 背景工作：NestJS 把要處理的資料放進 BullMQ `ai-jobs`；AI Worker 算完後把結果放進 `ai-results`；由 NestJS 取出並寫入資料庫。

## 理由

| 理由 | 說明 |
|---|---|
| 資料表只有一個來源 | 由 Prisma migration 定義與型別化；Python 直接寫 SQL 時欄位改名不會事先報錯（microservices.io 所說的開發期耦合） |
| 權限規則集中 | 聊天室成員、配對有效、雙向封鎖等規則在 `Social.access()`；讓 Python 直接讀 `messages` 等於繞過 |
| AI 服務簡單好測 | 輸入 → 輸出，不需要資料庫連線；可用 Pydantic AI 的測試模型離線測試 |
| 避免互相卡住 | 大量批次查詢或長交易不會鎖住 NestJS 正在用的表（執行期耦合） |
| 結果不會遺失 | BullMQ 官方建議跨服務時用專門的結果佇列，而不是只靠 `completed` 事件 |

## 做法

- 向量欄位在 `schema.prisma` 宣告為 `Unsupported("vector(768)")?`，並啟用 `postgresqlExtensions` 預覽功能與 `extensions = [vector]`。
- 寫入：npm `pgvector` 的 `toSql()` ＋ `$executeRaw ... ${v}::vector`。
- 搜尋：`$queryRaw ... ORDER BY embedding <=> ${q}::vector LIMIT k`（cosine 距離）；先用 `conversation_id`／`user_id` 的 B-tree 索引縮小範圍做精確搜尋，資料量大到變慢時才加 HNSW。
- 所有原生 SQL 集中在同一個 `VectorStore` service，避免散落各處。
- `dating` 資料庫已安裝 pgvector 0.8.6。

## 影響

- `ARCHITECTURE.md` 的非同步流程改為「NestJS → BullMQ → AI Worker → 結果佇列 → NestJS → PostgreSQL」。
- AI 服務的依賴中不包含任何資料庫驅動程式（psycopg、pgvector-python 都不需要）。
- 每次背景工作的資料由 job 攜帶；2 萬則訊息（平均 7.1 字）約 1 MB 以內，Redis 可以承受，再大時拆成多個 job。

## 考慮過的替代方案

| 方案 | 不採用的原因 |
|---|---|
| Python 直接讀寫資料庫 | 兩種語言共用資料表，改結構要同步兩邊；繞過 NestJS 的權限規則 |
| Python 用受限的資料庫帳號（只讀 messages、只寫 AI 表） | 批次處理最有效率，但目前資料量用不到；之後單次工作要搬幾十萬則時再評估 |
| Python 透過 NestJS 內部 API 取資料 | 大量資料會需要很多次 HTTP 呼叫 |
| 另外架向量資料庫（Qdrant、Milvus） | 多一套基礎設施與資料同步；已有 pgvector |
