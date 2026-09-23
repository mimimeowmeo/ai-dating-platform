# AI 服務

包含兩個功能：
- **真人驗證**（Phase 1–7，本文件前半）。
- **AI 推薦回覆**（本文件最後的「AI 推薦回覆」一節；完整規格見 [REPLY-SUGGESTIONS-SPEC](../../docs/ai/REPLY-SUGGESTIONS-SPEC.md)）。

依 [ADR 0002](../../docs/architecture/adr/0002-db-ownership.md)，AI 服務不連資料庫，只做運算。

提供私有 FastAPI 真人驗證介面與實際消費 Redis BullMQ 的 Python worker。預設沒有辨識模型；有效影像回 `unavailable / MODEL_NOT_CONFIGURED`，不會因為格式正確、存在人臉或開發模式而標為 `verified`。目前完成的是可接模型、可測試的服務邊界，**尚未完成真實模型的真人驗證驗收**。

## 執行

在本目錄執行：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn app.main:app --port 8000 --no-access-log
.venv/bin/python -m app.worker
.venv/bin/python -m unittest discover -s tests -v
REDIS_URL=redis://127.0.0.1:6379/0 .venv/bin/python -m tests.integration_queue
AI_INTERNAL_TOKEN='<測試 token>' .venv/bin/python -m tests.integration_http
```

從專案根目錄建置：`docker build -f services/ai/Dockerfile -t dating-ai .`。AI 容器使用預設指令，worker 容器覆寫為 `python -m app.worker`，worker healthcheck 為 `python -m app.worker_health`。

## 環境變數

| 名稱 | 用途 |
| --- | --- |
| `AI_INTERNAL_TOKEN` | NestJS → AI 的 `X-Internal-Token`；未設定時私有 API 回 503 |
| `AI_VERIFICATION_PROVIDER_URL` | 選定 provider 的完整 POST URL；空白表示未配置模型 |
| `AI_VERIFICATION_PROVIDER_TOKEN` | 呼叫 provider 的 Bearer token；URL 配置時必須同時設定 |
| `REDIS_URL` | worker 的 Redis URL，預設 `redis://localhost:6379/0` |

`GET /health` 不需要 token，回傳程序健康與 provider 是否已配置。這個健康檢查不宣稱模型已驗收。外部 provider 應使用 HTTPS；HTTP 僅適合受控的本機容器網路。

## 私有驗證介面

`POST /internal/ai/face/verify`，header `X-Internal-Token`，JSON：

```json
{
  "imageBase64": "<base64>", "mimeType": "image/jpeg", "requestId": "<UUID>",
  "referenceImages": [{"imageBase64": "<base64>", "mimeType": "image/jpeg"}],
  "liveCapture": {"challengeId": "<UUID>", "frames": [{"action": "turn_left", "imageBase64": "<base64>", "mimeType": "image/jpeg"}]}
}
```

`liveCapture` 只在即時鏡頭驗證時出現：`imageBase64` 是正面影格，`frames` 是每個動作各一張（`turn_left`、`turn_right`、`look_up`、`look_down`，最多 3 張、每張 1 MiB），錯誤碼加上 `FRAME_` 前綴。轉給 provider 時沒有 `liveCapture` 就省略這個欄位。

`referenceImages` 是身分參照（使用者的第一張主照片），最多 1 張、1 MiB 以下。NestJS 固定送 1 張：使用者沒有照片時 API 直接回 409 `AVATAR_REQUIRED`，不會呼叫這裡。欄位仍可省略（預設空陣列），交給 provider 判斷。

JPEG、PNG、WebP 必須完整解碼成功，宣告 MIME 須符合實際格式，只接受單張影像，5 MiB 以下、每邊 64–4096 像素、最多 1600 萬像素；參照照片套用相同規則，錯誤碼加上 `REFERENCE_` 前綴（例如 `REFERENCE_INVALID_IMAGE`）。不保存檔案、不提供自拍 URL、不記錄影像、EXIF 或生物特徵。格式錯誤回 400/422；輸入大小超限回 413/422；所有驗證錯誤回應不包含原始輸入。

回應包含 `status, reasonCode, modelName, modelVersion` 與可選的 `livenessScore, faceMatchScore`。無模型的 modelName/modelVersion 為 null。Provider 逾時、連線錯誤、重導向、格式錯誤一律 `unavailable`，不會退回模擬成功。

## Provider 的明確契約

傳給 provider 的輸入與上述相同（含 `referenceImages`），認證為 `Authorization: Bearer <token>`。專案內建一個自架 provider：[`services/face`](../face/README.md)。回應必須為 HTTP 200、`application/json`，完整回應最多 16 KiB，整個請求最多 10 秒。未配置時不對外連線。

Provider 除了上述 API 欄位，還必須回：

```json
{
  "status": "verified",
  "reasonCode": "VERIFICATION_PASSED",
  "modelName": "approved-provider-model",
  "modelVersion": "version-from-provider",
  "verificationType": "identity_verification",
  "livenessVerified": true,
  "identityVerified": true,
  "livenessScore": 0.99,
  "faceMatchScore": 0.98
}
```

數值僅示範 schema，不是驗證門檻。`verified` 必須有明確的活體與身分驗證決策、兩個 0–1 的有限分數以及模型名稱／版本；模型判為 `rejected` 同樣須帶名稱／版本。只回 face detection、缺少證據、布林字串、額外欄位或未知狀態都會拒絕。Provider 專用的 3 個證據欄位不會轉傳到產品 API。

服務不自行決定辨識門檻。真正啟用前仍須選定模型、建立參考身分／比對證據的來源與 provider 協定、取得適當同意並驗收活體、防冒用與錯誤率。只有自拍本身並不構成身分證據。

## BullMQ worker

- Queue：`ai-verification`；job name：`verify`；data 與 HTTP request 相同。
- Worker 使用同一份驗證流程；不包含 Phase 8/9 的向量、推薦或對話分析。
- Producer 必須將 `removeOnComplete: true`、`removeOnFail: true`，並避免重試以影像為輸入的工作。自拍在尚未消費時暫存在 Redis；Redis 必須在受控私有網路中並配置資料保留政策。預設產品流程使用同步私有 HTTP，不把自拍排入佇列。
- Worker 開始處理時先清除 job data 的影像內容，僅使用記憶體副本。佇列結果不保留任何生物特徵分數；失敗記錄使用固定代碼。
- 只有 Redis ping 成功且 worker 消費迴圈正在執行，才每 20 秒更新 `/tmp/ai-worker-heartbeat`；Redis 失聯會移除 heartbeat，60 秒未更新則健康檢查失敗。SIGINT/SIGTERM 會正常關閉 consumer。
- Compose 請設定 worker `stop_grace_period: 30s`，讓最多 15 秒的收尾時間完成。

## 已執行驗證

- 真人驗證 30 項 unittest（2026-09-23 加入 `referenceImages` 後）在本機 Python 3.14 與 Docker Python 3.12 均通過。
- 2026-09-23 接上自架 provider（`services/face`）做端對端檢查：API 讀主照片 → AI 服務 → provider，驗證紀錄正確寫入狀態、代碼、兩個分數與模型版本，`isVerified` 未被設成 true。
- Docker 映像建置成功；真實 Uvicorn HTTP 測試確認 `/health`、401、有效 token 與無模型回應。
- 隔離 Redis 的整合測試確認 Python BullMQ worker 實際取出並完成工作，工作影像已從 Redis job data 去除。
- 獨立 worker 程序的 heartbeat 在 Redis 正常時通過；停止 Redis 後檢查失敗，符合預期。
- 尚未選定真人辨識模型、尚未驗證其準確率與實際活體流程；Node producer 與 Python consumer 的互通測試仍待產品後端完成。

官方 BullMQ Python 文件：<https://docs.bullmq.io/python/introduction>。

## AI 推薦回覆

程式在 `app/reply/`，每個模組開頭都有中文說明，每個函式都有註解。

### 執行

```sh
.venv/bin/python -m uvicorn app.main:app --port 8000 --no-access-log   # API（與真人驗證同一個程序）
.venv/bin/python -m app.reply.worker                                  # 背景 worker（queue ai-jobs → ai-results）
.venv/bin/python -m app.reply.worker_health                           # worker 健康檢查
REDIS_URL=redis://127.0.0.1:6379/0 .venv/bin/python -m tests.integration_reply_queue
```

- 背景萃取預設用 **Ollama Cloud**：`OLLAMA_BASE_URL=https://ollama.com/v1`、`OLLAMA_API_KEY`、`AI_EXTRACTION_MODELS=ollama:gemma4:31b`。
- 想離線開發時才用本機 Ollama：請安裝**原生 App**（Mac 的 Docker 用不到 GPU），`OLLAMA_BASE_URL=http://localhost:11434/v1`；
  AI 服務跑在容器裡時改用 `http://host.docker.internal:11434/v1`。雲端用 `ollama/ollama` 容器，網址 `http://ollama:11434/v1`。
- 容器的 worker 指令覆寫為 `python -m app.reply.worker`，healthcheck 為 `python -m app.reply.worker_health`。

### 環境變數

| 名稱 | 預設 | 用途 |
| --- | --- | --- |
| `GEMINI_API_KEY` | （空） | Google AI Studio 的 key；也接受 `GOOGLE_API_KEY` |
| `AI_REPLY_MODELS` | `ollama:gemma4:31b,gemini-3.8-flash` | 線上推薦的備援鏈（依序嘗試）；設成 `-` 代表停用 |
| `AI_REPLY_MODEL_TIMEOUT_SECONDS` | `12` | 備援鏈中每個模型各自的逾時，太慢就換下一個 |
| `AI_REPLY_THINKING_LEVEL` | `LOW` | Gemini 3.x Flash 的思考程度 |
| `AI_EXTRACTION_MODELS` | `ollama:gemma4:31b` | 背景萃取（摘要、風格卡）的模型鏈 |
| `OLLAMA_BASE_URL` | （空） | Ollama 的 OpenAI 相容端點；Ollama Cloud 用 `https://ollama.com/v1`；沒有 `/v1` 會自動補上 |
| `OLLAMA_API_KEY` | （空） | 只有 Ollama Cloud 需要；雲端模型名稱用 `/api/tags` 列出的名字（例：`ollama:gemma4:31b`） |
| `AI_EMBEDDING_MODEL` | `gemini-embedding-2` | 向量模型；所有向量必須同一個模型 |
| `AI_EMBEDDING_DIMENSIONS` | `768` | 與 pgvector 的 `vector(768)` 一致 |
| `AI_LLM_TIMEOUT_SECONDS` | `25` | 線上推薦整次請求（含備援與重試）的總逾時；也是向量化每批的逾時 |
| `AI_EXTRACTION_TIMEOUT_SECONDS` | `180` | 背景萃取每次模型呼叫的逾時 |

### 內部 API（都需要 `X-Internal-Token`）

| 路徑 | 用途 |
| --- | --- |
| `POST /internal/ai/reply-suggestions` | 產生 3～5 則推薦（第 1 則 B 100%，其餘 A 80%／B 20%） |
| `POST /internal/ai/embed` | 文字轉向量（`purpose`: `query`／`document`） |
| `POST /internal/ai/chunks` | 對話切片（30 分鐘／12 則／400 tokens／重疊 2 則），可一併向量化 |
| `POST /internal/ai/topic-spans` | 找出 AI 推薦開啟的話題區段 |
| `POST /internal/ai/conversation-summary` | 更新聊天室摘要（Ollama） |
| `POST /internal/ai/style-profile` | 萃取風格卡（Ollama＋向量化） |

請求與回應格式以 `app/reply/schemas.py` 為準。可預期的錯誤回 503 與固定代碼
（`LLM_NOT_CONFIGURED`、`LLM_UNAVAILABLE`、`EMBEDDING_*`、`EXTRACTION_*`）；格式錯誤回 422 且不回顯輸入。

### 背景工作

- Queue `ai-jobs`，job 名稱：`chunk-embed`、`topic-spans`、`summarize`、`build-style`；job data 與對應 HTTP 請求相同。
- 結果放進 queue `ai-results`（同名 job），資料為 `{sourceJobId, name, key, result}`，由 NestJS 取出寫入資料庫。
- 失敗時丟出固定代碼，錯誤紀錄不含聊天內容；worker concurrency 為 1（萃取模型一次處理一個請求）。

### 測試

- `tests/test_reply_*.py`：63 項單元測試，全部用 Pydantic AI 的 `FunctionModel` 與假的向量服務，**不會連網、不花額度**。
- `tests/live_reply_smoke.py`：手動執行、會呼叫**真的模型**（Ollama／Gemini，依 `.env` 設定），用來確認模型真的接上；不會被自動執行。
- `tests/integration_reply_queue.py`：真實 Redis 的 BullMQ 串接測試（用唯一 queue 名稱，結束時只清掉自己的 queue）。
- 2026-09-23 本機 Python 3.14 與 Docker Python 3.12：108 項單元測試（含真人驗證 30 項）全數通過；Redis 整合測試在加入 `referenceImages` 前通過；
  `live_reply_smoke.py` 實測 Ollama Cloud（gemma4:31b）與 Gemini 向量化正常。
