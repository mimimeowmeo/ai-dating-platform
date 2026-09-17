# AI 服務（Phase 1–7）

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
{"imageBase64":"<base64>","mimeType":"image/jpeg","requestId":"<UUID>"}
```

JPEG、PNG、WebP 必須完整解碼成功，宣告 MIME 須符合實際格式，只接受單張影像，5 MiB 以下、每邊 64–4096 像素、最多 1600 萬像素。不保存檔案、不提供自拍 URL、不記錄影像、EXIF 或生物特徵。格式錯誤回 400/422；輸入大小超限回 413/422；所有驗證錯誤回應不包含原始輸入。

回應包含 `status, reasonCode, modelName, modelVersion` 與可選的 `livenessScore, faceMatchScore`。無模型的 modelName/modelVersion 為 null。Provider 逾時、連線錯誤、重導向、格式錯誤一律 `unavailable`，不會退回模擬成功。

## Provider 的明確契約

傳給 provider 的輸入與上述相同，認證為 `Authorization: Bearer <token>`。回應必須為 HTTP 200、`application/json`，完整回應最多 16 KiB，整個請求最多 10 秒。未配置時不對外連線。

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

- 24 項 unittest 在本機 Python 3.14 與 Docker Python 3.12 均通過。
- Docker 映像建置成功；真實 Uvicorn HTTP 測試確認 `/health`、401、有效 token 與無模型回應。
- 隔離 Redis 的整合測試確認 Python BullMQ worker 實際取出並完成工作，工作影像已從 Redis job data 去除。
- 獨立 worker 程序的 heartbeat 在 Redis 正常時通過；停止 Redis 後檢查失敗，符合預期。
- 尚未選定真人辨識模型、尚未驗證其準確率與實際活體流程；Node producer 與 Python consumer 的互通測試仍待產品後端完成。

官方 BullMQ Python 文件：<https://docs.bullmq.io/python/introduction>。
