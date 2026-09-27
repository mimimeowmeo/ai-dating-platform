# 推薦 worker（外貌向量）

把使用者的主照片算成「只看臉」的外貌向量，給探索頁排序用（排序在 [`apps/api/src/social.ts`](../../apps/api/src/social.ts)、[`apps/api/src/discovery-rank.ts`](../../apps/api/src/discovery-rank.ts)）。
只負責算向量：不碰資料庫（ADR 0002），向量由 NestJS 寫進 `appearance_embeddings`。

| 步驟 | 做法 | 模型與授權 |
| --- | --- | --- |
| 1. 找臉 | 信心 0.6 以上、取最大的一張臉；找不到就回 `embedding: null` | YuNet `face_detection_yunet_2023mar.onnx`（MIT） |
| 2. 去背只留臉 | 只留臉部皮膚，加上眼睛高度的配件（眼鏡）；補成凸包、背景填灰。分割失敗時退回橢圓裁切 | MediaPipe `selfie_multiclass_256x256.tflite`（Apache-2.0） |
| 3. 算向量 | 512 維、長度正規化成 1 | CLIP ViT-B/32，Hugging Face revision `3d74acf`（MIT） |
| 4. 扣掉「戴眼鏡」方向 | 由 NestJS 做（[`rec-jobs.ts`](../../apps/api/src/rec-jobs.ts)）：資料庫有同版本的方向才扣 | — |

授權全文在 `third_party/licenses/`，隨映像一起散布。模型檔在 build 時下載並驗證大小與 sha256（[`tools/fetch_models.py`](tools/fetch_models.py)），執行期不連網。
做法和 `ml/experiments/appearance_face_clip.py` 的 `--mask seg-face` 相同，裁切程式是它的移植版。

模型版本：`clip-vit-b32@3d74acf+seg-face+seg3`；NestJS 扣掉戴眼鏡方向後存成 `…+glasses-project`。
改了裁切規則要把 [`appearance/pipeline.py`](appearance/pipeline.py) 的 `SEG_VERSION` 加一，版本字串才分得出新舊向量。

## 執行

屬於 `recommendation` profile，`docker compose up -d` 預設不會啟動：

```bash
docker compose --profile recommendation up -d recommendation-worker
```

- 映像約 3 GB（PyTorch CPU 版與 605 MB 的 CLIP 權重），第一次 build 要下載約 1 GB。
- 沒啟動時探索頁照常運作：API 不把照片放進佇列，沒有向量的人外貌分數給中間值。啟動後 API 每分鐘自動補排。
- CI、`docker-compose.test.yml`、GCP 測試站（`docker-compose.ci.yml`）都沒有這個服務。

## 環境變數

| 變數 | 預設 | 用途 |
| --- | --- | --- |
| `REDIS_URL` | （必填） | BullMQ 佇列 |
| `REC_MODELS_DIR` | `/app/models` | 模型資料夾（映像裡已放好） |

## 背景工作

| 佇列 | 方向 | job 名稱 | data |
| --- | --- | --- | --- |
| `rec-jobs` | NestJS → worker | `embed-appearance` | `{photoId, image}`，image 是主照片的 JPEG 原檔（base64；上傳時已轉成長邊 1600px 以內） |
| `rec-results` | worker → NestJS | `embed-appearance` | `{photoId, modelVersion, embedding}`，找不到臉時 embedding 是 `null` |

- **在線訊號**：worker 每 20 秒寫 `rec-worker:online`（60 秒過期），關閉時先刪掉。NestJS 看到它才排照片，所以 worker 沒在跑時照片不會堆在 Redis。
  不用 BullMQ 的 `getWorkers()`：Node 版用 base64 的佇列名稱辨識 worker 連線，Python 版的名稱對不上。
- **送原檔、不重新壓縮**：`ml/` 離線算種子向量時讀的也是原檔。重新壓縮會讓少數照片的去背範圍改變
  （實測一張戴運動太陽眼鏡的照片，額頭一下算進臉、一下不算，相似度只剩 0.93）。
- **照片不留在 Redis**：worker 取到 job 就把 data 覆寫成 `{"redacted": true}`；NestJS 那邊 `attempts: 1`、成功或失敗都立刻刪除 job。
- **誰來排**：主照片上傳、刪掉主照片由下一張遞補時由 NestJS 排；漏掉的（worker 停機期間上傳的）由 NestJS 每分鐘補排一批（20 張，佇列清空才排下一批）。
  找不到臉的照片也會記一列（向量是空的），不會一直重排。
- **錯誤**：只丟固定代碼 `UNSUPPORTED_JOB`、`INVALID_JOB_INPUT`、`INVALID_IMAGE`，日誌不帶 job 內容。

## 測試

```bash
docker compose --profile recommendation run --rm --no-deps recommendation-worker python -m unittest discover -s tests -v
```

| 檔案 | 內容 |
| --- | --- |
| `tests/test_cutout.py` | 去背規則：留臉與眼鏡、不留頭帶、找不到臉或範圍太大就退回橢圓裁切 |
| `tests/test_worker.py` | 先清掉照片再處理、輸入驗證、錯誤只丟固定代碼、在線訊號 |
| `tests/test_real_models.py` | 用映像裡的真模型：空白圖找不到臉、壞檔判為無效圖片、版本字串 |

NestJS 端的整合測試在 `apps/api/test/integration.test.mjs`（「外貌向量：worker 在線才排照片…」）：本機開著 worker 時用真的，沒開時測試自己模擬一個。

## 已執行驗證

2026-09-27，在本機堆疊（1 萬個種子帳號）上：

| 項目 | 結果 |
| --- | --- |
| 單元測試（容器內） | 16 項全過 |
| 裁切程式 vs `ml/` 實驗版 | 500 張資料庫照片逐像素相同 |
| 線上路徑 vs 離線匯入的向量 | 40 張種子主照片經 API → worker → 扣眼鏡方向寫回，和 `ml/` 匯入的向量相似度全部 1.0000 |
| 補排 | 啟動 worker 後 49 張沒有向量的主照片約 3 分鐘處理完；其中 3 張找不到臉，記成空的一列 |
| API 整合測試 | 開著真 worker 與關掉 worker（測試自己模擬）兩種情況都過 |

一開始照片先縮到 800px、重新壓縮再送，40 張裡有 1 張相似度只剩 0.93，其餘 0.985 以上，所以改成送原檔。

## 已知限制

- **CLIP 不是為上線產品設計的**：官方 model card 寫明任何部署用途目前都不在預期範圍，監控與人臉辨識類用途則一律不在範圍內。內部測試可以用；公開上線前要換模型或做法律評估。
- **「戴眼鏡」方向用 CelebA 的標註估出**：CelebA 限非商業研究，方向只存在本機資料庫、不進版控；真實使用者沒有標註，上線前要另想估法。
- **深膚色的去背可能較差**：分割模型 model card 的評估裡，256×256 版本在最深膚色組（Monk 9–10）的 IoU 是 68.25，全體是 77.23。
- **只用 CPU、一次一張**（`concurrency 1`）；數千張的大量重算請用 `ml/` 的離線腳本算好再 `pnpm db:import-appearance`。
- **換版本不會自動重算**：排序時不分 `model_version`，新舊版本的向量會混在一起比；換模型或裁切規則後要整批重算。
