# 人臉驗證 provider（即時鏡頭動作挑戰＋大頭貼比對）

自架的真人驗證 provider，實作 [AI 服務的 provider 契約](../ai/README.md#provider-的明確契約)：
AI 服務把正面影格（或上傳的自拍）、使用者的主照片（`referenceImages`）與即時鏡頭的動作影格（`liveCapture`）POST 到 `/verify`，這裡回傳 `ProviderResult`。

| 步驟 | 模型 | 授權 |
| --- | --- | --- |
| 人臉偵測、5 點對齊 | YuNet `face_detection_yunet_2023mar.onnx`（OpenCV `FaceDetectorYN`） | MIT |
| 1:1 比對 | SFace `face_recognition_sface_2021dec.onnx`（OpenCV `FaceRecognizerSF`） | Apache-2.0 |
| 被動防偽 | MiniFASNet V2＋V1SE（Silent-Face-Anti-Spoofing，build 時轉成 ONNX） | Apache-2.0 |

授權全文在 `third_party/licenses/`，隨映像一起散布。**程式碼與權重的授權允許商用，但 SFace 權重的訓練資料集沒有公開**
（opencv_zoo issue #124、#313 都沒有回覆），正式上線前需要法律評估。

## 判定政策（版本 3）

請求有兩種：
- **即時鏡頭**（前端現在的流程）：`imageBase64` 是正面影格，`liveCapture.frames` 是每個動作各一張影格。
  每張動作影格都用 YuNet 的 5 個點重算頭部角度，和正面影格比較（算法與門檻見 [`app/pose.py`](app/pose.py)、[`app/policy.py`](app/policy.py)），
  也檢查和正面影格是同一個人。角度在「兩眼連線」座標系計算，歪頭或轉動照片不會改變 yaw／pitch；
  動作影格和正面影格的側傾差不能超過 15°，正面影格本身也要大致正對鏡頭。
  全部通過、每張影格的被動防偽與大頭貼比對也通過，才回 `verified`。
- **只有上傳的自拍檔**（沒有 `liveCapture`）：無法證明是活人當下拍攝，**全部通過也只回 `unavailable`**。

| 情況 | 回傳 |
| --- | --- |
| 沒有主照片（API 已先回 409 擋下，這裡是直接呼叫 provider 時的保險） | `unavailable / REFERENCE_PHOTO_REQUIRED` |
| 自拍沒有臉／多張臉／臉太小 | `rejected / NO_FACE_DETECTED`、`MULTIPLE_FACES_DETECTED`、`FACE_TOO_SMALL` |
| 主照片同上 | `unavailable`，代碼加 `REFERENCE_` 前綴（問題在照片，不撤銷既有的驗證狀態） |
| 動作影格沒有臉／多張臉／臉太小 | 同上代碼加 `ACTION_` 前綴 |
| 動作沒做到（角度變化不夠、方向相反、側傾變化超過 15°），或正面影格沒有正對鏡頭 | `rejected / CHALLENGE_FAILED` |
| 動作影格和正面影格不是同一人 | `rejected / FACE_CHANGED_DURING_CAPTURE` |
| 任何一張影格（正面或動作）被動防偽判為翻拍 | `rejected / SPOOF_SUSPECTED`（`livenessScore` 是所有影格中最低的真人機率） |
| 臉不是同一人 | `rejected / FACE_MISMATCH` |
| 只有上傳的自拍檔，其他都通過 | `unavailable / LIVE_CAPTURE_REQUIRED`（分數照樣回傳，記在驗證紀錄） |
| 即時鏡頭，全部通過 | `verified / VERIFICATION_PASSED` |

`rejected` 會讓 API 把 `isVerified` 設成 false；`unavailable` 不改變驗證狀態。

**動作挑戰只擋得住呈現攻擊（presentation attack）**：拿照片、螢幕或預錄影片對著真的鏡頭。
**擋不住注入攻擊（injection attack）**：伺服器無法證明影格是鏡頭當下拍的，繞過前端直接呼叫 API 送出事先準備好的影格，
或用虛擬攝影機、即時換臉（deepfake），都能避開動作挑戰；被動防偽對「沒有翻拍過的原始數位照片」也會判成真人。
要擋這類攻擊需要伺服器驅動、事先無法準備的挑戰，或商用的注入攻擊偵測（injection attack detection）。

門檻集中在 [`app/policy.py`](app/policy.py)，每個數值都註明出處，**都還沒用台灣使用者的資料校準**。
依 [AI-SPEC](../../docs/ai/AI-SPEC.md)，改門檻要有測試、升 `POLICY_VERSION`，並經人工核准。

## 啟用

預設不啟動（compose 的 `face` profile），CI 與測試用的 compose 也不包含它。

```sh
pnpm setup                                   # 產生 AI_VERIFICATION_PROVIDER_TOKEN（已有就保留）
# .env 設定 AI_VERIFICATION_PROVIDER_URL=http://face:8100/verify
docker compose --profile face up -d --build face ai
```

啟用後 `pnpm test` 的真人驗證測試會失敗：測試用純色圖當自拍，provider 會回 `rejected / NO_FACE_DETECTED`，
而測試預期的是沒有模型時的 `unavailable`。跑 `pnpm test` 前請把 URL 清空並重啟 ai。

## 建置

```sh
docker build -f services/face/Dockerfile -t dating-face .   # 從專案根目錄
```

分兩個階段：

1. **models**：`tools/fetch_models.py` 從鎖定 commit 的網址下載 4 個模型檔並驗證 sha256；
   `tools/export_minifasnet.py` 用 PyTorch 2.5.1（CPU）把 MiniFASNet 轉成 ONNX，並檢查 onnxruntime 與 PyTorch 的輸出一致，不一致就讓 build 失敗。
2. **執行期**：只有 OpenCV、onnxruntime 與轉好的模型，不含 torch。

## 測試

```sh
docker compose --profile face run --rm --no-deps -v "$PWD/services/face/tests:/app/tests:ro" \
  face python -m unittest discover -s tests -t . -v
```

- `test_pipeline.py`、`test_live.py`、`test_api.py`、`test_crop.py`、`test_imaging.py`：用假模型測每一種判定（含動作挑戰與頭部角度）、HTTP 邊界、回應契約、裁切與 BGR／EXIF 處理。
- `test_real_models.py`：在容器內載入真的模型，用合成影像確認模型接得上、輸出格式正確。依 SECURITY.md，repo 不放人臉照片，所以這裡不測準確度。
- `smoke_local_images.py`：手動用本機的兩張照片跑真模型，只印出判定結果（用法見檔案開頭）。

## 已知限制

- 被動防偽（MiniFASNet）沒有 iBeta／ISO 30107-3 認證，跨環境的準確度不穩定；擋不住即時換臉與虛擬攝影機。
- MiniFASNet 是用 RetinaFace 的框訓練的，這裡改用 YuNet 的框，裁切範圍會有差異，門檻需要校準。
- 同一組模型一次只處理一個請求（YuNet 的 `setInputSize` 會改內部狀態）；排隊超過 10 秒，AI 服務會記成 `PROVIDER_TIMEOUT`。
