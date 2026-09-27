# 人臉驗證 provider（即時鏡頭動作挑戰＋大頭貼比對）

自架的真人驗證 provider，實作 [AI 服務的 provider 契約](../ai/README.md#provider-的明確契約)：
AI 服務把正面影格（或上傳的自拍）、使用者的主照片（`referenceImages`）與即時鏡頭的動作影格（`liveCapture`）POST 到 `/verify`，這裡回傳 `ProviderResult`。

| 步驟 | 模型 | 授權 |
| --- | --- | --- |
| 人臉偵測、5 點對齊 | YuNet `face_detection_yunet_2023mar.onnx`（OpenCV `FaceDetectorYN`） | MIT |
| 1:1 比對 | SFace `face_recognition_sface_2021dec.onnx`（OpenCV `FaceRecognizerSF`） | Apache-2.0 |
| 被動防偽 | MiniFASNet V2＋V1SE（Silent-Face-Anti-Spoofing，build 時轉成 ONNX） | Apache-2.0 |
| 動作挑戰的頭部角度 | MediaPipe Face Landmarker `face_landmarker.task`（float16 第 1 版，和前端同一個檔案） | Apache-2.0（套件與三個模型的 model card） |

授權全文在 `third_party/licenses/`，隨映像一起散布。**程式碼與權重的授權允許商用，但 SFace 權重的訓練資料集沒有公開**
（opencv_zoo issue #124、#313 都沒有回覆），正式上線前需要法律評估。

## 判定政策（版本 4）

請求有兩種：
- **即時鏡頭**（前端現在的流程）：`imageBase64` 是正面影格，`liveCapture.frames` 是每個動作各一張影格。
  正面影格與每張動作影格都用 MediaPipe Face Landmarker 重算頭部角度，和正面影格比較（算法與門檻見 [`app/pose.py`](app/pose.py)、[`app/policy.py`](app/policy.py)）。
  前端 [`apps/web/lib/head-pose.ts`](../../apps/web/lib/head-pose.ts) 用**同一個模型檔、同一組點、同一套公式**，而且每個門檻都比這裡嚴（動作要 1.5 倍變化量），
  所以前端判定做到的動作，這裡原則上也會通過；剩下的差異來自前端 VIDEO 模式（跨幀追蹤）與這裡 IMAGE 模式、JPEG 壓縮。
  版本 3 以前改用 YuNet 的 5 個點重算，和前端量到的變化量對不上：policy-3 的 42 筆紀錄有 28 筆 `CHALLENGE_FAILED`，`FACE_MISMATCH` 是 0 筆。
  也檢查動作影格和正面影格是同一個人。角度在「兩眼連線」座標系計算，歪頭或轉動照片不會改變 yaw／pitch；
  動作影格和正面影格的側傾差不能超過 15°，正面影格本身也要大致正對鏡頭。
  全部通過、每張影格的被動防偽與大頭貼比對也通過，才回 `verified`。
- **只有上傳的自拍檔**（沒有 `liveCapture`）：無法證明是活人當下拍攝，**全部通過也只回 `unavailable`**。

| 情況 | 回傳 |
| --- | --- |
| 沒有主照片（API 已先回 409 擋下，這裡是直接呼叫 provider 時的保險） | `unavailable / REFERENCE_PHOTO_REQUIRED` |
| 自拍沒有臉／多張臉／臉太小 | `rejected / NO_FACE_DETECTED`、`MULTIPLE_FACES_DETECTED`、`FACE_TOO_SMALL` |
| 主照片同上 | `unavailable`，代碼加 `REFERENCE_` 前綴（問題在照片，不撤銷既有的驗證狀態） |
| 動作影格沒有臉／多張臉／臉太小 | 同上代碼加 `ACTION_` 前綴 |
| 動作沒做到（角度變化不夠、方向相反、側傾變化超過 15°）、正面影格沒有正對鏡頭，或 Face Landmarker 沒有剛好找到一張臉 | `rejected / CHALLENGE_FAILED`（另在 log 印一行診斷資料，見下節） |
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

### 動作挑戰的診斷 log

對外只有 `CHALLENGE_FAILED` 一個原因碼。沒過時，容器 log 會多一行 JSON，`requestId` 就是 `verification_records.id`：

```json
{"event": "CHALLENGE_FAILED", "requestId": "<驗證紀錄 id>", "policy": "4", "reason": "INSUFFICIENT", "action": "look_up", "yawChange": 0.02, "pitchChange": -0.05, "rollChange": 1.3}
```

| `reason` | 意思 | 附帶欄位 |
| --- | --- | --- |
| `NEUTRAL_POSE_UNAVAILABLE` | 正面影格量不到角度（Face Landmarker 沒有剛好找到一張臉） | — |
| `NEUTRAL_NOT_FACING` | 正面影格沒有正對鏡頭（yaw 絕對值 > 0.15 或 roll 絕對值 > 20°） | `neutralYaw`、`neutralRoll` |
| `POSE_UNAVAILABLE` | 動作影格量不到角度 | `action` |
| `ROLL_CHANGED` | 側傾變化超過 15° | `action`、三個變化量 |
| `INSUFFICIENT` | 變化量沒有朝正確方向達到門檻（方向相反看正負號） | `action`、三個變化量 |

只記變化量與正面影格的朝向，不記影像、臉部點座標與 pitch 的絕對值（它反映五官比例）；通過的驗證不印任何東西。

```sh
docker compose logs face | grep CHALLENGE_FAILED                    # 在專案根目錄執行
docker logs ai-dating-platform-face-1 2>&1 | grep CHALLENGE_FAILED  # 任何資料夾都能執行
```

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

1. **models**：`tools/fetch_models.py` 從鎖定 commit（或固定版本）的網址下載 5 個模型檔並驗證 sha256；
   `face_landmarker.task` 必須和前端 [`prepare-mediapipe.mjs`](../../apps/web/scripts/prepare-mediapipe.mjs) 鎖定同一個檔案，更新時兩邊一起改。
   `tools/export_minifasnet.py` 用 PyTorch 2.5.1（CPU）把 MiniFASNet 轉成 ONNX，並檢查 onnxruntime 與 PyTorch 的輸出一致，不一致就讓 build 失敗。
2. **執行期**：只有 OpenCV、onnxruntime、MediaPipe 與準備好的模型，不含 torch。
   - MediaPipe 用 `--no-deps` 安裝（[`requirements-nodeps.txt`](requirements-nodeps.txt)）：它宣告的 `opencv-contrib-python` 會和 `opencv-python-headless` 搶同一個 `cv2`。
   - MediaPipe 的原生函式庫要 `libEGL.so.1`、`libGLESv2.so.2`（只用 CPU，不會真的呼叫）。`libegl1` 強制相依 Mesa（約 190 MB），
     所以只解出它的檔案、不裝 Mesa，細節見 Dockerfile。映像約 920 MB（加 MediaPipe 前約 650 MB）。

## 測試

```sh
docker compose --profile face run --rm --no-deps -v "$PWD/services/face/tests:/app/tests:ro" \
  face python -m unittest discover -s tests -t . -v
```

- `test_pipeline.py`、`test_live.py`、`test_api.py`、`test_crop.py`、`test_imaging.py`：用假模型測每一種判定（含動作挑戰與頭部角度）、HTTP 邊界、回應契約、裁切與 BGR／EXIF 處理。
- `test_real_models.py`：在容器內載入真的模型，用合成影像確認模型接得上、輸出格式正確，Face Landmarker 能從別的執行緒呼叫、關閉時不留錯誤。依 SECURITY.md，repo 不放人臉照片，所以這裡不測準確度。
- `smoke_local_images.py`：手動用本機的兩張照片跑真模型，只印出判定結果（用法見檔案開頭）。

## 已知限制

- 被動防偽（MiniFASNet）沒有 iBeta／ISO 30107-3 認證，跨環境的準確度不穩定；擋不住即時換臉與虛擬攝影機。
- MiniFASNet 是用 RetinaFace 的框訓練的，這裡改用 YuNet 的框，裁切範圍會有差異，門檻需要校準。
- 同一組模型一次只處理一個請求（YuNet 的 `setInputSize` 會改內部狀態）；排隊超過 10 秒，AI 服務會記成 `PROVIDER_TIMEOUT`。
- YuNet 與 Face Landmarker 是分開偵測的：YuNet 找到一張臉、Face Landmarker 卻沒有剛好找到一張時，回 `CHALLENGE_FAILED`（fail closed）。
- 前端 VIDEO 模式與這裡 IMAGE 模式量到的角度不會完全相同；前端 1.5 倍的餘裕夠不夠，要看上線後診斷 log 的 `INSUFFICIENT` 數量。
