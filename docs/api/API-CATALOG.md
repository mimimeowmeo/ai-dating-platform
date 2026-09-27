# API Catalog — MVP v1

Base path: `/api/v1`

## Authentication

- `POST /auth/register`
- `POST /auth/login`
- `POST /auth/logout`
- `POST /auth/refresh`
- `GET /auth/me`

## Media

- `GET /media/:photoId?u=<viewerId>&e=<expires>&s=<signature>` — 照片。`<img>` 送不出
  Authorization header，所以改用簽章授權：URL 由 API 產生、綁定觀看者、每 6 小時換一次，
  被對方封鎖後同一組 URL 會回 404。沒有簽章或簽章不符一律 403。

## Profile

- `GET /profile`
- `PUT /profile` — 建檔與編輯都要帶齊必填欄位，缺的話回 400：
  `heightCm`（100–250）、`bio`（去掉頭尾空白後至少 20 個字，以使用者看到的字〔grapheme〕計算，❤️、👍🏻、國旗都算一個字）、
  `datingGoals`（1–2 項），以及 `traits` 內 `personality`／`diet`／`value`／`lifestyle` 各至少 1 項、`interest` 至少 3 項。
  分兩階段回報：先檢查欄位格式與必填（`VALIDATION_ERROR`，同一階段的問題一起列出）；都通過後，
  再檢查各類小熱愛的數量（`TRAITS_REQUIRED`，一次列出所有不足的類別）。
- `GET /profile/:userId` — 回傳對方的卡片，含 `heightCm`（舊資料沒填時為 `null`）。
- `POST /profile/photos`
- `DELETE /profile/photos/:photoId` — 軟刪除，照片記錄與 MinIO 物件保留。

## Preferences

- `GET /preferences`
- `PUT /preferences` — `minAge`／`maxAge` 收 18–130，`minHeightCm`／`maxHeightCm` 收 130–250（沒帶就回到 130／250＝不限），
  兩組都要求下限不大於上限；`maxDistanceKm` 1–20000。身高拉桿停在 130／250 代表那一端不限
  （身高必填後，低於 130 的人才不會被預設偏好擋掉）；身高條件只過濾有填身高的人。

## Traits

- `GET /traits` — 回傳 `traits` 資料表的全部代碼（`{ category, code, label }`）。
  `dating_goal` 供「想遇見的關係」與探索偏好的「關係期待」使用，其餘五類（`personality`／`diet`／`value`／`lifestyle`／`interest`）供「我的小熱愛」使用。
- `PUT /profile` 以 `traits`（最多 40 項）與 `datingGoals`（1–2 項）送出選擇，每次都整組換掉；各類最低數量見上方 Profile。`GET /profile`、`GET /profile/:userId`、`GET /discovery` 會一併回傳。
- `PUT /preferences` 的 `preferredDatingIntent` 收 `any` 或 `dating_goal` 的代碼。

## Interests / Hobbies / Foods（舊欄位，僅供匯入相容）

- `GET /interests`
- `PUT /me/interests`
- `GET /hobbies`
- `PUT /me/hobbies`
- `GET /foods`
- `PUT /me/foods`

## Verification

- `POST /verification/challenge`（即時鏡頭：領隨機動作挑戰）
- `POST /onboarding/live`（即時鏡頭：送出正面與動作影格）
- `POST /onboarding/selfie`（舊流程：上傳自拍檔）
- `GET /verification/status`
- `POST /verification/retry`

## Discovery / Interactions

- `GET /discovery` — 最多 30 張卡片。先排除自己、按過喜歡／略過的人、雙向封鎖與配對過的人，再依三個開關處理，開關預設都開：
  - 硬篩選：雙向的年齡、性別、交往目的、身高、距離，從最新註冊的 500 人裡篩。
  - 外貌分數：候選人主照片和「我最近按喜歡的 20 人」的向量比，取最高相似度（`appearance_embeddings`），再扣略過分數（PASS V2：候選人附近有 5 個以上我最近略過的人才扣，附近也有我喜歡過的人會減輕；原規格最多扣 0.20，依種子分布乘上「1 − 同性別隨機兩人相似度的中位數」換算，女最多約 0.034、男約 0.051；「附近」門檻女 0.901、男 0.845）。
  - 興趣分數：五類標籤各算 Jaccard，依 70/15/2/5/8 加權。
  兩個分數先換成候選池裡的百分位，都開時外貌 60%、興趣 40%，只開一個就是 100%；還沒按過喜歡就只看興趣，兩個都關就是原本的最新註冊在前。
  外貌開著時保留探索名額：依按過的喜歡數 0／1／2／3／4／5+，個人化 0／60／65／70／75／80%（興趣也開著時至少 60%），其餘從剩下的人用固定種子挑（先避開被略過扣分的人），平均穿插在清單中；同一天沒有新的喜歡時探索名單不變。
  外貌向量：主照片上傳或遞補時由 API 排進 BullMQ `rec-jobs`，推薦 worker（`services/recommendation`）算好放進 `rec-results`，API 扣掉戴眼鏡方向後寫回（找不到臉也記一列、向量是空的）；刪照片時一起刪除。worker 沒在跑時不排，API 每分鐘替還沒處理過的主照片補排一批（20 張，佇列清空才排下一批）。
  測試用參數 `?hardfilter=false`、`?appearance=false`、`?interest=false` 可以個別關掉；前端網址帶同樣的參數會原樣轉過來。關掉硬篩選時整個使用者池都是候選人。
- `GET /discovery/search?q=` — 測試用：以顯示名稱或 email 搜尋除了自己以外的全部使用者（不分大小寫、部分符合），不論按過什麼、是否配對、偏好是否相符、是否封鎖都列出；沒有個人檔案的人不列出，最多 20 筆。每筆多一個 `searchState`：`action`（`like`／`pass`／`null`）、`match`（`active`／`ended`／`null`）、`eligible`、`blocked`。前端只在網址帶 `?muggle=false` 時顯示搜尋列，並隱藏原本的探索卡片。
- `POST /discovery/search/interactions` — 測試用：搜尋列的喜歡／略過，body 同 `POST /interactions`，但規則放寬：先解除雙方的封鎖（兩個方向）、不檢查雙方偏好、已配對時按略過會解除配對、配對已結束時互相喜歡會恢復原本的配對（聊天紀錄一併回來）。
- `POST /interactions`
- `GET /likes` — 我按過喜歡的人，`status` 為 `waiting`（等對方回應）或 `matched`（附 `matchId`／`conversationId`）。
  雙向封鎖、沒有個人檔案、配對已結束的人不列出；依按喜歡的時間由新到舊，最多 200 筆。
- `GET /matches`
- `GET /matches/:matchId`
- `DELETE /matches/:matchId`
- `POST /blocks`
- `DELETE /blocks/:blockedUserId`

## Conversations / Chat

- `GET /conversations`
- `POST /conversations`
- `GET /conversations/:conversationId`
- `GET /conversations/:conversationId/messages`

Socket events should handle realtime delivery, typing, read receipts, and presence.

## Internal AI Endpoints

These are internal service calls and are not browser-facing.

- `POST /internal/ai/face/verify`
- `POST /internal/ai/face/liveness`
- `POST /internal/ai/image/embed`
- `POST /internal/ai/conversation/analyze`
- `POST /internal/ai/users/:userId/recompute-features`
- `POST /internal/ai/clustering/recompute`
- `POST /internal/ai/recommendations/generate`
