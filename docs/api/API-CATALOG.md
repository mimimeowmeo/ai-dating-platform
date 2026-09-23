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

- `GET /discovery`
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
