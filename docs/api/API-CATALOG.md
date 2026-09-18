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
- `PUT /profile`
- `GET /profile/:userId`
- `POST /profile/photos`
- `DELETE /profile/photos/:photoId`

## Preferences

- `GET /preferences`
- `PUT /preferences` — `minAge`／`maxAge` 收 18–130，`minHeightCm`／`maxHeightCm` 收 130–250（沒帶就回到 130／250＝不限），
  兩組都要求下限不大於上限；`maxDistanceKm` 1–20000。身高條件只過濾有填身高的人。

## Traits

- `GET /traits` — 回傳 `traits` 資料表的全部代碼（`{ category, code, label }`）。
  `dating_goal` 供「想遇見的關係」與探索偏好的「關係期待」使用，其餘五類（`personality`／`diet`／`value`／`lifestyle`／`interest`）供「我的小熱愛」使用。
- `PUT /profile` 以 `traits`（最多 40 項）與 `datingGoals`（最多 2 項）送出選擇，`GET /profile`、`GET /profile/:userId`、`GET /discovery` 會一併回傳。
- `PUT /preferences` 的 `preferredDatingIntent` 收 `any` 或 `dating_goal` 的代碼。

## Interests / Hobbies / Foods（舊欄位，僅供匯入相容）

- `GET /interests`
- `PUT /me/interests`
- `GET /hobbies`
- `PUT /me/hobbies`
- `GET /foods`
- `PUT /me/foods`

## Verification

- `POST /onboarding/selfie`
- `GET /verification/status`
- `POST /verification/retry`

## Discovery / Interactions

- `GET /discovery`
- `POST /interactions`
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
