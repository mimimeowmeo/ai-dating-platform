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

- `GET /discovery` — 最多 30 張卡片。先排除自己、按過喜歡／略過的人、雙向封鎖與配對過的人，再依網址參數處理（`prefs`、`rank`、`debug`，說明見 [`docs/testing/QUERY-PARAMS.md`](../testing/QUERY-PARAMS.md)）：
  - 硬篩選（`prefs=on`，預設）：雙向的年齡、性別、交往目的、身高、距離，從全部使用者裡篩（不限註冊時間；SQL 版規則見 `apps/api/src/eligibility.ts`，和按喜歡時的檢查相同）。`prefs=off` 時整個使用者池都是候選人。
  - 外貌分數（`rank=face`）：候選人主照片和「我最近按喜歡的 20 人」的向量比，取最高相似度（`appearance_embeddings`），再扣略過分數（PASS V2：候選人附近有 5 個以上我最近略過的人才扣，附近也有我喜歡過的人會減輕；原規格最多扣 0.20，依種子分布乘上「1 − 同性別隨機兩人相似度的中位數」換算，女最多約 0.034、男約 0.051；「附近」門檻女 0.901、男 0.845）。
  - 個人標籤分數（`rank=tags`）：五類標籤各算 Jaccard，依 70/15/2/5/8 加權。
  `rank` 可多選、預設兩個都用。兩個分數先換成候選池裡的百分位，都用時外貌 60%、個人標籤 40%，只選一個就是 100%；還沒按過喜歡就只看個人標籤。排序出錯時退回最新註冊在前（來源記 `latest`）。
  推薦卡與未推薦卡穿插：推薦卡照合併分數從排名最前面依序放；每 5 張推薦卡之後插 1 張未推薦卡（探索卡），從推薦卡之後的排名用固定種子隨機挑（先避開被略過扣分的人），讓使用者有機會看到其他類型。前端一次只顯示第一張、每滑一張就重抓，所以位置依「已滑張數」決定：第 6、12、18… 次滑卡看到的是未推薦卡，重新整理不會換。退回最新註冊時不穿插。
  外貌向量：主照片上傳或遞補時由 API 排進 BullMQ `rec-jobs`，推薦 worker（`services/recommendation`）算好放進 `rec-results`，API 扣掉戴眼鏡方向後寫回（找不到臉也記一列、向量是空的）；刪照片時一起刪除。worker 沒在跑時不排，API 每分鐘替還沒處理過的主照片補排一批（20 張，佇列清空才排下一批）。
  參數名稱或值不認得回 400 `VALIDATION_ERROR`（例如「rank 只能是 face、tags（可用逗號多選）」）；舊的參數名稱（`hardfilter`、`appearance`、`interest`、`test`）回 400 並附上新寫法。前端網址帶的 `prefs`、`rank`、`debug` 會原樣轉過來。
  `debug=explain`：每張卡多一個 `explain`（推薦卡或未推薦卡、排名、合併分數與比重、外貌百分位與相似度、最像的那位喜歡過的人、略過扣分、各類標籤的共同項目與 Jaccard），前端卡片右上角顯示推薦卡／未推薦卡與分數，並標亮共同的標籤。公開上線前要移除或只開給管理者。
  每次回傳時，把每張卡當下的推薦狀態存在 Redis（`discovery:served:<userId>`，只留最後一次的清單，24 小時過期），給滑卡紀錄用。
  最像的那位喜歡過的人只列「我的配對」看得到的人：雙方封鎖或配對已結束時 `anchor` 是 `null`。
- `GET /discovery/explain-appearance?candidate=<userId>` — 測試用「像在哪裡」：探索頁 `?debug=explain` 點「最像你喜歡過的 ○○」並排比較時呼叫，框出兩張臉最像的部位。
  只能問上一次探索清單裡的卡（`discovery:served` 對不到回 404），比的是那張卡當時最像的那位喜歡過的人，不能任意指定兩個人；雙方封鎖或配對已結束也回 404。任一方沒有主照片回 409 `NO_AVATAR`，推薦 worker 沒在跑回 503 `REC_WORKER_OFFLINE`。
  回傳 `{status, result?, anchorUserId, candidatePhotoUrl, anchorPhotoUrl}`：`pending` 時前端每秒再問一次；`ready` 時 `result` 是 7 個臉部區域（眉毛、眼睛、鼻子、嘴唇、左右臉頰、下巴，左右照畫面來分）各自遮住後外貌相似度下降多少（`drop`，兩張臉各遮一次的平均），以及兩張照片上的框（原始像素 `[x, y, w, h]`），依下降量由大到小；任一張找不到臉部特徵點時 `result` 是 `null`。框的座標以回傳的兩張照片為準。
  第一次問時 API 把兩張主照片原檔與戴眼鏡方向排進 `rec-jobs`（`explain-appearance`），結果只在 Redis 放 1 小時（`rec:explain:<照片 id>:<照片 id>`），不寫資料庫。公開上線前要移除或只開給管理者。
- `GET /discovery/search?q=` — 測試用：以顯示名稱或 email 搜尋除了自己以外的全部使用者（不分大小寫、部分符合），不論按過什麼、是否配對、偏好是否相符、是否封鎖都列出；沒有個人檔案的人不列出，最多 20 筆。每筆多一個 `searchState`：`action`（`like`／`pass`／`null`）、`match`（`active`／`ended`／`null`）、`eligible`、`blocked`。前端只在網址帶 `?debug=search` 時顯示搜尋列，並隱藏原本的探索卡片。
- `POST /discovery/search/interactions` — 測試用：搜尋列的喜歡／略過，body 同 `POST /interactions`，但規則放寬：先解除雙方的封鎖（兩個方向）、不檢查雙方偏好、已配對時按略過會解除配對、配對已結束時互相喜歡會恢復原本的配對（聊天紀錄一併回來）。
- `POST /interactions` — 喜歡／略過真的寫入時，一併在 `swipe_logs` 記一筆：那張卡當下的推薦狀態（來源 `recommended`／`exploration`／`latest`、位置、排名、合併分數、外貌與興趣的百分位與原始分數、略過扣分、當時的喜歡數與已滑張數、開關、排序規則版本）。對不到最後一次清單時來源記 `unknown`；已配對時再按不會寫入，也不記。從測試用搜尋列（`POST /discovery/search/interactions`）滑的記 `search`、沒有分數。
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
- `POST /conversations/:conversationId/reply-suggestions` — AI 推薦回覆（`docs/ai/REPLY-SUGGESTIONS-SPEC.md`）。回傳 `{requestId, status, mode, notice, canRegenerate, suggestions[]}`，最多 5 則。
  - 每一輪只產生一次（2026-09-28）：「一輪」＝聊天室的最後一則訊息相同。同一個人在同一輪已經有成功的推薦（ok／partial／empty）時，再按會回傳同一批（同樣的 requestId 與推薦 id），不呼叫 AI、不新增紀錄。失敗（503）不算用掉這一輪。
  - `canRegenerate` 目前固定 `false`（「換一批」鎖住，`apps/api/src/ai-reply.ts` 的 `ALLOW_REGENERATE`）。

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
