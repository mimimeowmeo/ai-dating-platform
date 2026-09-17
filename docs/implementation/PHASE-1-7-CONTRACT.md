# Phase 1–7 實作契約

本次使用者已授權直接推進 Phase 1–7，取代原始文件逐項停下的節奏。本地 MVP 使用真實 PostgreSQL、Redis、MinIO；沒有模型時不得標記使用者已通過真人驗證。

## 共用約定

- API `/api/v1`；JSON 回應直接為物件或陣列，錯誤 `{ code, message }`（message 為繁中，不含英文欄位名）。日期為 ISO 字串。框架層錯誤統一為 `BAD_REQUEST`（400，例如 JSON 格式錯誤）、`NOT_FOUND`（404）、`PAYLOAD_TOO_LARGE`（413：照片超過 8 MB 或 JSON 超過 32KB）。
- API 前面只經過一層反向代理（Docker 為 Nginx、開發模式為 Next.js），以 `trust proxy = 1` 取得真實來源 IP 作為限流依據；API 埠不得直接對外公開。
- 認證使用記憶體中的 bearer access token、HttpOnly refresh cookie；cookie 經 refresh 可恢復登入。
- 性別 `woman | man | nonbinary`；交友目的 `serious | casual | friendship`；偏好另接受 `any`。
- 最低年齡 18 歲；城市與經緯度由使用者選擇／提供。探索排除自己、未完成資料、雙向封鎖、已互動與不符合雙方偏好的人。
- 陌生人的回應不得含電子郵件、完整生日、精確座標或私密自拍檔案。
- 目前沒有正式生物特徵政策與驗證模型，驗證狀態必須如實顯示；不使用模擬成功作為真實驗證。

## 帳號

- `POST /auth/register` `{email,password}` → `{accessToken,user}`，同時設定 refresh cookie。
- `POST /auth/login` 同上；`POST /auth/refresh` → 同上；`POST /auth/logout` → `{ok:true}`。
- refresh 在同一個 session 上換新 refresh token（session id 不變，既有 access token 與 Socket 連線維持有效到過期）。換新後 30 秒內，舊 token 或目前 token 再次 refresh 只回傳新的 access token、不換發 cookie（多分頁同時 refresh 不會互相登出）；超過 30 秒使用舊 token 回 401。登出撤銷整個 session。
- 前端只有 refresh 回 401／403 才視為登出；429、5xx 與網路錯誤保留目前狀態並稍後重試。
- `GET /auth/me` → `{id,email,isVerified}`。

## 資料與偏好

- `GET /profile`、`PUT /profile` → 自己的 Profile。
- Profile：`userId,displayName,birthDate,gender,bio,city,latitude,longitude,datingIntent,heightCm,occupation,education,interests,hobbies,foods,photos`；標籤為字串陣列。
- `PUT /profile` 接受上述可編輯欄位，不接受 userId、photos、isVerified。
- `GET /profile/:userId` → 公開卡片 `{userId,displayName,age,gender,bio,city,datingIntent,interests,hobbies,foods,photos,isVerified}`。
- Photo：`id,url,isAvatar,displayOrder`；`POST /profile/photos` multipart `file` → Photo；`DELETE /profile/photos/:photoId` → `{ok:true}`。
- 上傳照片前必須已有個人檔案，否則回 409 `PROFILE_REQUIRED`。`displayOrder` 取現有最大值 + 1，不會重複；照片依 `displayOrder`、建立時間排序，第一張即主照片。圖片錯誤分別回報無法讀取、格式（僅 JPEG／PNG／WebP 靜態圖）、尺寸（至少 64×64）與解析度（至多 2400 萬像素）。
- `GET/PUT /preferences` → `{minAge,maxAge,preferredGender,maxDistanceKm,preferredDatingIntent}`。
- `GET /interests`、`/hobbies`、`/foods` → 字串陣列；`PUT /me/:category` `{values:string[]}` → 字串陣列。

## 真人驗證

- `POST /onboarding/selfie` multipart `file` → Verification。
- `GET /verification/status` → Verification；`POST /verification/retry` → Verification。
- Verification：`{id?,status,reasonCode?,modelName?,modelVersion?,createdAt?}`；狀態 `not_started | pending | verified | rejected | unavailable`。
- API 將影像透過私有 HTTP 傳至 AI `/internal/ai/face/verify`，`X-Internal-Token` 驗證；JSON `{imageBase64,mimeType,requestId}`。回應 `{status,reasonCode,modelName,modelVersion,livenessScore?,faceMatchScore?}`。
- 無設定的模型回 `unavailable / MODEL_NOT_CONFIGURED`，不得回 verified。
- 只有 `verified`／`rejected` 會更新使用者的 `isVerified`；`unavailable`（未接模型、逾時、服務中斷）只寫入紀錄，不改變既有驗證狀態。

## 探索與配對

- `GET /discovery` → 公開卡片陣列，最多 30 位。
- `POST /interactions` `{targetUserId,action:"like"|"pass"}` → `{matched:boolean,matchId?:string}`；互讚只產生一次配對與聊天室。對象不存在回 404 `NOT_FOUND`，不符合雙方偏好回 409 `NOT_ELIGIBLE`。
- `GET /matches`、`GET /matches/:matchId` → `{id,createdAt,otherUser:公開卡片,conversationId}` 陣列／物件。
- `DELETE /matches/:matchId` → `{ok:true}`。
- `POST /blocks` `{blockedUserId}`；`DELETE /blocks/:blockedUserId` → `{ok:true}`；`GET /blocks` → `{blockedUserId,displayName}` 陣列。

## 聊天

- `GET /conversations` → `{id,matchId,otherUser,lastMessage,unreadCount,otherLastReadAt}` 陣列；`otherLastReadAt` 為對方最後已讀時間（ISO 字串或 null），用於重新整理後恢復已讀狀態。
- `POST /conversations` `{matchId}` → 對應聊天室，必須是有效配對成員。
- `GET /conversations/:id` → 同上單一聊天室。
- `GET /conversations/:id/messages?before=<ISO>&beforeId=<messageId>` → Message 陣列（依時間、id 升序，單頁最多 50）。游標為目前最舊訊息的 `createdAt` 與 `id`，時間相同的訊息不會在分頁邊界漏掉；只給 `before` 時沿用舊行為。
- `POST /conversations/:id/messages` `{content,clientId}` → Message。
- `POST /conversations/:id/read` → `{ok:true}`。
- Message：`{id,conversationId,senderId,content,clientId,createdAt}`；clientId 為 UUID，用於重送去重。
- `GET /notifications` → `{id,type,payload,readAt,createdAt}` 陣列；`POST /notifications/:id/read` → `{ok:true}`。
- Socket.IO path `/socket.io`，auth `{token}`；加入 `conversation:join` `{conversationId}`；傳送 `message:send` `{conversationId,content,clientId}`，ack `{ok,message? ,error?}`。
- `typing` `{conversationId,isTyping}`、`conversation:read` `{conversationId}`；server 發出 `message:new`、`typing`、`conversation:read`、`notification:new`、`presence`。
- Socket 與 HTTP 皆須逐次檢查成員、配對有效與封鎖狀態；封鎖／取消配對後禁止既有連線收發。
- 推播時只有連線的 access token 失效才中斷該連線（前端會換新 token 後重連）；無權存取某聊天室時只略過該事件。
- `presence`：使用者第一條連線建立時，通知其所有有效聊天室「上線」；最後一條連線中斷 3 秒後仍未重連，才通知「離線」。前端只採用對方的 `presence` 與 `typing` 事件。

## 執行環境

- web 3000、api 3001、ai 8000，nginx 本機 8080。
- API 環境變數：`DATABASE_URL,REDIS_URL,S3_ENDPOINT,S3_ACCESS_KEY,S3_SECRET_KEY,S3_BUCKET,JWT_SECRET,AI_INTERNAL_URL,AI_INTERNAL_TOKEN,WEB_ORIGIN,PORT,NODE_ENV`。
- 瀏覽器透過同來源 `/api/v1`、`/socket.io`，媒體透過 API 授權／公開照片代理。
- 檔案型自拍不提供公開 URL；一般照片需驗證圖片類型與大小。
- AI 使用 `AI_INTERNAL_TOKEN`，預設 provider disabled；提供 `/health`，Python BullMQ worker 使用 `REDIS_URL`。
