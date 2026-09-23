# 驗收手冊（API × 資料表 × 畫面）

這份文件記錄「怎麼確認整個網站真的可以用」，以及每一項驗收實際檢查了什麼。
對應的腳本放在 `scripts/verify/`，全部只用真的 HTTP 請求、真的瀏覽器、真的資料庫查詢，不做 mock。

## 目錄

1. [前置條件](#前置條件)
2. [一鍵執行](#一鍵執行)
3. [各腳本涵蓋範圍](#各腳本涵蓋範圍)
4. [API ↔ 資料表欄位對照](#api--資料表欄位對照)
5. [畫面 ↔ API 對照](#畫面--api-對照)
6. [冗餘資料表／欄位報表](#冗餘資料表欄位報表)
7. [已知待辦](#已知待辦)

---

## 前置條件

```bash
docker compose up -d --build --wait
docker restart ai-dating-platform-nginx-1   # 重建 web/api 後 nginx 會指著舊 IP，必須重啟
curl -s localhost:8080/api/v1/health         # 預期 {"status":"ok","service":"dating-api"}
```

- 驗收預設打 `http://localhost:8080`，資料庫查詢預設走 `docker exec heartlink-pg psql -d dating`。
  要換環境用環境變數：`VERIFY_API_BASE`、`VERIFY_ORIGIN`、`VERIFY_PG_CONTAINER`、`VERIFY_PG_USER`、`VERIFY_PG_PASSWORD`、`VERIFY_PG_DATABASE`。
- `VERIFY_ORIGIN` 必須與後端的 `WEB_ORIGIN` 一致，否則所有寫入請求都會被 `checkOrigin` 擋掉。
- 驗收會建立測試帳號，一律使用 `e2e-<RUN_ID>-<用途>@example.test`，跑完由 `apps/api/test/cleanup-e2e.mjs` 依 `E2E_RUN_ID` 連同照片一起刪除。
- 後端有限流：`/auth/*` 每個 IP 300 秒 20 次、每個帳號每分鐘 120 次請求。連續重跑會拿到 429，等視窗過了再跑。

## 一鍵執行

```bash
node scripts/with-env.mjs node scripts/verify/run-all.mjs
```

會依序跑：API 端點 → 照片存取控制 → 雙人聊天與配對 → 畫面互動 → Playwright e2e → 清理測試帳號 → 印出冗餘報表。
任何一段失敗都會在最後列出來，離開碼為 1。

單獨執行某一段：

```bash
node scripts/verify/api-endpoints.mjs      # 每一支 API 與對應的資料表欄位
node scripts/verify/media-access.mjs       # 照片簽章與封鎖後失效
node scripts/verify/chat-and-match.mjs     # 互讚 → 配對 → 即時聊天 → 歷史紀錄
node scripts/verify/ui-interactions.mjs    # 完成度關卡、traits、拉桿、日夜模式、手機溢出
node scripts/verify/redundancy-report.mjs  # 唯讀，產生冗餘欄位報表
E2E_RUN_ID=<剛才那組> node apps/api/test/cleanup-e2e.mjs   # 手動清理
```

## 各腳本涵蓋範圍

| 腳本                                 | 檢查項目                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `api-endpoints.mjs`                  | 逐支呼叫所有 HTTP 端點，**每一個寫入端點都回資料庫確認欄位真的變了**：註冊同時建立 `users`／`sessions`／`preferences`、`PUT /profile` 寫 `profiles` 與 `user_traits`、照片寫 `user_photos`、偏好寫 `preferences` 七個欄位、互讚寫 `likes`→`matches`→`notifications`、訊息寫 `messages`（含 `clientId` 冪等）、已讀寫 `conversation_members.last_read_at`、封鎖寫 `blocks` 並把 `matches.status` 改成 `blocked`、解除配對寫 `unmatched_at`、登出寫 `sessions.revoked_at` 且舊 token 立刻失效                                                                     |
| `media-access.mjs`                   | 照片網址一定要帶簽章；竄改簽章、換掉觀看者 id、過期都必須 403；被對方封鎖後同一組還沒過期的網址要立刻變 404，解除封鎖後恢復                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `chat-and-match.mjs`                 | **① 雙人聊天**：兩個瀏覽器同時開，A 送 B 即時收到（socket.io，不重新整理）<br>**② 聊天室資料正確**：`conversations`／`conversation_members`（剛好兩人）／`conversations.match_id` 對得上<br>**③ 聊天結果正確儲存**：`messages` 的 `sender_id` 與 `content` 逐筆比對、已讀寫回 `last_read_at`<br>**④ 歷史紀錄**：登出再登入後兩則訊息都還在且順序正確，API 讀出來的內容與畫面一致<br>**⑤ 互讚配對**：單方面喜歡只寫 `likes` 不產生配對；雙方都按喜歡才出現 `matches` 與兩筆 `match` 通知，且兩邊的「我的配對」都看得到對方<br>**⑥ 權限**：非成員讀這個對話回 404 |
| `ui-interactions.mjs`                | 新帳號的完成度關卡（沒填完會被導回個人檔案、側欄鎖住三個連結）、必填檢查（一次列出缺的項目、自我介紹顯示字數）、補照片後解鎖、我的小熱愛依類別顯示小標、交友目標上限兩項、身高欄位存得起來、關係期待選項只有「都可以」＋ dating_goal、年齡／身高是同一條軌道兩個把手且會互相帶動、偏好重新整理後保留、日夜模式切換與記憶、六個頁面在 390px 沒有橫向溢出、探索卡片照片實際載入                                                                                                                                                                                   |
| `apps/web/tests/*.spec.ts`           | Playwright e2e：首頁與受保護頁面、註冊到探索的完整流程、雙人互讚與即時訊息、照片上傳不會清掉未儲存的輸入、多分頁不會登出、聊天歷史                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `apps/api/test/integration.test.mjs` | 後端整合測試：session 輪替、封鎖、訊息去重、分頁游標、照片權限（簽章／竄改／封鎖）、BullMQ 與 Python worker 互通                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `redundancy-report.mjs`              | 唯讀報表：各表筆數、舊欄位還有多少資料、只寫不讀的欄位、匯入流程留下的 `_map` 與 `hl` schema、資料完整性                                                                                                                                                                                                                                                                                                                                                                                                                                                        |

## API ↔ 資料表欄位對照

Prisma model 與實體表名（`@@map`）：

| Model                                 | 資料表                                   | 主要欄位                                                                                                                                                               |
| ------------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `User`                                | `users`                                  | `id` `email` `password_hash` `is_verified`                                                                                                                             |
| `Session`                             | `sessions`                               | `token_hash` `previous_token_hash` `rotated_at` `expires_at` `revoked_at`                                                                                              |
| `Profile`                             | `profiles`                               | `display_name` `birth_date` `gender` `bio` `city` `latitude` `longitude` `height_cm` `occupation` `education` `dating_intent`（舊）`interests`/`hobbies`/`foods`（舊） |
| `Preference`                          | `preferences`                            | `min_age` `max_age` `preferred_gender` `max_distance_km` `min_height_cm` `max_height_cm` `preferred_dating_intent`                                                     |
| `Trait` / `UserTrait`                 | `traits` / `user_traits`                 | `category` `code` `label_zh` ／ `user_id`＋`trait_id`                                                                                                                  |
| `Photo`                               | `user_photos`                            | `storage_key` `mime_type` `is_avatar` `display_order` `deleted_at`                                                                                                     |
| `Verification`                        | `verification_records`                   | `status` `reason_code` `model_name` `model_version`                                                                                                                    |
| `Interaction`                         | `likes`                                  | `from_user_id` `to_user_id` `action`（like／pass）                                                                                                                     |
| `Match`                               | `matches`                                | `user_a_id` `user_b_id` `status` `unmatched_at`                                                                                                                        |
| `Block`                               | `blocks`                                 | `user_id` `blocked_user_id`                                                                                                                                            |
| `Conversation` / `ConversationMember` | `conversations` / `conversation_members` | `match_id` ／ `last_read_at`                                                                                                                                           |
| `Message`                             | `messages`                               | `sender_id` `content` `client_id`                                                                                                                                      |
| `Notification`                        | `notifications`                          | `type` `payload` `read_at`                                                                                                                                             |

端點對照（全部掛在 `/api/v1`，除了標註公開的都需要 `Authorization: Bearer`）：

| 端點                                                 | 讀                                                                                      | 寫                                                                                           |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `POST /auth/register`（公開）                        | `users`（email 唯一）                                                                   | `users`、`sessions`、`preferences`（預設值）                                                 |
| `POST /auth/login`（公開）                           | `users`                                                                                 | `sessions`                                                                                   |
| `POST /auth/refresh`（公開，靠 cookie）              | `sessions`                                                                              | `sessions.token_hash` `previous_token_hash` `rotated_at`                                     |
| `POST /auth/logout`（公開，靠 cookie）               | —                                                                                       | `sessions.revoked_at`                                                                        |
| `GET /auth/me`                                       | `users`                                                                                 | —                                                                                            |
| `GET /health`（公開）                                | `select 1` + Redis + S3                                                                 | —                                                                                            |
| `GET /media/:id`（公開，靠簽章）                     | `user_photos`、`blocks`                                                                 | —                                                                                            |
| `GET /profile`                                       | `profiles` `user_photos` `user_traits` `traits`                                         | —                                                                                            |
| `PUT /profile`                                       | `traits`（驗證 code、類別與各類最低數量）                                               | `profiles`（upsert）、`user_traits`（整組換掉）                                              |
| `GET /profile/:id`                                   | `blocks` `profiles` `user_photos` `user_traits`                                         | —                                                                                            |
| `POST /profile/photos`                               | `profiles` `user_photos`                                                                | `user_photos` + MinIO 物件                                                                   |
| `DELETE /profile/photos/:id`                         | `user_photos`                                                                           | `user_photos.deleted_at`、`is_avatar=false`（軟刪除，記錄與 MinIO 物件保留；必要時改下一張為主照片） |
| `GET` / `PUT /preferences`                           | `traits`（dating_goal）                                                                 | `preferences`                                                                                |
| `GET /traits`                                        | `traits`                                                                                | —                                                                                            |
| `GET /interests`、`/hobbies`、`/foods`               | 程式內建清單（不碰資料庫）                                                              | —                                                                                            |
| `PUT /me/interests`、`/hobbies`、`/foods`            | `profiles`                                                                              | `profiles.interests`／`hobbies`／`foods`（舊欄位相容）                                       |
| `GET /verification/status`                           | `verification_records`                                                                  | —                                                                                            |
| `POST /onboarding/selfie`                            | `verification_records`                                                                  | `verification_records`、必要時 `users.is_verified`                                           |
| `POST /verification/retry`                           | —                                                                                       | —（固定回覆，不碰資料庫）                                                                    |
| `GET /discovery`                                     | `users` `profiles` `preferences` `user_photos` `user_traits` `likes` `blocks` `matches` | —                                                                                            |
| `POST /interactions`                                 | `blocks` `likes` `matches`                                                              | `likes`、成立配對時 `matches`＋`conversations`＋`conversation_members`＋兩筆 `notifications` |
| `GET /matches`、`GET /matches/:id`                   | `matches` `conversations` `blocks`                                                      | —                                                                                            |
| `DELETE /matches/:id`                                | `matches`                                                                               | `matches.status='unmatched'`、`unmatched_at`                                                 |
| `GET` / `POST /blocks`、`DELETE /blocks/:id`         | `blocks` `matches`                                                                      | `blocks`、`matches.status='blocked'`                                                         |
| `GET /conversations`                                 | `conversations` `conversation_members` `messages` `matches` `blocks`                    | —                                                                                            |
| `POST /conversations`                                | `matches` `conversations`                                                               | —（只是用 `matchId` 取回既有聊天室）                                                         |
| `GET /conversations/:id`、`/messages`                | `conversations` `messages`                                                              | —                                                                                            |
| `POST /conversations/:id/messages`                   | `messages`（`sender_id`+`client_id` 冪等）                                              | `messages`、`conversations.updated_at`、`notifications`                                      |
| `POST /conversations/:id/read`                       | —                                                                                       | `conversation_members.last_read_at`                                                          |
| `GET /notifications`、`POST /notifications/:id/read` | `notifications`                                                                         | `notifications.read_at`                                                                      |

Socket.IO（握手用同一組 JWT）：`conversation:join`、`message:send`、`typing`、`conversation:read`；
伺服器推播 `message:new`、`notification:new`、`conversation:read`、`conversation:closed`、`presence`、`typing`。
前端目前只 emit `conversation:join` 與 `typing`，送訊息與標記已讀都走 REST。

### 照片網址的授權方式

`<img>` 不會帶 `Authorization` header，所以照片不掛 `AuthGuard`，改用 API 產生的簽章網址：

```
/api/v1/media/<photoId>?u=<觀看者id>&e=<到期時間>&s=<HMAC-SHA256(JWT_SECRET) 前 32 碼>
```

- 到期時間對齊 6 小時整點 → 同一段時間內網址固定，瀏覽器快取得住。
- 讀取時除了驗簽，還會查封鎖關係：**被封鎖後，對方手上那組還沒過期的網址立刻變 404**。
- 沒有簽章或簽章不符一律 403（`scripts/verify/media-access.mjs` 會逐項驗證）。

## 畫面 ↔ API 對照

| 頁面                  | 互動元素                                 | 觸發                                                                                          |
| --------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------- |
| landing `/`           | 開始你的故事／登入                       | 只換路由                                                                                      |
| `/register`、`/login` | 表單送出                                 | `POST /auth/register`／`/auth/login` → 寫入 zustand 後導向                                    |
| 全站外框              | 側欄四個主要連結、探索偏好、真人驗證     | 只換路由；個人檔案沒完成時渲染成不可點的 `.nav-item.locked`                                   |
| 全站外框              | 日夜模式                                 | 只改 `document.documentElement.dataset.theme` 與 `localStorage`                               |
| 全站外框              | 通知鈴鐺／登出／頭像                     | 路由／`POST /auth/logout`／路由                                                               |
| `/discover`           | 略過、喜歡                               | `POST /interactions` → invalidate `/discovery`、`/matches`                                    |
| `/profile`            | 儲存個人檔案                             | `PUT /profile`（含 `traits`、`datingGoals`、`heightCm`）→ invalidate `/profile`、`/discovery` |
| `/profile`            | 加入照片／刪除照片                       | `POST` / `DELETE /profile/photos` → invalidate `/profile`                                     |
| `/profile`            | 標籤按鈕（交友目標、五類小熱愛）         | 只改前端狀態，選項來自 `GET /traits`，送出時一起寫進 `user_traits`                            |
| `/preferences`        | 年齡／身高／距離拉桿、性別偏好、關係期待 | 只改前端狀態；儲存時 `PUT /preferences`                                                       |
| `/preferences`        | 解除封鎖                                 | `DELETE /blocks/:id` → invalidate `/blocks`                                                   |
| `/verification`       | 提交驗證                                 | `POST /onboarding/selfie` → 再 `GET /auth/me` 更新驗證狀態                                    |
| `/matches`            | 開始聊天／結束配對／封鎖                 | 路由／`DELETE /matches/:id`／`POST /blocks`                                                   |
| `/messages/:id`       | 傳送訊息                                 | `POST /conversations/:id/messages`（帶 `clientId` 冪等）                                      |
| `/messages/:id`       | 載入較早訊息                             | `GET /conversations/:id/messages?before=&beforeId=`                                           |
| `/messages/:id`       | 進入對話／分頁重新可見                   | `POST /conversations/:id/read`                                                                |
| `/notifications`      | 點一則通知                               | `POST /notifications/:id/read` → 導向對話或配對頁                                             |

## 冗餘資料表／欄位報表

```bash
node scripts/verify/redundancy-report.mjs
```

輸出是 markdown，分成四段：各表筆數、舊欄位使用狀況、只寫不讀的欄位、匯入流程留下的東西，最後是資料完整性檢查（五個數字都該是 0）。

目前已知的結論：

| 對象                                                       | 狀態                                                                         |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `profiles.interests` / `hobbies` / `foods`                 | 匯入資料還留著，但畫面與配對都已改用 `user_traits`，只剩 `/me/*` 舊 API 會寫 |
| `profiles.dating_intent`                                   | 仍會寫入與回傳，但配對條件已改看 traits 的 `dating_goal`                     |
| `verification_records.storage_key`                         | 永遠 NULL——自拍只送給 AI，沒有存檔                                           |
| `verification_records.liveness_score` / `face_match_score` | 只寫不讀，回應不含這兩欄                                                     |
| `user_photos.mime_type` / `verification_records.mime_type` | 永遠 `image/jpeg`，讀取端也寫死                                              |
| `conversation_members.joined_at`、`matches.unmatched_at`   | 只寫不讀                                                                     |
| `likes.id`、`blocks.id`                                    | 代理鍵，所有操作都走複合唯一鍵                                               |
| `public._map`、`hl` schema                                 | 匯入與回填用；刪掉就無法再對回舊 id                                          |

## 已知待辦

| 項目                                                                                      | 影響                                                               |
| ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `POST /verification/retry` 是固定回覆的空殼                                               | 與 `GET /verification/status` 的結果可能互相矛盾；畫面目前沒有入口 |
| 註冊頁「我已滿 18 歲」與驗證頁「我同意…」兩個 checkbox 沒有接 state                       | 勾選狀態不會送出，只靠瀏覽器原生 `required` 擋                     |
| `/matches` 的結束配對、封鎖與 socket `conversation:closed` 用無參數 `invalidateQueries()` | 會把快取一小時的 `/traits` 一起打掉重查                            |
| 在 `/discover` 按喜歡成立配對後沒有 invalidate `/conversations`                           | 同頁右側「最近的對話」會停在舊資料約 15 秒                         |
| 探索卡片不能點開完整檔案                                                                  | `GET /profile/:id` 目前沒有畫面入口                                |
| 重建 `web` 或 `api` 容器後 nginx 仍指著舊 IP                                              | 會出現 502，必須 `docker restart ai-dating-platform-nginx-1`       |
