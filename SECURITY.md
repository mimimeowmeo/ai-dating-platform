# 開發安全邊界

目前是本機 MVP。正式公開服務前仍需獨立安全審查。

- 密碼使用 bcrypt；refresh token 僅存 SHA-256 雜湊，輪替採原子交易並以 session 層級的 advisory lock 排隊。
- refresh 在同一個 session 上輪替 token；上一把 token 只在輪替後 30 秒內可換 access token（不換發 cookie），用於多分頁同時 refresh，超過即拒絕。登出撤銷整個 session。
- Access token 存前端記憶體，15 分鐘過期；每次 API／Socket 操作驗證伺服器端 session。
- API 設定 `trust proxy = 1`，限流以代理轉送的真實來源 IP 計算。前提是 API 只能經由一層反向代理存取；若直接公開 API 埠，`X-Forwarded-For` 可被偽造。
- Refresh cookie 為 HttpOnly、SameSite=Lax；本機 HTTP 允許非 Secure，正式 HTTPS 必須設定 COOKIE_SECURE=true 與正確 WEB_ORIGIN。
- API 驗證輸入白名單、使用 Prisma 參數化 SQL、對認證／訊息／上傳設 rate limit。
- 私人資料只限本人；聊天室僅限有效配對成員，雙向封鎖與取消配對立即限制後續操作。
- 影像用 Sharp／Pillow 實際解碼並限制尺寸，公開照片重新編碼以移除 EXIF；MinIO bucket 不公開，透過照片 id 代理讀取。
- 公開照片 URL 屬可分享資源，知悉 URL 的人可以讀取；封鎖不撤銷已保存的照片或 URL。
- 不得提交 .env、密碼、影像或生物特徵；錯誤回應不得包含資料庫連線、stack trace。
- 所有 Docker 主機埠限 127.0.0.1；MinIO 管理帳密僅限本機。

正式上線缺口：TLS、金鑰輪替、密碼重設／電子郵件驗證、內容審核與檢舉處理、部署與備份、外部驗證供應商審查、多實例支援及專業安全測試。

## 2026-09-16 依賴掃描

已固定修補版本：Sharp 0.35.4、Multer 2.4.0、deepmerge-ts 8.0.0。正式執行依賴無高／嚴重公告。

仍有兩項中度間接依賴公告，來自 MinIO SDK 8.0.7：`decode-uri-component` 與 `stream-json`。前者用於 SDK URL 處理，後者的受影響深層 JSON 篩選功能未在本期照片上傳／讀取中使用；MinIO endpoint 只由管理者設定且僅連內部服務。尚未以可能不相容的主版本強制替換它們，正式部署前需升級相容 SDK 並驗證。

公告：<https://github.com/advisories/GHSA-vcc3-ghjq-m6fr>、<https://github.com/advisories/GHSA-528h-pc64-c93x>。
