# 網址參數（Query Parameters）

網站頁面上只有探索頁（`/discover`）讀網址參數，共三個：`prefs`、`rank`、`debug`，都是測試用。
前端把它們原樣轉給 `GET /discovery`（`debug=search` 只影響前端），值不對時 API 回錯誤訊息，顯示在探索頁上方。
其他頁面（我的配對、聊天室、個人檔案、探索偏好、真人驗證）不讀任何網址參數。

## 探索頁 `/discover`

| 參數 | 可用的值 | 沒寫時 | 功能 |
| --- | --- | --- | --- |
| `prefs` | `on`、`off` | `on` | `off`：不套用雙向偏好篩選（年齡、性別、交往目的、身高、距離），全部使用者都是候選人。自己、按過喜歡或略過的人、任一方封鎖、配對過的人照樣排除；按喜歡時後端仍會檢查雙向偏好，不符合回 409 `NOT_ELIGIBLE`。 |
| `rank` | `face`、`tags`，可多選 | 兩個都用 | 排序用哪些分數：`face` 外貌（和最近 20 個喜歡裡最像那位的相似度，扣掉略過扣分）、`tags` 個人標籤（五類標籤「共同 ÷ 合計」依 70/15/2/5/8 加權）。兩個分數各自換成候選人裡的百分位再合併：都選時外貌 60%、個人標籤 40%，只選一個就是 100%。 |
| `debug` | `explain`、`search`，可多選 | 不開 | `explain`：每張卡附上推薦依據（右上角推薦卡／未推薦卡與分數、共同標籤標亮、「為什麼出現這張卡」、點「最像你喜歡過的 ○○」並排比較並框出最像的部位）。`search`：隱藏探索卡片，改顯示測試用搜尋列（用名稱或 email 搜全部使用者，任何狀態都能重新按喜歡／略過）。 |

多選的寫法：逗號分隔（`rank=face,tags`）或重複參數（`rank=face&rank=tags`）都可以，重複的值只算一次。

偏好篩選與排序的順序：先排除固定不看的人 → `prefs=on` 時做雙向偏好篩選 → 用 `rank` 選的分數排序 →
前 25 名是推薦卡，每滑 6 張遇到 1 張未推薦卡（從第 26 名之後隨機挑）→ 回傳 30 張。
排序出錯時退回最新註冊的順序（推薦依據的來源顯示 `latest`），探索頁不會壞掉。

### 範例

| 網址 | 意思 |
| --- | --- |
| `/discover` | 平常的探索頁：偏好篩選開、外貌＋個人標籤 |
| `/discover?debug=explain` | 平常的排序，卡片附上推薦依據 |
| `/discover?rank=tags&debug=explain` | 只用個人標籤排序，附推薦依據 |
| `/discover?rank=face` | 只用外貌排序 |
| `/discover?prefs=off&rank=face&debug=explain` | 不套用偏好，從全部使用者裡只用外貌排序 |
| `/discover?debug=search` | 測試用搜尋列 |
| `/discover?debug=explain,search` | 搜尋列；切回探索卡片時也附推薦依據 |

### 打錯時

參數名稱或值不認得，`GET /discovery` 回 400 `VALIDATION_ERROR`，探索頁上方顯示：

| 網址 | 訊息 |
| --- | --- |
| `?rank=foo` | rank 只能是 face、tags（可用逗號多選） |
| `?rank=` | rank 至少要選一個：face、tags |
| `?prefs=0` | prefs 只能是 on 或 off |
| `?debug=verbose` | debug 只能是 explain、search（可用逗號多選） |
| `?colour=red` | 不認得的參數：colour（可用 prefs、rank、debug） |

### 舊參數（2026-09-28 改名）

舊名字已經沒有作用。網址帶到舊名字時，探索頁上方會提示新寫法；直接呼叫 API 帶舊名字回 400 並附上新寫法。

| 舊 | 新 |
| --- | --- |
| `test=true` | `debug=explain` |
| `muggle=false` | `debug=search` |
| `hardfilter=false` | `prefs=off` |
| `appearance=false` | `rank=tags` |
| `interest=false` | `rank=face` |
| `appearance=false&interest=false`（照最新註冊排） | 已取消，`rank` 至少要選一個 |

## 注意：沒有權限檢查

這些參數只是畫面上的開關。後端的測試功能沒有權限檢查（2026-09-28 決定），任何登入的人都能直接呼叫 API：

| 功能 | 任何登入的人都能做到 |
| --- | --- |
| 測試用搜尋列的喜歡／略過（`POST /discovery/search/interactions`） | 刪掉雙方的封鎖、跳過偏好檢查、恢復已結束的配對 |
| 測試用搜尋（`GET /discovery/search`） | 用 email 的一部分搜全部使用者，可以試出某個 email 有沒有註冊 |
| `prefs=off` | 看到偏好不接受自己的人的卡片 |
| `debug=explain` 與「像在哪裡」 | 看到推薦分數、名次、最像你喜歡過的誰；請推薦 worker 分析臉部特徵點（生物特徵） |

公開上線前要移除這些測試功能，或加上權限（只開給管理者或測試帳號）。

## API 的查詢參數

前端會自動帶，列出來方便對照。

| API | 參數 | 功能 |
| --- | --- | --- |
| `GET /discovery` | `prefs`、`rank`、`debug` | 同上表；`debug=search` API 收到不做事 |
| `GET /discovery/search` | `q` | 搜尋列的關鍵字：顯示名稱或 email，部分符合、不分大小寫，最多 20 筆 |
| `GET /discovery/explain-appearance` | `candidate` | 「像在哪裡」要分析哪張卡（卡上那個人的 id），只能問上一次探索清單裡的卡 |
| `GET /conversations/:id/messages` | `before`、`beforeId` | 聊天室往上捲載入更早的訊息：從這個時間、這則訊息之前開始拿（分頁用，兩個要一起給） |
| `GET /media/:id` | `u`、`e`、`s` | 照片網址的簽章：誰在看、何時過期（約 6～12 小時後）、簽章；少了或被竄改回 403，被封鎖後回 404 |

程式：參數解析在 `apps/api/src/social.ts`（`discoveryOptions`），前端轉送在 `apps/web/components/dating-app.tsx`（`discoveryPath`），
偏好篩選規則在 `apps/api/src/eligibility.ts`。
