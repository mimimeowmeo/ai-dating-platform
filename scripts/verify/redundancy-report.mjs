// 冗餘資料表／欄位報表：把「還在用」「只寫不讀」「已被取代」分開列出來。
// 只做查詢，不會修改任何資料。用法：node scripts/verify/redundancy-report.mjs
import { db } from "./lib.mjs";

const rows = (sql) =>
  db(sql)
    .split("\n")
    .filter(Boolean)
    .map((line) => line.split("|"));

console.log("# 冗餘資料表／欄位報表");
console.log(
  `\n產生時間：${new Date().toISOString().slice(0, 19).replace("T", " ")}\n`,
);

console.log("## 各資料表筆數\n");
console.log("| 資料表 | 筆數 |");
console.log("|---|---|");
for (const [name, count] of rows(
  "select relname, n_live_tup from pg_stat_user_tables where schemaname='public' order by n_live_tup desc",
))
  console.log(`| \`${name}\` | ${Number(count).toLocaleString()} |`);

console.log("\n## 舊欄位的實際使用狀況\n");
const profiles = db(
  `select count(*)||'|'||
     count(*) filter (where cardinality(interests)>0)||'|'||
     count(*) filter (where cardinality(hobbies)>0)||'|'||
     count(*) filter (where cardinality(foods)>0)||'|'||
     count(height_cm)||'|'||count(occupation)||'|'||count(education)
   from profiles`,
).split("|");
console.log("| 欄位 | 有資料的筆數 | 狀態 |");
console.log("|---|---|---|");
console.log(
  `| \`profiles.interests\` | ${profiles[1]} / ${profiles[0]} | 🟡 已被 \`user_traits\` 取代，畫面不再讀寫 |`,
);
console.log(
  `| \`profiles.hobbies\` | ${profiles[2]} / ${profiles[0]} | 🟡 同上 |`,
);
console.log(
  `| \`profiles.foods\` | ${profiles[3]} / ${profiles[0]} | 🟡 同上 |`,
);
console.log(
  `| \`profiles.height_cm\` | ${profiles[4]} / ${profiles[0]} | 🟢 使用中（探索偏好的身高篩選） |`,
);
console.log(
  `| \`profiles.occupation\` | ${profiles[5]} / ${profiles[0]} | 🟢 個人檔案可填、檢視頁顯示 |`,
);
console.log(
  `| \`profiles.education\` | ${profiles[6]} / ${profiles[0]} | 🟢 個人檔案可填 |`,
);

const intents = rows(
  "select dating_intent, count(*) from profiles group by 1 order by 2 desc",
)
  .map(([v, c]) => `${v}=${c}`)
  .join("、");
const preferred = rows(
  "select preferred_dating_intent, count(*) from preferences group by 1 order by 2 desc",
)
  .map(([v, c]) => `${v}=${c}`)
  .join("、");
console.log(
  `| \`profiles.dating_intent\` | ${intents} | 🟡 保留欄位，但配對條件已改看 traits 的 \`dating_goal\` |`,
);
console.log(
  `| \`preferences.preferred_dating_intent\` | ${preferred} | 🟢 使用中，值為 \`any\` 或 dating_goal 的 code |`,
);

console.log("\n## 只寫不讀／永遠是空值的欄位\n");
const verification = db(
  `select count(*)||'|'||count(storage_key)||'|'||count(liveness_score)||'|'||count(face_match_score)||'|'||count(distinct mime_type) from verification_records`,
).split("|");
const photoMime = rows(
  "select mime_type, count(*) from user_photos group by 1 order by 2 desc",
)
  .map(([v, c]) => `${v}=${c}`)
  .join("、");
console.log("| 欄位 | 現況 | 建議 |");
console.log("|---|---|---|");
console.log(
  `| \`verification_records.storage_key\` | ${verification[0]} 筆中有值 ${verification[1]} 筆 | 自拍只送給 AI、從未存檔；要嘛真的存、要嘛刪欄位 |`,
);
console.log(
  `| \`verification_records.liveness_score\` / \`face_match_score\` | 有值 ${verification[2]} / ${verification[3]} 筆 | 寫得進去但沒有端點讀回來 |`,
);
console.log(
  `| \`user_photos.mime_type\` | ${photoMime} | 永遠寫 \`image/jpeg\`，\`GET /media/:id\` 也是寫死，等於常數 |`,
);
console.log(
  `| \`conversation_members.joined_at\` | 只有預設值，程式零引用 | 保留作稽核或刪除 |`,
);
console.log(
  `| \`matches.unmatched_at\` | 解除配對時會寫，但沒有端點讀 | 保留作稽核 |`,
);
console.log(
  `| \`likes.id\` / \`blocks.id\` | 所有操作都走複合唯一鍵 | 代理鍵沒被讀過，可移除 |`,
);

console.log("\n## 匯入流程留下的東西\n");
const map = db("select count(*) from _map").trim();
const foreign = db(
  "select count(*) from information_schema.foreign_tables where foreign_table_schema='hl'",
).trim();
const userTraits = db("select count(*) from user_traits").trim();
console.log("| 對象 | 現況 | 建議 |");
console.log("|---|---|---|");
console.log(
  `| \`public._map\` | ${Number(map).toLocaleString()} 列（舊 id ↔ 新 uuid） | 回填完成後可刪，但刪掉就無法再對回舊資料 |`,
);
console.log(
  `| \`hl\` schema（${foreign} 張外部表） | 匯入與回填的來源 | 同上 |`,
);
console.log(
  `| \`user_traits\` | ${Number(userTraits).toLocaleString()} 列 | 🟢 使用中，畫面的「我的小熱愛」與配對條件都靠它 |`,
);

console.log("\n## 資料完整性\n");
const integrity = db(
  `select
     (select count(*) from users u where not exists (select 1 from profiles p where p.user_id=u.id))||'|'||
     (select count(*) from profiles p where not exists (select 1 from preferences pr where pr.user_id=p.user_id))||'|'||
     (select count(*) from profiles where latitude is null or longitude is null)||'|'||
     (select count(*) from profiles where gender not in ('woman','man','nonbinary'))||'|'||
     (select count(*) from users where email like '%@example.test')`,
).split("|");
console.log("| 檢查 | 筆數（預期 0） |");
console.log("|---|---|");
console.log(`| 有帳號但沒有個人檔案 | ${integrity[0]} |`);
console.log(`| 有檔案但沒有探索偏好 | ${integrity[1]} |`);
console.log(`| 缺經緯度 | ${integrity[2]} |`);
console.log(`| 性別不在三種之內 | ${integrity[3]} |`);
console.log(`| 殘留的測試帳號 | ${integrity[4]} |`);
