// 雙人流程：互讚 → 配對 → 即時聊天 → 資料庫內容 → 登出再登入看歷史。
// 用真的瀏覽器操作（含 socket.io），每一步都回資料庫對照。
// 用法：node scripts/verify/chat-and-match.mjs
import { chromium } from "../../apps/web/node_modules/@playwright/test/index.mjs";
import { call, checker, db, makeAccount, ORIGIN } from "./lib.mjs";

const check = checker("雙人聊天與配對");
const tag = String(Date.now()).slice(-6);
const nameA = `配對甲${tag}`;
const nameB = `配對乙${tag}`;
const A = await makeAccount("chat-a", { displayName: nameA, gender: "woman" });
const B = await makeAccount("chat-b", { displayName: nameB, gender: "man" });
check("兩個可進站的帳號建立完成", !!A.id && !!B.id);

const browser = await chromium.launch();
const login = async (user) => {
  const page = await (await browser.newContext({ baseURL: ORIGIN })).newPage();
  await page.goto("/login");
  await page.getByLabel("電子郵件", { exact: true }).fill(user.email);
  await page.getByLabel("密碼", { exact: true }).fill(user.password);
  await page.getByRole("button", { name: "登入" }).click();
  await page.waitForURL(/discover|profile/);
  return page;
};
const a = await login(A);
const b = await login(B);

// 1) 單方面喜歡：只寫 likes，還不算配對
await a.goto("/discover");
await a.getByRole("heading", { name: new RegExp(nameB) }).waitFor();
await a.getByRole("button", { name: "喜歡", exact: true }).click();
await a.waitForTimeout(800);
check(
  "A 按喜歡 → likes 一筆",
  db(
    `select count(*) from likes where from_user_id='${A.id}' and to_user_id='${B.id}'`,
  ) === "1",
);
check(
  "單方面喜歡不會產生配對",
  db(
    `select count(*) from matches where (user_a_id='${A.id}' and user_b_id='${B.id}') or (user_a_id='${B.id}' and user_b_id='${A.id}')`,
  ) === "0",
);

// 2) 互相喜歡 → matches + 雙方通知
await b.goto("/discover");
await b.getByRole("heading", { name: new RegExp(nameA) }).waitFor();
await b.getByRole("button", { name: "喜歡", exact: true }).click();
await b.getByRole("status").filter({ hasText: "互相喜歡" }).waitFor();
await b.waitForTimeout(800);
check(
  "互相喜歡 → matches 一筆",
  db(
    `select count(*) from matches where (user_a_id='${A.id}' and user_b_id='${B.id}') or (user_a_id='${B.id}' and user_b_id='${A.id}')`,
  ) === "1",
);
check(
  "雙方都收到配對通知",
  db(
    `select count(*) from notifications where type='match' and user_id in ('${A.id}','${B.id}')`,
  ) === "2",
);

// 3) 兩邊的「我的配對」都看得到對方
await a.goto("/matches");
await a.getByText(nameB).first().waitFor({ timeout: 10000 });
check("A 的配對清單有 B", true);
await b.goto("/matches");
await b.getByText(nameA).first().waitFor({ timeout: 10000 });
check("B 的配對清單有 A", true);

// 4) 即時聊天：兩邊都不重新整理也要收得到
await a.getByRole("link", { name: "開始聊天" }).first().click();
await b.getByRole("link", { name: "開始聊天" }).first().click();
const msgA = `你好，我是 ${nameA}`;
const msgB = `收到了，我是 ${nameB}`;
await a.getByRole("textbox", { name: "訊息內容" }).fill(msgA);
await a.getByRole("button", { name: "傳送訊息" }).click();
await b.getByText(msgA, { exact: true }).first().waitFor({ timeout: 10000 });
check("A 送出的訊息，B 即時收到（socket）", true, msgA);
await b.getByRole("textbox", { name: "訊息內容" }).fill(msgB);
await b.getByRole("button", { name: "傳送訊息" }).click();
await a.getByText(msgB, { exact: true }).first().waitFor({ timeout: 10000 });
check("B 回覆的訊息，A 即時收到（socket）", true, msgB);
await a.waitForTimeout(1200);

// 5) 聊天室資料正確：conversations / conversation_members / messages
const conversationId = db(
  `select c.id from conversations c
   join conversation_members m1 on m1.conversation_id=c.id and m1.user_id='${A.id}'
   join conversation_members m2 on m2.conversation_id=c.id and m2.user_id='${B.id}'`,
);
check(
  "conversations + conversation_members 成對建立",
  conversationId.length === 36,
  conversationId,
);
check(
  "對話成員剛好兩人",
  db(
    `select count(*) from conversation_members where conversation_id='${conversationId}'`,
  ) === "2",
);
check(
  "對話掛在正確的配對上",
  db(
    `select count(*) from conversations c join matches m on m.id=c.match_id where c.id='${conversationId}'`,
  ) === "1",
);
const stored = db(
  `select sender_id||'|'||content from messages where conversation_id='${conversationId}' order by created_at, id`,
);
check(
  "messages 的發送者與內容都正確",
  stored === `${A.id}|${msgA}\n${B.id}|${msgB}`,
  stored.replace(/\n/g, " ／ "),
);

// 6) 已讀寫回資料庫
await a.reload();
await a.getByText(/已讀/).first().waitFor({ timeout: 10000 });
check(
  "已讀時間寫回 conversation_members.last_read_at",
  Number(
    db(
      `select count(*) from conversation_members where conversation_id='${conversationId}' and last_read_at is not null`,
    ),
  ) >= 1,
);

// 7) 登出再登入，歷史紀錄還在
await a.locator('.icon-button[aria-label="登出"]').click();
await a.waitForURL(/login/);
await a.getByLabel("電子郵件", { exact: true }).fill(A.email);
await a.getByLabel("密碼", { exact: true }).fill(A.password);
await a.getByRole("button", { name: "登入" }).click();
await a.waitForURL(/discover|profile/);
await a.goto("/messages");
await a
  .getByRole("link", { name: new RegExp(nameB) })
  .first()
  .click();
const bubbles = a.locator(".bubble");
await bubbles.filter({ hasText: msgA }).first().waitFor({ timeout: 10000 });
await bubbles.filter({ hasText: msgB }).first().waitFor({ timeout: 10000 });
check(
  "重新登入後聊天歷史完整",
  (await bubbles.count()) === 2,
  `${await bubbles.count()} 則`,
);

// 8) API 讀出來的內容與畫面一致，且非成員讀不到
const history = await call(`/conversations/${conversationId}/messages`, {
  token: A.token,
});
const contents = (history.data || []).map((m) => m.content).join(" ／ ");
check(
  "GET /conversations/:id/messages 與畫面一致",
  history.status === 200 &&
    history.data.length === 2 &&
    contents.includes(msgA) &&
    contents.includes(msgB),
  contents,
);
const stranger = await makeAccount("chat-c", {
  displayName: `路人丙${tag}`,
  gender: "man",
});
const denied = await call(`/conversations/${conversationId}/messages`, {
  token: stranger.token,
});
check(
  "非成員讀不到這個對話",
  denied.status === 403 || denied.status === 404,
  `status=${denied.status}`,
);

await browser.close();
const result = check.summary();
process.exitCode = result.failed ? 1 : 0;
