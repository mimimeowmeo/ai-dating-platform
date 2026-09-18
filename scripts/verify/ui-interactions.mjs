// 畫面互動驗收：完成度關卡、我的小熱愛分類、年齡／身高雙把手拉桿、
// 關係期待選項、日夜模式、以及六個頁面在手機寬度沒有橫向溢出。
// 用法：node scripts/verify/ui-interactions.mjs
import { randomUUID } from "node:crypto";
import { chromium } from "../../apps/web/node_modules/@playwright/test/index.mjs";
import { checker, ORIGIN, testEmail } from "./lib.mjs";

const check = checker("畫面互動");
const email = testEmail("ui");
const password = `Safe-${randomUUID()}`;
const photo = {
  name: "photo.png",
  mimeType: "image/png",
  buffer: Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAA1BMVEXZj3op6S/DAAAACXBIWXMAAAPoAAAD6AG1e1JrAAAAG0lEQVRYw+3BgQAAAADDoPlT3+AEVQEAAAB8AxBAAAEZszF2AAAAAElFTkSuQmCC",
    "base64",
  ),
};
// 每一類各挑一個，順便確認檢視頁的小標分類
const picks = [
  ["個性", "幽默"],
  ["飲食", "愛海鮮"],
  ["價值觀", "愛分享日常"],
  ["生活型態", "朝九晚五"],
  ["興趣", "登山"],
];
const goals = ["認真交往", "先做朋友"];

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));

// 1) 註冊後被留在個人檔案頁，其他頁面鎖住
await page.goto(`${ORIGIN}/register`);
await page.getByLabel("電子郵件", { exact: true }).fill(email);
await page.getByLabel("密碼", { exact: true }).fill(password);
await page.getByRole("checkbox").check();
await page.getByRole("button", { name: "建立帳號" }).click();
await page.waitForURL(/profile/);
check("新帳號進站後停在個人檔案頁", true);
check(
  "側欄鎖住三個主要連結",
  (await page.locator(".side-nav .nav-item.locked").count()) === 3,
);
await page.goto(`${ORIGIN}/discover`);
await page.waitForURL(/profile/);
check("直接輸入網址也會被導回個人檔案", true);

// 2) 必填檢查
await page.getByLabel("顯示名稱", { exact: true }).fill("畫面驗收");
await page.getByLabel("生日", { exact: false }).fill("1995-04-02");
await page.getByRole("button", { name: "儲存個人檔案" }).click();
await page.getByText("請寫一段自我介紹。").waitFor();
check("沒填自我介紹會被擋下", true);
await page.getByLabel("自我介紹").fill("畫面驗收用的帳號。");
await page.getByLabel("身高（選填）").fill("172");
for (const label of goals)
  await page.getByRole("button", { name: label, exact: true }).click();
check(
  "第三個交友目標被停用（上限兩項）",
  await page
    .getByRole("button", { name: "只想聊天", exact: true })
    .isDisabled(),
);
for (const [group, label] of picks)
  await page
    .getByRole("group", { name: group })
    .getByRole("button", { name: label, exact: true })
    .click();
await page.getByRole("button", { name: "儲存個人檔案" }).click();
await page.getByRole("status").filter({ hasText: "個人檔案已儲存" }).waitFor();
check(
  "存檔後仍鎖住（還缺照片）",
  (await page.locator(".side-nav .nav-item.locked").count()) === 3,
);

// 3) 補上照片後解鎖
await page.locator(".photo-upload input[type=file]").setInputFiles(photo);
await page.getByRole("status").filter({ hasText: "個人檔案完成了" }).waitFor();
check(
  "補上照片後側欄解鎖",
  (await page.locator(".side-nav .nav-item.locked").count()) === 0,
);

// 4) 檢視頁：交友目標與分類小標
await page.reload();
await page.locator(".profile-loves").waitFor();
await page.waitForTimeout(1500);
const main = await page.locator(".main-content").innerText();
const loves = await page.locator(".profile-loves").innerText();
check(
  "檢視頁顯示交友目標",
  goals.every((g) => main.includes(g)),
  goals.join("、"),
);
check(
  "我的小熱愛列出五個類別小標",
  picks.every(([g]) => loves.includes(g)),
  loves.replace(/\n/g, " / "),
);
check(
  "每個小標下是自己類別的標籤",
  picks.every(([g, l]) => loves.indexOf(l) > loves.indexOf(g)),
);
await page.getByRole("button", { name: /編輯/ }).first().click();
check(
  "重新編輯時身高保留",
  (await page.getByLabel("身高（選填）").inputValue()) === "172",
);
const pressed = (
  await page
    .locator('.tags.selectable button[aria-pressed="true"]')
    .allInnerTexts()
).map((t) => t.trim());
check(
  "重新編輯時選取狀態保留",
  [...goals, ...picks.map((p) => p[1])].every((l) => pressed.includes(l)),
);

// 5) 探索偏好：關係期待選項、雙把手拉桿
await page.goto(`${ORIGIN}/preferences`);
await page.getByLabel(/關係期待/).waitFor();
await page.waitForTimeout(1200);
const options = (
  await page
    .getByLabel(/關係期待/)
    .locator("option")
    .allInnerTexts()
).map((t) => t.trim());
check(
  "關係期待只有「都可以」＋ dating_goal",
  options.length === 6 && options[0] === "都可以",
  options.join("、"),
);
check(
  "年齡與身高各是一條軌道兩個把手",
  (await page.locator(".range-pair").count()) === 2 &&
    (await page
      .locator(".range-pair")
      .first()
      .locator("input[type=range]")
      .count()) === 2,
);
const minAge = page.getByLabel("最小年齡");
const maxAge = page.getByLabel("最大年齡");
check(
  "年齡範圍 18–130",
  (await minAge.getAttribute("min")) === "18" &&
    (await minAge.getAttribute("max")) === "130",
);
check(
  "身高範圍 130–250",
  (await page.getByLabel("最低身高").getAttribute("min")) === "130" &&
    (await page.getByLabel("最高身高").getAttribute("max")) === "250",
);
await minAge.fill("30");
await maxAge.fill("28");
check(
  "拖過頭時另一支被帶著走",
  (await minAge.inputValue()) === "28" && (await maxAge.inputValue()) === "28",
);
await maxAge.fill("45");
await page.getByLabel("最低身高").fill("165");
await page.getByLabel("最高身高").fill("180");
await page.getByRole("button", { name: "儲存探索偏好" }).click();
await page.getByRole("status").filter({ hasText: "探索偏好已更新" }).waitFor();
await page.reload();
await page.getByLabel("最低身高").waitFor();
await page.waitForTimeout(1200);
check(
  "偏好重新整理後保留",
  (await page.getByLabel("最低身高").inputValue()) === "165" &&
    (await page.getByLabel("最高身高").inputValue()) === "180",
);

// 6) 日夜模式
const html = page.locator("html");
await page.getByRole("button", { name: "切換到夜間模式" }).click();
check("切到夜間模式", (await html.getAttribute("data-theme")) === "dark");
await page.reload();
await page.waitForTimeout(800);
check(
  "重新整理後記得夜間模式",
  (await html.getAttribute("data-theme")) === "dark",
);
await page.getByRole("button", { name: "切換到日間模式" }).click();
check("切回日間模式", (await html.getAttribute("data-theme")) === "light");
await page.close();

// 7) 手機寬度：六個頁面都不能有橫向捲動
const phone = await browser.newPage({ viewport: { width: 390, height: 844 } });
await phone.goto(`${ORIGIN}/login`);
await phone.getByLabel("電子郵件", { exact: true }).fill(email);
await phone.getByLabel("密碼", { exact: true }).fill(password);
await phone.getByRole("button", { name: "登入" }).click();
await phone.waitForURL(/discover|profile/);
let worst = 0;
for (const path of [
  "/discover",
  "/matches",
  "/messages",
  "/profile",
  "/preferences",
  "/notifications",
]) {
  await phone.goto(ORIGIN + path);
  await phone.locator(".main-content").waitFor();
  await phone.waitForTimeout(900);
  const overflow = await phone.evaluate(
    () =>
      document.documentElement.scrollWidth -
      document.documentElement.clientWidth,
  );
  worst = Math.max(worst, overflow);
  console.log(`   ${path}：溢出 ${overflow}px`);
}
check("手機版六個頁面都沒有橫向溢出", worst === 0, `最大 ${worst}px`);
await phone.goto(`${ORIGIN}/discover`);
await phone.locator(".person-photo img").first().waitFor();
check(
  "探索卡片的照片實際載入成功",
  (await phone.evaluate(
    () => document.querySelector(".person-photo img")?.naturalWidth || 0,
  )) > 0,
);

check("整個流程沒有前端錯誤", errors.length === 0, errors.join(" | "));
await browser.close();
const result = check.summary();
process.exitCode = result.failed ? 1 : 0;
