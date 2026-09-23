import {
  test,
  expect,
  devices,
  type APIRequestContext,
  type Page,
} from "@playwright/test";
import { randomUUID } from "node:crypto";
const photo = {
  name: "photo.png",
  mimeType: "image/png",
  buffer: Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAA1BMVEXZj3op6S/DAAAACXBIWXMAAAPoAAAD6AG1e1JrAAAAG0lEQVRYw+3BgQAAAADDoPlT3+AEVQEAAAB8AxBAAAEZszF2AAAAAElFTkSuQmCC",
    "base64",
  ),
};
// 新帳號要填完個人檔案（含一張照片）才進得了站，API 建立的測試帳號也要補齊。
async function seedProfile(
  request: APIRequestContext,
  headers: Record<string, string>,
  data: Record<string, unknown>,
) {
  const profile = await request.put("/api/v1/profile", {
    headers,
    data: {
      city: "台北市",
      latitude: 25.033,
      longitude: 121.5654,
      heightCm: 170,
      // 必填：個性／飲食／價值觀／生活型態各一項，興趣三項。
      traits: [
        "humorous",
        "likes_hotpot",
        "values_communication",
        "nine_to_five",
        "coffee",
        "movies",
        "reading",
      ],
      datingGoals: ["serious_relationship"],
      ...data,
    },
  });
  expect(profile.status()).toBe(200);
  const uploaded = await request.post("/api/v1/profile/photos", {
    headers,
    multipart: {
      file: {
        name: photo.name,
        mimeType: photo.mimeType,
        buffer: photo.buffer,
      },
    },
  });
  expect(uploaded.status()).toBe(201);
}
// 在表單上選齊必填的小熱愛（興趣刻意不選「旅行」，後面的測試要用它當未儲存的改動）。
const requiredPicks = [
  "幽默",
  "愛火鍋",
  "重視溝通",
  "朝九晚五",
  "咖啡",
  "電影",
  "閱讀",
];
async function pickRequiredTraits(page: Page, labels = requiredPicks) {
  for (const label of labels)
    await page.getByRole("button", { name: label, exact: true }).click();
}
const newEmail = () =>
  `e2e-${process.env.E2E_RUN_ID || "manual"}-${randomUUID()}@example.test`;
test("首頁、登入與受保護頁面", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: /不是找一個人。\s*是遇見懂你的人。/ }),
  ).toBeVisible();
  await page.getByRole("link", { name: "開始你的故事" }).click();
  await expect(page).toHaveURL(/register/);
  await page.getByRole("button", { name: "建立帳號" }).click();
  await expect(page).toHaveURL(/register/);
  await page.goto("/messages");
  await expect(page).toHaveURL(/login/);
  await expect(page.getByRole("heading", { name: "歡迎回來。" })).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
  expect(errors).toEqual([]);
});
test("建立帳號、儲存檔案與偏好、重新整理恢復登入", async ({ page }) => {
  const email = `e2e-${process.env.E2E_RUN_ID || "manual"}-${randomUUID()}@example.test`;
  const password = `Safe-${randomUUID()}`;
  await page.goto("/register");
  await page.getByLabel("電子郵件", { exact: true }).fill(email);
  await page.getByLabel("密碼", { exact: true }).fill(password);
  await page.getByRole("checkbox").check();
  await page.getByRole("button", { name: "建立帳號" }).click();
  await expect(page).toHaveURL(/profile/);
  await page.getByLabel("顯示名稱", { exact: true }).fill("瀏覽器測試");
  await page.getByLabel("生日", { exact: false }).fill("1998-06-15");
  await page.getByLabel("身高").fill("168");
  await page
    .getByLabel("自我介紹")
    .fill("本機自動化測試帳號，喜歡散步、咖啡與看展，想認識聊得來的人。");
  await page.getByRole("button", { name: "認真交往", exact: true }).click();
  await pickRequiredTraits(page);
  await page.getByRole("button", { name: "儲存個人檔案" }).click();
  await expect(page.getByRole("status")).toContainText("個人檔案已儲存");
  // 還缺一張照片，其他頁面仍鎖著，直接輸入網址也會被帶回個人檔案。
  await expect(page.locator(".side-nav .nav-item.locked")).toHaveCount(3);
  await page.goto("/discover");
  await expect(page).toHaveURL(/profile/);
  await page.locator(".photo-upload input[type=file]").setInputFiles(photo);
  await expect(page.getByRole("status")).toContainText("個人檔案完成了");
  await expect(page.locator(".side-nav .nav-item.locked")).toHaveCount(0);
  await page.goto("/preferences");
  await page.getByLabel("最小年齡").fill("22");
  await page.getByLabel("最大年齡").fill("38");
  await page.getByRole("button", { name: "儲存探索偏好" }).click();
  await expect(page.getByRole("status")).toContainText("探索偏好已更新");
  await page.reload();
  await expect(page.getByLabel("最小年齡")).toHaveValue("22");
  await expect(page.getByLabel("最高身高")).toHaveValue("250");
  // 個人頁要顯示性別（表單預設「女性」，這個流程沒有改）。
  await page.goto("/profile");
  await expect(page.locator(".profile-summary .gender-tag")).toHaveText("女性");
  await page.goto("/verification");
  await expect(
    page.getByRole("heading", { name: "尚未驗證", exact: true }),
  ).toBeVisible();
  await page.goto("/discover");
  await expect(page.getByRole("heading", { name: "探索心動" })).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
  // 日夜模式：切換後重新整理仍然記得。
  const html = page.locator("html");
  await expect(html).toHaveAttribute("data-theme", "light");
  await page.getByRole("button", { name: "切換到夜間模式" }).click();
  await expect(html).toHaveAttribute("data-theme", "dark");
  await page.reload();
  await expect(html).toHaveAttribute("data-theme", "dark");
  await page.getByRole("button", { name: "切換到日間模式" }).click();
  await expect(html).toHaveAttribute("data-theme", "light");
  // 根目錄 pnpm test:e2e 會依本次 E2E_RUN_ID 清除此測試帳號。
});
test("兩個瀏覽器帳號互讚，配對後收到即時訊息", async ({ browser }, info) => {
  test.skip(
    info.project.name === "mobile",
    "完整雙人流程在桌面執行，手機測試涵蓋註冊與RWD。",
  );
  const base = process.env.E2E_BASE_URL || "http://localhost:8080";
  const contexts = [
    await browser.newContext({ baseURL: base }),
    await browser.newContext({ baseURL: base }),
  ];
  const people = [];
  try {
    for (let i = 0; i < 2; i++) {
      const email = `e2e-${process.env.E2E_RUN_ID || "manual"}-${randomUUID()}@example.test`;
      const auth = await contexts[i].request.post("/api/v1/auth/register", {
        data: { email, password: `Safe-${randomUUID()}` },
      });
      expect(auth.status()).toBe(201);
      const user = await auth.json();
      const name = i === 0 ? "雙人測試甲" : "雙人測試乙";
      const gender = i === 0 ? "woman" : "man";
      await seedProfile(
        contexts[i].request,
        { Authorization: `Bearer ${user.accessToken}` },
        {
          displayName: name,
          birthDate: "1996-06-15",
          gender,
          bio: "本機瀏覽器雙人測試，互讚之後要能即時收到對方的訊息。",
        },
      );
      people.push(user);
    }
    const a = await contexts[0].newPage(),
      b = await contexts[1].newPage();
    await a.clock.install();
    await a.goto("/discover");
    await b.goto("/discover");
    await expect(a.getByRole("heading", { name: /雙人測試乙/ })).toBeVisible();
    // 每個出現對方的地方都要標性別：探索卡片、側欄新配對、配對卡、對話列表、聊天室標題。
    await expect(a.locator(".person-card .gender-tag")).toHaveText("男性");
    await a.getByRole("button", { name: "喜歡", exact: true }).click();
    await expect(b.getByRole("heading", { name: /雙人測試甲/ })).toBeVisible();
    await expect(b.locator(".person-card .gender-tag")).toHaveText("女性");
    await b.getByRole("button", { name: "喜歡", exact: true }).click();
    await expect(b.getByRole("status")).toContainText("互相喜歡");
    await expect(b.locator(".mini-matches .gender-tag")).toHaveText("女性");
    await a.goto("/matches");
    await expect(a.locator(".match-card .gender-tag")).toHaveText("男性");
    await a.getByRole("link", { name: "開始聊天" }).click();
    await expect(a.locator(".conversation .gender-tag")).toHaveText("男性");
    await expect(a.locator(".chat-header .gender-tag")).toHaveText("男性");
    await b.goto("/matches");
    await b.getByRole("link", { name: "開始聊天" }).click();
    await expect(b.locator(".chat-header .gender-tag")).toHaveText("女性");
    await a
      .getByRole("textbox", { name: "訊息內容" })
      .fill("你好，這是一段真實的即時對話！");
    await a.getByRole("button", { name: "傳送訊息" }).click();
    // 收窄到訊息氣泡：同一段文字也會出現在左側對話列表的預覽，
    // 只用 getByText 會同時命中兩個元素而觸發 strict mode violation。
    await expect(
      b.locator(".bubble", { hasText: "你好，這是一段真實的即時對話！" }),
    ).toBeVisible();
    await expect(a.getByText(/已讀/)).toBeVisible();
    // AI 推薦回覆（用 b 這一頁測：a 裝了假時鐘，打字動畫的計時器不會照實際時間跑）。
    const ai = b.getByRole("button", { name: "AI 推薦回覆" });
    const box = b.getByRole("textbox", { name: "訊息內容" });
    // 按鈕要在輸入框裡面，而不是跟傳送鍵並排在外面。
    await expect(b.locator(".ai-shell .ai-suggest")).toBeVisible();
    // 先把推薦 API 的請求扣住，確認完「等待中」的畫面再放行：
    // CI 沒有 ai 服務，API 幾十毫秒就回 503，不扣住的話等待狀態一閃而過、測不到。
    let release: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      release = () => resolve();
    });
    await b.route("**/reply-suggestions", async (route) => {
      await held;
      await route.continue();
    });
    await box.fill("等待中按 Enter 不能送出");
    await ai.click();
    // 等待期間輸入框會跑彩光（busy），AI 鈕與傳送鍵都停用。
    await expect(b.locator(".ai-shell")).toHaveClass(/busy/);
    await expect(ai).toBeDisabled();
    await expect(b.getByRole("button", { name: "傳送訊息" })).toBeDisabled();
    // 按 Enter 也不能繞過停用的傳送鍵。
    await box.press("Enter");
    release();
    const chip = b.locator(".ai-chips button").first();
    const failed = b.locator(".chat-panel .error");
    // AI 找不到依據時不硬湊，只顯示這句提示（規格 4.1）。
    const none = b.locator(".ai-note", { hasText: "沒有可推薦的句子" });
    // CI 的 compose 沒有 ai 服務，這時要如實顯示錯誤而不是卡住或假裝成功。
    await expect(chip.or(failed).or(none)).toBeVisible({ timeout: 45000 });
    await b.unroute("**/reply-suggestions");
    await expect(
      b.locator(".bubble", { hasText: "等待中按 Enter 不能送出" }),
    ).toHaveCount(0);
    if (await chip.isVisible()) {
      // 有建議時：第 1 則會被打字填進輸入框，點上方的按鈕可以換成另一則。
      await expect(box).not.toHaveValue("");
      const other = ((await chip.textContent()) ?? "").trim();
      await chip.click();
      await expect(box).toHaveValue(other);
    }
    await box.fill("");
    await a.reload();
    await expect(a.getByText(/已讀/)).toBeVisible();
    const renewed = a.waitForResponse(
      (response) =>
        response.url().endsWith("/api/v1/auth/refresh") &&
        response.request().method() === "POST" &&
        response.status() === 201,
    );
    await a.clock.fastForward(12 * 60 * 1000);
    await renewed;
    await a
      .getByRole("textbox", { name: "訊息內容" })
      .fill("登入續期後仍可聊天");
    await a.getByRole("button", { name: "傳送訊息" }).click();
    await expect(
      b.locator(".bubble", { hasText: "登入續期後仍可聊天" }),
    ).toBeVisible();
  } finally {
    await Promise.all(contexts.map((c) => c.close()));
  }
});
test("個人檔案：存檔後才能上傳，上傳照片不會清掉未儲存的輸入；年齡與身高拉桿互相夾住", async ({
  page,
}, info) => {
  test.skip(info.project.name === "mobile", "表單邏輯在桌面驗證即可。");
  await page.goto("/register");
  await page.getByLabel("電子郵件", { exact: true }).fill(newEmail());
  await page.getByLabel("密碼", { exact: true }).fill(`Safe-${randomUUID()}`);
  await page.getByRole("checkbox").check();
  await page.getByRole("button", { name: "建立帳號" }).click();
  await expect(page).toHaveURL(/profile/);
  const upload = page.locator(".photo-upload input[type=file]");
  await expect(upload).toBeDisabled();
  await expect(
    page.getByText("先儲存「關於我」，就能加入照片。"),
  ).toBeVisible();
  const name = page.getByLabel("顯示名稱", { exact: true });
  await name.fill("回歸測試");
  await page.getByLabel("生日", { exact: false }).fill("1995-03-20");
  // 必填沒填齊就存不了，而且一次列出所有缺的項目。
  await page.getByRole("button", { name: "儲存個人檔案" }).click();
  await expect(
    page.getByText(
      "請完成：身高、自我介紹至少 20 字（目前 0 字）、想遇見的關係選 1～2 項、個性至少選 1 項、飲食至少選 1 項、價值觀至少選 1 項、生活型態至少選 1 項、興趣至少選 3 項。",
    ),
  ).toBeVisible();
  // ❤️ 由兩個 code point 組成，但使用者看到的是一個字，計數也只算一個。
  await page.getByLabel("身高").fill("175");
  await page.getByLabel("自我介紹").fill(`${"字".repeat(18)}❤️`);
  await expect(page.getByText("至少 20 字，目前 19 字。")).toBeVisible();
  await page.getByRole("button", { name: "認真交往", exact: true }).click();
  await pickRequiredTraits(page, requiredPicks.slice(0, -1));
  await page.getByRole("button", { name: "儲存個人檔案" }).click();
  await expect(
    page.getByText(
      "請完成：自我介紹至少 20 字（目前 19 字）、興趣至少選 3 項。",
    ),
  ).toBeVisible();
  await page.getByLabel("自我介紹").fill(`${"字".repeat(19)}❤️`);
  await expect(page.getByText("至少 20 字，目前 20 字。")).toBeVisible();
  await pickRequiredTraits(page, requiredPicks.slice(-1));
  await page.getByRole("button", { name: "儲存個人檔案" }).click();
  await expect(page.getByRole("status")).toContainText("個人檔案已儲存");
  // 還缺照片，所以留在表單，上傳後才解鎖其他頁面。
  await expect(upload).toBeEnabled();
  await name.fill("尚未儲存的名字");
  await page.getByRole("button", { name: "旅行", exact: true }).click();
  await upload.setInputFiles(photo);
  await expect(page.locator(".photo-tile img")).toHaveCount(1);
  await expect(name).toHaveValue("尚未儲存的名字");
  await expect(
    page.getByRole("button", { name: "旅行", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await page.goto("/discover");
  // 側欄下半部與探索頁各有一個入口，這裡走側欄。
  await page
    .locator(".sidebar-bottom")
    .getByRole("link", { name: "探索偏好", exact: true })
    .click();
  // 年齡與身高都是拉桿，拖過頭時另一支被帶著走，範圍不會反過來。
  const minAge = page.getByLabel("最小年齡");
  const maxAge = page.getByLabel("最大年齡");
  await expect(minAge).toHaveValue("18");
  await minAge.fill("25");
  await expect(minAge).toHaveValue("25");
  await expect(maxAge).toHaveValue("99");
  await maxAge.fill("20");
  await expect(maxAge).toHaveValue("20");
  await expect(minAge).toHaveValue("20");
  const minHeight = page.getByLabel("最低身高");
  const maxHeight = page.getByLabel("最高身高");
  await minHeight.fill("190");
  await expect(maxHeight).toHaveValue("250");
  await maxHeight.fill("170");
  await expect(minHeight).toHaveValue("170");
  await page.getByRole("button", { name: "儲存探索偏好" }).click();
  await expect(page.getByRole("status")).toContainText("探索偏好已更新");
  // 照片由根目錄 pnpm test:e2e 的清理步驟連同帳號一起刪除。
});
test("多分頁同時開啟不會登出、仍即時收訊；手機開啟不存在的對話會顯示列表", async ({
  browser,
}, info) => {
  test.skip(
    info.project.name === "mobile",
    "流程內另外建立手機 context 驗證手機版面。",
  );
  const base = process.env.E2E_BASE_URL || "http://localhost:8080";
  const contexts = [
    await browser.newContext({ baseURL: base }),
    await browser.newContext({ baseURL: base }),
  ];
  try {
    const people = [];
    for (let i = 0; i < 2; i++) {
      const email = newEmail();
      const password = `Safe-${randomUUID()}`;
      const auth = await contexts[i].request.post("/api/v1/auth/register", {
        data: { email, password },
      });
      expect(auth.status()).toBe(201);
      const { accessToken, user } = await auth.json();
      const headers = { Authorization: `Bearer ${accessToken}` };
      await seedProfile(contexts[i].request, headers, {
        displayName: i === 0 ? "多分頁甲" : "多分頁乙",
        birthDate: "1994-08-08",
        gender: i === 0 ? "woman" : "man",
        bio: "多分頁回歸測試：同一個帳號開好幾個分頁也不會被登出。",
      });
      people.push({ email, password, id: user.id, headers });
    }
    for (const [from, to] of [
      [0, 1],
      [1, 0],
    ]) {
      const liked = await contexts[from].request.post("/api/v1/interactions", {
        headers: people[from].headers,
        data: { targetUserId: people[to].id, action: "like" },
      });
      expect(liked.status()).toBe(201);
    }
    const [conversation] = await (
      await contexts[0].request.get("/api/v1/conversations", {
        headers: people[0].headers,
      })
    ).json();
    const a1 = await contexts[0].newPage();
    const a2 = await contexts[0].newPage();
    await Promise.all([
      a1.goto(`/messages/${conversation.id}`),
      a2.goto("/matches"),
    ]);
    await expect(a1.locator(".chat-header")).toContainText("多分頁乙");
    await expect(
      a2.getByRole("heading", { name: "我的配對", exact: true }),
    ).toBeVisible();
    const b = await contexts[1].newPage();
    await b.goto(`/messages/${conversation.id}`);
    await expect(a1.locator(".chat-header")).toContainText("上線");
    await b.getByRole("textbox", { name: "訊息內容" }).fill("多分頁也收得到");
    await b.getByRole("button", { name: "傳送訊息" }).click();
    await expect(
      a1
        .locator(".message-scroll")
        .getByText("多分頁也收得到", { exact: true }),
    ).toBeVisible();
    await b.close();
    await expect(a1.locator(".chat-header")).toContainText("離線");
    const { defaultBrowserType: _ignored, ...iphone } = devices["iPhone 13"];
    const mobile = await browser.newContext({ ...iphone, baseURL: base });
    contexts.push(mobile);
    const login = await mobile.request.post("/api/v1/auth/login", {
      data: { email: people[0].email, password: people[0].password },
    });
    expect(login.status()).toBe(201);
    const m = await mobile.newPage();
    await m.goto(`/messages/${randomUUID()}`);
    await expect(m.getByRole("status")).toContainText("這段對話已結束或不存在");
    await expect(m.locator(".conversation-list")).toBeVisible();
    await expect(m.locator("a.conversation")).toHaveCount(1);
  } finally {
    await Promise.all(contexts.map((c) => c.close()));
  }
});
test("我的配對：已配對與我喜歡的放同一頁，用 tag 區分與篩選", async ({
  browser,
}, info) => {
  test.skip(
    info.project.name === "mobile",
    "雙人流程在桌面執行，手機版面由拉桿與 RWD 測試涵蓋。",
  );
  const base = process.env.E2E_BASE_URL || "http://localhost:8080";
  const contexts = [
    await browser.newContext({ baseURL: base }),
    await browser.newContext({ baseURL: base }),
  ];
  try {
    const people = [];
    for (let i = 0; i < 2; i++) {
      const auth = await contexts[i].request.post("/api/v1/auth/register", {
        data: { email: newEmail(), password: `Safe-${randomUUID()}` },
      });
      expect(auth.status()).toBe(201);
      const { accessToken, user } = await auth.json();
      await seedProfile(
        contexts[i].request,
        { Authorization: `Bearer ${accessToken}` },
        {
          displayName: i === 0 ? "標籤測試甲" : "標籤測試乙",
          birthDate: "1996-06-15",
          // 乙用最長的性別標籤「非二元性別」，窄欄位的版面才測得到。
          gender: i === 0 ? "woman" : "nonbinary",
          bio: "本機瀏覽器配對頁標籤測試，已配對與我喜歡的要分開顯示。",
        },
      );
      people.push({ id: user.id, accessToken });
    }
    const a = await contexts[0].newPage();
    await a.goto("/matches");
    await expect(
      a.getByRole("heading", { name: "你的下一個火花，還在路上" }),
    ).toBeVisible();
    // 主選單只有四項，「我喜歡的」不再是獨立頁面。
    await expect(a.locator(".side-nav > a, .side-nav > span")).toHaveCount(4);
    await a.goto("/discover");
    await expect(a.getByRole("heading", { name: /標籤測試乙/ })).toBeVisible();
    await a.getByRole("button", { name: "喜歡", exact: true }).click();
    await a
      .locator(".side-nav")
      .getByRole("link", { name: "我的配對" })
      .click();
    const tags = a.getByRole("group", { name: "篩選" });
    const card = a.locator(".match-card", { hasText: "標籤測試乙" });
    // 只有「已配對」與「我喜歡的」兩個 tag，預設停在已配對。
    await expect(tags.getByRole("button")).toHaveCount(2);
    await expect(
      tags.getByRole("button", { name: "已配對 0" }),
    ).toHaveAttribute("aria-pressed", "true");
    await expect(card).toBeHidden();
    await expect(
      a.getByRole("heading", { name: "還沒有互相喜歡的人" }),
    ).toBeVisible();
    // 切到「我喜歡的」：卡片出現，可以看對方的檔案。
    await tags.getByRole("button", { name: "我喜歡的 1" }).click();
    await expect(card.locator(".badge")).toHaveText("我喜歡的");
    await expect(card.locator(".gender-tag")).toHaveText("非二元性別");
    await expect(card).toContainText("今天按下喜歡");
    await card.getByRole("button", { name: "看看檔案" }).click();
    await expect(a.getByRole("dialog")).toContainText("標籤測試乙");
    await expect(a.getByRole("dialog")).toContainText("170 公分");
    await expect(a.getByRole("dialog").locator(".gender-tag")).toHaveText(
      "非二元性別",
    );
    await a.getByRole("button", { name: "關閉" }).click();
    await expect(a.getByRole("dialog")).toBeHidden();
    // 對方也按了喜歡：不重新整理，卡片就從「我喜歡的」移到「已配對」。
    const liked = await contexts[1].request.post("/api/v1/interactions", {
      headers: { Authorization: `Bearer ${people[1].accessToken}` },
      data: { targetUserId: people[0].id, action: "like" },
    });
    expect(liked.status()).toBe(201);
    await expect(
      tags.getByRole("button", { name: "我喜歡的 0" }),
    ).toBeVisible();
    await expect(card).toBeHidden();
    await expect(
      a.getByRole("heading", { name: "目前沒有等待回應的喜歡" }),
    ).toBeVisible();
    await tags.getByRole("button", { name: "已配對 1" }).click();
    await expect(card.locator(".badge")).toHaveText("已配對");
    await card.getByRole("link", { name: "開始聊天" }).click();
    await expect(a).toHaveURL(/\/messages\//);
    // 窄版面的對話列表：名字不能被性別標籤擠掉，最長的「非二元性別」也不能壓到右邊的時間欄。
    // 1100px 是平板（列表欄只剩約 40px，標籤固定換行、放不下以省略號收尾）；
    // 320px 是小手機（列表整頁寬，放不下才換行，名字要完整）。
    for (const [width, minName] of [
      [1100, 20],
      [320, 60],
    ]) {
      // 手機寬度下選了對話就只顯示聊天室，回到沒有選取的列表頁才量得到。
      if (width < 640) await a.goto("/messages");
      await a.setViewportSize({ width, height: 800 });
      const row = a.locator(".conversation").first();
      const rect = async (selector: string) =>
        (await row.locator(selector).boundingBox())!;
      const name = await rect("b");
      const tag = await rect(".gender-tag");
      const text = await rect(".conversation-text");
      expect(name.width).toBeGreaterThan(minName);
      expect(tag.x + tag.width).toBeLessThanOrEqual(text.x + text.width + 1);
      await expect(row.locator(".gender-tag")).toHaveAttribute(
        "title",
        "非二元性別",
      );
    }
  } finally {
    await Promise.all(contexts.map((c) => c.close()));
  }
});
test("探索偏好：點軌道會把最近的那顆把手移過去，按住可以直接拖", async ({
  browser,
}) => {
  const base = process.env.E2E_BASE_URL || "http://localhost:8080";
  const context = await browser.newContext({ baseURL: base });
  try {
    const auth = await context.request.post("/api/v1/auth/register", {
      data: { email: newEmail(), password: `Safe-${randomUUID()}` },
    });
    expect(auth.status()).toBe(201);
    const user = await auth.json();
    await seedProfile(
      context.request,
      { Authorization: `Bearer ${user.accessToken}` },
      {
        displayName: "拉桿測試",
        birthDate: "1996-06-15",
        gender: "woman",
        bio: "本機瀏覽器拉桿測試，點軌道或拖曳都要移動最近的把手。",
      },
    );
    const page = await context.newPage();
    await page.goto("/preferences");
    const minAge = page.getByLabel("最小年齡");
    const maxAge = page.getByLabel("最大年齡");
    await expect(minAge).toHaveValue("18");
    await expect(maxAge).toHaveValue("99");
    const track = page.locator(".range-pair").first();
    const box = (await track.boundingBox())!;
    const thumb = 22;
    // 把手中心的行程是扣掉把手寬度後的那一段，跟元件的換算一致。
    const xAt = (ratio: number) =>
      box.x + thumb / 2 + ratio * (box.width - thumb);
    const valueAt = (ratio: number) => Math.round(18 + ratio * (130 - 18));
    const near = (value: number) =>
      new RegExp(`^(${value - 1}|${value}|${value + 1})$`);
    // 按在軌道 70% 的位置（約 96 歲）：離最大年齡（99）近，最小年齡不動。
    await page.mouse.click(xAt(0.7), box.y + box.height / 2);
    await expect(maxAge).toHaveValue(near(valueAt(0.7)));
    await expect(minAge).toHaveValue("18");
    // 按在 15%（約 35 歲）：離最小年齡（18）近，換它過去。
    await page.mouse.click(xAt(0.15), box.y + box.height / 2);
    await expect(minAge).toHaveValue(near(valueAt(0.15)));
    await expect(maxAge).toHaveValue(near(valueAt(0.7)));
    // 從最小把手附近按住往右拖過最大把手：最小跟著走，最大被推著走，範圍不會反過來。
    await page.mouse.move(xAt(0.15), box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(xAt(0.5), box.y + box.height / 2, { steps: 6 });
    await page.mouse.move(xAt(0.9), box.y + box.height / 2, { steps: 6 });
    await page.mouse.up();
    await expect(minAge).toHaveValue(near(valueAt(0.9)));
    await expect(maxAge).toHaveValue(near(valueAt(0.9)));
    // 鍵盤仍然可用：剛拖過的把手已取得焦點，方向鍵能微調。
    await page.keyboard.press("ArrowLeft");
    await expect(minAge).toHaveValue(near(valueAt(0.9) - 1));
    // 身高那條也是同一個元件：點 50% 的位置，離較近的最高身高過去。
    const minHeight = page.getByLabel("最低身高");
    const maxHeight = page.getByLabel("最高身高");
    const heights = page.locator(".range-pair").nth(1);
    const hb = (await heights.boundingBox())!;
    await page.mouse.click(
      hb.x + thumb / 2 + 0.6 * (hb.width - thumb),
      hb.y + hb.height / 2,
    );
    await expect(maxHeight).toHaveValue(near(Math.round(130 + 0.6 * 120)));
    await expect(minHeight).toHaveValue("130");
  } finally {
    await context.close();
  }
});
