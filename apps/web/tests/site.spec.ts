import { test, expect, devices } from "@playwright/test";
import { randomUUID } from "node:crypto";
const photo = {
  name: "photo.png",
  mimeType: "image/png",
  buffer: Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAA1BMVEXZj3op6S/DAAAACXBIWXMAAAPoAAAD6AG1e1JrAAAAG0lEQVRYw+3BgQAAAADDoPlT3+AEVQEAAAB8AxBAAAEZszF2AAAAAElFTkSuQmCC",
    "base64",
  ),
};
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
  await page.getByLabel("自我介紹").fill("本機自動化測試帳號。");
  await page.getByRole("button", { name: "咖啡", exact: true }).click();
  await page.getByRole("button", { name: "儲存個人檔案" }).click();
  await expect(page.getByRole("status")).toContainText("個人檔案已儲存");
  await page.goto("/preferences");
  await page.getByLabel("最小年齡").fill("22");
  await page.getByLabel("最大年齡").fill("38");
  await page.getByRole("button", { name: "儲存探索偏好" }).click();
  await expect(page.getByRole("status")).toContainText("探索偏好已更新");
  await page.reload();
  await expect(page.getByLabel("最小年齡")).toHaveValue("22");
  await page.goto("/verification");
  await expect(
    page.getByRole("heading", { name: "尚未驗證", exact: true }),
  ).toBeVisible();
  await page.goto("/discover");
  await expect(
    page.getByRole("heading", { name: "今天，想遇見誰？" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
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
      const profile = await contexts[i].request.put("/api/v1/profile", {
        headers: { Authorization: `Bearer ${user.accessToken}` },
        data: {
          displayName: name,
          birthDate: "1996-06-15",
          gender,
          city: "台北市",
          latitude: 25.033,
          longitude: 121.5654,
          bio: "本機瀏覽器雙人測試",
          datingIntent: "serious",
          interests: ["咖啡"],
          hobbies: [],
          foods: [],
        },
      });
      expect(profile.status()).toBe(200);
      people.push(user);
    }
    const a = await contexts[0].newPage(),
      b = await contexts[1].newPage();
    await a.clock.install();
    await a.goto("/discover");
    await b.goto("/discover");
    await expect(a.getByRole("heading", { name: /雙人測試乙/ })).toBeVisible();
    await a.getByRole("button", { name: "喜歡", exact: true }).click();
    await expect(b.getByRole("heading", { name: /雙人測試甲/ })).toBeVisible();
    await b.getByRole("button", { name: "喜歡", exact: true }).click();
    await expect(b.getByRole("status")).toContainText("互相喜歡");
    await a.goto("/matches");
    await a.getByRole("link", { name: "說聲你好" }).click();
    await b.goto("/matches");
    await b.getByRole("link", { name: "說聲你好" }).click();
    await a
      .getByRole("textbox", { name: "訊息內容" })
      .fill("你好，這是一段真實的即時對話！");
    await a.getByRole("button", { name: "傳送訊息" }).click();
    await expect(
      b.getByText("你好，這是一段真實的即時對話！", { exact: true }),
    ).toBeVisible();
    await expect(a.getByText(/已讀/)).toBeVisible();
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
      b.getByText("登入續期後仍可聊天", { exact: true }),
    ).toBeVisible();
  } finally {
    await Promise.all(contexts.map((c) => c.close()));
  }
});
test("個人檔案：存檔後才能上傳，上傳照片不會清掉未儲存的輸入；年齡欄位可清空重打", async ({
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
  await page.getByRole("button", { name: "儲存個人檔案" }).click();
  await expect(page.getByRole("status")).toContainText("個人檔案已儲存");
  await expect(upload).toBeEnabled();
  await name.fill("尚未儲存的名字");
  await page.getByRole("button", { name: "旅行", exact: true }).click();
  await upload.setInputFiles(photo);
  await expect(page.locator(".photo-tile img")).toHaveCount(1);
  await expect(name).toHaveValue("尚未儲存的名字");
  await expect(
    page.getByRole("button", { name: "旅行", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("link", { name: "探索偏好", exact: true }).click();
  const minAge = page.getByLabel("最小年齡");
  await expect(minAge).toHaveValue("18");
  await minAge.fill("");
  await expect(minAge).toHaveValue("");
  await minAge.pressSequentially("25");
  await expect(minAge).toHaveValue("25");
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
      const profile = await contexts[i].request.put("/api/v1/profile", {
        headers,
        data: {
          displayName: i === 0 ? "多分頁甲" : "多分頁乙",
          birthDate: "1994-08-08",
          gender: i === 0 ? "woman" : "man",
          city: "台北市",
          latitude: 25.033,
          longitude: 121.5654,
          bio: "多分頁回歸測試",
          datingIntent: "serious",
        },
      });
      expect(profile.status()).toBe(200);
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
      a2.getByRole("heading", { name: "剛好，你們也喜歡彼此。" }),
    ).toBeVisible();
    const b = await contexts[1].newPage();
    await b.goto(`/messages/${conversation.id}`);
    await expect(a1.locator(".chat-header")).toContainText("在線上");
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
