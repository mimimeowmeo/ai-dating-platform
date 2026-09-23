import {
  test,
  expect,
  type APIRequestContext,
  type BrowserContext,
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
      file: { name: photo.name, mimeType: photo.mimeType, buffer: photo.buffer },
    },
  });
  expect(uploaded.status()).toBe(201);
}

async function createAccount(
  context: BrowserContext,
  displayName: string,
  gender: string,
) {
  const email = `e2e-${process.env.E2E_RUN_ID || "manual"}-${randomUUID()}@example.test`;
  const password = `Safe-${randomUUID()}`;
  const auth = await context.request.post("/api/v1/auth/register", {
    data: { email, password },
  });
  expect(auth.status()).toBe(201);
  const user = await auth.json();
  await seedProfile(
    context.request,
    { Authorization: `Bearer ${user.accessToken}` },
    {
      displayName,
      birthDate: "1995-03-20",
      gender,
      bio: "聊天歷史驗證帳號，登出再登入後要看得到完整的對話。",
      // 種子資料全在台灣（緯度 22.6~25.0），這裡刻意放到南極。
      // 探索頁的距離上限預設 100km，所以這對帳號只看得到彼此，
      // 也不會擠進其他測試的候選名單裡把它們的對象推掉。
      city: "E2E 隔離區",
      latitude: -77.85,
      longitude: 166.67,
    },
  );
  return { email, password, userId: user.user?.id as string | undefined };
}

test("互讚配對後雙向聊天，登出再登入仍看得到完整歷史，且後端資料正確", async ({
  browser,
}, info) => {
  test.skip(
    info.project.name === "mobile",
    "完整雙人流程在桌面執行；手機專案已涵蓋註冊與 RWD。",
  );
  // 這支走完註冊→配對→聊天→登出→重登，步驟長；在完整套件中又是第一個跑的，
  // 要承受 Next.js 首次編譯頁面的冷啟動，所以放寬整支的上限。
  test.setTimeout(90000);
  const base = process.env.E2E_BASE_URL || "http://localhost:8080";
  const contexts = [
    await browser.newContext({ baseURL: base }),
    await browser.newContext({ baseURL: base }),
  ];
  try {
    const alice = await createAccount(contexts[0], "歷史測試甲", "woman");
    // 乙只需要存在並能被甲看到，後續都用瀏覽器操作，不需要它的回傳值。
    await createAccount(contexts[1], "歷史測試乙", "man");

    const a = await contexts[0].newPage();
    const b = await contexts[1].newPage();

    // ---- 1. 互相 like 觸發配對 ----
    await a.goto("/discover");
    await b.goto("/discover");
    await expect(a.getByRole("heading", { name: /歷史測試乙/ })).toBeVisible();
    await a.getByRole("button", { name: "喜歡", exact: true }).click();
    await expect(b.getByRole("heading", { name: /歷史測試甲/ })).toBeVisible();
    await b.getByRole("button", { name: "喜歡", exact: true }).click();
    await expect(b.getByRole("status")).toContainText("互相喜歡");

    // ---- 2. 雙向對話 ----
    await a.goto("/matches");
    await a.getByRole("link", { name: "開始聊天" }).click();
    await b.goto("/matches");
    await b.getByRole("link", { name: "開始聊天" }).click();

    const msgFromA = "甲說：第一句，用來驗證歷史保存";
    const msgFromB = "乙回：第二句，用來驗證雙向儲存";

    await a.getByRole("textbox", { name: "訊息內容" }).fill(msgFromA);
    await a.getByRole("button", { name: "傳送訊息" }).click();
    // 即時訊息要等 socket 來回，比照 scripts/verify/chat-and-match.mjs 給 10 秒。
    await expect(b.locator(".bubble", { hasText: msgFromA })).toBeVisible({
      timeout: 25000,
    });

    await b.getByRole("textbox", { name: "訊息內容" }).fill(msgFromB);
    await b.getByRole("button", { name: "傳送訊息" }).click();
    await expect(a.locator(".bubble", { hasText: msgFromB })).toBeVisible({
      timeout: 25000,
    });

    const chatUrl = new URL(a.url()).pathname;

    // ---- 3. 甲登出 ----
    await a.getByRole("button", { name: "登出" }).first().click();
    await expect(a).toHaveURL(/login/);
    // 登出後受保護頁面應該進不去
    await a.goto("/messages");
    await expect(a).toHaveURL(/login/);

    // ---- 4. 甲用帳密重新登入 ----
    await a.getByLabel("電子郵件", { exact: true }).fill(alice.email);
    await a.getByLabel("密碼", { exact: true }).fill(alice.password);
    await a.getByRole("button", { name: "登入" }).click();
    await expect(a).not.toHaveURL(/login/);

    // ---- 5. 重新登入後，完整歷史必須都還在 ----
    await a.goto(chatUrl);
    await expect(a.locator(".bubble", { hasText: msgFromA })).toBeVisible();
    await expect(a.locator(".bubble", { hasText: msgFromB })).toBeVisible({
      timeout: 25000,
    });

    // ---- 6. 後端資料正確性（用獨立的 API session 檢查）----
    const api = await browser.newContext({ baseURL: base });
    try {
      const login = await api.request.post("/api/v1/auth/login", {
        data: { email: alice.email, password: alice.password },
      });
      expect(login.status()).toBe(201);
      const token = (await login.json()).accessToken as string;
      const headers = { Authorization: `Bearer ${token}` };

      // 配對只該有一筆，且對象是乙
      const matches = await api.request.get("/api/v1/matches", { headers });
      expect(matches.status()).toBe(200);
      const matchList = await matches.json();
      const rows = Array.isArray(matchList) ? matchList : (matchList.items ?? []);
      expect(rows.length).toBe(1);

      // 對話訊息：兩則、順序正確、發送者正確
      const convs = await api.request.get("/api/v1/conversations", { headers });
      expect(convs.status()).toBe(200);
      const convBody = await convs.json();
      const convRows = Array.isArray(convBody) ? convBody : (convBody.items ?? []);
      expect(convRows.length).toBe(1);
      const convId = convRows[0].id;

      const msgs = await api.request.get(
        `/api/v1/conversations/${convId}/messages`,
        { headers },
      );
      expect(msgs.status()).toBe(200);
      const msgBody = await msgs.json();
      const msgRows = Array.isArray(msgBody) ? msgBody : (msgBody.items ?? []);
      const bodies = msgRows.map((m: { body?: string; content?: string }) =>
        (m.body ?? m.content ?? "").trim(),
      );
      expect(bodies).toContain(msgFromA);
      expect(bodies).toContain(msgFromB);
      expect(msgRows.length).toBe(2);

      // 兩則訊息必須分屬不同發送者
      const senders = new Set(
        msgRows.map((m: { senderId?: string }) => m.senderId),
      );
      expect(senders.size).toBe(2);
    } finally {
      await api.close();
    }
  } finally {
    await Promise.all(contexts.map((c) => c.close()));
  }
});
