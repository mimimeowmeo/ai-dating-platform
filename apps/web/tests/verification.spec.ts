import { test, expect, type APIRequestContext } from "@playwright/test";
import { randomUUID } from "node:crypto";

// Chromium 的假相機：自動允許相機權限，畫面是內建的測試圖樣（沒有人臉）。
// 用來確認即時鏡頭的流程接得起來；真人做動作的「通過」路徑由後端單元測試涵蓋。
test.use({
  permissions: ["camera"],
  launchOptions: {
    args: [
      "--use-fake-ui-for-media-stream",
      "--use-fake-device-for-media-stream",
    ],
    ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE }
      : {}),
  },
});

const photo = {
  name: "photo.png",
  mimeType: "image/png",
  buffer: Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAA1BMVEXZj3op6S/DAAAACXBIWXMAAAPoAAAD6AG1e1JrAAAAG0lEQVRYw+3BgQAAAADDoPlT3+AEVQEAAAB8AxBAAAEZszF2AAAAAElFTkSuQmCC",
    "base64",
  ),
};

/** 註冊並填好個人檔案（還沒有照片）；回傳帶 token 的 headers。 */
async function createUser(request: APIRequestContext) {
  const auth = await request.post("/api/v1/auth/register", {
    data: {
      email: `e2e-${process.env.E2E_RUN_ID || "manual"}-${randomUUID()}@example.test`,
      password: `Safe-${randomUUID()}`,
    },
  });
  expect(auth.status()).toBe(201);
  const headers = {
    Authorization: `Bearer ${(await auth.json()).accessToken}`,
  };
  const profile = await request.put("/api/v1/profile", {
    headers,
    data: {
      displayName: "真人驗證測試",
      birthDate: "1996-06-15",
      gender: "woman",
      bio: "本機瀏覽器測試：上傳大頭貼之後才可以開始真人驗證。",
      city: "台北市",
      latitude: 25.033,
      longitude: 121.5654,
      heightCm: 165,
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
    },
  });
  expect(profile.status()).toBe(200);
  return headers;
}

async function uploadPhoto(
  request: APIRequestContext,
  headers: Record<string, string>,
) {
  const uploaded = await request.post("/api/v1/profile/photos", {
    headers,
    multipart: { file: photo },
  });
  expect(uploaded.status()).toBe(201);
}

test("上傳大頭貼之後才出現真人驗證；開鏡頭即時框臉，模型由本站提供", async ({
  page,
}) => {
  const headers = await createUser(page.request);

  // 還沒有照片：API 不給驗證，個人檔案頁也沒有驗證入口。
  const status = await page.request.get("/api/v1/verification/status", {
    headers,
  });
  expect(await status.json()).toEqual({
    status: "not_started",
    reasonCode: "AVATAR_REQUIRED",
    canVerify: false,
  });
  const challenge = await page.request.post("/api/v1/verification/challenge", {
    headers,
  });
  expect(challenge.status()).toBe(409);
  expect((await challenge.json()).code).toBe("AVATAR_REQUIRED");
  await page.goto("/profile");
  // 先等沒照片時一定會出現的說明，再確認驗證入口（表單側欄、檔案卡、左側使用者列）都不存在。
  await expect(
    page.getByText("上傳第一張照片（大頭貼）後，就能開始真人驗證。"),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: "查看真人驗證" })).toHaveCount(0);
  await expect(page.getByRole("link", { name: "前往真人驗證" })).toHaveCount(0);
  await expect(page.locator(".side-user-status a")).toHaveCount(0);
  // 直接輸入網址也會被帶回個人檔案（沒有照片時其他頁面都鎖著）。
  await page.goto("/verification");
  await expect(page).toHaveURL(/profile/);

  // 上傳第一張照片後，兩個入口都出現，也能進驗證頁，看得到要比對的大頭貼。
  await uploadPhoto(page.request, headers);
  await page.goto("/profile");
  // 照片補齊後新手流程完成，個人檔案頁改顯示檔案卡，入口是「前往真人驗證」。
  await expect(page.getByRole("link", { name: "前往真人驗證" })).toBeVisible();
  await expect(page.locator(".side-user-status a")).toHaveCount(1);
  const requests: string[] = [];
  page.on("request", (r) => requests.push(r.url()));
  await page.goto("/verification");
  await expect(
    page.getByRole("heading", { name: "尚未驗證", exact: true }),
  ).toBeVisible();
  await expect(page.getByAltText("用來比對的大頭貼")).toBeVisible();

  // 沒勾同意不能開始。
  const start = page.getByRole("button", { name: "開始真人驗證" });
  await expect(start).toBeDisabled();
  await page.getByLabel("我同意使用鏡頭畫面進行本次真人驗證。").check();
  const issued = page.waitForResponse(
    (r) =>
      r.url().endsWith("/api/v1/verification/challenge") &&
      r.request().method() === "POST",
  );
  await start.click();

  // 相機與模型準備好才領挑戰；假相機的畫面沒有人臉，會即時提示找不到臉。
  await expect(page.locator(".live-stage.on video")).toBeVisible();
  const response = await issued;
  expect(response.status()).toBe(201);
  const body = await response.json();
  expect(body.actions).toHaveLength(2);
  await expect(page.locator(".live-hint")).toContainText("找不到臉", {
    timeout: 30000,
  });
  await expect(page.locator(".live-steps li.current")).toContainText(
    "正對鏡頭",
  );

  // MediaPipe 的 WASM 與模型都從本站載入，沒有連到 Google。
  expect(
    requests.some((u) => u.includes("/mediapipe/face_landmarker.task")),
  ).toBe(true);
  expect(requests.some((u) => u.includes("/mediapipe/wasm/"))).toBe(true);
  expect(
    requests.filter((u) => /googleapis\.com|gstatic\.com/.test(u)),
  ).toEqual([]);

  // 取消會關掉鏡頭，回到開始前的畫面。
  await page.getByRole("button", { name: "取消" }).click();
  await expect(page.locator(".live-stage.on")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "開始真人驗證" }),
  ).toBeVisible();
});

test("驗證結果依原因代碼顯示說明", async ({ page }, info) => {
  test.skip(
    info.project.name === "mobile",
    "文字判斷與版面無關，桌面測一次即可。",
  );
  const headers = await createUser(page.request);
  await uploadPhoto(page.request, headers);
  const cases = [
    {
      state: { status: "verified", reasonCode: "VERIFICATION_PASSED" },
      heading: "已通過驗證",
      text: "驗證完成",
    },
    {
      state: { status: "rejected", reasonCode: "CHALLENGE_FAILED" },
      heading: "未通過驗證",
      text: "沒有偵測到指定的動作",
    },
    {
      state: { status: "rejected", reasonCode: "REFERENCE_NO_FACE_DETECTED" },
      heading: "未通過驗證",
      text: "你的大頭貼無法用來比對",
    },
    {
      state: { status: "unavailable", reasonCode: "MODEL_NOT_CONFIGURED" },
      heading: "這次無法完成驗證",
      text: "驗證服務暫時無法使用",
    },
  ];
  for (const { state, heading, text } of cases) {
    await page.route("**/api/v1/verification/status", (route) =>
      route.fulfill({ json: { ...state, canVerify: true } }),
    );
    await page.goto("/verification");
    await expect(
      page.getByRole("heading", { name: heading, exact: true }),
    ).toBeVisible();
    await expect(page.locator(".verification-status")).toContainText(text);
    await page.unroute("**/api/v1/verification/status");
  }
});
