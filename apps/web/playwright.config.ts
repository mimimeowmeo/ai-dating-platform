import { defineConfig, devices } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  // 這些測試會走完整的註冊→配對→socket 推播流程，經過 nginx 與（做展示時）
  // 外部通道，5 秒的預設 expect 上限對即時訊息太緊。
  // 專案自己的驗收腳本 scripts/verify/chat-and-match.mjs 也是給 10 秒。
  timeout: 60000,
  expect: { timeout: 15000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: process.env.E2E_BASE_URL || "http://localhost:8080",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    launchOptions: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE }
      : {},
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    {
      name: "mobile",
      use: { ...devices["iPhone 13"], defaultBrowserType: "chromium" },
    },
  ],
});
