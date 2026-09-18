// 一次跑完整套驗收，最後自動清掉這輪建立的測試帳號。
// 用法：node scripts/with-env.mjs node scripts/verify/run-all.mjs
// 前提：docker compose 已啟動，且 :8080 可以連線。
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";

const runId = process.env.E2E_RUN_ID || randomUUID();
const env = { ...process.env, E2E_RUN_ID: runId };
const run = (command, args, label) =>
  new Promise((resolve) => {
    console.log(`\n▶ ${label}`);
    const child = spawn(command, args, { env, stdio: "inherit" });
    child.once("error", (error) => {
      console.error(error.message);
      resolve(1);
    });
    child.once("exit", (code) => resolve(code ?? 1));
  });

const steps = [
  [
    "node",
    ["scripts/verify/api-endpoints.mjs"],
    "每一支 API 與對應的資料表欄位",
  ],
  ["node", ["scripts/verify/media-access.mjs"], "照片存取控制"],
  ["node", ["scripts/verify/chat-and-match.mjs"], "雙人聊天與配對"],
  ["node", ["scripts/verify/ui-interactions.mjs"], "畫面互動"],
];

// 這一組要在探索頁乾淨的狀態下跑，時機見下方的清理步驟。
const browserSteps = [
  [
    "node",
    [
      "apps/web/node_modules/@playwright/test/cli.js",
      "test",
      "--config",
      "apps/web/playwright.config.ts",
    ],
    "Playwright e2e",
  ],
];

// 註冊／登入的限流是每個來源 IP 20 次 / 5 分鐘（apps/api/src/auth.ts 的 limit(`auth:${ip}`, 20, 300)）。
// 跑完一輪完整驗收要建 20 個以上的帳號，一定會撞上限，撞到之後的測試會以
// 「註冊後沒跳轉」之類的假性失敗收場。這裡只清 rate:* 計數器，不動佇列或其他資料。
const resetRateLimit = () =>
  run(
    "docker",
    [
      "compose",
      "exec",
      "-T",
      "redis",
      "sh",
      "-c",
      "redis-cli --scan --pattern 'rate:*' | xargs -r redis-cli DEL",
    ],
    "重置 rate limit 計數",
  );

await resetRateLimit();

const failures = [];
for (const [command, args, label] of steps) {
  const code = await run(command, args, label);
  if (code !== 0) failures.push(label);
}

// 瀏覽器測試會挑探索頁的第一張卡片，上面各套留下的測試帳號會擠進候選名單
// 把對方擠掉，所以先清一次再跑，否則 Playwright 會假性失敗。
console.log("\n▶ 清掉上面各套的測試帳號（避免污染探索頁）");
await run("node", ["apps/api/test/cleanup-e2e.mjs"], "cleanup-e2e");

// 上面四套已經用掉大半的註冊額度，瀏覽器測試還要再建 8~9 個帳號。
await resetRateLimit();

for (const [command, args, label] of browserSteps) {
  const code = await run(command, args, label);
  if (code !== 0) failures.push(label);
}

console.log("\n▶ 清理這輪建立的測試帳號");
await run("node", ["apps/api/test/cleanup-e2e.mjs"], "cleanup-e2e");

console.log("\n▶ 冗餘資料表／欄位報表");
await run(
  "node",
  ["scripts/verify/redundancy-report.mjs"],
  "redundancy-report",
);

console.log(
  failures.length
    ? `\n✗ 有 ${failures.length} 項沒過：${failures.join("、")}`
    : "\n✓ 全部通過",
);
process.exitCode = failures.length ? 1 : 0;
