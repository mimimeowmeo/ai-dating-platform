// 一次跑完整套驗收，最後自動清掉這輪建立的測試帳號。
// 用法：node scripts/with-env.mjs node scripts/verify/run-all.mjs
// 前提：docker compose 已啟動，且受測站台（預設 :8080）可以連線。
// 驗別的 stack 時，除了 lib.mjs 列的 VERIFY_*，這支還會用到：
//   E2E_BASE_URL           Playwright 打的站台，沒設時跟著 VERIFY_ORIGIN
//   VERIFY_REDIS_CONTAINER 重置限流計數的 Redis 容器，沒設時的找法見 resetRateLimit 上方
//   DATABASE_URL、S3_*     清理腳本用（.env），要和站台同一個 stack，開跑前會先確認
import { execFileSync, spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { API, ORIGIN, PG_CONTAINER } from "./lib.mjs";

const runId = process.env.E2E_RUN_ID || randomUUID();
// apps/web/playwright.config.ts 沒拿到 E2E_BASE_URL 就打 :8080（本機的主 stack）。
// 這裡改成跟著 VERIFY_ORIGIN，免得 API 檢查打受測環境、瀏覽器測試卻跑去 :8080 建帳號。
const site = process.env.E2E_BASE_URL || ORIGIN;
const env = { ...process.env, E2E_RUN_ID: runId, E2E_BASE_URL: site };
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

const failures = [];
const step = async (command, args, label) => {
  const code = await run(command, args, label);
  if (code !== 0 && !failures.includes(label)) failures.push(label);
  return code;
};
const cleanup = () =>
  step("node", ["apps/api/test/cleanup-e2e.mjs"], "cleanup-e2e");

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

// 註冊／登入的限流是每個來源 IP 20 次 / 5 分鐘（apps/api/src/auth.ts 的 limit(`auth:${ip}`, 20, 300)）。
// 跑完一輪完整驗收要建 20 個以上的帳號，一定會撞上限，撞到之後的測試會以
// 「註冊後沒跳轉」之類的假性失敗收場。這裡只清 rate:* 計數器，不動佇列或其他資料。
// 要清的是受測 stack 的 Redis，依序採用：
//   1. VERIFY_REDIS_CONTAINER（和 VERIFY_PG_CONTAINER 一樣直接指定容器）
//   2. 有設 COMPOSE_PROJECT_NAME／COMPOSE_FILE 時交給 docker compose，它自己會讀這兩個變數
//   3. VERIFY_PG_CONTAINER 所在 compose 專案裡的 redis 服務
//   4. 都找不到才用目前目錄預設 compose 專案的 redis
const redisNextToPostgres = () => {
  const docker = (...args) =>
    execFileSync("docker", args, { stdio: ["ignore", "pipe", "ignore"] })
      .toString()
      .trim();
  try {
    const project = docker(
      "inspect",
      "--format",
      '{{index .Config.Labels "com.docker.compose.project"}}',
      PG_CONTAINER,
    );
    if (!project) return undefined;
    return (
      docker(
        "ps",
        "--filter",
        `label=com.docker.compose.project=${project}`,
        "--filter",
        "label=com.docker.compose.service=redis",
        "--format",
        "{{.Names}}",
      ).split("\n")[0] || undefined
    );
  } catch {
    return undefined;
  }
};
const redisContainer =
  process.env.VERIFY_REDIS_CONTAINER ||
  (process.env.COMPOSE_PROJECT_NAME || process.env.COMPOSE_FILE
    ? undefined
    : redisNextToPostgres());
const flushRateKeys =
  "redis-cli --scan --pattern 'rate:*' | xargs -r redis-cli DEL";
const resetRateLimit = () =>
  step(
    "docker",
    redisContainer
      ? ["exec", redisContainer, "sh", "-c", flushRateKeys]
      : ["compose", "exec", "-T", "redis", "sh", "-c", flushRateKeys],
    "重置 rate limit 計數",
  );

// 清理腳本刪的是 DATABASE_URL 那個資料庫，瀏覽器測試的帳號卻建在站台背後的資料庫；
// 兩邊不是同一個 stack 時，帳號會留在沒人清的地方（2026-09-27 就這樣在本機主 stack 留下 17 個）。
// 所以開跑前先經由站台註冊一個探測帳號、跑一次清理，再確認它已經登入不了。
const cleanupProblem = async () => {
  const origin = new URL(site).origin;
  const probe = {
    email: `e2e-${runId}-probe@example.test`,
    password: `Safe-${randomUUID()}`,
  };
  const post = (path) =>
    fetch(`${origin}/api/v1${path}`, {
      method: "POST",
      headers: { origin, "content-type": "application/json" },
      body: JSON.stringify(probe),
    }).then(
      (res) => res.status,
      (error) => error.cause?.code || error.cause?.message || error.message,
    );
  // 清不掉時帳號留在站台那邊，用那個 stack 的 DATABASE_URL 跑一次清理就好。
  const leftover = `探測帳號 ${probe.email} 可能還在，用 ${origin} 那個 stack 的 DATABASE_URL 執行 E2E_RUN_ID=${runId} node apps/api/test/cleanup-e2e.mjs 清掉`;
  console.log(`\n▶ 確認清理碰得到受測站台 ${origin}`);
  const registered = await post("/auth/register");
  if (registered !== 201)
    return `經由 ${origin} 註冊探測帳號失敗：${registered}`;
  if (
    (await run("node", ["apps/api/test/cleanup-e2e.mjs"], "cleanup-e2e")) !== 0
  )
    return `清理腳本執行失敗；${leftover}`;
  const login = await post("/auth/login");
  if (login === 401) return null;
  return login === 201
    ? `DATABASE_URL 不是 ${origin} 背後的資料庫；${leftover}`
    : `確認不了探測帳號有沒有被清掉（登入回 ${login}）；${leftover}`;
};

console.log(
  `受測站台 ${site}（API ${API}）、資料庫容器 ${PG_CONTAINER}、Redis ${redisContainer ?? "docker compose 的 redis 服務"}`,
);
await resetRateLimit();
const problem = await cleanupProblem();
if (problem) {
  console.log(
    `\n✗ ${problem}。\n  DATABASE_URL、S3_* 要和 E2E_BASE_URL（沒設時是 VERIFY_ORIGIN）指到同一個 stack，這輪不跑會建帳號的步驟。`,
  );
  failures.push("清理碰不到受測站台");
} else {
  for (const [command, args, label] of steps) await step(command, args, label);

  // 瀏覽器測試會挑探索頁的第一張卡片，上面各套留下的測試帳號會擠進候選名單
  // 把對方擠掉，所以先清一次再跑，否則 Playwright 會假性失敗。
  console.log("\n▶ 清掉上面各套的測試帳號（避免污染探索頁）");
  await cleanup();

  // desktop 與 mobile 分兩次跑，和 CI（.github/workflows/ci.yml）一樣：整套 e2e 的註冊／登入
  // 合計超過 20 次，一次跑完，排在後面的 mobile 會撞到限流（429）。每次開跑前清計數、
  // 跑完清帳號，mobile 的探索頁才不會被 desktop 留下的帳號擠滿。
  for (const project of ["desktop", "mobile"]) {
    await resetRateLimit();
    await step(
      "node",
      [
        "apps/web/node_modules/@playwright/test/cli.js",
        "test",
        "--config",
        "apps/web/playwright.config.ts",
        "--project",
        project,
      ],
      `Playwright e2e（${project}）`,
    );
    console.log(`\n▶ 清理 ${project} 建立的測試帳號`);
    await cleanup();
  }
}

console.log("\n▶ 冗餘資料表／欄位報表");
await step(
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
