// ai-priority.ts 的單元測試：不需要 Redis，用假的 store 記錄呼叫順序。
// 這個標記決定背景 worker 要不要讓路；標了沒移除，背景工作會一直等（最多 30 秒），
// 所以「失敗也要移除」「Redis 壞了不能影響推薦」都要測到。
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
require("ts-node/register/transpile-only");
const {
  trackOnlineRequest,
  ONLINE_REQUESTS_KEY,
  STALE_MS,
} = require("../src/ai-priority.ts");

/** 假的 Redis：記下每個指令；fail=true 時每個指令都失敗。 */
function fakeStore({ fail = false } = {}) {
  const calls = [];
  const record =
    (name) =>
    async (...args) => {
      calls.push([name, ...args]);
      if (fail) throw new Error("redis down");
      return 1;
    };
  return {
    calls,
    zadd: record("zadd"),
    zrem: record("zrem"),
    zremrangebyscore: record("zremrangebyscore"),
  };
}

test("推薦期間標記，結束後移除；寫入前先清掉殘骸", async () => {
  const store = fakeStore();
  const now = () => 1_000_000;
  let seenDuringWork;
  const result = await trackOnlineRequest(
    store,
    "req-1",
    async () => {
      seenDuringWork = store.calls.map((call) => call[0]);
      return "suggestions";
    },
    now,
  );
  assert.equal(result, "suggestions");
  assert.deepEqual(seenDuringWork, ["zremrangebyscore", "zadd"]);
  assert.deepEqual(store.calls, [
    ["zremrangebyscore", ONLINE_REQUESTS_KEY, "-inf", 1_000_000 - STALE_MS],
    ["zadd", ONLINE_REQUESTS_KEY, 1_000_000, "req-1"],
    ["zrem", ONLINE_REQUESTS_KEY, "req-1"],
  ]);
});

test("推薦失敗時也要移除標記，錯誤照常往外丟", async () => {
  const store = fakeStore();
  await assert.rejects(
    trackOnlineRequest(store, "req-2", async () => {
      throw new Error("AI_UNAVAILABLE");
    }),
    /AI_UNAVAILABLE/,
  );
  assert.deepEqual(store.calls.at(-1), ["zrem", ONLINE_REQUESTS_KEY, "req-2"]);
});

test("Redis 壞掉不影響推薦本身", async () => {
  const store = fakeStore({ fail: true });
  const result = await trackOnlineRequest(store, "req-3", async () => "ok");
  assert.equal(result, "ok");
});
