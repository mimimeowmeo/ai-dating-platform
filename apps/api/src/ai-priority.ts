/**
 * 線上推薦優先（2026-09-23）：告訴 Python AI worker「現在有人在等推薦」，讓背景工作先讓路。
 *
 * 為什麼需要：萃取模型與線上推薦都用 Ollama Cloud，而這個帳號同時只處理 1 個請求
 * （實測同時送 3 個，完成時間依序錯開）。背景 worker 萃取風格卡時，如果剛好有人按「AI 推薦」，
 * 推薦就得排在後面，可能超過單一模型 12 秒的逾時而回 503。
 *
 * 做法：呼叫推薦 API 的期間，把 requestId 記在 Redis 的 sorted set（分數是開始時間，毫秒），
 * 結束後移除。worker（services/ai/app/reply/priority.py）每次呼叫模型前看這個集合，
 * 有推薦在進行就先等。兩邊的 key 與殘骸判定時間必須一致。
 */

/** Redis sorted set 的 key；跟 priority.py 的 ONLINE_REQUESTS_KEY 一致。 */
export const ONLINE_REQUESTS_KEY = "ai:online-requests";
/**
 * 超過這麼久的紀錄視為殘骸（程序在推薦途中當掉、來不及移除），寫入前順手清掉；
 * worker 那邊也會忽略它們（priority.py 的 STALE_AFTER_MS）。
 */
export const STALE_MS = 45_000;

/** 只用到這三個 Redis 指令；ioredis 的連線符合這個形狀，測試可以傳假的物件。 */
export type OnlineRequestStore = {
  zadd(key: string, score: number, member: string): Promise<unknown>;
  zrem(key: string, member: string): Promise<unknown>;
  zremrangebyscore(
    key: string,
    min: number | string,
    max: number | string,
  ): Promise<unknown>;
};

/**
 * 執行 work（呼叫推薦 API）的期間，把 requestId 標成「線上推薦進行中」。
 *
 * - 開始前：清掉殘骸，再把 requestId 加進 sorted set。
 * - 結束後（不論成功或失敗）：把 requestId 移除。
 * - Redis 出錯只會讓背景工作這一次不讓路，不影響推薦本身，所以錯誤一律吞掉。
 */
export async function trackOnlineRequest<T>(
  store: OnlineRequestStore,
  requestId: string,
  work: () => Promise<T>,
  now: () => number = Date.now,
): Promise<T> {
  const started = now();
  try {
    await store.zremrangebyscore(
      ONLINE_REQUESTS_KEY,
      "-inf",
      started - STALE_MS,
    );
    await store.zadd(ONLINE_REQUESTS_KEY, started, requestId);
  } catch {
    // 記不進去：這次背景工作不會讓路，推薦照常進行。
  }
  try {
    return await work();
  } finally {
    await store.zrem(ONLINE_REQUESTS_KEY, requestId).catch(() => undefined);
  }
}
