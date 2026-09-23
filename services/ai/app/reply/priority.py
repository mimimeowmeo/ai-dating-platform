"""線上推薦優先：背景工作呼叫模型前，先讓正在進行的「AI 推薦」跑完（2026-09-23）。

為什麼需要：Ollama Cloud 這個帳號同時只處理 1 個請求。2026-09-23 實測同時送 3 個推薦，
完成時間依序錯開（1.7／3.2／4.9 秒）。背景 worker 萃取風格卡或整理摘要時，如果剛好有人按
「AI 推薦」，推薦就得排在背景請求後面；再加上備援的 Gemini 被限流，推薦就會回 503。

做法：
- NestJS 在呼叫推薦 API 的期間，把 requestId 記在 Redis 的 sorted set 裡（分數是開始時間，
  毫秒），結束後移除（`apps/api/src/ai-priority.ts`）。
- worker 每次呼叫模型前先執行 `OnlinePriority.wait_for_idle()`：有推薦正在進行，就每隔
  POLL_SECONDS 看一次，等到沒有為止；最多等 MAX_WAIT_SECONDS，避免推薦很頻繁時背景工作永遠做不完。
- 超過 STALE_AFTER_MS 還沒移除的紀錄，視為 NestJS 在推薦途中當掉留下的，不再等它。
- Redis 出錯時不等（fail open）：寧可偶爾跟推薦搶到，也不能讓背景工作卡住。

已經送出的背景請求無法中途插隊，所以推薦最多還是要等「正在跑的那一批」。
實測最大的一批萃取（約 5,300 tokens）在 Ollama 閒置時約 2.8 秒，遠低於推薦的 12 秒逾時。
"""

import asyncio
import time
from typing import Protocol

# 跟 apps/api/src/ai-priority.ts 的 ONLINE_REQUESTS_KEY 一致；兩邊要一起改。
ONLINE_REQUESTS_KEY = "ai:online-requests"
# 超過這麼久還在的紀錄視為殘骸（NestJS 那邊同一個數字：STALE_MS）。
STALE_AFTER_MS = 45_000
# 有推薦在進行時，多久再看一次。
POLL_SECONDS = 0.25
# 最多讓多久；推薦一次約 2～3 秒，30 秒已經足夠讓一連串的「換一批」跑完。
MAX_WAIT_SECONDS = 30.0


class SortedSetCounter(Protocol):
    """wait_for_idle 只需要 Redis 的 ZCOUNT；測試可以用假的物件代替 redis.asyncio 的連線。"""

    async def zcount(self, name: str, min: float | str, max: float | str) -> int:
        """回傳 sorted set 裡分數介於 min 與 max 之間的成員數。"""
        ...


class OnlinePriority:
    """讓背景工作在呼叫模型前，先等正在進行的線上推薦結束。"""

    def __init__(
        self,
        client: SortedSetCounter,
        *,
        key: str = ONLINE_REQUESTS_KEY,
        poll_seconds: float = POLL_SECONDS,
        max_wait_seconds: float = MAX_WAIT_SECONDS,
        stale_after_ms: int = STALE_AFTER_MS,
    ):
        """client 是 redis.asyncio 的連線（或測試用的假物件）；其餘參數讓測試可以把時間調短。"""
        self.client = client
        self.key = key
        self.poll_seconds = poll_seconds
        self.max_wait_seconds = max_wait_seconds
        self.stale_after_ms = stale_after_ms

    async def online_count(self) -> int:
        """目前有幾個線上推薦正在進行；只算 stale_after_ms 以內開始的，更早的視為殘骸。"""
        now_ms = int(time.time() * 1000)
        return int(await self.client.zcount(self.key, now_ms - self.stale_after_ms, "+inf"))

    async def wait_for_idle(self) -> float:
        """等到沒有線上推薦在進行，回傳實際等了幾秒。

        - 沒有推薦在進行：立刻回傳（只查一次 Redis）。
        - 有推薦在進行：每 poll_seconds 再查一次，最多等 max_wait_seconds。
        - 查 Redis 出錯：立刻回傳，不讓背景工作因為 Redis 的問題卡住。
        """
        loop = asyncio.get_running_loop()
        started = loop.time()
        while True:
            try:
                if await self.online_count() == 0:
                    break
            except Exception:  # noqa: BLE001 — 任何 Redis 錯誤都放行（fail open）
                break
            waited = loop.time() - started
            if waited >= self.max_wait_seconds:
                break
            await asyncio.sleep(min(self.poll_seconds, self.max_wait_seconds - waited))
        return loop.time() - started
