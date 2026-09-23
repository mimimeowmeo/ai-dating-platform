import time
import unittest

from app.reply.priority import ONLINE_REQUESTS_KEY, OnlinePriority
from app.reply.service import ReplyAIService
from tests.reply_helpers import SETTINGS, FakeEmbedder


class FakeRedis:
    """假的 Redis：zcount 依序回傳 answers（最後一個會一直重複）；給 error 時每次都丟出那個錯誤。"""

    def __init__(self, answers, error: Exception | None = None):
        self.answers = list(answers)
        self.error = error
        self.calls: list[tuple] = []

    async def zcount(self, name, min, max):
        self.calls.append((name, min, max))
        if self.error is not None:
            raise self.error
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


class OnlinePriorityTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_immediately_when_no_online_request(self):
        redis = FakeRedis([0])
        waited = await OnlinePriority(redis, poll_seconds=0.01).wait_for_idle()
        self.assertLess(waited, 0.05)
        self.assertEqual(len(redis.calls), 1)
        name, low, high = redis.calls[0]
        self.assertEqual((name, high), (ONLINE_REQUESTS_KEY, "+inf"))
        # 只算 45 秒內開始的推薦；更早的視為 NestJS 在推薦途中當掉留下的殘骸。
        self.assertAlmostEqual(low, time.time() * 1000 - 45_000, delta=2_000)

    async def test_waits_until_online_requests_finish(self):
        redis = FakeRedis([1, 2, 0])
        waited = await OnlinePriority(redis, poll_seconds=0.01, max_wait_seconds=5).wait_for_idle()
        self.assertEqual(len(redis.calls), 3)
        self.assertGreaterEqual(waited, 0.015)

    async def test_gives_up_after_max_wait_so_background_work_still_finishes(self):
        redis = FakeRedis([1])  # 一直有推薦在進行
        started = time.perf_counter()
        waited = await OnlinePriority(redis, poll_seconds=0.01, max_wait_seconds=0.05).wait_for_idle()
        self.assertGreaterEqual(waited, 0.05)
        self.assertLess(time.perf_counter() - started, 0.5)

    async def test_redis_error_does_not_block_background_work(self):
        redis = FakeRedis([1], error=ConnectionError("redis down"))
        waited = await OnlinePriority(redis, poll_seconds=0.01).wait_for_idle()
        self.assertLess(waited, 0.05)
        self.assertEqual(len(redis.calls), 1)


class ServiceWiringTests(unittest.TestCase):
    def test_only_background_components_yield_to_online_requests(self):
        service = ReplyAIService(SETTINGS, embedder=FakeEmbedder())

        async def wait():
            return 0.0

        service.yield_to_online_requests(wait)
        self.assertIs(service.style_builder.before_model_call, wait)
        self.assertIs(service.summarizer.before_model_call, wait)
        # 線上推薦自己不等：它就是別人要讓的對象。
        self.assertFalse(hasattr(service.suggester, "before_model_call"))


if __name__ == "__main__":
    unittest.main()
