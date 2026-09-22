"""REDIS_URL 指向測試 Redis 時執行；驗證推薦回覆 worker 真的能透過 BullMQ 收 job、把結果放進結果佇列。

用唯一的 queue 名稱隔離，不影響既有的 queue 或業務資料；結束時只清掉本次建立的 queue。
向量服務用假的（不連 Gemini），只驗證佇列串接本身。

執行：REDIS_URL=redis://127.0.0.1:6379/0 .venv/bin/python -m tests.integration_reply_queue
"""

import asyncio
import os
import uuid

import redis.asyncio as redis
from bullmq import Job, Queue, Worker

from app.reply.service import ReplyAIService
from app.reply.worker import make_processor
from tests.reply_helpers import SETTINGS, FakeEmbedder
from tests.test_reply_api import chunk_payload


async def main():
    """送一個 chunk-embed job，等 worker 處理完，再到結果佇列確認結果內容。"""
    redis_url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
    # 先做有界的連線檢查，避免 BullMQ 的長時間重試把測試卡住。
    probe = redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
    try:
        async with asyncio.timeout(5):
            await probe.ping()
    finally:
        await probe.aclose()
    suffix = uuid.uuid4()
    jobs = Queue(f"ai-jobs-test-{suffix}", {"connection": redis_url})
    results = Queue(f"ai-results-test-{suffix}", {"connection": redis_url})
    service = ReplyAIService(SETTINGS, embedder=FakeEmbedder())
    worker = Worker(f"ai-jobs-test-{suffix}", make_processor(service, results), {"connection": redis_url})
    try:
        job = await jobs.add("chunk-embed", chunk_payload(True), {"attempts": 1})
        async with asyncio.timeout(20):
            while await job.getState() != "completed":
                await asyncio.sleep(0.05)
        stored = await Job.fromId(jobs, job.id)
        assert stored.returnvalue["ok"] is True, stored.returnvalue
        waiting = await results.getJobs(["waiting"])
        assert len(waiting) == 1, waiting
        data = waiting[0].data
        assert data["key"] == "c-1" and data["name"] == "chunk-embed", data
        assert len(data["result"]["chunks"]) == 2 and data["result"]["chunks"][0]["vector"], data
        print("PASS: 真實 Redis BullMQ：ai-jobs 消費、結果放進 ai-results、job 回傳值不含大資料")
    finally:
        await worker.close()
        # 只移除本次測試的 UUID queue，保留所有其他 queue。
        for queue in (jobs, results):
            await queue.obliterate(force=True)
            await queue.close()


if __name__ == "__main__":
    asyncio.run(main())
