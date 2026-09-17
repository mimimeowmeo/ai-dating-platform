"""REDIS_URL 指向測試 Redis 時執行；驗證真實 BullMQ consumer。

以唯一 queue 隔離，不清除既有 queue 或業務資料。
"""

import asyncio
import base64
import io
import os
import uuid

import redis.asyncio as redis
from bullmq import Job, Queue, Worker
from PIL import Image

from app.config import Settings
from app.verification import VerificationService
from app.worker import make_processor


async def main():
    redis_url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
    # 先做有界連線檢查，避免 BullMQ 的長時間重試把測試卡住。
    probe = redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
    try:
        async with asyncio.timeout(5):
            await probe.ping()
    finally:
        await probe.aclose()
    queue_name = f"ai-verification-test-{uuid.uuid4()}"
    queue = Queue(queue_name, {"connection": redis_url})
    worker = Worker(
        queue_name,
        make_processor(VerificationService(Settings())),
        {"connection": redis_url},
    )
    buffer = io.BytesIO()
    Image.new("RGB", (128, 128)).save(buffer, format="PNG")
    job = None
    try:
        job = await queue.add("verify", {
            "imageBase64": base64.b64encode(buffer.getvalue()).decode(),
            "mimeType": "image/png", "requestId": str(uuid.uuid4()),
        }, {"attempts": 1, "removeOnFail": True})
        async with asyncio.timeout(20):
            while await job.getState() != "completed":
                await asyncio.sleep(0.05)
        stored = await Job.fromId(queue, job.id)
        assert stored.returnvalue["status"] == "unavailable", stored.returnvalue
        assert stored.returnvalue["reasonCode"] == "MODEL_NOT_CONFIGURED"
        assert stored.data == {"redacted": True}, "Image remained in Redis job data"
        assert "faceMatchScore" not in stored.returnvalue
        print("PASS: 真實 Redis BullMQ 消費、無模型決策、影像去敏")
    finally:
        await worker.close()
        if job:
            await job.remove()
        # 僅移除本次測試的 UUID queue，保留所有其他 queue。
        await queue.obliterate(force=True)
        await queue.close()


if __name__ == "__main__":
    asyncio.run(main())
