"""只消費真人驗證工作；不提前啟用推薦、向量或聊天分析。"""

import asyncio
import contextlib
import signal
import time
from pathlib import Path

import redis.asyncio as redis
from bullmq import Worker
from pydantic import ValidationError
from redis.exceptions import RedisError

from .config import Settings
from .schemas import VerificationRequest
from .verification import InvalidImage, VerificationService

QUEUE_NAME = "ai-verification"
JOB_NAME = "verify"
HEARTBEAT_FILE = Path("/tmp/ai-worker-heartbeat")


def make_processor(service: VerificationService):
    async def process(job, _token):
        data = job.data
        # 拿到工作後立即移除 Redis job hash 中的自拍；推論只使用記憶體副本。
        await job.updateData({"redacted": True})
        if job.name != JOB_NAME:
            raise ValueError("UNSUPPORTED_JOB")
        try:
            request = VerificationRequest.model_validate(data)
            result = await service.verify(request)
        except (ValidationError, InvalidImage):
            # Pydantic 的預設例外會包含輸入，不得進入 BullMQ 的失敗紀錄。
            raise ValueError("INVALID_VERIFICATION_INPUT") from None
        # 佇列結果只存決策與版本，不在 Redis 留下生物特徵分數。
        return result.model_dump(exclude={"livenessScore", "faceMatchScore"})

    return process


async def maintain_heartbeat(client, worker, stopped: asyncio.Event, path: Path = HEARTBEAT_FILE):
    while not stopped.is_set():
        try:
            async with asyncio.timeout(5):
                await client.ping()
            if worker.running and not worker.closing:
                path.write_text(str(time.time()), encoding="ascii")
            else:
                path.unlink(missing_ok=True)
        except (RedisError, TimeoutError, OSError):
            path.unlink(missing_ok=True)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stopped.wait(), timeout=20)


async def main():
    settings = Settings.from_env()
    HEARTBEAT_FILE.unlink(missing_ok=True)
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stopped.set)
    client = redis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=3, socket_timeout=3)
    worker = Worker(
        QUEUE_NAME, make_processor(VerificationService(settings)),
        {"connection": settings.redis_url, "concurrency": 2, "autorun": False},
    )
    # 只記錄固定代碼，避免第三方例外把影像或連線憑證寫入日誌。
    worker.on("error", lambda *_args: print("AI_WORKER_QUEUE_ERROR", flush=True))
    run_task = asyncio.create_task(worker.run())
    heartbeat = asyncio.create_task(maintain_heartbeat(client, worker, stopped))
    stop_task = asyncio.create_task(stopped.wait())
    try:
        done, _ = await asyncio.wait({run_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        if run_task in done and not stopped.is_set():
            raise RuntimeError("AI_WORKER_STOPPED") from None
    finally:
        stopped.set()
        heartbeat.cancel()
        stop_task.cancel()
        HEARTBEAT_FILE.unlink(missing_ok=True)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(worker.close(), timeout=15)
        run_task.cancel()
        await asyncio.gather(run_task, heartbeat, stop_task, return_exceptions=True)
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
