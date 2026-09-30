"""推薦 worker：消費 BullMQ 的 rec-jobs，把照片算成外貌向量，結果放進 rec-results。

- job 名稱：`embed-appearance`；data：`{photoId, image}`，image 是主照片在物件儲存裡的 JPEG 原檔（base64，上傳時已轉成長邊 1600px 以內）。
- 結果：`{photoId, modelVersion, embedding}`；找不到臉時 embedding 是 null。由 NestJS 寫進資料庫
  （ADR 0002：Python 不碰資料庫），「扣掉戴眼鏡方向」也在 NestJS 做。
- job 名稱：`explain-appearance`（測試畫面「像在哪裡」）；data：`{requestId, candidate, anchor, direction?, directionVersion?}`，
  兩張主照片原檔。結果：`{requestId, result}`，result 是各臉部區域遮住後相似度下降多少與框的座標（見 appearance/explain.py），
  任一張找不到臉時是 null。不寫資料庫，NestJS 只短暫快取在 Redis。
- 佇列和 AI 服務的 ai-jobs／ai-results 分開：兩邊的 worker 遇到不認得的 job 都會直接失敗。

隱私：取到 job 就把 Redis 裡的 data 覆寫成 {"redacted": True}，照片不留在 Redis；
錯誤只丟固定代碼、日誌只印固定代碼，不帶 job 內容。代價是覆寫後的 job 無法重試，
NestJS 也只排一次（attempts 1），漏掉的照片由 NestJS 每分鐘補排。

在線訊號：心跳時在 Redis 寫 `rec-worker:online`（60 秒過期），NestJS 看到它才把照片放進佇列，
worker 沒在跑時照片不會堆在 Redis。不用 BullMQ 的 getWorkers()：Node 版用 base64 的連線名稱辨識 worker，
Python 版的名稱對不上。

執行：`python -m app.worker`；健康檢查：`python -m app.worker_health`。
"""
import asyncio
import base64
import binascii
import contextlib
import signal
import time
from pathlib import Path

import redis.asyncio as redis
from bullmq import Queue, Worker
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from redis.exceptions import RedisError

from appearance.pipeline import MODEL_VERSION, AppearanceEmbedder, InvalidImage
from appearance.explain import REGIONS

from .config import Settings
from .worker_health import HEARTBEAT_FILE

QUEUE_NAME = "rec-jobs"
RESULTS_QUEUE = "rec-results"
JOB_NAME = "embed-appearance"
EXPLAIN_JOB = "explain-appearance"
ONLINE_KEY = "rec-worker:online"
ONLINE_TTL_SECONDS = 60
# NestJS 處理結果失敗時由 BullMQ 以指數退避重試，成功後刪除。結果裡沒有照片，只有向量。
RESULT_JOB_OPTIONS = {"attempts": 5, "backoff": {"type": "exponential", "delay": 2000}, "removeOnComplete": True}
# 長邊 1600px 以內的 JPEG 通常不到 1 MB；base64 會再大三分之一。
MAX_IMAGE_BASE64 = 4 * 1024 * 1024


UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"


class EmbedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    photoId: str = Field(pattern=rf"^{UUID}$")
    image: str = Field(min_length=1, max_length=MAX_IMAGE_BASE64)


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    # 卡片上的人與喜歡過的人的主照片 id，NestJS 用它當快取鍵。
    requestId: str = Field(pattern=rf"^{UUID}:{UUID}$")
    candidate: str = Field(min_length=1, max_length=MAX_IMAGE_BASE64)
    anchor: str = Field(min_length=1, max_length=MAX_IMAGE_BASE64)
    # 戴眼鏡方向；JSON 裡的 0 會是整數，所以這欄不用嚴格型別。
    direction: list[float] | None = Field(default=None, min_length=512, max_length=512, strict=False)
    directionVersion: str | None = Field(default=None, max_length=200)


def make_processor(embedder: AppearanceEmbedder, results: Queue):
    async def process(job, _token):
        data = job.data
        await job.updateData({"redacted": True})
        if job.name == EXPLAIN_JOB:
            return await explain(data)
        if job.name != JOB_NAME:
            raise ValueError("UNSUPPORTED_JOB")
        try:
            request = EmbedRequest.model_validate(data)
        except ValidationError:
            raise ValueError("INVALID_JOB_INPUT") from None
        try:
            image = base64.b64decode(request.image, validate=True)
            embedding = await asyncio.to_thread(embedder.embed, image)
        except (binascii.Error, ValueError, InvalidImage):
            raise ValueError("INVALID_IMAGE") from None
        await results.add(
            JOB_NAME,
            {"photoId": request.photoId, "modelVersion": MODEL_VERSION, "embedding": embedding},
            RESULT_JOB_OPTIONS,
        )
        return {"ok": True, "faceFound": embedding is not None}

    async def explain(data):
        try:
            request = ExplainRequest.model_validate(data)
        except ValidationError:
            raise ValueError("INVALID_JOB_INPUT") from None
        # 方向和目前的模型同版本才扣，數字才會和存進資料庫的向量一致。
        direction = request.direction if request.directionVersion == MODEL_VERSION else None
        try:
            candidate = base64.b64decode(request.candidate, validate=True)
            anchor = base64.b64decode(request.anchor, validate=True)
            result = await asyncio.to_thread(embedder.explain, candidate, anchor, direction)
        except (binascii.Error, ValueError, InvalidImage):
            raise ValueError("INVALID_IMAGE") from None
        await results.add(EXPLAIN_JOB, {"requestId": request.requestId, "result": result}, RESULT_JOB_OPTIONS)
        return {"ok": True, "explained": result is not None, "regions": len(REGIONS)}

    return process


async def maintain_heartbeat(client, worker, stopped: asyncio.Event, path: Path = HEARTBEAT_FILE):
    """Redis 連得上、worker 正在跑時才寫心跳檔與在線訊號（約每 20 秒）；否則刪掉，讓健康檢查判定不健康。"""
    while not stopped.is_set():
        try:
            async with asyncio.timeout(5):
                if worker.running and not worker.closing:
                    await client.set(ONLINE_KEY, "1", ex=ONLINE_TTL_SECONDS)
                    path.write_text(str(time.time()), encoding="ascii")
                else:
                    await client.delete(ONLINE_KEY)
                    path.unlink(missing_ok=True)
        except (RedisError, OSError, TimeoutError):
            path.unlink(missing_ok=True)
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(20):
                await stopped.wait()
    path.unlink(missing_ok=True)


async def main():
    settings = Settings.from_env()
    HEARTBEAT_FILE.unlink(missing_ok=True)
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stopped.set)
    # 模型在開始接 job 之前載入；載入失敗就讓程序結束，容器會被判定不健康。
    embedder = await asyncio.to_thread(AppearanceEmbedder.from_dir, settings.models_dir)
    client = redis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=3, socket_timeout=3)
    results = Queue(RESULTS_QUEUE, {"connection": settings.redis_url})
    worker = Worker(
        QUEUE_NAME,
        make_processor(embedder, results),
        {"connection": settings.redis_url, "concurrency": 1, "autorun": False},
    )
    worker.on("error", lambda *_args: print("REC_WORKER_QUEUE_ERROR", flush=True))
    run_task = asyncio.create_task(worker.run())
    heartbeat = asyncio.create_task(maintain_heartbeat(client, worker, stopped))
    stop_task = asyncio.create_task(stopped.wait())
    try:
        done, _ = await asyncio.wait({run_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        if run_task in done and not stopped.is_set():
            print("REC_WORKER_STOPPED", flush=True)
    finally:
        stopped.set()
        # 先撤掉在線訊號，NestJS 不再排新照片，再等手上的 job 做完。
        with contextlib.suppress(Exception):
            async with asyncio.timeout(3):
                await client.delete(ONLINE_KEY)
        with contextlib.suppress(Exception):
            async with asyncio.timeout(15):
                await worker.close()
        heartbeat.cancel()
        stop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await heartbeat
        await results.close()
        await client.aclose()
        HEARTBEAT_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
