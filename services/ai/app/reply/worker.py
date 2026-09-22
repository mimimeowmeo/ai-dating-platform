"""AI 推薦回覆的背景 worker：消費 BullMQ queue `ai-jobs`，把結果放進 `ai-results`。

依 ADR 0002，worker 不碰資料庫：NestJS 把要處理的資料放在 job 裡，worker 算完後把結果
放進結果佇列，由 NestJS 取出寫入資料庫。這樣就算 NestJS 暫時停機，結果也不會遺失。

執行：`python -m app.reply.worker`；健康檢查：`python -m app.reply.worker_health`。
萃取模型（Ollama）一次只能好好處理一個請求，所以 concurrency 設為 1。
"""

import asyncio
import contextlib
import signal
from pathlib import Path

import redis.asyncio as redis
from bullmq import Queue, Worker
from pydantic import ValidationError

from ..config import Settings
from ..worker import maintain_heartbeat
from .errors import AIServiceError
from .schemas import ChunkRequest, StyleProfileRequest, SummaryRequest, TopicSpanRequest
from .service import ReplyAIService, build_service

QUEUE_NAME = "ai-jobs"
RESULTS_QUEUE = "ai-results"
HEARTBEAT_FILE = Path("/tmp/ai-reply-worker-heartbeat")

# job 名稱 →（請求格式, 服務方法名稱, 用來當結果 key 的欄位）。規格第 8 節。
JOBS = {
    "chunk-embed": (ChunkRequest, "chunks", "conversationId"),
    "topic-spans": (TopicSpanRequest, "topic_spans", "conversationId"),
    "summarize": (SummaryRequest, "summarize", "conversationId"),
    "build-style": (StyleProfileRequest, "style_profile", "userId"),
}

# 結果 job 的設定：NestJS 處理失敗時由 BullMQ 以指數退避重試，成功後自動刪除以保持 Redis 乾淨。
RESULT_JOB_OPTIONS = {"attempts": 5, "backoff": {"type": "exponential", "delay": 2000}, "removeOnComplete": True}


def make_processor(service: ReplyAIService, results: Queue):
    """建立 BullMQ 的處理函式。

    流程：依 job 名稱找到對應的請求格式與服務方法 → 驗證 job 資料 → 執行 → 把結果放進結果佇列。
    錯誤一律轉成固定代碼（不含 job 內容），讓 BullMQ 的失敗紀錄不會留下聊天內容。
    job 本身的回傳值只放「結果在哪裡」，避免同一份大資料在 Redis 存兩份。
    """

    async def process(job, _token):
        """處理一個 job：驗證 → 呼叫服務 → 結果放進結果佇列；job 回傳值只標示結果的去處。"""
        spec = JOBS.get(job.name)
        if spec is None:
            raise ValueError("UNSUPPORTED_JOB")
        request_model, method_name, key_field = spec
        try:
            request = request_model.model_validate(job.data)
        except ValidationError:
            raise ValueError("INVALID_JOB_INPUT") from None
        try:
            result = await getattr(service, method_name)(request)
        except AIServiceError as error:
            raise RuntimeError(error.code) from None
        await results.add(
            job.name,
            {
                "sourceJobId": job.id,
                "name": job.name,
                "key": getattr(request, key_field),
                "result": result.model_dump(mode="json"),
            },
            RESULT_JOB_OPTIONS,
        )
        return {"ok": True, "resultQueue": RESULTS_QUEUE}

    return process


async def main():
    """啟動 worker：連 Redis、開始消費 ai-jobs、維持心跳檔，收到 SIGTERM／SIGINT 時優雅關閉。

    心跳機制沿用 app.worker 的 maintain_heartbeat：Redis 連得上且 worker 正在跑時才更新心跳檔，
    健康檢查（worker_health.py）看心跳檔是否在 60 秒內更新過。
    """
    settings = Settings.from_env()
    HEARTBEAT_FILE.unlink(missing_ok=True)
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stopped.set)
    client = redis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=3, socket_timeout=3)
    results = Queue(RESULTS_QUEUE, {"connection": settings.redis_url})
    worker = Worker(
        QUEUE_NAME,
        make_processor(build_service(settings), results),
        {"connection": settings.redis_url, "concurrency": 1, "autorun": False},
    )
    # 只記錄固定代碼，避免第三方例外把聊天內容或連線憑證寫進日誌。
    worker.on("error", lambda *_args: print("AI_REPLY_WORKER_QUEUE_ERROR", flush=True))
    run_task = asyncio.create_task(worker.run())
    heartbeat = asyncio.create_task(maintain_heartbeat(client, worker, stopped, HEARTBEAT_FILE))
    stop_task = asyncio.create_task(stopped.wait())
    try:
        done, _ = await asyncio.wait({run_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        if run_task in done and not stopped.is_set():
            raise RuntimeError("AI_REPLY_WORKER_STOPPED") from None
    finally:
        stopped.set()
        heartbeat.cancel()
        stop_task.cancel()
        HEARTBEAT_FILE.unlink(missing_ok=True)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(worker.close(), timeout=15)
        run_task.cancel()
        await asyncio.gather(run_task, heartbeat, stop_task, return_exceptions=True)
        await results.close()
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
