"""只消費真人驗證工作；不提前啟用推薦、向量或聊天分析。

這是真人驗證的 BullMQ 佇列 worker（背景工作程式）。BullMQ 是以 Redis 為基礎的工作佇列：
生產者把「job」（工作，包含名稱與資料）放進佇列，worker 從佇列取出 job、處理完再把結果存回 Redis。

- 佇列名稱：`ai-verification`；job 名稱：`verify`；job 的 data 格式和 HTTP 路由
  `POST /internal/ai/face/verify` 的請求（VerificationRequest）完全相同，處理邏輯也共用 VerificationService。
- 執行方式：`python -m app.worker`；健康檢查：`python -m app.worker_health`（讀本檔寫出的心跳檔）。
- 目前產品流程走同步 HTTP：NestJS（apps/api/src/profiles.ts）直接呼叫 `POST /internal/ai/face/verify`，
  專案裡沒有程式把 job 放進 `ai-verification` 佇列；docker-compose.yml 的 ai-worker 容器跑的是
  AI 推薦回覆的 worker（`python -m app.reply.worker`），不是這支。
- 這支只處理 `verify` 這一種 job；推薦回覆、向量、聊天分析等背景工作由 app/reply/worker.py 另外負責。
- 本檔的 `maintain_heartbeat` 也被 app/reply/worker.py import 重複使用。

隱私設計（這支 worker 最重要的原則）：
1. 取到 job 後立刻把 Redis 裡的 job data 覆寫成 {"redacted": True}，自拍等影像不留在 Redis。
2. 失敗時只丟出固定錯誤碼，Pydantic 例外裡的輸入內容不會進入 BullMQ 的失敗紀錄。
3. 回傳值（BullMQ 會存進 Redis）不含 livenessScore／faceMatchScore 這類生物特徵分數。
4. 錯誤日誌只印固定代碼，不印第三方例外的內容。
"""

# asyncio：Python 內建的非同步（async/await）執行框架，相當於 JavaScript 的 event loop 與 Promise。
import asyncio
# contextlib：這裡用它的 suppress()，在 with 區塊裡「忽略指定的例外」。
import contextlib
# signal：作業系統訊號（SIGTERM、SIGINT 等）的常數；用來在容器停止或按 Ctrl+C 時優雅結束。
import signal
# time：這裡用 time.time() 取得目前的 Unix 時間戳（秒），寫進心跳檔。
import time
# Path：以物件方式操作檔案路徑（讀寫、刪除檔案）。
from pathlib import Path

# redis.asyncio：redis-py 的非同步版本客戶端；這裡只拿來 ping Redis、確認連線還活著（心跳用）。
import redis.asyncio as redis
# Worker：BullMQ 的 Python 版 worker，負責從佇列取 job 並呼叫我們提供的處理函式。
from bullmq import Worker
# ValidationError：Pydantic 驗證資料失敗時丟出的例外（例外訊息會包含輸入值，所以不能外洩）。
from pydantic import ValidationError
# RedisError：redis-py 所有錯誤的基底類別（包含連線失敗 ConnectionError 等）。
from redis.exceptions import RedisError

# Settings：從環境變數讀取的服務設定（redis_url、provider 網址與 token 等）。
from .config import Settings
# VerificationRequest：真人驗證請求的 Pydantic 模型；job data 要先通過它的驗證。
from .schemas import VerificationRequest
# InvalidImage：影像不合格的例外；VerificationService：實際做驗證（驗影像 → 呼叫 provider）的服務。
from .verification import InvalidImage, VerificationService

# 這個 worker 監聽的 BullMQ 佇列名稱（生產者要把 job 放進同名佇列）。
QUEUE_NAME = "ai-verification"
# 唯一接受的 job 名稱；其他名稱一律拒絕。
JOB_NAME = "verify"
# 心跳檔的位置：worker 健康時定期把目前時間寫進去，worker_health.py 讀它判斷 worker 是否健康。
# 放在 /tmp 是因為容器以非 root 使用者（ai）執行，/tmp 一定可寫。
HEARTBEAT_FILE = Path("/tmp/ai-worker-heartbeat")


def make_processor(service: VerificationService):
    """建立交給 BullMQ Worker 的 job 處理函式（processor）。

    參數：
        service：用來執行驗證的 VerificationService；和 HTTP 路由用的是同一個類別，所以兩條路的判定邏輯一致。
    回傳：
        async 函式 `process(job, _token)`，BullMQ 每取到一個 job 就呼叫一次。
    設計理由：用「回傳內部函式」（closure，閉包）的方式把 service 綁進 process，
        測試時就能傳入自己建立的 service，不用改全域狀態。
    """

    async def process(job, _token):
        """處理一個真人驗證 job。

        參數：
            job：BullMQ 的 Job 物件；會用到 job.name（job 名稱）、job.data（job 資料 dict）、
                job.updateData()（覆寫 Redis 裡的 job 資料）。
            _token：BullMQ 的鎖定 token（lock token），這裡用不到，名稱前的底線表示刻意不使用。
        回傳：
            驗證結果的 dict（status、reasonCode、modelName、modelVersion），BullMQ 會把它存成 job 的 returnvalue。
            刻意排除 livenessScore 與 faceMatchScore。
        可能丟出：
            ValueError("UNSUPPORTED_JOB")：job 名稱不是 verify。
            ValueError("INVALID_VERIFICATION_INPUT")：job 資料格式不符，或影像不合格。
            丟出例外時 BullMQ 會把 job 標成 failed，並把例外訊息（failedReason）與錯誤追蹤（stacktrace）存進 Redis，
            所以訊息只能是固定代碼。
        """
        # 先把 job 資料留一份在記憶體（區域變數），因為下一行就會覆寫 Redis 與 job 物件上的資料。
        data = job.data
        # 拿到工作後立即移除 Redis job hash 中的自拍；推論只使用記憶體副本。
        # （updateData 會把 Redis 裡這個 job 的 data 整個換成 {"redacted": True}；
        # 這一步刻意放在檢查 job 名稱之前，連不支援的 job 也不會把資料留在 Redis。）
        await job.updateData({"redacted": True})
        # 只處理名稱為 verify 的 job；放錯佇列或名稱打錯的 job 直接失敗。
        if job.name != JOB_NAME:
            # 固定代碼，不包含 job 內容。
            raise ValueError("UNSUPPORTED_JOB")
        # 下面兩步都可能因輸入不合格而丟例外，所以包在 try 裡統一轉成固定錯誤碼。
        try:
            # 用 Pydantic 模型驗證 job 資料（欄位、型別、長度等）；不合格會丟 ValidationError。
            request = VerificationRequest.model_validate(data)
            # 執行驗證：檢查影像 → 轉給 provider → 回傳 VerificationResult；影像不合格會丟 InvalidImage。
            result = await service.verify(request)
        # 兩種「輸入不合格」的例外都轉成同一個固定錯誤碼。
        except (ValidationError, InvalidImage):
            # Pydantic 的預設例外會包含輸入，不得進入 BullMQ 的失敗紀錄。
            # `from None`：讓新例外不顯示「由哪個例外引起」（設定 __suppress_context__）。BullMQ 失敗時除了 failedReason，
            # 還會用 traceback.format_exc() 把錯誤追蹤存進 Redis；加了 from None，追蹤裡就不會帶出原本含輸入內容的例外。
            raise ValueError("INVALID_VERIFICATION_INPUT") from None
        # 佇列結果只存決策與版本，不在 Redis 留下生物特徵分數。
        # model_dump 把 Pydantic 物件轉成 dict；exclude 指定要排除的欄位。
        return result.model_dump(exclude={"livenessScore", "faceMatchScore"})

    # 回傳處理函式本身（不是呼叫它），交給 BullMQ Worker 使用。
    return process


async def maintain_heartbeat(client, worker, stopped: asyncio.Event, path: Path = HEARTBEAT_FILE):
    """定期更新心跳檔，讓 Docker 健康檢查知道 worker 是否真的能工作。

    參數：
        client：redis.asyncio 客戶端（或任何有 async `ping()` 的物件），用來確認 Redis 連得上。
        worker：BullMQ Worker（或任何有 `running`、`closing` 屬性的物件），用來確認 worker 正在執行、沒有在關閉中。
        stopped：asyncio.Event，被 set 時代表程序要結束，迴圈就停止。
        path：心跳檔路徑，預設 HEARTBEAT_FILE；app/reply/worker.py 會傳入自己的路徑重複使用這個函式。
    回傳：None（直到 stopped 被 set 才結束）。
    不會丟出 Redis 或檔案錯誤：這些錯誤都被捕捉，並以「刪掉心跳檔」表示不健康。

    規則：只有「Redis ping 成功」而且「worker 正在執行、沒有在關閉」時才寫入目前時間；
    否則刪掉心跳檔。每輪之間最多等 20 秒；worker_health.py 認為 60 秒內的心跳才算健康，
    所以偶爾一輪延遲不會馬上被判定為不健康。
    設計理由：只檢查「程序還活著」不夠——Redis 斷線或 worker 迴圈已停止時，程序可能還在但完全不會處理 job。
    """
    # 只要還沒收到停止訊號，就一直循環。
    while not stopped.is_set():
        # 這一輪的檢查包在 try 裡，任何連線或檔案錯誤都當作不健康處理。
        try:
            # 最多等 5 秒；超過就丟 TimeoutError，避免 ping 卡住讓心跳迴圈停在這裡。
            async with asyncio.timeout(5):
                # 對 Redis 送 PING；連不上會丟 RedisError（或 OSError）。
                await client.ping()
            # Redis 正常，再確認 BullMQ worker 的狀態：running＝取 job 的主迴圈正在跑；closing＝正在關閉中。
            if worker.running and not worker.closing:
                # 健康：把目前的 Unix 時間（秒，含小數）以 ASCII 文字寫進心跳檔（覆寫舊內容）。
                path.write_text(str(time.time()), encoding="ascii")
            # worker 沒在跑或正在關閉。
            else:
                # 不健康：刪除心跳檔；missing_ok=True 表示檔案本來就不存在也不報錯。
                path.unlink(missing_ok=True)
        # Redis 錯誤、ping 逾時，或讀寫檔案失敗（OSError）。
        except (RedisError, TimeoutError, OSError):
            # 都視為不健康：刪除心跳檔。
            path.unlink(missing_ok=True)
        # 等待下一輪：wait_for 最多等 20 秒；若這期間 stopped 被 set，會立刻返回，迴圈條件不成立就結束。
        # 等滿 20 秒時 wait_for 會丟 TimeoutError，這是正常情況，用 suppress 忽略它。
        with contextlib.suppress(TimeoutError):
            # stopped.wait() 會一直等到 Event 被 set。
            await asyncio.wait_for(stopped.wait(), timeout=20)


async def main():
    """worker 程序的主函式：建立連線與 Worker，一直執行到收到停止訊號或 worker 意外停止。

    流程：
        1. 讀設定、刪掉舊心跳檔、註冊 SIGTERM／SIGINT 處理。
        2. 建立 Redis 客戶端（心跳用）與 BullMQ Worker（concurrency 2，一次最多處理 2 個 job）。
        3. 同時啟動三個背景 task：worker 主迴圈、心跳迴圈、等待停止訊號。
        4. 等「worker 主迴圈結束」或「收到停止訊號」其中一個先發生。
        5. 無論如何都在 finally 裡依序收尾：停止心跳、刪心跳檔、關閉 worker（最多等 15 秒）、關 Redis 連線。
    回傳：None。
    可能丟出：RuntimeError("AI_WORKER_STOPPED")——worker 主迴圈在沒有收到停止訊號時就結束了（異常），
        讓程序以錯誤結束，交給容器的重啟政策處理。
    """
    # 從環境變數讀設定（REDIS_URL、provider 網址與 token 等）。
    settings = Settings.from_env()
    # 刪掉上一次執行留下的心跳檔；容器重啟時 /tmp 可能還在，舊檔案會讓健康檢查在 worker 就緒前就誤判為健康。
    HEARTBEAT_FILE.unlink(missing_ok=True)
    # 「該停止了」的旗標；收到訊號或結束時會被 set，所有迴圈都看它決定是否結束。
    stopped = asyncio.Event()
    # 取得目前正在執行的 event loop，下面要在它上面註冊訊號處理器。
    loop = asyncio.get_running_loop()
    # SIGTERM：docker stop／容器停止時送出；SIGINT：在終端機按 Ctrl+C 時送出。
    for sig in (signal.SIGTERM, signal.SIGINT):
        # 收到訊號時只 set stopped 旗標，讓 main 走正常收尾流程，而不是被直接中斷
        # （loop.add_signal_handler 只在 Unix 類系統可用，容器環境符合）。
        loop.add_signal_handler(sig, stopped.set)
    # 建立 Redis 客戶端，只用來 ping 做心跳：
    # decode_responses=True 讓回應自動解碼成字串；連線與讀寫逾時都設 3 秒，避免 Redis 沒回應時卡住。
    client = redis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=3, socket_timeout=3)
    # 建立 BullMQ Worker（它會自己建立連到 Redis 的連線）。
    worker = Worker(
        # 第一個參數：佇列名稱；第二個參數：處理函式（用設定建立 VerificationService 後綁進 processor）。
        QUEUE_NAME, make_processor(VerificationService(settings)),
        # 選項：connection＝Redis 連線網址；concurrency＝同時最多處理 2 個 job；
        # autorun=False＝建立時不要自動開始取 job，下面改用 create_task 自己啟動，才能拿到 task 監看它是否意外結束。
        {"connection": settings.redis_url, "concurrency": 2, "autorun": False},
    )
    # 只記錄固定代碼，避免第三方例外把影像或連線憑證寫入日誌。
    # worker.on("error", 回呼) 註冊錯誤事件；回呼收到的參數（例外、job）全部忽略（*_args），只印固定字串。
    # flush=True 讓輸出立刻寫出，不被緩衝，docker logs 才看得到。
    worker.on("error", lambda *_args: print("AI_WORKER_QUEUE_ERROR", flush=True))
    # 在背景啟動 worker 主迴圈（持續從佇列取 job 並處理）；create_task 會立刻返回一個 Task 物件。
    run_task = asyncio.create_task(worker.run())
    # 在背景啟動心跳迴圈，使用預設的 HEARTBEAT_FILE。
    heartbeat = asyncio.create_task(maintain_heartbeat(client, worker, stopped))
    # 把「等待停止訊號」也包成 task，才能和 run_task 一起交給 asyncio.wait 比誰先完成。
    stop_task = asyncio.create_task(stopped.wait())
    # try/finally：不論正常結束或發生例外，finally 裡的收尾一定會執行。
    try:
        # 等 run_task 或 stop_task 其中一個先完成（FIRST_COMPLETED）；done 是已完成的 task 集合，另一個回傳值用 _ 忽略。
        done, _ = await asyncio.wait({run_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        # worker 主迴圈先結束了，但並不是因為收到停止訊號 → 屬於異常狀況。
        if run_task in done and not stopped.is_set():
            # 丟出固定代碼讓程序以錯誤結束；此時沒有正在處理的例外，`from None` 只是明確表示不串接任何其他例外。
            raise RuntimeError("AI_WORKER_STOPPED") from None
    # 收尾（每一步都要執行，順序有意義）。
    finally:
        # 設定停止旗標，讓心跳迴圈在下一次檢查時結束（若是異常路徑進來，這裡才第一次 set）。
        stopped.set()
        # 取消心跳 task，不必等它最多 20 秒的等待結束。
        heartbeat.cancel()
        # 取消等待停止訊號的 task（若它已完成，cancel 不會有作用）。
        stop_task.cancel()
        # 立刻刪除心跳檔，讓健康檢查在關閉期間就回報不健康，不會把關閉中的 worker 當成健康。
        HEARTBEAT_FILE.unlink(missing_ok=True)
        # 關閉 worker：close() 會停止取新 job，並等正在處理的 job 完成；最多等 15 秒，逾時就放棄等待（忽略 TimeoutError）。
        with contextlib.suppress(TimeoutError):
            # wait_for 逾時時會取消 close() 並丟 TimeoutError（由外層 suppress 忽略）。
            await asyncio.wait_for(worker.close(), timeout=15)
        # 取消 worker 主迴圈 task（如果它還在跑）。
        run_task.cancel()
        # 等三個 task 都真正結束；return_exceptions=True 讓取消（CancelledError）或其他例外被收集成回傳值，而不是往外丟。
        await asyncio.gather(run_task, heartbeat, stop_task, return_exceptions=True)
        # 關閉心跳用的 Redis 連線，釋放資源。
        await client.aclose()


# 只有直接執行這個檔案（python -m app.worker）時才啟動；被 import（例如測試、worker_health.py）時不會執行。
if __name__ == "__main__":
    # asyncio.run 建立 event loop、執行 main() 直到結束，最後關閉 event loop。
    asyncio.run(main())
