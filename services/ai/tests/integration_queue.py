"""REDIS_URL 指向測試 Redis 時執行；驗證真實 BullMQ consumer。

以唯一 queue 隔離，不清除既有 queue 或業務資料。

這支檔案是「整合測試腳本」，不是 unittest 測試：
- 檔名不是 test*.py，所以 `python -m unittest discover -s tests` 不會自動執行它。
- 需要一個真的 Redis，手動執行：
  `REDIS_URL=redis://127.0.0.1:6379/0 .venv/bin/python -m tests.integration_queue`（見 services/ai/README.md）。

背景：BullMQ 是用 Redis 當儲存的工作佇列（job queue）。「生產者」用 Queue.add 放入工作，
「消費者」Worker 從 Redis 取出工作、交給處理函式（processor）執行，再把回傳值存回 Redis。
app/worker.py 的 make_processor 就是真人驗證的處理函式（產品流程目前用同步 HTTP，這個 worker 是備用路徑）。

這裡驗證 make_processor 在真實 Redis + 真實 BullMQ 下的行為：
1. Worker 真的會消費工作並把結果存回 Redis。
2. 沒有配置模型（Settings() 沒有 provider_url）時，結果是 unavailable / MODEL_NOT_CONFIGURED。
3. 工作資料在 Redis 裡被換成 {"redacted": True}，自拍的 Base64 不會留在 Redis。
4. 存回 Redis 的結果不包含 faceMatchScore（生物特徵相關的分數不留在佇列）。

任何一個 assert 不成立會丟出 AssertionError，腳本以非 0 結束；全部通過才印出 PASS。
注意：Python 用 `-O` 參數執行時會略過所有 assert，所以這支腳本不要用 -O 執行。
"""

# asyncio：Python 內建的非同步執行框架（async／await），類似 JS 的 event loop 與 Promise。
import asyncio
# base64：把影像位元組編成 Base64 字串。
import base64
# io：提供 BytesIO（存在記憶體裡的檔案）。
import io
# os：讀取 REDIS_URL 環境變數。
import os
# uuid：產生隨機且幾乎不會重複的 UUID，用來當 queue 名稱與 requestId。
import uuid

# redis.asyncio：非同步版的 Redis 用戶端，這裡只用來做連線檢查（ping）。
import redis.asyncio as redis
# BullMQ 的 Python 版：Job（單一工作）、Queue（放工作的佇列）、Worker（取工作來執行的消費者）。
from bullmq import Job, Queue, Worker
# Pillow 的 Image：產生測試用圖片。
from PIL import Image

# Settings：AI 服務設定；這裡用預設值（沒有 provider），模擬「未配置模型」。
from app.config import Settings
# VerificationService：真正的驗證服務（驗影像、呼叫 provider）。
from app.verification import VerificationService
# make_processor：產生 BullMQ Worker 要用的處理函式（和正式 worker 用同一個）。
from app.worker import make_processor


async def main():
    """在一個唯一命名的測試 queue 上，跑一次真實的 BullMQ 生產／消費流程並檢查結果。

    參數：無（Redis 位址從 REDIS_URL 環境變數讀取，沒設定時用 redis://127.0.0.1:6379/0）。

    回傳：None；全部檢查通過時印出 PASS 訊息。

    可能丟出：
        redis 的連線例外（例如 ConnectionError）或 TimeoutError：Redis 連不上，或 ping 超過 5 秒。
        TimeoutError：工作 20 秒內沒有變成 completed（例如 worker 沒在跑，或工作失敗變成 failed）。
        AssertionError：結果或 Redis 中的工作資料不符合預期。

    設計理由：
        - 用 uuid 產生唯一的 queue 名稱，和其他測試或正式的 "ai-verification" queue 完全隔離。
        - finally 區塊一定會清理：關閉 worker、刪除這次的工作與這個測試 queue，不留垃圾在 Redis。
    """
    # 讀取 Redis 位址；沒設定時用本機預設埠 6379、第 0 號資料庫。
    redis_url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
    # 先做有界連線檢查，避免 BullMQ 的長時間重試把測試卡住。
    # 建立一個「探測用」的 Redis 用戶端：連線與每次指令最多各等 2 秒。
    probe = redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
    # try/finally：不論 ping 成功或失敗，都要關閉這個探測連線。
    try:
        # asyncio.timeout(5)：區塊內的所有等待加起來超過 5 秒就丟 TimeoutError（Python 3.11 以上才有）。
        async with asyncio.timeout(5):
            # 送 PING 指令確認 Redis 可用；連不上會丟例外，腳本直接失敗，不會卡在後面的 BullMQ。
            await probe.ping()
    # 無論成功與否都執行。
    finally:
        # 關閉探測用的連線，釋放資源。
        await probe.aclose()
    # 產生這次測試專用的 queue 名稱，例如 "ai-verification-test-3f2b..."，保證不會和既有 queue 撞名。
    queue_name = f"ai-verification-test-{uuid.uuid4()}"
    # 建立生產者端的 Queue 物件，用來放入工作；connection 指定 Redis 位址。
    queue = Queue(queue_name, {"connection": redis_url})
    # 建立消費者 Worker：監聽同一個 queue，收到工作就交給處理函式執行。
    # 沒有傳 autorun 選項時預設為 True，建立後就會自動開始取工作。
    worker = Worker(
        # 監聽的 queue 名稱，必須和上面的 Queue 相同。
        queue_name,
        # 處理函式：和正式 worker 相同的 make_processor；Settings() 沒有 provider，所以結果會是 MODEL_NOT_CONFIGURED。
        make_processor(VerificationService(Settings())),
        # Worker 的選項：同一個 Redis 位址。
        {"connection": redis_url},
    )
    # 建立記憶體中的檔案，準備放測試圖片。
    buffer = io.BytesIO()
    # 產生一張 128×128 的純黑 PNG（尺寸在 64～4096 px 的合法範圍內）。
    Image.new("RGB", (128, 128)).save(buffer, format="PNG")
    # 先設成 None：如果 queue.add 之前就出錯，finally 裡才知道不需要刪除工作。
    job = None
    # try/finally：不論測試成功或失敗，都要清理 worker、工作與測試 queue。
    try:
        # 放入一個名稱為 "verify" 的工作（worker.py 只接受這個名稱，其他名稱會丟 UNSUPPORTED_JOB）。
        job = await queue.add("verify", {
            # 圖片的 Base64 字串。
            "imageBase64": base64.b64encode(buffer.getvalue()).decode(),
            # 宣告格式與實際 PNG 一致；requestId 用隨機 UUID（只含英數字與連字號，符合格式限制）。
            "mimeType": "image/png", "requestId": str(uuid.uuid4()),
        # 工作選項：attempts=1 表示失敗不重試；removeOnFail=True 表示失敗的工作直接從 Redis 刪掉。
        }, {"attempts": 1, "removeOnFail": True})
        # 最多等 20 秒讓 worker 處理完；超過就丟 TimeoutError，測試失敗而不是永遠卡住。
        async with asyncio.timeout(20):
            # 反覆查詢工作狀態，直到變成 "completed"（完成）。
            # 注意：如果工作失敗（狀態是 "failed" 或已被刪除），這個迴圈不會提早結束，會等到 20 秒逾時。
            while await job.getState() != "completed":
                # 每次查詢之間暫停 0.05 秒，避免密集查詢 Redis。
                await asyncio.sleep(0.05)
        # 從 Redis 重新讀取這個工作，拿到 worker 寫回的最新 data 與 returnvalue（回傳值）。
        stored = await Job.fromId(queue, job.id)
        # 檢查 2：沒有配置模型，狀態必須是 unavailable；失敗時把整個回傳值印進錯誤訊息方便除錯。
        assert stored.returnvalue["status"] == "unavailable", stored.returnvalue
        # 原因代碼必須是 MODEL_NOT_CONFIGURED。
        assert stored.returnvalue["reasonCode"] == "MODEL_NOT_CONFIGURED"
        # 檢查 3：處理函式一拿到工作就呼叫 job.updateData({"redacted": True})，
        # 所以 Redis 裡的工作資料只剩這個標記，自拍影像已被移除。
        assert stored.data == {"redacted": True}, "Image remained in Redis job data"
        # 檢查 4：make_processor 回傳前排除了 livenessScore 與 faceMatchScore，存回 Redis 的結果不能有分數。
        assert "faceMatchScore" not in stored.returnvalue
        # 走到這裡代表全部檢查通過。
        print("PASS: 真實 Redis BullMQ 消費、無模型決策、影像去敏")
    # 無論成功與否都執行清理。
    finally:
        # 先關閉 worker，讓它停止從 queue 取新工作。
        await worker.close()
        # 如果工作已經建立，就把它從 Redis 刪掉。
        if job:
            # 刪除這個工作的資料。
            await job.remove()
        # 僅移除本次測試的 UUID queue，保留所有其他 queue。
        # obliterate 會刪除這個 queue 的所有內容；force=True 表示即使還有執行中的工作也強制刪除。
        await queue.obliterate(force=True)
        # 關閉 Queue 物件的 Redis 連線。
        await queue.close()


# 只有直接執行這個檔案（python -m tests.integration_queue）時才執行；被 import 時不會自動跑。
if __name__ == "__main__":
    # asyncio.run：建立 event loop、執行 async 的 main()，結束後關閉 event loop。
    asyncio.run(main())
