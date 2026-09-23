"""真人驗證 worker（app/worker.py）的健康檢查程式。

執行方式：`python -m app.worker_health`。程序結束碼 0 代表健康、1 代表不健康；
Docker 的 HEALTHCHECK（或 docker-compose 的 healthcheck）就是依結束碼判斷容器狀態。

判斷方式：讀 worker 寫出的心跳檔（HEARTBEAT_FILE，內容是最後一次心跳的 Unix 時間戳），
心跳時間距離現在不到 60 秒才算健康。worker 只有在「Redis ping 成功且 worker 正在執行」時才會寫心跳，
大約每 20 秒一次，所以 60 秒的容忍度可以吸收偶爾一兩輪的延遲。

設計理由：健康檢查另開一個程序執行，不能直接問 worker 程序的狀態，所以用「檔案」當兩個程序之間的溝通方式；
這支程式不連 Redis、不做網路請求，執行快又不會被外部服務拖慢。

注意：目前 docker-compose.yml 的 ai-worker 跑的是 AI 推薦回覆的 worker，
健康檢查用的是 app/reply/worker_health.py（讀另一個心跳檔）；這支對應的是真人驗證 worker。
"""

# sys：這裡用 sys.exit() 設定程序的結束碼。
import sys
# time：這裡用 time.time() 取得目前的 Unix 時間戳（秒）。
import time

# 從 worker 模組取得心跳檔路徑，確保兩邊讀寫的是同一個檔案（路徑只在 worker.py 定義一次）。
from .worker import HEARTBEAT_FILE


def healthy() -> bool:
    """判斷真人驗證 worker 目前是否健康。

    回傳：
        True：心跳檔存在、內容是合法數字，而且心跳時間在「現在往前 60 秒內」。
        False：心跳檔不存在（worker 沒啟動、正在關閉，或 Redis 連不上時會被刪掉）、內容壞掉，
            或心跳太舊（worker 卡住）。
    不會丟出例外：讀檔與轉數字的錯誤都被捕捉並回傳 False。
    """
    # 讀檔或轉換數字都可能失敗，包在 try 裡。
    try:
        # 讀出心跳檔的文字（worker 以 ASCII 寫入的時間戳），轉成浮點數，再用現在時間減掉，得到心跳距今幾秒。
        age = time.time() - float(HEARTBEAT_FILE.read_text(encoding="ascii"))
        # 必須 0 ≤ age < 60：負數代表心跳時間在未來（時鐘異常或內容不可信），也視為不健康。
        return 0 <= age < 60
    # OSError：檔案不存在或無法讀取；ValueError：內容不是合法數字（或不是 ASCII 文字）。
    except (OSError, ValueError):
        # 任何讀取問題都當作不健康。
        return False


# 只有直接執行這個檔案（python -m app.worker_health）時才執行；被 import 時不執行。
if __name__ == "__main__":
    # 健康就以結束碼 0 結束，不健康就以 1 結束，讓 Docker 依結束碼判斷。
    sys.exit(0 if healthy() else 1)
