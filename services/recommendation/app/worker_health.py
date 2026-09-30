"""推薦 worker 的健康檢查：心跳檔在 60 秒內更新過就是健康（結束碼 0），否則 1。"""
import sys
import time
from pathlib import Path

# 路徑定義在這裡、由 worker.py 反過來 import：import worker.py 會載入外貌模型相關套件，
# 機器忙時健康檢查會超過 5 秒的時限而被判定不健康。
HEARTBEAT_FILE = Path("/tmp/rec-worker-heartbeat")


def healthy() -> bool:
    try:
        age = time.time() - float(HEARTBEAT_FILE.read_text(encoding="ascii"))
        return 0 <= age < 60
    except (OSError, ValueError):
        return False


if __name__ == "__main__":
    sys.exit(0 if healthy() else 1)
