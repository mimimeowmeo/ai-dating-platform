"""推薦回覆 worker 的健康檢查：心跳檔在 60 秒內更新過才算健康（給 Docker HEALTHCHECK 使用）。"""

import sys
import time

from .worker import HEARTBEAT_FILE


def healthy() -> bool:
    """讀取心跳檔的時間戳記；檔案不存在、格式錯誤或超過 60 秒沒更新都算不健康。"""
    try:
        age = time.time() - float(HEARTBEAT_FILE.read_text(encoding="ascii"))
        return 0 <= age < 60
    except (OSError, ValueError):
        return False


if __name__ == "__main__":
    sys.exit(0 if healthy() else 1)
