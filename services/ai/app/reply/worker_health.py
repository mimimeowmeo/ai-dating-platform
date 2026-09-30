"""推薦回覆 worker 的健康檢查：心跳檔在 60 秒內更新過才算健康（給 Docker HEALTHCHECK 使用）。"""

import sys
import time
from pathlib import Path

# 路徑定義在這裡、由 worker.py 反過來 import：import worker.py 會載入 pydantic_ai、openai 等套件，
# 機器忙時（例如部署重建映像）健康檢查會超過 5 秒的時限而被判定不健康。
HEARTBEAT_FILE = Path("/tmp/ai-reply-worker-heartbeat")


def healthy() -> bool:
    """讀取心跳檔的時間戳記；檔案不存在、格式錯誤或超過 60 秒沒更新都算不健康。"""
    try:
        age = time.time() - float(HEARTBEAT_FILE.read_text(encoding="ascii"))
        return 0 <= age < 60
    except (OSError, ValueError):
        return False


if __name__ == "__main__":
    sys.exit(0 if healthy() else 1)
