import sys
import time

from .worker import HEARTBEAT_FILE


def healthy() -> bool:
    try:
        age = time.time() - float(HEARTBEAT_FILE.read_text(encoding="ascii"))
        return 0 <= age < 60
    except (OSError, ValueError):
        return False


if __name__ == "__main__":
    sys.exit(0 if healthy() else 1)
