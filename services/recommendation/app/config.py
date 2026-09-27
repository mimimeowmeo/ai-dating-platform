"""推薦 worker 的設定，全部來自環境變數。"""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    redis_url: str = "redis://localhost:6379"
    models_dir: Path = Path("/app/models")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            redis_url=os.getenv("REDIS_URL") or cls.redis_url,
            models_dir=Path(os.getenv("REC_MODELS_DIR") or cls.models_dir),
        )
