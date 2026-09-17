import os
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    internal_token: str = ""
    provider_url: str = ""
    provider_token: str = ""
    provider_timeout_seconds: float = 10.0
    redis_url: str = "redis://localhost:6379/0"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            internal_token=os.getenv("AI_INTERNAL_TOKEN", ""),
            provider_url=os.getenv("AI_VERIFICATION_PROVIDER_URL", "").strip(),
            provider_token=os.getenv("AI_VERIFICATION_PROVIDER_TOKEN", ""),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        )

    @property
    def provider_configured(self) -> bool:
        try:
            parsed = urlsplit(self.provider_url)
        except ValueError:
            return False
        return bool(
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and not parsed.username
            and not parsed.password
            and not parsed.fragment
            and self.provider_token
        )
