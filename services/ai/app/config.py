import os
from dataclasses import dataclass
from urllib.parse import urlsplit

# AI 推薦回覆的預設模型鏈；依序嘗試，前一個失敗（錯誤、額度用完或超過單一模型逾時）才換下一個。
# 2026-09-23 實測：Ollama Cloud gemma4:31b 約 2 秒；gemini-3.8-flash 當時 503（需求量大），所以 Ollama 在前、Gemini 當備用。
DEFAULT_REPLY_MODELS = ("ollama:gemma4:31b", "gemini-3.8-flash")
# 背景萃取（摘要、風格卡）預設交給 Ollama Cloud 的 gemma4:31b（2026-09-23 實測：本機 12b 在這台 Mac
# 記憶體不足、逾時失敗；雲端 2.5 秒完成）。OLLAMA_BASE_URL 仍需設定成 https://ollama.com/v1 並提供 key。
DEFAULT_EXTRACTION_MODELS = ("ollama:gemma4:31b",)


def _model_list(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    """把「逗號分隔的模型清單」環境變數轉成 tuple。

    例如 `gemini-3.8-flash, ollama:gemma4:12b` → `("gemini-3.8-flash", "ollama:gemma4:12b")`。
    沒設定或全是空白時回傳預設值；設成單一個 `-` 代表刻意停用（回傳空 tuple）。
    """
    if value is None or not value.strip():
        return default
    if value.strip() == "-":
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _positive_float(value: str | None, default: float) -> float:
    """讀取正數設定；格式錯誤或不是正數時退回預設值，避免一個打錯的環境變數讓服務起不來。"""
    try:
        number = float(value) if value is not None else default
    except ValueError:
        return default
    return number if number > 0 else default


def _positive_int(value: str | None, default: int) -> int:
    """讀取正整數設定；規則同 `_positive_float`。"""
    try:
        number = int(value) if value is not None else default
    except ValueError:
        return default
    return number if number > 0 else default


@dataclass(frozen=True)
class Settings:
    """AI 服務的所有設定（不可變）；正式執行用 from_env 讀環境變數，測試直接建構並只填需要的欄位。"""

    internal_token: str = ""
    provider_url: str = ""
    provider_token: str = ""
    provider_timeout_seconds: float = 10.0
    redis_url: str = "redis://localhost:6379/0"
    # ---- AI 推薦回覆（詳見 docs/ai/REPLY-SUGGESTIONS-SPEC.md 第 10 節） ----
    gemini_api_key: str = ""
    reply_models: tuple[str, ...] = DEFAULT_REPLY_MODELS
    reply_thinking_level: str = "LOW"
    extraction_models: tuple[str, ...] = DEFAULT_EXTRACTION_MODELS
    ollama_base_url: str = ""
    ollama_api_key: str = ""
    embedding_model: str = "gemini-embedding-2"
    embedding_dimensions: int = 768
    llm_timeout_seconds: float = 25.0
    reply_model_timeout_seconds: float = 12.0
    extraction_timeout_seconds: float = 180.0

    @classmethod
    def from_env(cls) -> "Settings":
        """從環境變數建立設定；每個變數的意義與預設值見 docs/ai/REPLY-SUGGESTIONS-SPEC.md 第 10 節。"""
        return cls(
            internal_token=os.getenv("AI_INTERNAL_TOKEN", ""),
            provider_url=os.getenv("AI_VERIFICATION_PROVIDER_URL", "").strip(),
            provider_token=os.getenv("AI_VERIFICATION_PROVIDER_TOKEN", ""),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            # Google 官方範例用 GEMINI_API_KEY；Pydantic AI 預設讀 GOOGLE_API_KEY，兩個都接受。
            gemini_api_key=(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip(),
            reply_models=_model_list(os.getenv("AI_REPLY_MODELS"), DEFAULT_REPLY_MODELS),
            reply_thinking_level=(os.getenv("AI_REPLY_THINKING_LEVEL") or "LOW").strip().upper(),
            extraction_models=_model_list(os.getenv("AI_EXTRACTION_MODELS"), DEFAULT_EXTRACTION_MODELS),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "").strip(),
            # 只有 Ollama Cloud（https://ollama.com/v1）需要；本機 Ollama 不用 key。
            ollama_api_key=os.getenv("OLLAMA_API_KEY", "").strip(),
            embedding_model=(os.getenv("AI_EMBEDDING_MODEL") or "gemini-embedding-2").strip(),
            embedding_dimensions=_positive_int(os.getenv("AI_EMBEDDING_DIMENSIONS"), 768),
            llm_timeout_seconds=_positive_float(os.getenv("AI_LLM_TIMEOUT_SECONDS"), 25.0),
            # 備援鏈裡「每個模型」各自的逾時；要比總逾時短，慢的模型才有機會讓給下一個。
            reply_model_timeout_seconds=_positive_float(os.getenv("AI_REPLY_MODEL_TIMEOUT_SECONDS"), 12.0),
            extraction_timeout_seconds=_positive_float(os.getenv("AI_EXTRACTION_TIMEOUT_SECONDS"), 180.0),
        )

    @property
    def provider_configured(self) -> bool:
        """真人驗證 provider 是否設定完整：必須是 http(s) 網址、不含帳密與片段，而且有 token。"""
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

    @property
    def gemini_configured(self) -> bool:
        """有沒有 Gemini API key；沒有的話，所有 `gemini-*` 模型與向量化都視為未設定。"""
        return bool(self.gemini_api_key)

    @property
    def ollama_is_cloud(self) -> bool:
        """OLLAMA_BASE_URL 是否指向 Ollama Cloud（ollama.com），而不是本機或自架的 Ollama。"""
        try:
            hostname = urlsplit(self.ollama_base_url).hostname or ""
        except ValueError:
            return False
        return hostname == "ollama.com" or hostname.endswith(".ollama.com")

    @property
    def ollama_configured(self) -> bool:
        """Ollama 是否可用：網址必須是合法的 http(s)；Ollama Cloud 另外必須有 OLLAMA_API_KEY。"""
        try:
            parsed = urlsplit(self.ollama_base_url)
        except ValueError:
            return False
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        return bool(self.ollama_api_key) if self.ollama_is_cloud else True
