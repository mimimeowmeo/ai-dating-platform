"""建立語言模型與執行 agent。

模型用「規格字串」描述，方便用環境變數設定：
- `gemini-3.8-flash`、`google:gemini-3.8-flash` → Google Gemini（需要 GEMINI_API_KEY）。
- `ollama:gemma4:12b` → Ollama（需要 OLLAMA_BASE_URL）。可以是本機／自架的 Ollama，
  也可以是 Ollama Cloud（OLLAMA_BASE_URL=https://ollama.com/v1，另外需要 OLLAMA_API_KEY）。
多個規格組成「備援鏈」：Pydantic AI 的 FallbackModel 會依序嘗試，前一個丟出 API 錯誤
（例如 429 額度用完、503 需求量大）才換下一個。沒設定的模型會被略過，全部都沒設定時回傳 None。
鏈裡的每個模型都包一層 TimeoutModel：單一模型太慢（而不是直接回錯誤）時也會被當成失敗，
讓下一個模型接手，不會把整條鏈的總逾時耗光。
"""

import asyncio
from typing import Literal

from pydantic_ai import Agent, NativeOutput, ToolOutput
from pydantic_ai.exceptions import AgentRunError, FallbackExceptionGroup, ModelAPIError, UnexpectedModelBehavior
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.settings import ModelSettings

from ..config import Settings
from .errors import AIServiceError
from .schemas import UsageInfo

Purpose = Literal["reply", "extraction"]

# 產生推薦要有一點變化，溫度高一些；萃取與摘要要穩定，溫度低。
TEMPERATURE = {"reply": 0.8, "extraction": 0.2}


def normalize_ollama_base_url(url: str) -> str:
    """把 Ollama 網址整理成 OpenAI 相容端點的格式（結尾必須是 /v1）。

    例：`http://localhost:11434` → `http://localhost:11434/v1`；已經有 /v1 的原樣保留。
    Pydantic AI 的 OllamaProvider 需要 /v1 結尾的網址。
    """
    base = url.strip().rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _google_settings(model_name: str, settings: Settings, purpose: Purpose) -> GoogleModelSettings:
    """組出 Gemini 模型的設定：溫度，以及（適用時）思考程度。

    Gemini 3.x Flash 的思考不能關，最低是 LOW（3.8 Flash）；產生推薦不需要長推理，
    設低可以縮短等待。Flash-Lite 的思考參數在官方文件沒有列出，保守起見不設定，交給模型預設。
    """
    options: GoogleModelSettings = {"temperature": TEMPERATURE[purpose]}
    if model_name.startswith("gemini-3") and "lite" not in model_name and settings.reply_thinking_level:
        options["google_thinking_config"] = {"thinking_level": settings.reply_thinking_level}
    return options


def build_model(spec: str, settings: Settings, purpose: Purpose) -> Model | None:
    """把一個模型規格字串變成 Pydantic AI 的模型物件；缺少對應設定時回傳 None。"""
    spec = spec.strip()
    if spec.startswith("ollama:"):
        if not settings.ollama_configured:
            return None
        # 本機 Ollama 不需要 key（傳 None 讓 provider 用預設值）；Ollama Cloud 用 OLLAMA_API_KEY 認證。
        provider = OllamaProvider(
            base_url=normalize_ollama_base_url(settings.ollama_base_url),
            api_key=settings.ollama_api_key or None,
        )
        return OllamaModel(
            spec.removeprefix("ollama:"),
            provider=provider,
            settings=ModelSettings(temperature=TEMPERATURE[purpose]),
        )
    name = spec.removeprefix("google:")
    if not name.startswith("gemini") or not settings.gemini_configured:
        return None
    provider = GoogleProvider(api_key=settings.gemini_api_key)
    return GoogleModel(name, provider=provider, settings=_google_settings(name, settings, purpose))


def structured_output(output_cls: type, specs: tuple[str, ...], settings: Settings):
    """決定要用哪種方式請模型回傳結構化結果。

    - Gemini 與本機／自架 Ollama：用 NativeOutput，模型會被強制照 JSON schema 輸出。
    - 模型鏈裡有 Ollama Cloud：改用 ToolOutput（用工具呼叫回傳結果）。Pydantic AI 官方文件說明，
      Ollama Cloud 會接受 JSON schema 參數但不會真的強制套用，用 NativeOutput 可能拿到不合格式的輸出。
    兩種方式最後都會經過 Pydantic 驗證，不合格一樣會觸發重試。
    """
    uses_cloud = settings.ollama_is_cloud and any(spec.strip().startswith("ollama:") for spec in specs)
    return ToolOutput(output_cls) if uses_cloud else NativeOutput(output_cls)


class TimeoutModel(WrapperModel):
    """替單一模型加上逾時：每次請求超過 timeout_seconds 就中止，並丟出 ModelAPIError。

    為什麼要轉成 ModelAPIError：FallbackModel 預設只在收到 ModelAPIError 時才換下一個模型。
    如果讓逾時直接變成 TimeoutError，外層的總逾時會先觸發，整條鏈直接失敗，備用模型根本沒機會上場。
    其他屬性（model_name、profile、settings…）都由 WrapperModel 轉給被包住的模型。
    """

    def __init__(self, wrapped: Model, timeout_seconds: float):
        """包住一個模型；timeout_seconds 是這個模型每次請求最多能花的秒數。"""
        super().__init__(wrapped)
        self.timeout_seconds = timeout_seconds

    async def request(self, messages, model_settings, model_request_parameters):
        """照常轉送請求給被包住的模型，但超過時間就放棄，改丟出可以觸發備援的 ModelAPIError。"""
        try:
            async with asyncio.timeout(self.timeout_seconds):
                return await super().request(messages, model_settings, model_request_parameters)
        except TimeoutError:
            raise ModelAPIError(self.model_name, f"timed out after {self.timeout_seconds:g}s") from None


def per_model_timeout(settings: Settings, purpose: Purpose) -> float:
    """每個模型各自的逾時：產生推薦用 AI_REPLY_MODEL_TIMEOUT_SECONDS，背景萃取用 AI_EXTRACTION_TIMEOUT_SECONDS。"""
    return settings.reply_model_timeout_seconds if purpose == "reply" else settings.extraction_timeout_seconds


def build_chain(specs: tuple[str, ...], settings: Settings, purpose: Purpose) -> Model | None:
    """依序建立多個模型並組成備援鏈；每個模型都包上各自的逾時。

    只有一個可用時直接回傳它（仍然有逾時），全部不可用回傳 None。
    """
    timeout = per_model_timeout(settings, purpose)
    models = [
        TimeoutModel(model, timeout)
        for model in (build_model(spec, settings, purpose) for spec in specs)
        if model is not None
    ]
    if not models:
        return None
    if len(models) == 1:
        return models[0]
    return FallbackModel(models[0], *models[1:])


async def run_agent(
    agent: Agent,
    prompt: str,
    model: Model | None,
    timeout_seconds: float,
    not_configured_code: str,
    unavailable_code: str,
):
    """執行一次 agent，並把各種模型錯誤轉成固定代碼的 AIServiceError。

    - model 為 None（完全沒設定模型）→ not_configured_code。
    - API 錯誤（含 429）、備援鏈全部失敗、輸出驗證重試用完、逾時 → unavailable_code。
    轉換時不保留原始例外訊息，因為裡面可能夾帶 prompt 片段（也就是使用者的聊天內容）。
    """
    if model is None:
        raise AIServiceError(not_configured_code)
    try:
        async with asyncio.timeout(timeout_seconds):
            return await agent.run(prompt, model=model)
    except (ModelAPIError, FallbackExceptionGroup, UnexpectedModelBehavior, AgentRunError, TimeoutError):
        raise AIServiceError(unavailable_code) from None


def usage_info(result) -> UsageInfo:
    """從 agent 結果取出 token 用量；Pydantic AI 2.x 的 usage 是屬性（不是方法）。"""
    usage = result.usage
    return UsageInfo(
        inputTokens=int(getattr(usage, "input_tokens", 0) or 0),
        outputTokens=int(getattr(usage, "output_tokens", 0) or 0),
        requests=int(getattr(usage, "requests", 0) or 0),
    )


def model_name_of(result) -> str | None:
    """取出實際回應的模型名稱；用備援鏈時可以知道最後是哪一個模型成功。"""
    response = getattr(result, "response", None)
    return getattr(response, "model_name", None)
