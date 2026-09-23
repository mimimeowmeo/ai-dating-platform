"""AI 私有服務的設定：從環境變數讀取，整理成一個不可變的 Settings 物件。

包含兩大類設定：
- 真人驗證：內部 token（NestJS 呼叫本服務時要帶的 X-Internal-Token）、人臉驗證 provider 的網址與 token、
  provider 逾時秒數。
- AI 推薦回覆：Gemini／Ollama 的金鑰與網址、模型備援鏈、各種逾時、向量模型
  （每個變數的意義見 docs/ai/REPLY-SUGGESTIONS-SPEC.md 第 10 節）。

設計理由：讀取時對格式錯誤的數字寬鬆處理（退回預設值），避免一個打錯的環境變數讓整個服務起不來；
但「是否可用」（例如 provider_configured、ollama_configured）則嚴格判斷，設定不完整就視為未設定。
"""

# os：讀取環境變數（os.getenv）。
import os
# dataclass：自動產生建構子等樣板程式碼的裝飾器，類似用一個 class 宣告「有型別的設定物件」。
from dataclasses import dataclass
# urlsplit：把網址拆成 scheme、hostname、username、password、fragment 等部分，用來檢查網址是否合法。
from urllib.parse import urlsplit

# AI 推薦回覆的預設模型鏈；依序嘗試，前一個失敗（錯誤、額度用完或超過單一模型逾時）才換下一個。
# 2026-09-23 實測：Ollama Cloud gemma4:31b 約 2 秒；gemini-3.8-flash 當時 503（需求量大），所以 Ollama 在前、Gemini 當備用。
# 格式：「ollama:」開頭的交給 Ollama（後面是 Ollama 的模型名稱），「gemini」開頭的交給 Google Gemini（見 reply/llm.py 的 build_model）。
DEFAULT_REPLY_MODELS = ("ollama:gemma4:31b", "gemini-3.8-flash")
# 背景萃取（摘要、風格卡）預設交給 Ollama Cloud 的 gemma4:31b（2026-09-23 實測：本機 12b 在這台 Mac
# 記憶體不足、逾時失敗；雲端 2.5 秒完成）。OLLAMA_BASE_URL 仍需設定成 https://ollama.com/v1 並提供 key。
# 注意 ("…",) 結尾的逗號：Python 只有一個元素的 tuple 必須這樣寫，否則括號只是一般的括號。
DEFAULT_EXTRACTION_MODELS = ("ollama:gemma4:31b",)


def _model_list(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    """把「逗號分隔的模型清單」環境變數轉成 tuple。

    例如 `gemini-3.8-flash, ollama:gemma4:12b` → `("gemini-3.8-flash", "ollama:gemma4:12b")`。
    沒設定或全是空白時回傳預設值；設成單一個 `-` 代表刻意停用（回傳空 tuple）。

    參數：
        value：環境變數的原始值；沒設定時是 None。
        default：沒設定時使用的預設模型鏈。

    回傳：模型名稱的 tuple（不可變的序列），每個名稱都去掉前後空白，空項目（例如 "a,,b" 中間那個）會被略過。

    可能丟出：無。

    設計理由：用 tuple 而不是 list，配合 frozen 的 Settings，設定建立後就不會被意外修改。
    """
    # 沒設定這個環境變數，或值只有空白：使用預設模型鏈。
    if value is None or not value.strip():
        # 回傳預設值。
        return default
    # 去掉前後空白後剛好是 "-"：使用者刻意停用這項功能。
    if value.strip() == "-":
        # 空 tuple 代表沒有任何模型可用。
        return ()
    # 以逗號切開，每一項去掉前後空白，並略過去掉空白後是空字串的項目；
    # tuple(... for ...) 是「產生器運算式」，效果類似 JS 的 value.split(",").map(trim).filter(Boolean)。
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _positive_float(value: str | None, default: float) -> float:
    """讀取正數設定；格式錯誤或不是正數時退回預設值，避免一個打錯的環境變數讓服務起不來。

    參數：
        value：環境變數的原始字串；沒設定時是 None。
        default：沒設定、格式錯誤或不是正數時使用的預設值。

    回傳：大於 0 的浮點數，或 default。

    可能丟出：無（ValueError 會在內部被捕捉）。
    """
    # 嘗試把字串轉成浮點數。
    try:
        # 有值就用 float() 轉換（例如 "12.5" → 12.5）；沒設定（None）就直接用預設值。
        number = float(value) if value is not None else default
    # 字串不是數字（例如 "abc"），float() 會丟出 ValueError。
    except ValueError:
        # 格式錯誤：退回預設值。
        return default
    # 大於 0 才採用；0、負數（以及和任何數比較都為 False 的 NaN）都退回預設值，因為逾時秒數必須是正數。
    return number if number > 0 else default


def _positive_int(value: str | None, default: int) -> int:
    """讀取正整數設定；規則同 `_positive_float`。

    參數：
        value：環境變數的原始字串；沒設定時是 None。
        default：沒設定、格式錯誤或不是正整數時使用的預設值。

    回傳：大於 0 的整數，或 default。

    可能丟出：無（ValueError 會在內部被捕捉）。
    """
    # 嘗試把字串轉成整數。
    try:
        # 有值就用 int() 轉換（例如 "768" → 768；"1.5" 這種小數字串會丟 ValueError）；沒設定就用預設值。
        number = int(value) if value is not None else default
    # 字串不是整數格式。
    except ValueError:
        # 格式錯誤：退回預設值。
        return default
    # 大於 0 才採用，否則退回預設值。
    return number if number > 0 else default


# frozen=True：建立後所有欄位都不能再修改（指派會丟出 FrozenInstanceError），避免執行中設定被意外改掉。
@dataclass(frozen=True)
class Settings:
    """AI 服務的所有設定（不可變）；正式執行用 from_env 讀環境變數，測試直接建構並只填需要的欄位。

    每個欄位的 `= 值` 是預設值；測試可以只指定需要的欄位，例如 `Settings(internal_token="x")`。
    """

    # NestJS 呼叫本服務 /internal/* 時要帶的 X-Internal-Token；空字串代表未設定，main.py 會讓內部 API 回 503。
    internal_token: str = ""
    # 人臉驗證 provider 的網址（例如 services/face 的 /verify）；空字串代表沒接 provider，驗證回 MODEL_NOT_CONFIGURED。
    provider_url: str = ""
    # 呼叫 provider 時放在 Authorization: Bearer 標頭的 token。
    provider_token: str = ""
    # 呼叫 provider 的逾時秒數（verification.py 同時用在 httpx 的單步逾時與整體逾時）。
    # 注意：from_env 沒有讀這個欄位，正式執行固定是 10 秒，只有直接建構 Settings 時才能改。
    provider_timeout_seconds: float = 10.0
    # Redis 連線網址；BullMQ 佇列 worker（驗證與推薦回覆）用它連 Redis。
    redis_url: str = "redis://localhost:6379/0"
    # ---- AI 推薦回覆（詳見 docs/ai/REPLY-SUGGESTIONS-SPEC.md 第 10 節） ----
    # Google AI Studio 的 API key；空字串代表 Gemini 模型與向量化都不可用。
    gemini_api_key: str = ""
    # 線上產生推薦回覆的模型備援鏈，依序嘗試。
    reply_models: tuple[str, ...] = DEFAULT_REPLY_MODELS
    # Gemini 3.x（非 Lite）模型的思考程度；reply/llm.py 只在這個值不是空字串時才送出思考設定。
    reply_thinking_level: str = "LOW"
    # 背景萃取（摘要、風格卡）的模型鏈。
    extraction_models: tuple[str, ...] = DEFAULT_EXTRACTION_MODELS
    # Ollama 的 API 網址（本機、自架或 Ollama Cloud）；空字串代表不用 Ollama。
    ollama_base_url: str = ""
    # Ollama Cloud 的 API key；本機或自架 Ollama 不需要。
    ollama_api_key: str = ""
    # 向量（embedding）模型名稱；所有向量必須用同一個模型產生，才能互相比較。
    embedding_model: str = "gemini-embedding-2"
    # 向量維度；pgvector 的 HNSW 索引對 vector 型別最多 2,000 維，所以用 768 而不是模型預設的 3,072。
    embedding_dimensions: int = 768
    # 線上產生推薦「整次請求」的總逾時秒數（含備援與重試），也是向量化每批的逾時。
    llm_timeout_seconds: float = 25.0
    # 備援鏈中「每個模型」各自的逾時秒數。
    reply_model_timeout_seconds: float = 12.0
    # 背景萃取每次模型呼叫的逾時秒數；背景工作不急，給得比較長。
    extraction_timeout_seconds: float = 180.0

    # classmethod：用類別本身呼叫（Settings.from_env()），cls 就是 Settings，常用來寫「另一種建構方式」。
    @classmethod
    def from_env(cls) -> "Settings":
        """從環境變數建立設定；每個變數的意義與預設值見 docs/ai/REPLY-SUGGESTIONS-SPEC.md 第 10 節。

        參數：無（cls 是 Settings 類別本身，由 Python 自動傳入）。

        回傳：依目前環境變數建立的 Settings。

        可能丟出：無；數字格式錯誤會退回預設值，不會讓服務啟動失敗。

        注意：provider_timeout_seconds 不從環境變數讀取，維持欄位預設值 10 秒。
        """
        # 用關鍵字參數建立 Settings；os.getenv(名稱, 預設) 在變數不存在時回傳預設值。
        return cls(
            # 內部 token：原樣讀取（不去空白）；沒設定時是空字串。
            internal_token=os.getenv("AI_INTERNAL_TOKEN", ""),
            # provider 網址：去掉前後空白，避免複製貼上時多帶的空白讓網址失效。
            provider_url=os.getenv("AI_VERIFICATION_PROVIDER_URL", "").strip(),
            # provider token：原樣讀取。
            provider_token=os.getenv("AI_VERIFICATION_PROVIDER_TOKEN", ""),
            # Redis 網址：沒設定時用本機預設的 Redis。
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            # Google 官方範例用 GEMINI_API_KEY；Pydantic AI 預設讀 GOOGLE_API_KEY，兩個都接受。
            # `or` 會取第一個「有值（非 None、非空字串）」的結果：GEMINI_API_KEY 優先，其次 GOOGLE_API_KEY，都沒有就是空字串；最後去掉前後空白。
            gemini_api_key=(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip(),
            # 推薦回覆的模型鏈：解析逗號分隔清單，沒設定用預設鏈，設 "-" 代表停用。
            reply_models=_model_list(os.getenv("AI_REPLY_MODELS"), DEFAULT_REPLY_MODELS),
            # 思考程度：沒設定或空字串時用 "LOW"，再去空白並轉成大寫（例如 "low" → "LOW"）。
            reply_thinking_level=(os.getenv("AI_REPLY_THINKING_LEVEL") or "LOW").strip().upper(),
            # 背景萃取的模型鏈：解析規則同 reply_models。
            extraction_models=_model_list(os.getenv("AI_EXTRACTION_MODELS"), DEFAULT_EXTRACTION_MODELS),
            # Ollama 網址：去掉前後空白；沒設定是空字串。
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "").strip(),
            # 只有 Ollama Cloud（https://ollama.com/v1）需要；本機 Ollama 不用 key。
            ollama_api_key=os.getenv("OLLAMA_API_KEY", "").strip(),
            # 向量模型名稱：沒設定或空字串時用 "gemini-embedding-2"，再去掉前後空白。
            embedding_model=(os.getenv("AI_EMBEDDING_MODEL") or "gemini-embedding-2").strip(),
            # 向量維度：必須是正整數，否則用 768。
            embedding_dimensions=_positive_int(os.getenv("AI_EMBEDDING_DIMENSIONS"), 768),
            # 推薦回覆整次請求的總逾時：必須是正數，否則用 25 秒。
            llm_timeout_seconds=_positive_float(os.getenv("AI_LLM_TIMEOUT_SECONDS"), 25.0),
            # 備援鏈裡「每個模型」各自的逾時；要比總逾時短，慢的模型才有機會讓給下一個。
            reply_model_timeout_seconds=_positive_float(os.getenv("AI_REPLY_MODEL_TIMEOUT_SECONDS"), 12.0),
            # 背景萃取每次模型呼叫的逾時：必須是正數，否則用 180 秒。
            extraction_timeout_seconds=_positive_float(os.getenv("AI_EXTRACTION_TIMEOUT_SECONDS"), 180.0),
        )

    # property：把方法變成「唯讀屬性」，使用時寫 settings.provider_configured（不用加括號），類似 JS 的 getter。
    @property
    def provider_configured(self) -> bool:
        """真人驗證 provider 是否設定完整：必須是 http(s) 網址、不含帳密與片段，而且有 token。

        回傳：True 代表可以呼叫 provider；False 時 verification.py 回 PROVIDER_CONFIGURATION_INVALID，
            main.py 的 /health 顯示 verificationProvider 為 unavailable。

        可能丟出：無（網址解析失敗會回 False）。

        設計理由：網址裡的帳密（user:pass@host）不該出現在設定中，#片段 對 API 請求沒有意義，
        都可能代表設定寫錯；缺 token 時 provider 會拒絕請求，所以直接視為未設定（fail closed）。
        """
        # 嘗試拆解網址。
        try:
            # urlsplit 把網址拆成各部分；格式嚴重錯誤（例如 IPv6 的方括號沒關）時會丟 ValueError。
            parsed = urlsplit(self.provider_url)
        # 網址無法解析。
        except ValueError:
            # 視為設定不完整。
            return False
        # 以下條件全部成立才算設定完整；bool(...) 把結果轉成真正的 True／False。
        return bool(
            # 協定只能是 http 或 https。
            parsed.scheme in {"http", "https"}
            # 必須有主機名稱（例如 face 或 provider.example.com）。
            and parsed.hostname
            # 網址不能帶使用者名稱。
            and not parsed.username
            # 網址不能帶密碼。
            and not parsed.password
            # 網址不能有 # 之後的片段。
            and not parsed.fragment
            # 必須有 provider token。
            and self.provider_token
        )

    @property
    def gemini_configured(self) -> bool:
        """有沒有 Gemini API key；沒有的話，所有 `gemini-*` 模型與向量化都視為未設定。

        回傳：gemini_api_key 不是空字串時為 True。
        """
        # 非空字串轉成 True、空字串轉成 False。
        return bool(self.gemini_api_key)

    @property
    def ollama_is_cloud(self) -> bool:
        """OLLAMA_BASE_URL 是否指向 Ollama Cloud（ollama.com），而不是本機或自架的 Ollama。

        回傳：主機名稱剛好是 ollama.com，或是它的子網域（*.ollama.com）時為 True。

        可能丟出：無（網址解析失敗會回 False）。

        設計理由：Ollama Cloud 需要 API key，而且結構化輸出的方式不同（見 reply/llm.py 的 structured_output）。
        """
        # 嘗試取出網址的主機名稱。
        try:
            # 拆解網址後取 hostname（會轉成小寫）；沒有主機名稱時是 None，用 `or ""` 換成空字串方便下面比較。
            hostname = urlsplit(self.ollama_base_url).hostname or ""
        # 網址無法解析。
        except ValueError:
            # 無法判斷就當作不是 Ollama Cloud。
            return False
        # 比對「完全等於 ollama.com」或「以 .ollama.com 結尾」；開頭加點是為了不把 evilollama.com 這種網域也算進來。
        return hostname == "ollama.com" or hostname.endswith(".ollama.com")

    @property
    def ollama_configured(self) -> bool:
        """Ollama 是否可用：網址必須是合法的 http(s)；Ollama Cloud 另外必須有 OLLAMA_API_KEY。

        回傳：True 代表 reply/llm.py 可以建立 Ollama 模型；False 時 reply/llm.py 的 build_model 對 ollama: 開頭的模型回傳 None（不建立）。

        可能丟出：無（網址解析失敗會回 False）。
        """
        # 嘗試拆解網址。
        try:
            # 拆成 scheme、hostname 等部分。
            parsed = urlsplit(self.ollama_base_url)
        # 網址無法解析。
        except ValueError:
            # 視為未設定。
            return False
        # 協定不是 http／https，或沒有主機名稱（包含沒設定網址的空字串）。
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            # 視為未設定。
            return False
        # Ollama Cloud 必須有 API key 才算可用；本機或自架的 Ollama 不需要 key，直接可用。
        return bool(self.ollama_api_key) if self.ollama_is_cloud else True
