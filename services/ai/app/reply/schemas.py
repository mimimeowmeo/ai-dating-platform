"""AI 推薦回覆的資料格式（對外契約）。

分成兩類：
1. `ApiModel` 子類別：AI 內部 HTTP API 與 BullMQ job 的請求／回應格式。欄位用 camelCase，
   跟 NestJS 的 JSON 一致；`extra="forbid"` 讓後端傳錯欄位名稱時立刻回 422，而不是被默默忽略。
2. `LlmModel` 子類別：要求語言模型輸出的結構化格式。模型偶爾會多吐欄位，所以用
   `extra="ignore"`；長度上限也放寬一些，真正的長度限制在後處理再做，減少不必要的重試。

時間欄位都用 `datetime`，後端傳 ISO 8601 字串即可（例如 Prisma 的 `createdAt`）。
"""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# 風格卡的特徵版本；計算方式或欄位意義改變時要升版（AI-SPEC：模型與特徵都要記版本）。
FEATURE_VERSION = "style-card-v1"

# 訊息來源：真人打的／AI 推薦原封不動送出／AI 推薦改過再送（由後端依相似度判斷）。
Origin = Literal["human", "ai_verbatim", "ai_edited"]
# 一則推薦的用途：回答、提問、呼應舊話題、幽默、邀約推進、分享自己。
Intent = Literal["answer", "question", "callback", "humor", "plan", "share"]
# 產生推薦的模式，判斷規則見 suggest.detect_mode。
Mode = Literal["opener", "reply", "follow_up", "revive"]
# 聊天室裡的角色：A 是按按鈕的人，B 是聊天對象。
Role = Literal["A", "B"]

Id = Annotated[str, Field(min_length=1, max_length=64)]
MessageText = Annotated[str, Field(min_length=1, max_length=2000)]
Ratio = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
Tag = Annotated[str, Field(min_length=1, max_length=40)]
ShortText = Annotated[str, Field(min_length=1, max_length=40)]


class ApiModel(BaseModel):
    """所有 API 格式的共同設定：拒絕未知欄位。"""

    model_config = ConfigDict(extra="forbid")


class LlmModel(BaseModel):
    """所有模型輸出格式的共同設定：忽略模型多吐的欄位。"""

    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------------------
# 共用的資料片段
# ---------------------------------------------------------------------------


class ChatMessage(ApiModel):
    """A–B 聊天室裡的一則訊息（產生推薦時使用）。

    `sender` 只分 A／B：後端先把 user id 對應成「按按鈕的人」與「聊天對象」，
    AI 服務因此不需要知道任何使用者 id。`origin` 讓 AI 知道哪幾則是 AI 推薦送出的。
    """

    id: Id
    sender: Role
    content: MessageText
    createdAt: datetime
    origin: Origin = "human"


class IndexMessage(ApiModel):
    """背景工作使用的訊息（對話切片、AI 話題區段、聊天室摘要）。

    背景工作處理的是「整個聊天室」，不站在 A 或 B 的角度，所以用使用者 id 與暱稱標示說話者。
    `suggestionIntent` 只有在這則訊息來自 AI 推薦時才有意義，代表那則推薦的用途，
    用來判斷它是不是「開啟了新話題」。
    """

    id: Id
    senderId: Id
    senderName: Annotated[str, Field(min_length=1, max_length=40)]
    content: MessageText
    createdAt: datetime
    origin: Origin = "human"
    suggestionIntent: Intent | None = None


class OwnMessage(ApiModel):
    """某位使用者「自己發出」的一則訊息（萃取風格卡時使用）。

    `inAiTopic` 由後端依 AI 話題區段標記：只有這則訊息的發送者正是區段發起者時才會是 True。
    這類訊息的寫法照用，話題喜好則降權（規格 5.3）。
    """

    id: Id
    conversationId: Id
    content: MessageText
    createdAt: datetime
    origin: Origin = "human"
    inAiTopic: bool = False


class ProfileSnapshot(ApiModel):
    """一位使用者的檔案快照：標籤與興趣都傳中文顯示名稱（例如「羽球」），不是代碼。"""

    displayName: Annotated[str, Field(min_length=1, max_length=40)]
    age: Annotated[int, Field(ge=18, le=120)] | None = None
    gender: Annotated[str, Field(max_length=20)] | None = None
    city: Annotated[str, Field(max_length=40)] | None = None
    bio: Annotated[str, Field(max_length=1000)] = ""
    occupation: Annotated[str, Field(max_length=80)] | None = None
    education: Annotated[str, Field(max_length=80)] | None = None
    heightCm: Annotated[int, Field(ge=100, le=250)] | None = None
    interests: list[Tag] = Field(default_factory=list, max_length=40)
    hobbies: list[Tag] = Field(default_factory=list, max_length=40)
    foods: list[Tag] = Field(default_factory=list, max_length=40)
    traits: list[Tag] = Field(default_factory=list, max_length=80)
    datingGoals: list[Tag] = Field(default_factory=list, max_length=10)


class StyleStats(ApiModel):
    """可量化的寫法統計，用來算「風格目標」與「風格距離」。

    - medianChars／meanChars：每則訊息字數的中位數與平均（不含空白）。
    - emojiPerMessage：每則平均幾個 emoji。
    - questionRatio／exclamationRatio／laughterRatio：含問句、驚嘆、笑聲詞的訊息比例。
    - burstMean：平均一次連發幾則（同一聊天室 60 秒內的連續訊息算同一次）。
    - particles：常用語助詞（啦、欸、喔…）出現在多少比例的訊息裡。
    """

    messageCount: Annotated[int, Field(ge=0)] = 0
    medianChars: Annotated[float, Field(ge=0, le=2000, allow_inf_nan=False)]
    meanChars: Annotated[float, Field(ge=0, le=2000, allow_inf_nan=False)]
    emojiPerMessage: Annotated[float, Field(ge=0, le=50, allow_inf_nan=False)]
    questionRatio: Ratio
    exclamationRatio: Ratio
    laughterRatio: Ratio
    burstMean: Annotated[float, Field(ge=1, le=50, allow_inf_nan=False)] = 1.0
    particles: dict[str, Ratio] = Field(default_factory=dict)

    @field_validator("particles")
    @classmethod
    def limit_particles(cls, value: dict[str, float]) -> dict[str, float]:
        """語助詞最多保留 12 個、每個鍵最多 4 個字，避免後端傳入異常大的字典。"""
        if len(value) > 12 or any(not key or len(key) > 4 for key in value):
            raise ValueError("particles must have at most 12 short keys")
        return value


class StyleFacet(ApiModel):
    """風格卡上的一條抽象特徵句，例如「聊到旅行會回得特別長」。

    kind：topic（喜歡的話題）／tone（語氣）／habit（聊天習慣）／avoid（會避開的話題）。
    weight 是 0～1 的重要度；evidence 是有幾批訊息支持這條特徵。
    """

    kind: Literal["topic", "tone", "habit", "avoid"]
    statement: ShortText
    weight: Ratio = 0.5
    evidence: Annotated[int, Field(ge=0)] = 0


class StyleCard(ApiModel):
    """一位使用者的風格卡（寫法＋喜好），A 與 B 共用同一種格式。

    - confidence：high（≥30 則真人訊息）／low（1～29 則，或只有 bio）／none（什麼都沒有）。
    - sampleSource：寫法樣本從哪來：chat／mixed（聊天＋bio 各半）／bio／none。
    - bioSample：冷啟動時放進 prompt 當寫法範例的 bio 片段。
    """

    featureVersion: Annotated[str, Field(min_length=1, max_length=40)] = FEATURE_VERSION
    userId: Id | None = None
    confidence: Literal["high", "low", "none"]
    sampleSource: Literal["chat", "mixed", "bio", "none"]
    stats: StyleStats
    voiceNotes: list[ShortText] = Field(default_factory=list, max_length=8)
    facets: list[StyleFacet] = Field(default_factory=list, max_length=40)
    bioSample: Annotated[str, Field(max_length=300)] = ""
    messageCount: Annotated[int, Field(ge=0)] = 0
    windowFrom: datetime | None = None
    windowTo: datetime | None = None
    modelName: Annotated[str, Field(max_length=120)] | None = None
    promptVersion: Annotated[str, Field(max_length=40)] | None = None


class RetrievedChunk(ApiModel):
    """後端用 pgvector 檢索到的一段舊對話；清單順序就是相關程度（第一個最相關）。"""

    content: Annotated[str, Field(min_length=1, max_length=4000)]
    lastAt: datetime | None = None
    score: Annotated[float, Field(allow_inf_nan=False)] | None = None


class BlendConfig(ApiModel):
    """風格混合比例：B 的權重（0＝完全照 A 的寫法，1＝完全照 B 喜歡的樣子）。規格 4.1、4.2。

    一批 5 則推薦共用同一個比例，而且只看「整個聊天室」有沒有訊息，跟是誰傳的無關：
    - firstMessagePartnerWeight：聊天室還沒有任何訊息、要寫整個聊天室的第一則訊息時用，
      預設 1.0（B 100%）。
    - laterPartnerWeight：只要有人傳過訊息（A 或 B 都算），之後一律用這個，預設 0.2（A 80%／B 20%）。
    """

    firstMessagePartnerWeight: Ratio = 1.0
    laterPartnerWeight: Ratio = 0.2


# ---------------------------------------------------------------------------
# POST /internal/ai/reply-suggestions
# ---------------------------------------------------------------------------


class ReplySuggestionRequest(ApiModel):
    """產生推薦所需的全部資料，由 NestJS 組好（它負責權限檢查與資料庫查詢）。

    - recentMessages：這個聊天室最近的訊息（舊到新或新到舊都可以，服務會自己排序）。
    - retrievedChunks：用 /embed 的查詢向量在 pgvector 找到的舊對話片段。
    - requesterStyle／partnerStyle：A、B 的風格卡；沒有時服務會用 bio 做冷啟動。
    - partnerFacets：用查詢向量檢索到、跟目前話題最相關的 B 特徵句。
    - siteStats：全站聊天平均，冷啟動時當數值目標；沒給就用內建暫定值。
    - excludeTexts：「換一批」時要避開的舊推薦。
    """

    requestId: Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")]
    mode: Mode | None = None
    requester: ProfileSnapshot
    partner: ProfileSnapshot
    sharedTags: list[Tag] = Field(default_factory=list, max_length=60)
    recentMessages: list[ChatMessage] = Field(default_factory=list, max_length=300)
    retrievedChunks: list[RetrievedChunk] = Field(default_factory=list, max_length=12)
    conversationSummary: Annotated[str, Field(max_length=3000)] | None = None
    requesterStyle: StyleCard | None = None
    partnerStyle: StyleCard | None = None
    partnerFacets: list[StyleFacet] = Field(default_factory=list, max_length=20)
    siteStats: StyleStats | None = None
    excludeTexts: list[Annotated[str, Field(min_length=1, max_length=300)]] = Field(
        default_factory=list, max_length=30
    )
    blend: BlendConfig = Field(default_factory=BlendConfig)
    now: datetime | None = None


class Suggestion(ApiModel):
    """一則推薦。rank 1 會被前端以打字動畫填入輸入框，其餘變成按鈕。

    styleTarget 是這一批共用的寫法規則（同一批 5 則都一樣）：
    - partner：要寫整個聊天室的第一則訊息，照 B 喜歡的樣子寫（B 100%）。
    - blend：其他情況，也就是 A 80%／B 20%；B 沒有足夠資料時只用 A 的寫法，也標成 blend。
    """

    rank: Annotated[int, Field(ge=1, le=5)]
    text: Annotated[str, Field(min_length=1, max_length=200)]
    intent: Intent
    styleTarget: Literal["partner", "blend"]
    styleDistance: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    reason: Annotated[str, Field(max_length=120)] = ""


class RejectedSuggestion(ApiModel):
    """被後處理刪掉的候選與原因代碼；後端要全部保存（所有 AI 推薦內容保存到專案結束）。"""

    text: Annotated[str, Field(max_length=400)]
    reasonCode: Annotated[str, Field(min_length=1, max_length=60)]


class UsageInfo(ApiModel):
    """這次請求用掉的模型 token 與請求次數（含驗證失敗的重試）。"""

    inputTokens: int = 0
    outputTokens: int = 0
    requests: int = 0


class StyleTarget(ApiModel):
    """這一批 5 則共用的風格目標（由 style.resolve_target 算出）。

    - rule：套用了哪一條規則。first_message＝聊天室還沒有任何訊息；later＝已經有人傳過訊息。
    - source：目標實際用了誰的寫法。partner＝B 喜歡的樣子；blend＝A 為主、帶一點 B；
      requester＝B 沒有足夠資料（沒聊過天、bio 也太短），只用 A 的寫法（規格 5.3）。
    - stats：數值目標本身（字數、emoji、語助詞…），排序時用來算每則候選的風格距離。
    """

    rule: Literal["first_message", "later"]
    source: Literal["partner", "blend", "requester"]
    stats: StyleStats


class ReplySuggestionResponse(ApiModel):
    """產生推薦的結果。

    status：ok（3～5 則）／partial（1～2 則，notice 會說明）／empty（0 則，notice 是「沒有可推薦的句子」）。
    modelName 是實際回應的模型（備援鏈中哪一個成功）；完全沒有資料根據、沒有呼叫模型時是 None，
    這時 usage 也全是 0。
    """

    requestId: str
    status: Literal["ok", "partial", "empty"]
    mode: Mode
    suggestions: list[Suggestion]
    rejected: list[RejectedSuggestion]
    notice: str | None = None
    modelName: str | None = None
    promptVersion: str
    usage: UsageInfo
    target: StyleTarget


# ---------------------------------------------------------------------------
# POST /internal/ai/embed
# ---------------------------------------------------------------------------


class EmbedRequest(ApiModel):
    """把文字轉成向量。purpose=query 用在檢索時的查詢；document 用在要存進資料庫的內容。"""

    texts: list[Annotated[str, Field(min_length=1, max_length=8000)]] = Field(min_length=1, max_length=100)
    purpose: Literal["query", "document"] = "document"
    title: Annotated[str, Field(min_length=1, max_length=60)] | None = None


class EmbedResponse(ApiModel):
    """向量結果；vectors 的順序與請求的 texts 相同。"""

    model: str
    dimensions: int
    vectors: list[list[float]]


# ---------------------------------------------------------------------------
# POST /internal/ai/chunks
# ---------------------------------------------------------------------------


class ChunkRequest(ApiModel):
    """把一個聊天室的訊息切成片段；embed=True 時順便向量化。

    增量更新時，後端從「上一次標為 isOpen 的片段」的第一則訊息開始傳即可。
    """

    conversationId: Id
    messages: list[IndexMessage] = Field(min_length=1, max_length=20000)
    embed: bool = True
    now: datetime | None = None


class ChunkOut(ApiModel):
    """一個對話片段。isOpen=True 表示這段可能還會長大，之後要刪掉重建。"""

    firstMessageId: str
    lastMessageId: str
    firstAt: datetime
    lastAt: datetime
    messageCount: int
    content: str
    tokenEstimate: int
    isOpen: bool
    vector: list[float] | None = None


class ChunkResponse(ApiModel):
    """切片結果；embeddingModel 在沒有向量化時為 null，後端要記錄它以便日後換模型時重建。"""

    conversationId: str
    chunkVersion: str
    embeddingModel: str | None = None
    chunks: list[ChunkOut]


# ---------------------------------------------------------------------------
# POST /internal/ai/topic-spans
# ---------------------------------------------------------------------------


class TopicSpanRequest(ApiModel):
    """找出 AI 推薦「開啟新話題」後的區段；messages 需包含 origin 與 suggestionIntent。"""

    conversationId: Id
    messages: list[IndexMessage] = Field(min_length=1, max_length=20000)


class TopicSpanOut(ApiModel):
    """一個 AI 話題區段。

    initiatingMessageId 是開啟話題的那則 AI 訊息；startMessageId～endMessageId 是它之後、
    仍在同一話題的訊息（含雙方）。後端只會把「initiatorId 本人」在區段內的訊息標為 inAiTopic。
    endReason：time_gap（隔太久）／topic_shift（換話題）／cap（達 30 則上限）／end_of_data（訊息用完）。
    """

    initiatingMessageId: str
    initiatorId: str
    startMessageId: str
    endMessageId: str
    messageCount: int
    endReason: Literal["time_gap", "topic_shift", "cap", "end_of_data"]


class TopicSpanResponse(ApiModel):
    """話題區段結果；usedVectors=False 代表只用時間與上限判斷（沒有向量服務）。"""

    conversationId: str
    detectorVersion: str
    usedVectors: bool
    spans: list[TopicSpanOut]


# ---------------------------------------------------------------------------
# POST /internal/ai/conversation-summary
# ---------------------------------------------------------------------------


class SummaryRequest(ApiModel):
    """用「舊摘要＋新訊息」更新聊天室摘要。"""

    conversationId: Id
    previousSummary: Annotated[str, Field(max_length=3000)] | None = None
    messages: list[IndexMessage] = Field(min_length=1, max_length=2000)
    maxChars: Annotated[int, Field(ge=100, le=1500)] = 600


class SummaryResponse(ApiModel):
    """新的摘要；untilMessageId 是這份摘要涵蓋到的最後一則訊息。"""

    conversationId: str
    summary: str
    untilMessageId: str
    modelName: str | None = None
    promptVersion: str


# ---------------------------------------------------------------------------
# POST /internal/ai/style-profile
# ---------------------------------------------------------------------------


class StyleProfileRequest(ApiModel):
    """萃取一位使用者的風格卡。

    messages 應該已由後端依視窗規則篩好（只含本人發出、human 的訊息）；服務仍會再排除一次
    AI 來源的訊息，當作保險。windowFrom／windowTo／capped 只是記錄用，會原樣寫進風格卡。
    """

    userId: Id
    bio: Annotated[str, Field(max_length=1000)] = ""
    messages: list[OwnMessage] = Field(default_factory=list, max_length=20000)
    windowFrom: datetime | None = None
    windowTo: datetime | None = None
    capped: bool = False
    siteStats: StyleStats | None = None
    embedFacets: bool = True


class StyleProfileResponse(ApiModel):
    """風格卡結果；facetVectors 與 card.facets 一一對應（沒有向量化時為 null）。"""

    card: StyleCard
    facetVectors: list[list[float]] | None = None
    embeddingModel: str | None = None
    rejectedStatements: int = 0


# ---------------------------------------------------------------------------
# 模型輸出格式（只在 AI 服務內部使用）
# ---------------------------------------------------------------------------


class DraftSuggestion(LlmModel):
    """模型產生的一則候選。

    寫法規則（B 100% 或 A 80%／B 20%）由程式依聊天室有沒有訊息決定，模型不用標示；
    模型如果照舊版格式多吐 styleTarget，LlmModel 會直接忽略。
    """

    text: Annotated[str, Field(min_length=1, max_length=300)]
    intent: Intent
    priority: Annotated[int, Field(ge=1, le=5)]
    reason: Annotated[str, Field(max_length=200)] = ""


class DraftBatch(LlmModel):
    """模型一次產生的候選清單：最多 5 則（容許 6 則），沒有依據時可以少於 5 則，甚至 0 則。

    刻意不設下限：以前要求至少 3 則、不足就重試，等於逼模型硬湊空泛的句子。
    現在找不到依據就回空清單，前端直接顯示「沒有可推薦的句子」（規格 4.1）。
    超過 6 則才算格式錯誤，會請模型重寫。
    """

    suggestions: list[DraftSuggestion] = Field(max_length=6)


class TopicHeat(LlmModel):
    """風格萃取時的一個話題與熱度（1～5）；fromAiTopic=True 表示主要來自 [AI話題] 訊息。"""

    topic: Annotated[str, Field(min_length=1, max_length=40)]
    heat: Annotated[int, Field(ge=1, le=5)]
    fromAiTopic: bool = False


class StyleMapResult(LlmModel):
    """風格萃取「一批訊息」的結果（map 步驟）；之後在程式裡合併（reduce）。"""

    voiceNotes: list[Annotated[str, Field(min_length=1, max_length=80)]] = Field(default_factory=list, max_length=8)
    topics: list[TopicHeat] = Field(default_factory=list, max_length=12)
    avoidTopics: list[Annotated[str, Field(min_length=1, max_length=40)]] = Field(default_factory=list, max_length=6)
    habits: list[Annotated[str, Field(min_length=1, max_length=80)]] = Field(default_factory=list, max_length=6)


class SummaryDraft(LlmModel):
    """模型產生的聊天室摘要。"""

    summary: Annotated[str, Field(min_length=1, max_length=4000)]
