"""把各元件組起來的服務層；HTTP API（api.py）與背景 worker（worker.py）共用同一個實例。

這一層只負責「呼叫哪個元件、把結果包成 API 格式」，實際邏輯都在各自的模組裡。
"""

from ..config import Settings
from .chunking import CHUNK_VERSION, build_chunks, sort_messages
from .embeddings import Embedder
from .errors import AIServiceError
from .extraction import ConversationSummarizer, StyleProfileBuilder
from .schemas import (
    ChunkOut,
    ChunkRequest,
    ChunkResponse,
    EmbedRequest,
    EmbedResponse,
    ReplySuggestionRequest,
    ReplySuggestionResponse,
    StyleProfileRequest,
    StyleProfileResponse,
    SummaryRequest,
    SummaryResponse,
    TopicSpanOut,
    TopicSpanRequest,
    TopicSpanResponse,
)
from .suggest import ReplySuggester
from .topics import DETECTOR_VERSION, cosine, plan_spans, resolve_span, similarity_texts

CHUNK_TITLE = "聊天片段"  # 片段向量化時的標題；片段內每行已經帶有日期時間
TOPIC_TITLE = "話題"


class ReplyAIService:
    """AI 推薦回覆的所有運算入口。

    各元件都可以從建構子注入（測試時換成假的模型或向量服務）；不注入時依 Settings 建立。
    """

    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        suggester: ReplySuggester | None = None,
        style_builder: StyleProfileBuilder | None = None,
        summarizer: ConversationSummarizer | None = None,
    ):
        """組合各元件；風格卡萃取預設與服務共用同一個向量服務，避免重複建立 client。"""
        self.settings = settings
        self.embedder = embedder or Embedder(settings)
        self.suggester = suggester or ReplySuggester(settings)
        self.style_builder = style_builder or StyleProfileBuilder(settings, embedder=self.embedder)
        self.summarizer = summarizer or ConversationSummarizer(settings)

    async def suggestions(self, request: ReplySuggestionRequest) -> ReplySuggestionResponse:
        """產生 3～5 則推薦（詳見 suggest.ReplySuggester.generate）。"""
        return await self.suggester.generate(request)

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        """把文字轉成向量；purpose=query 用於檢索查詢，document 用於要存進資料庫的內容。"""
        vectors = await self.embedder.embed(list(request.texts), request.purpose, request.title)
        return EmbedResponse(model=self.embedder.model, dimensions=self.embedder.dimensions, vectors=vectors)

    async def chunks(self, request: ChunkRequest) -> ChunkResponse:
        """把聊天室切成片段；embed=True 時一併向量化。

        向量化失敗時整個請求失敗（503），因為沒有向量的片段無法用於檢索；
        背景工作會由 BullMQ 依設定稍後重試。
        """
        drafts = build_chunks(request.messages, request.now)
        vectors: list[list[float]] | None = None
        if request.embed and drafts:
            vectors = await self.embedder.embed([draft.content for draft in drafts], "document", CHUNK_TITLE)
        return ChunkResponse(
            conversationId=request.conversationId,
            chunkVersion=CHUNK_VERSION,
            embeddingModel=self.embedder.model if vectors is not None else None,
            chunks=[
                ChunkOut(
                    firstMessageId=draft.messages[0].id,
                    lastMessageId=draft.messages[-1].id,
                    firstAt=draft.messages[0].createdAt,
                    lastAt=draft.messages[-1].createdAt,
                    messageCount=len(draft.messages),
                    content=draft.content,
                    tokenEstimate=draft.token_estimate,
                    isOpen=draft.is_open,
                    vector=vectors[index] if vectors is not None else None,
                )
                for index, draft in enumerate(drafts)
            ],
        )

    async def topic_spans(self, request: TopicSpanRequest) -> TopicSpanResponse:
        """找出 AI 開啟的話題區段。

        有向量服務時，每個區段把「起點＋每 3 則一組」一次送去向量化，用相似度判斷何時換話題；
        沒設定或向量化失敗時，退回只用時間間隔與 30 則上限判斷。
        usedVectors 只要有任何一個區段用到向量就是 True。
        """
        ordered = sort_messages(request.messages)
        spans: list[TopicSpanOut] = []
        used_vectors = False
        for plan in plan_spans(ordered):
            similarities = None
            if self.embedder.configured:
                try:
                    vectors = await self.embedder.embed(similarity_texts(ordered, plan), "document", TOPIC_TITLE)
                    similarities = [cosine(vectors[0], vector) for vector in vectors[1:]]
                    used_vectors = True
                except AIServiceError:
                    similarities = None
            span = resolve_span(ordered, plan, similarities)
            if span:
                spans.append(span)
        return TopicSpanResponse(
            conversationId=request.conversationId,
            detectorVersion=DETECTOR_VERSION,
            usedVectors=used_vectors,
            spans=spans,
        )

    async def summarize(self, request: SummaryRequest) -> SummaryResponse:
        """更新聊天室摘要（詳見 extraction.ConversationSummarizer）。"""
        return await self.summarizer.summarize(request)

    async def style_profile(self, request: StyleProfileRequest) -> StyleProfileResponse:
        """萃取一位使用者的風格卡（詳見 extraction.StyleProfileBuilder）。"""
        return await self.style_builder.build(request)

    def health(self) -> dict[str, str]:
        """回報各模型是否已設定（只看設定，不實際呼叫模型，也不代表額度還夠）。"""

        def state(ready: bool) -> str:
            """把布林值轉成 /health 使用的文字。"""
            return "configured" if ready else "unavailable"

        return {
            "replyModels": state(self.suggester.configured),
            "embedding": state(self.embedder.configured),
            "extraction": state(self.style_builder.model is not None),
        }


def build_service(settings: Settings) -> ReplyAIService:
    """依設定建立預設的服務實例（模型都延遲到第一次使用才建立）。"""
    return ReplyAIService(settings)
