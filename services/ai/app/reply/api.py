"""AI 推薦回覆的內部 HTTP API（路徑都在 /internal/ai/ 底下）。

認證與請求大小限制由 main.py 的 InternalBoundary 統一處理（需要 X-Internal-Token）；
格式錯誤由 main.py 的 RequestValidationError 處理器回 422，且不回顯輸入內容。
這裡只負責把請求交給服務層，並把 AIServiceError 轉成固定格式的 JSON 錯誤。
"""

from fastapi import APIRouter, FastAPI
from starlette.responses import JSONResponse

from .errors import AIServiceError
from .schemas import (
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
    TopicSpanRequest,
    TopicSpanResponse,
)
from .service import ReplyAIService


def build_router(service: ReplyAIService) -> APIRouter:
    """建立所有內部路由；每個路由都只是把請求轉交給服務層對應的方法。"""
    router = APIRouter(prefix="/internal/ai")

    @router.post("/reply-suggestions", response_model=ReplySuggestionResponse)
    async def reply_suggestions(request: ReplySuggestionRequest):
        """產生 3～5 則推薦。模型不可用時回 503（LLM_NOT_CONFIGURED／LLM_UNAVAILABLE）。"""
        return await service.suggestions(request)

    @router.post("/embed", response_model=EmbedResponse)
    async def embed(request: EmbedRequest):
        """把文字轉成向量（查詢或文件）。向量服務不可用時回 503（EMBEDDING_*）。"""
        return await service.embed(request)

    @router.post("/chunks", response_model=ChunkResponse)
    async def chunks(request: ChunkRequest):
        """把聊天室切成片段，可選擇一併向量化。"""
        return await service.chunks(request)

    @router.post("/topic-spans", response_model=TopicSpanResponse)
    async def topic_spans(request: TopicSpanRequest):
        """找出 AI 推薦開啟的話題區段；沒有向量服務時退回只用時間判斷。"""
        return await service.topic_spans(request)

    @router.post("/conversation-summary", response_model=SummaryResponse)
    async def conversation_summary(request: SummaryRequest):
        """用舊摘要＋新訊息更新聊天室摘要。萃取模型不可用時回 503（EXTRACTION_*）。"""
        return await service.summarize(request)

    @router.post("/style-profile", response_model=StyleProfileResponse)
    async def style_profile(request: StyleProfileRequest):
        """萃取一位使用者的風格卡（可順便把特徵句向量化）。"""
        return await service.style_profile(request)

    return router


def install_reply_api(app: FastAPI, service: ReplyAIService) -> None:
    """把推薦回覆的路由與錯誤處理器掛到 FastAPI app 上（由 main.create_app 呼叫）。"""
    app.include_router(build_router(service))

    @app.exception_handler(AIServiceError)
    async def ai_service_error(_request, error: AIServiceError):
        """把可預期的服務錯誤轉成 {"code", "message"}，狀態碼通常是 503。"""
        return JSONResponse({"code": error.code, "message": error.message}, status_code=error.http_status)
