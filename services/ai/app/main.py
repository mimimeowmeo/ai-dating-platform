import secrets

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from .config import Settings
from .reply.api import install_reply_api
from .reply.service import ReplyAIService, build_service
from .schemas import MAX_BODY_BYTES, VerificationRequest, VerificationResult
from .verification import InvalidImage, VerificationService


class InternalBoundary:
    """先驗證內部 token 與串流大小，再允許框架解析 JSON。"""

    def __init__(self, app, settings: Settings):
        self.app = app
        self.settings = settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/internal/"):
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        if not self.settings.internal_token:
            response = JSONResponse({"code": "INTERNAL_AUTH_NOT_CONFIGURED", "message": "內部服務尚未設定"}, status_code=503)
            return await response(scope, receive, send)
        if not secrets.compare_digest(
            headers.get(b"x-internal-token", b""), self.settings.internal_token.encode("utf-8")
        ):
            response = JSONResponse({"code": "UNAUTHORIZED", "message": "需要內部服務認證"}, status_code=401)
            return await response(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > MAX_BODY_BYTES:
                response = JSONResponse({"code": "PAYLOAD_TOO_LARGE", "message": "影像超過大小限制"}, status_code=413)
                return await response(scope, receive, send)
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)


def create_app(
    settings: Settings | None = None,
    service: VerificationService | None = None,
    reply_service: ReplyAIService | None = None,
) -> FastAPI:
    """建立 FastAPI app：真人驗證與 AI 推薦回覆共用同一個內部邊界（token 與大小限制）。

    reply_service 讓測試可以注入使用假模型的服務；不給時依設定建立（模型延遲到第一次使用才建立）。
    """
    settings = settings or Settings.from_env()
    app = FastAPI(title="AI 私有服務", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(InternalBoundary, settings=settings)
    verifier = service or VerificationService(settings)
    replies = reply_service or build_service(settings)

    @app.exception_handler(RequestValidationError)
    async def request_error(_request, _error):
        # FastAPI 的預設錯誤可能包含 input；自拍與 Base64 不可回顯。
        return JSONResponse({"code": "INVALID_REQUEST", "message": "請求格式不正確"}, status_code=422)

    @app.exception_handler(InvalidImage)
    async def image_error(_request, error):
        return JSONResponse({"code": str(error), "message": "影像格式、內容或尺寸不符要求"}, status_code=400)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": "ai",
            "verificationProvider": "configured" if settings.provider_configured else "unavailable",
            # 只回報「有沒有設定」，不實際呼叫模型；額度是否足夠要到 Ollama／AI Studio 的帳號頁面查看。
            "replySuggestions": replies.health(),
        }

    @app.post("/internal/ai/face/verify", response_model=VerificationResult)
    async def verify(request: VerificationRequest):
        return await verifier.verify(request)

    install_reply_api(app, replies)
    return app


app = create_app()
