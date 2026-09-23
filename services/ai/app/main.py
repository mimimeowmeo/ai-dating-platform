"""AI 私有服務的 HTTP 入口（FastAPI app）。

這個檔案做三件事：
1. 定義 `InternalBoundary`：一個 ASGI middleware（中介層），所有 `/internal/` 開頭的路徑都要先通過
   它的兩道檢查——X-Internal-Token 是否正確、請求 body 是否超過 `MAX_BODY_BYTES`——才會交給 FastAPI
   解析 JSON。這樣未授權或過大的請求在「讀進記憶體、解析 JSON」之前就被擋掉。
2. 定義 `create_app()`：組出 FastAPI app，掛上 middleware、錯誤處理器、`/health`、
   真人驗證路由 `POST /internal/ai/face/verify`，以及 AI 推薦回覆的路由（由 reply/api.py 提供）。
3. 在模組最底下建立 `app = create_app()`，讓 uvicorn 可以用 `app.main:app` 找到並啟動它
   （Dockerfile 的 CMD 就是 `python -m uvicorn app.main:app ...`）。

呼叫方：只有 NestJS API（apps/api/src/profiles.ts）會呼叫真人驗證路由，並帶上 X-Internal-Token。
這個服務不對外公開，所以也關掉了 FastAPI 自動產生的 /docs、/redoc、/openapi.json。
"""

# secrets 是 Python 標準函式庫；這裡只用它的 compare_digest 做「固定時間」的字串比對，避免時間側通道攻擊
# （timing attack：攻擊者靠比對花的時間長短，一個字元一個字元猜出 token）。
import secrets

# FastAPI 是這個服務使用的 Web 框架（類似 Node 的 Express／NestJS，但用 Python 型別宣告請求格式）。
from fastapi import FastAPI
# RequestValidationError：FastAPI 在「請求 body 不符合 Pydantic 模型」時丟出的例外；下面會自訂它的回應。
from fastapi.exceptions import RequestValidationError
# JSONResponse：Starlette（FastAPI 底層框架）的 JSON 回應物件；middleware 裡要自己組回應時會用到。
from starlette.responses import JSONResponse

# Settings：從環境變數讀進來的服務設定（內部 token、provider 網址等），定義在 config.py。
from .config import Settings
# install_reply_api：把「AI 推薦回覆」的所有 /internal/ai/... 路由掛到 app 上的函式。
from .reply.api import install_reply_api
# ReplyAIService：推薦回覆的服務層型別；build_service：依設定建立預設的推薦回覆服務。
from .reply.service import ReplyAIService, build_service
# MAX_BODY_BYTES：所有 /internal/* 請求 body 的總上限（位元組）；
# VerificationRequest／VerificationResult：真人驗證的請求與回應格式（Pydantic 模型）。
from .schemas import MAX_BODY_BYTES, VerificationRequest, VerificationResult
# InvalidImage：影像不合格時丟出的例外（訊息就是錯誤碼，例如 INVALID_IMAGE、FRAME_IMAGE_TOO_LARGE）；
# VerificationService：驗證影像後把請求轉給人臉驗證 provider 的服務。
from .verification import InvalidImage, VerificationService


class InternalBoundary:
    """先驗證內部 token 與串流大小，再允許框架解析 JSON。

    這是一個「純 ASGI middleware」。ASGI 是 Python 非同步 Web 伺服器與框架之間的介面標準：
    伺服器（uvicorn）對每個請求呼叫 `app(scope, receive, send)`——
        scope：這個連線的資訊（type、path、headers 等）的 dict。
        receive：非同步函式，每呼叫一次拿到一段請求 body（訊息 dict）。
        send：非同步函式，用來送出回應的狀態碼、headers 與 body。
    middleware 就是包在真正 app 外面的一層，同樣實作 `__call__(scope, receive, send)`，
    可以決定要自己回應（擋掉請求），還是把請求交給內層的 app。

    只處理 path 以 `/internal/` 開頭的 HTTP 請求，依序檢查：
        1. 伺服器沒設定 AI_INTERNAL_TOKEN → 503 INTERNAL_AUTH_NOT_CONFIGURED（寧可整個關閉，也不要在沒密碼時放行）。
        2. X-Internal-Token 不正確或缺少 → 401 UNAUTHORIZED。
        3. 一邊串流讀 body 一邊累計大小，超過 MAX_BODY_BYTES → 413 PAYLOAD_TOO_LARGE。
    全部通過後，才把已讀完的 body 交給 FastAPI 解析。

    為什麼不用 FastAPI 的依賴注入（Depends）來驗 token？因為 FastAPI 會先把整個 body 讀完並解析 JSON，
    才執行依賴；那樣未授權的人也能讓服務讀入、解析大量 JSON。在 ASGI 這層就能在讀 body 之前擋掉。
    為什麼不用 Content-Length 判斷大小？因為用 chunked 傳輸時可以沒有 Content-Length，或者謊報；
    所以實際累計收到的位元組數才可靠（tests/test_verification.py 的
    test_http_body_limit_even_without_content_length 驗證這點）。

    所有錯誤回應都是固定的 {"code", "message"}，不回顯任何請求內容。
    """

    def __init__(self, app, settings: Settings):
        """建立 middleware。

        參數：
            app：內層的 ASGI app（FastAPI／Starlette 建立 middleware 堆疊時自動傳入）。
            settings：服務設定；這裡只用到 internal_token。
        由 `app.add_middleware(InternalBoundary, settings=settings)` 註冊，
        Starlette 會在建構 middleware 堆疊時呼叫 `InternalBoundary(內層app, settings=settings)`。
        """
        # 保存內層 app，檢查通過後要把請求交給它。
        self.app = app
        # 保存設定，之後每個請求都要讀 internal_token。
        self.settings = settings

    async def __call__(self, scope, receive, send):
        """處理一個 ASGI 連線（每個 HTTP 請求都會呼叫一次）。

        參數：scope／receive／send 見 class 說明。
        回傳：None；回應是透過 send 送出，不是用 return 值。
        不會丟出自訂例外：所有擋下的情況都直接送出 JSON 錯誤回應。
        """
        # scope["type"] 可能是 "http"、"websocket" 或 "lifespan"（伺服器啟動／關閉事件）；
        # 非 HTTP，或路徑不是 /internal/ 開頭（例如 /health）時，完全不檢查，直接交給內層 app。
        if scope["type"] != "http" or not scope["path"].startswith("/internal/"):
            # 原封不動轉交；await 並 return，讓這個函式到此結束。
            return await self.app(scope, receive, send)
        # ASGI 的 headers 是 [(名稱bytes, 值bytes), ...] 的清單，名稱已轉成小寫；
        # 轉成 dict 方便用名稱查詢（同名 header 出現多次時，dict 會保留最後一個）。
        headers = dict(scope.get("headers", []))
        # 伺服器沒有設定內部 token（空字串）時，任何人都不該能呼叫內部 API。
        if not self.settings.internal_token:
            # 503 表示「服務目前無法提供」；代碼讓維運者知道是設定缺漏，而不是呼叫方的錯。
            response = JSONResponse({"code": "INTERNAL_AUTH_NOT_CONFIGURED", "message": "內部服務尚未設定"}, status_code=503)
            # JSONResponse 本身也是 ASGI app，呼叫它就會透過 send 把回應送出去。
            return await response(scope, receive, send)
        # 比對請求帶來的 X-Internal-Token 與設定值（都用 bytes 比）；
        # compare_digest 花的時間不隨「前幾個字元相同」而改變，避免被逐字元猜出 token。
        if not secrets.compare_digest(
            # 請求的 token；沒帶這個 header 時用空 bytes，比對一定失敗。
            headers.get(b"x-internal-token", b""), self.settings.internal_token.encode("utf-8")
        ):
            # token 錯誤或缺少：回 401，且在讀取 body 之前就結束，未授權者無法讓服務讀入大量資料。
            response = JSONResponse({"code": "UNAUTHORIZED", "message": "需要內部服務認證"}, status_code=401)
            # 送出 401 回應並結束。
            return await response(scope, receive, send)
        # 用 bytearray（可變的 bytes）累積整個請求 body。
        body = bytearray()
        # ASGI 的 body 是分段送來的，要一直呼叫 receive() 直到最後一段。
        while True:
            # 取得下一個訊息：通常是 {"type": "http.request", "body": ..., "more_body": ...}。
            message = await receive()
            # 客戶端在傳完 body 之前就斷線了：沒有人可以接收回應，直接結束。
            if message["type"] == "http.disconnect":
                # 不送任何回應，直接離開。
                return
            # 這一段的 body 內容；沒有 body 欄位時當作空 bytes。
            chunk = message.get("body", b"")
            # 「已累積的長度＋這一段」超過上限就立刻拒絕，不再讀剩下的資料，也不把這一段存進記憶體。
            if len(body) + len(chunk) > MAX_BODY_BYTES:
                # 413 Payload Too Large；訊息不含任何請求內容。
                response = JSONResponse({"code": "PAYLOAD_TOO_LARGE", "message": "影像超過大小限制"}, status_code=413)
                # 送出 413 回應並結束。
                return await response(scope, receive, send)
            # 還在上限內，把這一段接到累積的 body 後面。
            body.extend(chunk)
            # more_body 為 False（或沒有這個欄位）表示這是最後一段。
            if not message.get("more_body", False):
                # body 讀完了，跳出迴圈。
                break
        # 標記「完整 body 是否已經交給內層 app」；原本的 receive 已經被讀完，不能讓內層 app 再從頭讀一次。
        delivered = False

        async def bounded_receive():
            """給內層 app 用的替代 receive：第一次呼叫一次交出完整 body，之後改回原本的 receive。

            回傳：ASGI 訊息 dict。
            第一次回傳 {"type": "http.request", "body": 全部內容, "more_body": False}，
            FastAPI 因此會一次拿到整個 body 並解析 JSON。
            之後的呼叫交給原本的 receive，例如讓 Starlette 能收到 http.disconnect（客戶端斷線）訊息。
            """
            # nonlocal：宣告要修改外層函式的 delivered 變數，而不是在這裡建立同名的新區域變數。
            nonlocal delivered
            # 還沒交出 body 的話，這次就交出。
            if not delivered:
                # 記下已經交出，下一次呼叫不會重複送。
                delivered = True
                # 把 bytearray 轉成不可變的 bytes，一次送出全部內容，並用 more_body=False 表示沒有後續。
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            # body 已經交過了，之後的訊息（通常是 http.disconnect）照原本的 receive 取得。
            return await receive()

        # 所有檢查都通過：把請求交給內層 app（FastAPI），但把 receive 換成上面的 bounded_receive。
        await self.app(scope, bounded_receive, send)


def create_app(
    # 服務設定；不給時讀環境變數（正式執行）。測試會直接傳入自訂的 Settings。
    settings: Settings | None = None,
    # 真人驗證服務；不給時依設定建立。測試可以傳入使用假 provider 的服務。
    service: VerificationService | None = None,
    # AI 推薦回覆服務；不給時依設定建立。
    reply_service: ReplyAIService | None = None,
) -> FastAPI:
    """建立 FastAPI app：真人驗證與 AI 推薦回覆共用同一個內部邊界（token 與大小限制）。

    reply_service 讓測試可以注入使用假模型的服務；不給時依設定建立（模型延遲到第一次使用才建立）。

    參數：
        settings：服務設定；None 時用 `Settings.from_env()` 讀環境變數。
        service：真人驗證服務；None 時用 `VerificationService(settings)`。
        reply_service：AI 推薦回覆服務；None 時用 `build_service(settings)`。
    回傳：設定完成的 FastAPI app。
    路由：
        GET  /health                     不需 token 的健康檢查（Docker HEALTHCHECK 用）。
        POST /internal/ai/face/verify    真人驗證；需 X-Internal-Token。
        以及 install_reply_api 掛上的 /internal/ai/... 推薦回覆路由。
    設計理由：用「工廠函式」而不是直接在模組層建立，是為了讓測試能用不同設定建立多個獨立的 app。
    """
    # 呼叫方沒給設定就從環境變數讀（`a or b`：a 是 None 等「假值」時取 b，否則取 a）。
    settings = settings or Settings.from_env()
    # 建立 FastAPI app；docs_url／redoc_url／openapi_url 設為 None 會關閉自動產生的 API 文件頁與 schema，
    # 內部服務不需要對外揭露介面細節。
    app = FastAPI(title="AI 私有服務", docs_url=None, redoc_url=None, openapi_url=None)
    # 掛上內部邊界 middleware；settings=settings 會在 Starlette 建構 middleware 時傳給 InternalBoundary.__init__。
    app.add_middleware(InternalBoundary, settings=settings)
    # 真人驗證服務：有注入就用注入的（測試），否則依設定建立。
    verifier = service or VerificationService(settings)
    # AI 推薦回覆服務：有注入就用注入的（測試），否則依設定建立（模型延遲到第一次使用才建立）。
    replies = reply_service or build_service(settings)

    # 註冊「請求格式錯誤」的處理器：body 不是合法 JSON、缺欄位、多了欄位、型別不符都會走到這裡。
    @app.exception_handler(RequestValidationError)
    async def request_error(_request, _error):
        """把請求格式錯誤一律轉成固定的 422 回應。

        參數：_request（請求）與 _error（驗證錯誤）都刻意不使用，名稱前的底線表示「不會用到」。
        回傳：422 的 JSONResponse，內容固定為 {"code": "INVALID_REQUEST", "message": ...}。
        """
        # FastAPI 的預設錯誤可能包含 input；自拍與 Base64 不可回顯。
        # （預設的 422 回應會列出每個錯誤欄位及其輸入值，可能把整段自拍 Base64 或敏感內容送回去、寫進日誌。）
        return JSONResponse({"code": "INVALID_REQUEST", "message": "請求格式不正確"}, status_code=422)

    # 註冊「影像不合格」的處理器：verification.py 驗證影像失敗時丟出 InvalidImage。
    @app.exception_handler(InvalidImage)
    async def image_error(_request, error):
        """把 InvalidImage 轉成 400 回應。

        參數：
            _request：請求（不使用）。
            error：InvalidImage 例外；它的訊息就是錯誤碼，例如 INVALID_IMAGE、IMAGE_TYPE_MISMATCH、
                REFERENCE_IMAGE_TOO_LARGE、FRAME_INVALID_IMAGE_DIMENSIONS。
        回傳：400 的 JSONResponse，code 是錯誤碼，message 是固定的中文說明（不含影像內容）。
        """
        # str(error) 取出例外訊息（錯誤碼），讓 NestJS 知道是哪一張影像、哪一種問題。
        return JSONResponse({"code": str(error), "message": "影像格式、內容或尺寸不符要求"}, status_code=400)

    # 健康檢查路由；路徑不是 /internal/ 開頭，所以不需要 token（Docker HEALTHCHECK 直接呼叫它）。
    @app.get("/health")
    async def health():
        """回報服務是否活著，以及各項功能是否已設定。

        回傳：dict，FastAPI 會自動轉成 JSON：
            status：固定 "ok"（能回應就代表程序活著）。
            service：固定 "ai"，表示這是 AI 服務。
            verificationProvider：人臉驗證 provider 設定完整時是 "configured"，否則 "unavailable"。
            replySuggestions：推薦回覆各模型的設定狀態（replyModels／embedding／extraction）。
        不會實際呼叫 provider 或模型，所以很快、也不花額度。
        """
        # 回傳的 dict 就是回應 body。
        return {
            # 程序存活。
            "status": "ok",
            # 服務名稱。
            "service": "ai",
            # 只看設定（網址格式正確且有 token），不代表 provider 目前真的連得上。
            "verificationProvider": "configured" if settings.provider_configured else "unavailable",
            # 只回報「有沒有設定」，不實際呼叫模型；額度是否足夠要到 Ollama／AI Studio 的帳號頁面查看。
            "replySuggestions": replies.health(),
        }

    # 真人驗證路由。response_model=VerificationResult：FastAPI 會依這個模型檢查並輸出回應，
    # 回應只會有 VerificationResult 定義的欄位（status、reasonCode、modelName、modelVersion、兩個分數）。
    @app.post("/internal/ai/face/verify", response_model=VerificationResult)
    async def verify(request: VerificationRequest):
        """真人驗證：驗證影像後轉給人臉驗證 provider，回傳判定結果。

        參數：request——FastAPI 依型別註記自動把 JSON body 解析並驗證成 VerificationRequest
            （自拍或正面影格 imageBase64、mimeType、requestId、referenceImages、可選的 liveCapture）；
            格式不符時不會進到這裡，而是由上面的 request_error 回 422。
        回傳：VerificationResult（verified／rejected／unavailable 與原因代碼）。
        可能丟出：InvalidImage（影像不合格），由上面的 image_error 轉成 400。
        provider 沒設定、逾時、回應不合格等情況不會丟例外，而是回 status="unavailable" 與對應代碼（HTTP 200）。
        """
        # 所有實際工作都在 VerificationService.verify（見 verification.py）；await 等它完成再回傳結果。
        return await verifier.verify(request)

    # 掛上 AI 推薦回覆的路由與 AIServiceError 處理器；它們在 /internal/ai/ 底下，同樣受 InternalBoundary 保護。
    install_reply_api(app, replies)
    # 回傳組好的 app。
    return app


# 模組層級的 app：uvicorn 以 `app.main:app` 啟動時就是找這個變數。
# 注意：只要 import 這個模組就會執行 create_app()，並從環境變數讀設定。
app = create_app()
