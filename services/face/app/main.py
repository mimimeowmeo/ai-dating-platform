"""人臉驗證 provider 的 HTTP 入口（FastAPI app）。

這個檔案負責「對外的 HTTP 介面」，真正的判定邏輯在 pipeline.py（FaceVerifier）：
- ProviderBoundary：ASGI middleware。在 FastAPI 解析 JSON 之前，先檢查
  `Authorization: Bearer <token>` 與請求本文（body）大小，擋掉未授權或過大的請求。
- create_app()：建立 FastAPI app，註冊錯誤處理、`GET /health` 與 `POST /verify` 兩個路由。
- 模組最後的 `app = create_app()`：給 uvicorn 用（Dockerfile 的 CMD 是 `uvicorn app.main:app`）。

呼叫方是 AI 私有服務（services/ai 的 verification.py）：它把自拍／正面影格、主照片、
動作影格 POST 到 /verify，並只接受 HTTP 200 且格式完全符合 ProviderResult 的回應；
任何非 200 的狀態碼，AI 服務都會記成 PROVIDER_UNAVAILABLE。
因此這裡的設計原則是 fail closed：模型沒載入、token 沒設定、模型執行出錯時，
一律回錯誤狀態碼，絕不回一個「看起來通過」的結果。
"""

# asyncio：Python 內建的非同步（async/await）工具；這裡用 asyncio.to_thread 把耗 CPU 的模型推論丟到執行緒執行。
import asyncio
# secrets：Python 內建的安全相關工具；這裡用 compare_digest 做「固定時間」的字串比對，避免時序攻擊（timing attack）。
import secrets

# FastAPI：Web 框架本體，負責路由、請求解析（用 pydantic 驗證 JSON）與回應序列化。
from fastapi import FastAPI
# RequestValidationError：請求 JSON 不符合 pydantic schema（VerifyRequest）時，FastAPI 丟出的例外。
from fastapi.exceptions import RequestValidationError
# JSONResponse：Starlette（FastAPI 底層框架）的 JSON 回應物件，可以自己指定 status code。
from starlette.responses import JSONResponse

# policy：判定政策（門檻、模型名稱與版本字串等常數），見 policy.py。
from . import policy
# Settings：provider 設定（token、模型資料夾），見 config.py。
from .config import Settings
# InvalidImage：影像解碼或檢查失敗時丟出的例外，訊息本身就是錯誤碼（例如 "IMAGE_TOO_LARGE"），見 imaging.py。
from .imaging import InvalidImage
# load_models：從資料夾載入 YuNet、SFace、MiniFASNet 三組模型，見 models.py。
from .models import load_models
# FaceVerifier：整個判定流程（偵測、動作挑戰、防偽、比對、政策判定），見 pipeline.py。
from .pipeline import FaceVerifier
# MAX_BODY_BYTES：整個請求本文的位元組上限；VerifyRequest／VerifyResponse：請求與回應的 pydantic 模型（HTTP 契約），見 schemas.py。
from .schemas import MAX_BODY_BYTES, VerifyRequest, VerifyResponse

# 驗證端點的路徑；middleware 與路由共用同一個常數，確保「受保護的路徑」與「實際路由」不會寫得不一致。
VERIFY_PATH = "/verify"


class ProviderBoundary:
    """先驗證 Bearer token 與串流大小，再允許框架解析 JSON（同 services/ai 的 InternalBoundary 的做法）。

    這是一個「純 ASGI middleware」。ASGI 是 Python 非同步 Web 伺服器（uvicorn）與框架
    （FastAPI／Starlette）之間的介面：每個請求會以 `(scope, receive, send)` 三個參數呼叫 app。
    - scope：這次連線的描述（dict），例如 type（"http"、"websocket"、"lifespan"）、path、headers。
    - receive：非同步函式，每呼叫一次拿到一段請求本文（body 可能分成好幾段送達）。
    - send：非同步函式，用來送出回應。

    services/ai 的 InternalBoundary 用的是 X-Internal-Token 標頭；這裡是 provider 契約，
    AI 服務呼叫時帶的是 `Authorization: Bearer <token>`，所以檢查的是 Authorization。

    為什麼要自己寫 middleware，而不是在路由函式裡檢查？
    因為 FastAPI 會在進入路由函式「之前」就把整個 JSON 讀進來並解析。若在路由裡才檢查 token，
    未授權的請求也能讓伺服器讀完、解析一大包 Base64 影像。放在 middleware 可以：
    1. 沒帶正確 token 就直接回 401，根本不讀本文。
    2. 邊讀邊累計大小，一超過 MAX_BODY_BYTES 就回 413，不會先把超大本文整個放進記憶體。

    只攔截路徑是 /verify 的 HTTP 請求（不論 HTTP 方法）；/health 與其他路徑直接放行（/health 要給 Docker 健康檢查用，不需要 token）。

    可能的回應（皆為 JSON，body 格式為 {"code", "message"}）：
    - 503 PROVIDER_AUTH_NOT_CONFIGURED：伺服器沒設定 token（fail closed，不會變成「不需要驗證」）。
    - 401 UNAUTHORIZED：Authorization 標頭不對。
    - 413 PAYLOAD_TOO_LARGE：本文超過 MAX_BODY_BYTES。
    """

    def __init__(self, app, settings: Settings):
        """建立 middleware。

        參數：
            app：被包在裡面的下一層 ASGI app（也就是 FastAPI 本身）；檢查通過後要把請求交給它。
            settings：provider 設定，這裡只用到 settings.token。

        回傳：無（建構子）。
        由 FastAPI 的 `app.add_middleware(ProviderBoundary, settings=settings)` 呼叫，不需要手動建立。
        """
        # 記住下一層 app，檢查通過（或路徑不需檢查）時要呼叫它。
        self.app = app
        # 預先組好「預期的 Authorization 標頭值」並轉成 bytes（ASGI 的標頭值是 bytes，不是 str）。
        # 沒設定 token 時存成空 bytes（b""），後面用它判斷「尚未設定」並一律拒絕，避免空 token 被當成合法。
        self.expected = f"Bearer {settings.token}".encode("utf-8") if settings.token else b""

    async def __call__(self, scope, receive, send):
        """處理一個 ASGI 連線；定義 __call__ 讓這個物件本身可以被當成 ASGI app 呼叫。

        參數：
            scope：連線資訊（dict），含 type、path、headers 等。
            receive：讀取請求訊息（本文片段或斷線通知）的非同步函式。
            send：送出回應訊息的非同步函式。

        回傳：None。回應是透過 send 送出，不是用 return 的值。
        錯誤：這一層本身不主動丟例外；各種拒絕情況都直接送出對應的 JSON 錯誤回應。
        """
        # 不是 HTTP 請求（例如 lifespan 啟動／關閉事件），或路徑不是 /verify：不需檢查，原封不動交給下一層。
        # 注意 `or` 會短路：type 不是 "http" 時不會去讀 scope["path"]（lifespan 的 scope 沒有 path）。
        if scope["type"] != "http" or scope["path"] != VERIFY_PATH:
            # 直接把三個參數交給 FastAPI 處理，這個 middleware 不介入。
            return await self.app(scope, receive, send)
        # 伺服器沒設定 token（FACE_PROVIDER_TOKEN 為空）：不能讓任何人呼叫 /verify，回 503 表示「服務尚未就緒」。
        if not self.expected:
            # 建立 503 的 JSON 錯誤回應；code 給程式判斷，message 是給人看的說明。
            response = JSONResponse({"code": "PROVIDER_AUTH_NOT_CONFIGURED", "message": "provider 尚未設定"}, status_code=503)
            # Starlette 的 Response 物件本身也是 ASGI app，呼叫它就會透過 send 把回應送出去；送完就結束這個請求。
            return await response(scope, receive, send)
        # ASGI 的 headers 是 [(名稱 bytes, 值 bytes), ...] 的清單，名稱一律小寫；轉成 dict 方便用名稱查詢。
        # （同名標頭出現多次時，dict 會保留最後一個。）
        headers = dict(scope.get("headers", []))
        # 用 secrets.compare_digest 比對 Authorization 標頭與預期值：
        # 一般的 == 遇到第一個不同的位元組就會提早結束，攻擊者可從回應時間推測 token 內容；compare_digest 的耗時不受內容影響。
        # 沒帶 Authorization 標頭時用 b"" 代替，比對必然失敗。
        if not secrets.compare_digest(headers.get(b"authorization", b""), self.expected):
            # token 不對：回 401，而且完全沒有讀取本文。
            response = JSONResponse({"code": "UNAUTHORIZED", "message": "需要 provider 認證"}, status_code=401)
            # 送出 401 回應並結束。
            return await response(scope, receive, send)
        # 用來累積請求本文的可變位元組陣列（bytearray 可以原地 extend，不必每次產生新的 bytes）。
        body = bytearray()
        # 本文可能分成多段送達，所以用迴圈一直讀，直到最後一段。
        while True:
            # 讀取下一個 ASGI 訊息：通常是 {"type": "http.request", "body": ..., "more_body": ...}。
            message = await receive()
            # 客戶端中途斷線：已經沒有人可以接收回應，直接結束，不送任何東西。
            if message["type"] == "http.disconnect":
                # 結束這個請求的處理。
                return
            # 取出這一段的本文內容；沒有 body 欄位時當成空的 bytes。
            chunk = message.get("body", b"")
            # 先檢查「已累積＋這一段」是否會超過上限，超過就不再往下讀（不把超大本文放進記憶體）。
            # 這裡算的是實際收到的位元組，不相信客戶端自己宣稱的 Content-Length。
            if len(body) + len(chunk) > MAX_BODY_BYTES:
                # 回 413 Payload Too Large。
                response = JSONResponse({"code": "PAYLOAD_TOO_LARGE", "message": "影像超過大小限制"}, status_code=413)
                # 送出 413 回應並結束。
                return await response(scope, receive, send)
            # 沒超過上限，把這一段接到累積的本文後面。
            body.extend(chunk)
            # more_body 為 False（或不存在）代表這是最後一段，本文讀完了，跳出迴圈。
            if not message.get("more_body", False):
                # 離開 while 迴圈。
                break
        # 旗標：記錄「已經讀好的完整本文」是否已經交給下一層了。
        delivered = False

        async def bounded_receive():
            """替換原本的 receive，交給下一層（FastAPI）使用。

            原本的 receive 裡的本文已經被上面的迴圈讀光了，不能再讀一次；
            所以第一次被呼叫時，一次回傳整份已讀好（而且確認過大小）的本文。
            之後再被呼叫（例如框架在等待斷線通知），就轉回原本的 receive。

            回傳：ASGI 訊息 dict。
            """
            # delivered 是外層函式的區域變數；要在內層函式裡「重新指定」它，必須宣告 nonlocal，否則 Python 會把它當成新的區域變數。
            nonlocal delivered
            # 第一次呼叫：還沒交出本文。
            if not delivered:
                # 標記為已交出，之後的呼叫不會再重複回傳本文。
                delivered = True
                # 組出一個 ASGI 的 http.request 訊息：整份本文（轉成不可變的 bytes），並以 more_body=False 表示沒有下一段。
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            # 已經交出過本文：之後的呼叫交給原本的 receive（例如收到 http.disconnect）。
            return await receive()

        # 檢查都通過：把請求交給 FastAPI，但 receive 換成 bounded_receive，讓它讀到我們已緩衝好的本文。
        await self.app(scope, bounded_receive, send)


def create_app(settings: Settings | None = None, verifier: FaceVerifier | None = None) -> FastAPI:
    """建立 provider app；verifier 讓測試注入假模型，不給時從 FACE_MODEL_DIR 載入真模型。

    參數：
        settings：provider 設定。沒給（None）時用 Settings.from_env() 從環境變數讀
            （FACE_PROVIDER_TOKEN、FACE_MODEL_DIR）；測試可直接傳入自己建的 Settings。
        verifier：判定流程物件。沒給時載入真正的模型檔建立；測試可傳入用假模型建好的 FaceVerifier，
            就不需要真的模型檔。

    回傳：
        設定好 middleware、錯誤處理與路由的 FastAPI app。

    錯誤：
        不會因為模型載入失敗而丟例外。模型缺漏或載入失敗時 app 照樣建立，
        但 verifier 維持 None：/health 回 503，/verify 回 503 MODELS_UNAVAILABLE（fail closed）。
        這樣服務仍能啟動、健康檢查能反映「模型不可用」，而不是整個容器反覆崩潰重啟。
    """
    # 沒有傳入 settings 就從環境變數讀（`a or b`：a 是 None 時取 b）。
    settings = settings or Settings.from_env()
    # 建立 FastAPI app。docs_url、redoc_url、openapi_url 設成 None 會關掉自動產生的
    # Swagger UI（/docs）、ReDoc（/redoc）與 OpenAPI 規格（/openapi.json）：這是內部服務，不需要對外公開 API 文件。
    app = FastAPI(title="人臉驗證 provider", docs_url=None, redoc_url=None, openapi_url=None)
    # 掛上 ProviderBoundary middleware；FastAPI 會以 ProviderBoundary(下一層 app, settings=settings) 的方式建立它，
    # 讓每個請求都先經過 token 與大小檢查。
    app.add_middleware(ProviderBoundary, settings=settings)
    # 沒有注入 verifier（正式執行的情況）：自己載入模型。
    if verifier is None:
        # 模型載入可能失敗（檔案不存在、檔案損毀、OpenCV／onnxruntime 讀不進來），用 try 包起來。
        try:
            # load_models 讀取模型資料夾內的 YuNet、SFace、MiniFASNet 檔案；YuNet 的偵測信心門檻取自 policy。
            # 再把載入好的模型交給 FaceVerifier，得到可以執行 verify 的物件。
            verifier = FaceVerifier(load_models(settings.model_dir, policy.DETECTION_SCORE_THRESHOLD))
        # 接住所有例外（Exception）：不論哪種載入失敗，都不讓服務啟動失敗。
        except Exception:
            # 模型檔缺漏或載入失敗：服務照常啟動，但 /health 回 503、/verify 一律 fail closed。
            # 只印一個固定的代碼字串到標準輸出（會進容器 log）；flush=True 讓它立刻寫出，不留在緩衝區。
            print("FACE_MODELS_UNAVAILABLE", flush=True)

    # 註冊錯誤處理：請求 JSON 不符合 VerifyRequest（缺欄位、型別不對、多了欄位、字串過長等）時由這個函式產生回應。
    @app.exception_handler(RequestValidationError)
    async def request_error(_request, _error):
        """把 pydantic 驗證錯誤轉成固定內容的 422 回應。

        參數：
            _request：觸發錯誤的請求（沒用到，底線開頭表示刻意不使用）。
            _error：RequestValidationError（沒用到）。

        回傳：422 JSON 回應，code 固定為 INVALID_REQUEST。
        """
        # FastAPI 的預設錯誤可能包含 input；自拍與 Base64 不可回顯。
        # （預設的 422 回應會列出每個錯誤欄位的輸入值，可能把整段 Base64 影像原樣送回去，所以改成固定訊息。）
        return JSONResponse({"code": "INVALID_REQUEST", "message": "請求格式不正確"}, status_code=422)

    # 註冊錯誤處理：影像解碼或檢查失敗（InvalidImage）時由這個函式產生回應。
    @app.exception_handler(InvalidImage)
    async def image_error(_request, error):
        """把 InvalidImage 轉成 400 回應，錯誤碼沿用例外訊息。

        參數：
            _request：觸發錯誤的請求（沒用到）。
            error：InvalidImage 例外；它的訊息就是錯誤碼，例如 "IMAGE_TYPE_MISMATCH"，
                參照照片與動作影格的錯誤會分別帶 "REFERENCE_"、"FRAME_" 前綴（見 pipeline.py）。

        回傳：400 JSON 回應。
        """
        # str(error) 取出例外訊息（也就是錯誤碼）放進 code；message 用固定的中文說明，不帶任何影像內容。
        return JSONResponse({"code": str(error), "message": "影像格式、內容或尺寸不符要求"}, status_code=400)

    # 註冊 GET /health 路由：給 Docker HEALTHCHECK（見 Dockerfile）與維運檢查用；不經過 token 檢查。
    @app.get("/health")
    async def health():
        """回報服務與模型狀態。

        回傳：
            模型已載入：200，{"status": "ok", "service": "face", "models": "loaded", "modelVersion": ...}。
            模型未載入：503，{"status": "error", "service": "face", "models": "unavailable"}。
        """
        # verifier 是外層 create_app 的變數（閉包）；是 None 代表模型載入失敗。
        if verifier is None:
            # 回 503，讓 Docker 健康檢查判定為不健康，維運能看出「模型不可用」。
            return JSONResponse({"status": "error", "service": "face", "models": "unavailable"}, status_code=503)
        # 模型可用：回傳 dict，FastAPI 會自動轉成 200 的 JSON；附上模型與政策版本，方便確認部署的是哪一版。
        return {"status": "ok", "service": "face", "models": "loaded", "modelVersion": policy.MODEL_VERSION}

    # 註冊 POST /verify 路由。response_model=VerifyResponse：FastAPI 會用 VerifyResponse 驗證並序列化回傳值，
    # 確保送出去的欄位和 AI 服務的 ProviderResult 一致（直接回 JSONResponse 時則不經過這個驗證）。
    @app.post(VERIFY_PATH, response_model=VerifyResponse)
    async def verify(request: VerifyRequest):
        """執行真人驗證。

        參數：
            request：已通過 pydantic 驗證的 VerifyRequest（FastAPI 依型別註記自動從 JSON 本文解析）。
                內含 imageBase64（自拍或正面影格）、mimeType、requestId、referenceImages（主照片）
                與 liveCapture（即時鏡頭的動作影格，只上傳自拍時為 None）。

        回傳：
            正常：VerifyResponse（status 為 verified／rejected／unavailable 之一），HTTP 200。
            模型未載入：503 MODELS_UNAVAILABLE。
            模型執行出錯：503 MODEL_FAILED。

        錯誤：
            InvalidImage：影像不合格時再往外丟，交給上面的 image_error 轉成 400。
        """
        # 模型沒載入：不能判定，回 503（fail closed），不回任何判定結果。
        if verifier is None:
            # 回 503；AI 服務收到非 200 會記成 PROVIDER_UNAVAILABLE。
            return JSONResponse({"code": "MODELS_UNAVAILABLE", "message": "模型尚未載入"}, status_code=503)
        # 執行判定；可能丟出 InvalidImage 或模型執行時的各種例外，用 try 分別處理。
        try:
            # verifier.verify 是一般（同步）函式，而且是吃 CPU 的影像解碼與模型推論。
            # 直接在 async 函式裡呼叫會卡住整個事件迴圈（event loop），期間其他請求（包括 /health）都無法處理；
            # asyncio.to_thread 把它丟到另一個執行緒執行，await 等它完成並取得回傳的 VerifyResponse。
            # （同一組模型不能多執行緒同時使用，FaceVerifier 內部用 lock 保證一次只跑一個判定。）
            return await asyncio.to_thread(verifier.verify, request)
        # 影像不合格：這是客戶端的輸入問題，不是模型壞掉。
        except InvalidImage:
            # 原樣再丟出去，讓 image_error 處理成帶錯誤碼的 400，而不是被下面的 except 吃成 503。
            raise
        # 其他任何例外（OpenCV、onnxruntime、numpy 等執行時錯誤）。
        except Exception:
            # 模型執行中的任何錯誤都 fail closed；AI 服務收到非 200 會記成 PROVIDER_UNAVAILABLE。
            # 只印固定的代碼字串到 log（flush=True 立刻寫出），不印例外內容。
            print("FACE_MODEL_FAILED", flush=True)
            # 回 503 MODEL_FAILED，絕不回一個「通過」的結果。
            return JSONResponse({"code": "MODEL_FAILED", "message": "模型執行失敗"}, status_code=503)

    # 回傳設定完成的 app。
    return app


# 模組層級的 app：uvicorn 以 "app.main:app" 匯入這個模組時使用（見 Dockerfile 的 CMD）。
# 匯入時就會執行 create_app()：從環境變數讀設定並嘗試載入模型；測試則另外呼叫 create_app(settings, verifier) 建立自己的 app。
app = create_app()
