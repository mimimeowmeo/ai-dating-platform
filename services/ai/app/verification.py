"""真人驗證：先在本服務驗證影像，再把請求轉給人臉驗證 provider，並嚴格檢查 provider 的回應。

整體流程（VerificationService.verify）：
1. validate_image：把自拍（或即時鏡頭的正面影格）、參照照片、每張動作影格都 Base64 解碼，
   確認大小、格式、尺寸、不是動畫、像素能完整解碼。不合格就丟出 InvalidImage，
   main.py 會把它轉成 HTTP 400，錯誤碼就是例外訊息（參照照片加 REFERENCE_、動作影格加 FRAME_ 前綴）。
2. 檢查 provider 設定：沒設定網址回 MODEL_NOT_CONFIGURED；網址或 token 不合法回 PROVIDER_CONFIGURATION_INVALID。
3. 用 httpx 以串流方式 POST 到 provider（AI_VERIFICATION_PROVIDER_URL），整體有逾時，回應有大小上限。
4. 用 ProviderResult 嚴格驗證回應，再去掉 provider 專用欄位，轉成 VerificationResult 回給呼叫端。
   任何網路錯誤、逾時或格式不合格，都回 status="unavailable"（fail closed：出錯時一律不算通過）。

隱私：影像只存在記憶體（BytesIO），不寫檔、不記錄 EXIF 或生物特徵，錯誤訊息也不帶輸入內容。
"""

# asyncio：Python 內建的非同步（async/await）工具；這裡用 to_thread 把同步工作丟到執行緒、用 timeout 限制總時間。
import asyncio
# base64：解碼前端送來的 Base64 影像字串。
import base64
# binascii：base64 解碼失敗時丟出的 binascii.Error 定義在這裡，用來捕捉它。
import binascii
# io：用 io.BytesIO 把位元組包成「記憶體中的檔案」，讓 Pillow 讀取而不必寫到磁碟。
import io
# json：捕捉 json.JSONDecodeError（provider 回應不是合法 JSON 時的防禦性處理）。
import json
# warnings：把 Pillow 的「解壓縮炸彈」警告升級成例外，讓超大影像直接失敗。
import warnings

# httpx：支援 async 的 HTTP 用戶端，用來呼叫 provider。
import httpx
# Pillow（PIL）：Python 常用的影像函式庫；Image 用來開啟／檢查影像，
# UnidentifiedImageError 是「認不出這是什麼影像格式」時丟出的例外。
from PIL import Image, UnidentifiedImageError
# ValidationError：Pydantic 驗證失敗時丟出的例外（provider 回應格式不合格時會遇到）。
from pydantic import ValidationError

# Settings：服務設定（provider 網址、token、逾時秒數等），定義在 config.py。
from .config import Settings
# 從 schemas.py 匯入：三種影像的大小上限，以及請求與回應的資料格式。
from .schemas import (
    MAX_FRAME_BYTES, MAX_IMAGE_BYTES, MAX_REFERENCE_BYTES, ProviderResult, VerificationRequest, VerificationResult,
)

# Pillow 偵測到的檔案格式名稱 → 對應的 MIME 類型；只允許這三種。
# 用來確認「前端宣告的 mimeType」和「檔案實際格式」一致，例如宣告 PNG 卻其實是 JPEG 就拒絕。
FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
# 影像短邊至少 64px；太小的影像無法可靠偵測人臉。
MIN_DIMENSION = 64
# 影像長邊最多 4096px；避免超大影像耗掉大量記憶體與 CPU。
MAX_DIMENSION = 4096
# 總像素數（寬 × 高）最多 1,600 萬；即使長寬都在範圍內，也限制整體解碼後的記憶體用量。
MAX_PIXELS = 16_000_000
# provider 回應 body 最多 16 KiB；正常的判定 JSON 只有幾百 bytes，超過就視為不合格，避免被塞爆記憶體。
MAX_PROVIDER_BYTES = 16 * 1024


class InvalidImage(ValueError):
    """影像不合格時丟出的例外；例外訊息（str(error)）就是要回給呼叫端的錯誤碼。

    可能的錯誤碼：INVALID_IMAGE、IMAGE_TOO_LARGE、IMAGE_TYPE_MISMATCH、INVALID_IMAGE_DIMENSIONS、
    ANIMATED_IMAGE_NOT_ALLOWED；參照照片與動作影格會再加上 REFERENCE_／FRAME_ 前綴。

    繼承 ValueError，所以 _validate_one 必須先用 `except InvalidImage: raise` 把它原樣丟出，
    否則會被後面捕捉 ValueError 的 except 吃掉、變成通用的 INVALID_IMAGE。
    main.py 註冊了這個例外的處理器，會回 HTTP 400 與 {"code": 錯誤碼}；worker.py 則把它轉成通用錯誤。
    """

    # 不需要額外的屬性或方法，繼承 ValueError 的行為即可。
    pass


def validate_image(request: VerificationRequest) -> None:
    """驗證自拍、每張參照照片與每張動作影格；後兩者的錯誤碼加上 REFERENCE_／FRAME_ 前綴，才分得出是哪一張不合格。

    參數：
        request：已通過 Pydantic 格式驗證的驗證請求（Base64 長度、mimeType 已檢查過）。

    回傳：None；全部合格就正常結束。

    可能丟出：
        InvalidImage：任何一張影像不合格。自拍的錯誤碼沒有前綴（例如 IMAGE_TOO_LARGE），
            參照照片是 REFERENCE_ 開頭（例如 REFERENCE_IMAGE_TYPE_MISMATCH），
            動作影格是 FRAME_ 開頭（例如 FRAME_INVALID_IMAGE）。遇到第一張不合格的就停止。

    設計理由：Pydantic 只能檢查字串長度與宣告的格式，無法得知 Base64 內容是不是真的影像，
    所以要實際解碼檢查；先在本服務擋掉不合格的影像，就不會浪費 provider 的運算資源。
    這是同步、吃 CPU 的函式，VerificationService 會用 asyncio.to_thread 放到背景執行緒跑。
    """
    # 驗證自拍（或即時鏡頭的正面影格），上限 5 MiB；錯誤碼不加前綴。
    _validate_one(request.imageBase64, request.mimeType, MAX_IMAGE_BYTES)
    # 逐一驗證每張參照照片（目前最多 1 張）。
    for reference in request.referenceImages:
        # 參照照片上限 1 MiB，錯誤碼加 REFERENCE_ 前綴。
        _validate_prefixed(reference.imageBase64, reference.mimeType, MAX_REFERENCE_BYTES, "REFERENCE_")
    # 只有即時鏡頭流程才有 liveCapture；只上傳自拍時是 None，就跳過動作影格的檢查。
    if request.liveCapture is not None:
        # 逐一驗證每張動作影格。
        for frame in request.liveCapture.frames:
            # 動作影格上限 1 MiB，錯誤碼加 FRAME_ 前綴。
            _validate_prefixed(frame.imageBase64, frame.mimeType, MAX_FRAME_BYTES, "FRAME_")


def _validate_prefixed(image_base64: str, mime_type: str, max_bytes: int, prefix: str) -> None:
    """驗證一張影像，失敗時在錯誤碼前面加上前綴。

    參數：
        image_base64：影像的 Base64 字串。
        mime_type：宣告的 MIME 類型。
        max_bytes：解碼後允許的最大位元組數。
        prefix：錯誤碼前綴，例如 "REFERENCE_" 或 "FRAME_"。

    回傳：None。

    可能丟出：InvalidImage，訊息是「前綴 + 原本的錯誤碼」，例如 "FRAME_IMAGE_TOO_LARGE"。

    設計理由：共用 _validate_one 的檢查邏輯，只在這裡改寫錯誤碼，前端才能提示是哪一張照片有問題。
    """
    # 嘗試用共用的檢查函式驗證這張影像。
    try:
        # 實際的檢查都在 _validate_one。
        _validate_one(image_base64, mime_type, max_bytes)
    # 捕捉不合格的例外，拿到原本的錯誤碼。
    except InvalidImage as error:
        # f-string 把前綴接在原錯誤碼前面；`from None` 表示不附上原例外的追蹤鏈，讓錯誤訊息保持乾淨。
        raise InvalidImage(f"{prefix}{error}") from None


def _validate_one(image_base64: str, mime_type: str, max_bytes: int) -> None:
    """驗證完整解碼內容，不把影像落地、不輸出 EXIF 或生物特徵。

    依序檢查：
        1. Base64 能解碼（只允許 Base64 字元）→ 否則 INVALID_IMAGE。
        2. 解碼後不是空的、也不超過 max_bytes → 否則 IMAGE_TOO_LARGE（空內容也用這個代碼）。
        3. Pillow 認得的實際格式和宣告的 mime_type 一致 → 否則 IMAGE_TYPE_MISMATCH。
        4. 短邊 ≥ 64、長邊 ≤ 4096、總像素 ≤ 1,600 萬 → 否則 INVALID_IMAGE_DIMENSIONS。
        5. 只有 1 個影格（不是 GIF／APNG／動態 WebP 之類的動畫）→ 否則 ANIMATED_IMAGE_NOT_ALLOWED。
        6. 檔案結構檢查（verify）與實際解碼全部像素（load）都成功 → 否則 INVALID_IMAGE。

    參數：
        image_base64：影像的 Base64 字串。
        mime_type：宣告的 MIME 類型（image/jpeg、image/png、image/webp）。
        max_bytes：解碼後允許的最大位元組數。

    回傳：None。

    可能丟出：InvalidImage（訊息是上面列的錯誤碼，不含任何輸入內容）。

    設計理由：影像只在記憶體中處理（BytesIO），不寫到磁碟；Pillow 的各種例外統一轉成 INVALID_IMAGE，
    不把函式庫內部的錯誤訊息（可能含檔案細節）回給呼叫端。
    """
    # 第 1 步：Base64 解碼。
    try:
        # validate=True：遇到 Base64 字母表以外的字元（例如 HTML 標籤、空白）就丟錯，而不是默默略過。
        data = base64.b64decode(image_base64, validate=True)
    # binascii.Error：Base64 格式錯誤（非法字元或 padding 不對）；ValueError：字串含非 ASCII 字元。
    except (binascii.Error, ValueError):
        # 統一回 INVALID_IMAGE；from None 不附上原例外，避免輸入內容出現在錯誤鏈中。
        raise InvalidImage("INVALID_IMAGE") from None
    # 第 2 步：解碼後是空的，或超過這種影像的大小上限。
    if not data or len(data) > max_bytes:
        # 回 IMAGE_TOO_LARGE（空內容也歸在這個代碼）。
        raise InvalidImage("IMAGE_TOO_LARGE")
    # 第 3～6 步：用 Pillow 開啟並檢查影像；任何 Pillow 錯誤都在下面的 except 統一處理。
    try:
        # catch_warnings 建立一個暫時的警告設定範圍，離開 with 區塊後恢復原本設定，不影響程式其他地方。
        with warnings.catch_warnings():
            # 把 DecompressionBombWarning（像素數超過 Pillow 的安全門檻，預設約 8,900 萬）升級成例外，
            # 讓「檔案很小、解壓後卻極大」的解壓縮炸彈在開檔時就失敗；超過門檻兩倍時 Pillow 本來就會丟 DecompressionBombError。
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            # 從記憶體開啟影像；Image.open 是「延遲載入」，這時只讀檔頭（格式、尺寸），還沒解碼像素。
            # with 區塊結束時會自動關閉影像、釋放資源。
            with Image.open(io.BytesIO(data)) as image:
                # 從檔頭取得寬與高（像素）。
                width, height = image.size
                # 第 3 步：image.format 是 Pillow 偵測到的實際格式（例如 "PNG"）；
                # 查不到（例如 GIF、BMP）時 FORMATS.get 回 None，也會和宣告的 mime_type 不相等。
                if FORMATS.get(image.format) != mime_type:
                    # 宣告的格式和實際不一致（或不是允許的格式）。
                    raise InvalidImage("IMAGE_TYPE_MISMATCH")
                # 第 4 步：尺寸檢查，三個條件任一成立就不合格。
                if (
                    # 短邊小於 64px：太小，無法可靠偵測人臉。
                    min(width, height) < MIN_DIMENSION
                    # 長邊大於 4096px：太大。
                    or max(width, height) > MAX_DIMENSION
                    # 總像素超過 1,600 萬：解碼後太耗記憶體。
                    or width * height > MAX_PIXELS
                ):
                    # 尺寸不合格。
                    raise InvalidImage("INVALID_IMAGE_DIMENSIONS")
                # 第 5 步：n_frames 是影格數；沒有這個屬性的格式用 getattr 的預設值 1（視為單張）。
                if getattr(image, "n_frames", 1) != 1:
                    # 動畫影像（多個影格）不接受，驗證需要的是單一張照片。
                    raise InvalidImage("ANIMATED_IMAGE_NOT_ALLOWED")
                # 第 6 步（前半）：verify 檢查檔案結構是否完整，例如 PNG 會檢查每個區塊的 CRC 檢查碼；
                # 它不會解碼像素，而且依 Pillow 的規定，verify 之後同一個影像物件不能再拿來讀像素。
                image.verify()
            # verify 檢查結構，load 確認壓縮像素可實際完整解碼。
            # （verify 對 JPEG、WebP 幾乎不做檢查，截斷或損壞的像素資料只有真的解碼才會發現，
            # 而且 verify 後必須重新開檔，所以這裡再開一次。這次仍在 catch_warnings 範圍內。）
            with Image.open(io.BytesIO(data)) as image:
                # 實際解碼全部像素；資料不完整時會丟出 OSError 之類的例外。
                image.load()
    # InvalidImage 是 ValueError 的子類別，必須先原樣丟出，否則會被下一個 except 的 ValueError 捕捉而變成 INVALID_IMAGE。
    except InvalidImage:
        # 保留上面設定的具體錯誤碼。
        raise
    # 其他 Pillow 可能丟出的錯誤都視為「不是合法影像」：
    except (
        # UnidentifiedImageError：認不出格式；OSError：檔案損壞或截斷；SyntaxError、ValueError：解析檔頭失敗等。
        UnidentifiedImageError, OSError, SyntaxError, ValueError,
        # DecompressionBombError：像素數遠超安全門檻；DecompressionBombWarning：上面升級成例外的警告。
        Image.DecompressionBombError, Image.DecompressionBombWarning,
    ):
        # 統一回 INVALID_IMAGE，不附原例外，避免把函式庫內部訊息傳出去。
        raise InvalidImage("INVALID_IMAGE") from None


def unavailable(reason: str) -> VerificationResult:
    """建立一個「無法判定」的結果。

    參數：
        reason：原因代碼，例如 MODEL_NOT_CONFIGURED、PROVIDER_TIMEOUT、PROVIDER_INVALID_RESPONSE。

    回傳：status="unavailable" 的 VerificationResult；modelName、modelVersion、分數都是 None。

    可能丟出：ValidationError（只有 reason 不符合 reasonCode 格式時；這個檔案傳入的都是固定合法字串）。

    設計理由：本服務自己無法完成驗證時（沒設定、網路錯誤、回應不合格），一律回 unavailable，
    絕不回 verified，確保出錯時不會讓人意外通過（fail closed）。
    """
    # 用 Pydantic model 建立結果，建立時會檢查 reasonCode 的格式。
    return VerificationResult(status="unavailable", reasonCode=reason)


class VerificationService:
    """真人驗證服務：驗證影像後呼叫外部 provider，並把 provider 的決策安全地轉成公開結果。

    main.py 的 /internal/ai/face/verify 路由與 worker.py 的佇列 worker 都用這個類別。
    本服務不做人臉辨識、也不自己訂分數門檻，判定完全交給 provider（services/face）。
    """

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        """建立服務。

        參數：
            settings：服務設定，會用到 provider_url、provider_token、provider_timeout_seconds
                與 provider_configured。
            transport：可選的 httpx 傳輸層。正式執行時是 None（使用真的網路）；
                測試會傳入 httpx.MockTransport，讓請求不真的連網、直接回傳假回應。

        回傳：無（建構子）。
        """
        # 保存設定，verify 時讀取。
        self.settings = settings
        # 保存傳輸層（None 代表使用 httpx 預設的真實網路連線）。
        self.transport = transport

    async def verify(self, request: VerificationRequest) -> VerificationResult:
        """執行一次真人驗證。

        參數：
            request：已通過 Pydantic 格式驗證的驗證請求。

        回傳：VerificationResult。provider 正常回應時是它的判定（verified／rejected／unavailable），
            其他狀況都是 unavailable，reasonCode 可能是：
            - MODEL_NOT_CONFIGURED：沒有設定 provider 網址。
            - PROVIDER_CONFIGURATION_INVALID：網址或 token 設定不完整或不合法。
            - PROVIDER_UNAVAILABLE：HTTP 狀態不是 200（包含轉址），或網路錯誤。
            - PROVIDER_INVALID_RESPONSE：回應不是 JSON、超過 16 KiB，或內容不符合 ProviderResult。
            - PROVIDER_TIMEOUT：超過 provider_timeout_seconds。

        可能丟出：InvalidImage（影像不合格，由 validate_image 丟出，不在這裡捕捉，交給 main.py 回 400）。

        設計理由：影像檢查放在 try 之外，讓「使用者的影像有問題」（400）和「provider 出問題」（unavailable）
        分得清楚；provider 相關的每一種失敗都 fail closed，而且不把 provider 的錯誤訊息回給呼叫端。
        """
        # 影像檢查是同步、吃 CPU 的工作；用 to_thread 放到背景執行緒執行，避免卡住 async 事件迴圈（event loop），
        # 讓同一時間的其他請求還能被處理。await 會等它做完；不合格時 InvalidImage 會從這裡往外丟。
        await asyncio.to_thread(validate_image, request)
        # 沒設定 provider 網址：這個環境沒有接驗證模型，不發出任何網路請求。
        if not self.settings.provider_url:
            # 回 unavailable / MODEL_NOT_CONFIGURED。
            return unavailable("MODEL_NOT_CONFIGURED")
        # 有網址但設定不完整或不安全（不是 http(s)、含帳密或 #片段、缺 token），同樣不發請求。
        if not self.settings.provider_configured:
            # 回 unavailable / PROVIDER_CONFIGURATION_INVALID。
            return unavailable("PROVIDER_CONFIGURATION_INVALID")
        # 下面所有網路與解析錯誤都在最後的 except 轉成 unavailable。
        try:
            # 同時限制整個請求時間，避免 provider 持續零碎回應延長等待。
            # （httpx 的 timeout 是「每個步驟」的逾時，例如每次讀取之間的等待；對方每隔一下送一點資料就能一直拖下去。
            # asyncio.timeout 則限制整段 with 區塊的總時間，超過就取消並丟出 TimeoutError。）
            async with asyncio.timeout(self.settings.provider_timeout_seconds):
                # 建立 async HTTP 用戶端；離開 with 區塊時自動關閉連線。
                async with httpx.AsyncClient(
                    # 測試時是 MockTransport；正式執行是 None，使用真實網路。
                    transport=self.transport,
                    # 連線、讀取、寫入、連線池等待各自的逾時秒數。
                    timeout=self.settings.provider_timeout_seconds,
                    # 不自動跟隨轉址：避免帶著 provider token 被導到其他網址；3xx 會在下面被當成非 200 處理。
                    follow_redirects=False,
                    # 不讀取環境變數中的代理（HTTP_PROXY 等）、憑證路徑與 .netrc 設定，
                    # 讓請求只照程式指定的方式直接送到 provider，避免被環境設定改變去向或加上額外認證。
                    trust_env=False,
                ) as client:
                    # 用串流（stream）方式送出請求：回應 body 不會一次全部讀進記憶體，下面可以邊讀邊檢查大小。
                    async with client.stream(
                        # HTTP 方法與 provider 網址。
                        "POST", self.settings.provider_url,
                        # 沒有即時鏡頭時省略 liveCapture，不送出 null，和只上傳自拍時的請求格式一致。
                        # （model_dump 把 Pydantic 物件轉成 dict；exclude_none=True 會省略值是 None 的欄位，
                        # 這個請求中只有 liveCapture 可能是 None。httpx 會把 dict 轉成 JSON 並設定 Content-Type。）
                        json=request.model_dump(exclude_none=True),
                        # 請求標頭：
                        headers={
                            # 用 Bearer token 向 provider 證明身分；provider 會比對這個值。
                            "Authorization": f"Bearer {self.settings.provider_token}",
                            # 告訴 provider 我們只接受 JSON 回應。
                            "Accept": "application/json",
                        },
                    ) as response:
                        # 狀態碼不是 200（包含 3xx 轉址、401 認證失敗、500 伺服器錯誤）都視為 provider 無法使用。
                        if response.status_code != 200:
                            # 直接回 unavailable；離開 with 區塊時串流與連線會自動關閉。
                            return unavailable("PROVIDER_UNAVAILABLE")
                        # 取出 Content-Type 的主要部分（去掉 "; charset=utf-8" 之類的參數與前後空白），必須剛好是 application/json。
                        if response.headers.get("content-type", "").split(";")[0].strip() != "application/json":
                            # 不是 JSON 就不嘗試解析。
                            return unavailable("PROVIDER_INVALID_RESPONSE")
                        # bytearray 是可變的位元組陣列，用來一塊一塊累積回應內容。
                        payload = bytearray()
                        # aiter_bytes 逐塊讀取回應 body（已經過 gzip 等內容編碼的解壓縮），所以上限檢查的是解壓後的大小。
                        async for chunk in response.aiter_bytes():
                            # 加上這一塊之後會超過 16 KiB，就立刻停止讀取，不把超大回應留在記憶體裡。
                            if len(payload) + len(chunk) > MAX_PROVIDER_BYTES:
                                # 過大的回應視為不合格。
                                return unavailable("PROVIDER_INVALID_RESPONSE")
                            # 還沒超過上限，把這一塊接到 payload 後面。
                            payload.extend(chunk)
            # 已離開串流與逾時區塊：把 JSON 位元組解析並驗證成 ProviderResult。
            # strict、extra="forbid" 與 require_verified_evidence 規則不成立時會丟出 ValidationError。
            result = ProviderResult.model_validate_json(payload)
            # 轉成公開的 VerificationResult 回傳：
            return VerificationResult.model_validate(result.model_dump(exclude={
                # 去掉 provider 專用的三個決策欄位；VerificationResult 設了 extra="forbid"，不去掉會驗證失敗，
                # 而且這些內部判定細節也不需要回給 NestJS。
                "verificationType", "livenessVerified", "identityVerified",
            }))
        # 逾時：httpx 的單一步驟逾時（httpx.TimeoutException），或 asyncio.timeout 的總時間逾時（TimeoutError）。
        # 必須排在下一個 except 之前：httpx.TimeoutException 是 httpx.HTTPError 的子類別，TimeoutError 是 OSError 的子類別。
        except (httpx.TimeoutException, TimeoutError):
            # 回 PROVIDER_TIMEOUT；不帶原例外訊息，避免 provider 的錯誤細節外洩。
            return unavailable("PROVIDER_TIMEOUT")
        # 其他網路層錯誤：連不上、連線中斷、DNS 失敗等。
        except (httpx.HTTPError, OSError):
            # 回 PROVIDER_UNAVAILABLE。
            return unavailable("PROVIDER_UNAVAILABLE")
        # 回應內容不合格：Pydantic 驗證失敗（包含 JSON 語法錯誤，Pydantic 也是用 ValidationError 回報）；
        # 其他幾種是防禦性捕捉（JSON 解碼、文字編碼錯誤，以及一般的 ValueError）。
        except (ValidationError, json.JSONDecodeError, UnicodeError, ValueError):
            # 回 PROVIDER_INVALID_RESPONSE。
            return unavailable("PROVIDER_INVALID_RESPONSE")
