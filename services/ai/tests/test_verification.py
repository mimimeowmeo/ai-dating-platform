"""AI 私有服務「真人驗證」的單元測試（unittest）。

執行方式（在 services/ai 目錄）：`.venv/bin/python -m unittest discover -s tests -v`。
這些測試不需要網路、Redis 或真的模型：
- ApiTests：用 FastAPI 的 TestClient 在同一個程序裡呼叫 app（不開真的 HTTP 埠），
  檢查 main.py 的 InternalBoundary（內部 token、body 大小上限）、schemas.py 的欄位限制、
  verification.py 的影像檢查，以及「沒有模型時絕不會回 verified」。
- ProviderTests：用 httpx.MockTransport 假裝成人臉驗證 provider（services/face），
  檢查 VerificationService 送給 provider 的請求格式，以及對 provider 回應的嚴格檢查
  （任何缺證據、型別不對、多欄位、逾時、錯誤狀態碼都要「失敗即關閉」（fail closed），回 unavailable）。
- WorkerTests：用假的 job 物件測 worker.py 的 BullMQ 處理函式與心跳檔邏輯。

對前端工程師的對照：unittest.TestCase 類似 Jest 的 describe，每個 test_ 開頭的 method 類似一個 it()；
self.assertEqual(a, b) 類似 expect(a).toBe(b)；self.subTest(...) 讓同一個測試跑多組資料時，
每一組各自報告成功或失敗（類似 it.each），其中一組失敗也會繼續跑其他組。
"""

# asyncio：Python 的非同步框架；這裡用 asyncio.Event 控制心跳迴圈何時停止。
import asyncio
# base64：影像位元組與 Base64 字串互轉。
import base64
# io：BytesIO，在記憶體中當作檔案使用，讓 PIL 存圖片而不必寫硬碟。
import io
# json：手動把 dict 轉成 JSON 字串，或把請求 body 解析回 dict。
import json
# tempfile：建立測試用的暫存資料夾，結束後自動刪除（心跳檔測試用）。
import tempfile
# unittest：Python 內建的測試框架。
import unittest
# Path：物件化的檔案路徑，可以用 / 串接路徑。
from pathlib import Path
# SimpleNamespace：可以隨意加屬性的簡單物件，用來假造 BullMQ 的 job、worker 與 Redis client。
from types import SimpleNamespace
# AsyncMock：可以被 await 的假函式，會記錄被呼叫的參數與次數，類似 Jest 的 jest.fn()。
from unittest.mock import AsyncMock

# httpx：HTTP 用戶端；這裡主要用 httpx.MockTransport 攔截請求、回傳假的 provider 回應。
import httpx
# TestClient：FastAPI（Starlette）提供的測試用戶端，直接在程序內呼叫 app，不需要啟動伺服器。
from fastapi.testclient import TestClient
# PIL（Pillow）的 Image：產生測試用的圖片。
from PIL import Image

# Settings：AI 服務的設定（不可變的 dataclass），測試直接建構、只填需要的欄位。
from app.config import Settings
# create_app：建立 FastAPI app 的工廠函式，可以傳入測試用的設定。
from app.main import create_app
# 從 schemas.py 匯入大小上限常數與請求 model（括號讓 import 可以跨多行）。
from app.schemas import (
    # MAX_BASE64_LENGTH：自拍 Base64 最大長度；MAX_BODY_BYTES：HTTP body 總上限；
    # MAX_FRAME_BASE64_LENGTH／MAX_REFERENCE_BASE64_LENGTH：動作影格／參照照片的 Base64 最大長度；
    # VerificationRequest：驗證請求的 Pydantic model。
    MAX_BASE64_LENGTH, MAX_BODY_BYTES, MAX_FRAME_BASE64_LENGTH, MAX_REFERENCE_BASE64_LENGTH, VerificationRequest,
)
# VerificationService：驗證影像、呼叫 provider、檢查 provider 回應的服務類別。
from app.verification import VerificationService
# make_processor：產生 BullMQ 工作處理函式；maintain_heartbeat：維護 worker 健康檢查用的心跳檔。
from app.worker import make_processor, maintain_heartbeat


def image_payload(size=(128, 128), format="PNG"):
    """在記憶體中產生一張純色圖片，回傳 API 需要的 {imageBase64, mimeType}。

    參數：
        size：圖片的 (寬, 高)，單位 px，預設 128×128（在 verification.py 允許的 64～4096 px 內）。
        format：Pillow 的存檔格式，只能是 "PNG"、"JPEG"、"WEBP"（其他值查不到對應的 MIME，會丟 KeyError）。

    回傳：
        dict，包含 imageBase64（Base64 字串）與 mimeType（和實際格式一致的 MIME 類型）。

    設計理由：
        圖片是純色、沒有人臉；ApiTests 用它證明「格式合法的影像」在沒有模型時也不會通過。
    """
    # 建立記憶體中的空檔案，準備接收圖片位元組。
    buffer = io.BytesIO()
    # 產生指定尺寸、顏色為 RGB (50, 80, 100) 的純色圖片，以指定格式存進 buffer。
    Image.new("RGB", size, (50, 80, 100)).save(buffer, format=format)
    # 回傳 API 欄位需要的兩個值。
    return {
        # 取出圖片位元組 → Base64 編碼（得到 bytes）→ 用 ASCII 解碼成一般字串。
        "imageBase64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        # 依格式查出對應的 MIME 類型，讓宣告的格式和實際內容一致。
        "mimeType": {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[format],
    }


def image_request(size=(128, 128), format="PNG", references=0, actions=None):
    """組出一個合法的 VerificationRequest JSON（dict），各測試再依需要改掉其中一部分。

    參數：
        size、format：自拍（正面影格）的尺寸與格式，意義同 image_payload。
        references：要附幾張參照照片（referenceImages），每張都是 128×128 的 JPEG；預設 0 張。
        actions：動作名稱清單（例如 ["turn_left"]）；給了就加上 liveCapture，每個動作一張 JPEG 影格。
            預設 None 代表「只上傳自拍」，完全不放 liveCapture 欄位。

    回傳：
        dict，可以直接當 JSON 送出，或用 VerificationRequest(**dict) 建成 model。

    注意：這個函式不檢查參數是否合法，測試會刻意傳入超過上限的張數或不存在的動作，來確認服務會拒絕。
    """
    # 組出基本請求。
    request = {
        # ** 是字典展開（類似 JS 的 ...spread），把自拍的 imageBase64、mimeType 放進來；
        # requestId 是固定的測試編號（只含英數字與連字號，符合格式限制）。
        **image_payload(size, format), "requestId": "test-request-1",
        # 串列生成式：重複 references 次，每次產生一張 JPEG 參照照片；_ 表示用不到的迴圈變數。
        "referenceImages": [image_payload(format="JPEG") for _ in range(references)],
    }
    # 只有明確給了 actions（包含空清單）才加上即時鏡頭資料；None 代表只上傳自拍。
    if actions is not None:
        # 加上 liveCapture 欄位。
        request["liveCapture"] = {
            # 固定的測試挑戰編號（符合只含英數字、底線、連字號的規則）。
            "challengeId": "challenge-1",
            # 每個動作產生一張影格：{"action": 動作名稱, "imageBase64": ..., "mimeType": "image/jpeg"}。
            "frames": [{"action": action, **image_payload(format="JPEG")} for action in actions],
        }
    # 回傳組好的請求。
    return request


def decision(**overrides):
    """產生一個「完整且合法」的 provider 通過回應（ProviderResult 的 JSON），可以用參數覆蓋任何欄位。

    參數：
        **overrides：要覆蓋或新增的欄位，例如 decision(status="rejected")、decision(embedding=[1, 2, 3])。
            ** 會把所有具名參數收成一個 dict。

    回傳：
        dict；沒有覆蓋時會通過 schemas.py 的 ProviderResult 驗證（verified、兩個布林都是 True、兩個分數都有值）。

    設計理由：
        測試只要改一個欄位就能做出「剛好缺一項證據」的回應，確認每一項證據都是必要的。
    """
    # 回傳預設的通過回應，後面再用 overrides 覆蓋。
    return {
        # 判定狀態與原因代碼：通過。
        "status": "verified", "reasonCode": "VERIFICATION_PASSED",
        # 模型名稱與版本（verified／rejected 必須帶，才可追溯是哪個模型做的判定）。
        "modelName": "test-provider", "modelVersion": "test-version",
        # 驗證類型必須是身分驗證，而不只是人臉偵測。
        "verificationType": "identity_verification",
        # 身分比對與活體（防偽）兩個判定都通過。
        "identityVerified": True, "livenessVerified": True,
        # 兩個 0～1 的分數；這裡的數值只是示範，服務不會拿它們比門檻。
        "livenessScore": 0.9, "faceMatchScore": 0.9,
        # 放在最後展開：同名的鍵以後出現的為準，所以 overrides 會覆蓋上面的預設值。
        **overrides,
    }


# 只設定內部 token、沒有 provider 的設定：模擬「未配置模型」的服務。
SETTINGS = Settings(internal_token="internal-test-secret")
# 設定完整的 provider（網址 + token），讓 VerificationService 真的會去呼叫 provider。
PROVIDER_SETTINGS = Settings(
    # 內部 token 同上；provider 網址用 .invalid 保留網域（RFC 2606），即使不小心真的連線也不會連到任何人。
    internal_token="internal-test-secret", provider_url="https://provider.invalid/verify",
    # 呼叫 provider 用的 Bearer token；test_provider_request_contract 會檢查它有被帶上。
    provider_token="provider-test-secret",
)
# 呼叫私有 API 時要帶的 header（內部 token 正確）。
HEADERS = {"X-Internal-Token": SETTINGS.internal_token}
# 真人驗證的私有 API 路徑（在 /internal/ 底下，會經過 InternalBoundary 檢查）。
ENDPOINT = "/internal/ai/face/verify"


class ApiTests(unittest.TestCase):
    """透過 HTTP 介面（TestClient）測試 AI 服務的驗證 API。

    重點規則：
    - 內部 token 要在解析 body 之前檢查；沒設定 token 時整個私有 API 停用。
    - 沒有模型時，任何影像都只會得到 unavailable / MODEL_NOT_CONFIGURED。
    - 不合法的影像（壞 Base64、HTML、格式不符、尺寸不對、截斷、動畫）回 400 與明確的錯誤碼。
    - 錯誤回應不能回顯使用者送來的內容（自拍屬於敏感個資）。
    - body 大小上限要擋得住超大請求，但不能誤擋「最大的合法請求」。
    """

    def setUp(self):
        """每個測試開始前都會執行（類似 Jest 的 beforeEach）：用 SETTINGS 建一個新的 app 與測試用戶端。"""
        # 每個測試都拿到全新的 app，測試之間不會互相影響。
        self.client = TestClient(create_app(SETTINGS))

    def test_health_without_auth(self):
        """健康檢查不需要 token；沒有 provider 時要誠實回報 verificationProvider 是 unavailable。

        為什麼重要：容器的 healthcheck 不會帶內部 token；而且健康檢查不能宣稱模型可用。
        """
        # 不帶任何 header 呼叫 /health（它不在 /internal/ 底下，InternalBoundary 直接放行）。
        response = self.client.get("/health")
        # 斷言：HTTP 200，代表不需要認證也能呼叫。
        self.assertEqual(response.status_code, 200)
        # 斷言：SETTINGS 沒有 provider，所以回報 "unavailable" 而不是 "configured"。
        self.assertEqual(response.json()["verificationProvider"], "unavailable")

    def test_auth_checked_before_bad_body(self):
        """認證要在解析 body 之前檢查：沒 token 或 token 錯誤都回 401，不會先回 422（格式錯誤）。

        為什麼重要：未認證的呼叫者不該觸發 JSON 解析，也不該從錯誤訊息得知 API 的格式要求。
        """
        # 不帶 token，body 也不是 JSON。
        response = self.client.post(ENDPOINT, content="not json")
        # 斷言：回 401（未認證），而不是 422——證明 token 檢查發生在解析 JSON 之前。
        self.assertEqual(response.status_code, 401)
        # 送合法的請求，但 token 是錯的。
        response = self.client.post(ENDPOINT, json=image_request(), headers={"X-Internal-Token": "wrong"})
        # 斷言：token 錯誤一樣回 401。
        self.assertEqual(response.status_code, 401)

    def test_missing_configured_token_disables_internal_api(self):
        """服務本身沒設定 AI_INTERNAL_TOKEN 時，私有 API 一律回 503（停用），不能變成「不用認證」。

        為什麼重要：設定漏掉時要「失敗即關閉」，而不是讓任何人都能呼叫。
        """
        # Settings() 全用預設值，internal_token 是空字串。
        client = TestClient(create_app(Settings()))
        # 斷言：呼叫私有 API 得到 503（INTERNAL_AUTH_NOT_CONFIGURED）。
        self.assertEqual(client.post(ENDPOINT, json=image_request()).status_code, 503)

    def test_valid_image_never_passes_without_model(self):
        """三種合法格式的影像，在沒有模型時都只會得到 unavailable / MODEL_NOT_CONFIGURED。

        為什麼重要：不能因為「格式正確」或開發模式就把使用者標成已驗證。
        """
        # 對 PNG、JPEG、WEBP 各跑一次。
        for format in ("PNG", "JPEG", "WEBP"):
            # subTest：每種格式各自報告結果，一種失敗不會中斷其他格式。
            with self.subTest(format=format):
                # 用正確的 token 送出該格式的合法影像。
                response = self.client.post(ENDPOINT, json=image_request(format=format), headers=HEADERS)
                # 斷言：影像合法，所以 HTTP 200（判定結果放在 JSON 裡）。
                self.assertEqual(response.status_code, 200)
                # 斷言：狀態是 unavailable，不是 verified。
                self.assertEqual(response.json()["status"], "unavailable")
                # 斷言：原因明確是「模型未設定」。
                self.assertEqual(response.json()["reasonCode"], "MODEL_NOT_CONFIGURED")
                # 斷言：沒有模型，所以 modelName 是 null（JSON 的 null 在 Python 是 None）。
                self.assertIsNone(response.json()["modelName"])

    def test_bad_base64_and_html_cannot_pass(self):
        """壞掉的 Base64 與「偽裝成圖片的 HTML」都回 400 INVALID_IMAGE，而且回應不含原始內容。

        為什麼重要：只相信實際解碼得出的影像，不相信呼叫端的宣告；錯誤訊息不回顯輸入。
        """
        # 兩種內容："????" 含有 Base64 字母表以外的字元（嚴格解碼會失敗）；
        # 第二個是合法的 Base64，但解碼後是 HTML 文字，Pillow 認不出是圖片。
        for content in ("????", base64.b64encode(b"<html>not a photo</html>").decode()):
            # 每種內容各自報告結果。
            with self.subTest(content=content):
                # 複製一份合法請求，只把 imageBase64 換成測試內容（後面的鍵覆蓋前面的）。
                request = {**image_request(), "imageBase64": content}
                # 送出請求。
                response = self.client.post(ENDPOINT, json=request, headers=HEADERS)
                # 斷言：影像不合法回 400。
                self.assertEqual(response.status_code, 400)
                # 斷言：錯誤碼是 INVALID_IMAGE。
                self.assertEqual(response.json()["code"], "INVALID_IMAGE")
                # 斷言：回應內容不包含送來的原始字串（不回顯輸入）。
                self.assertNotIn(content, response.text)

    def test_mime_mismatch(self):
        """宣告的 mimeType 和實際檔案格式不一致時回 400 IMAGE_TYPE_MISMATCH。

        為什麼重要：provider 會依宣告的格式處理影像，宣告必須可信。
        """
        # 實際內容是 PNG（image_request 預設），但宣告成 image/jpeg。
        request = {**image_request(), "mimeType": "image/jpeg"}
        # 送出請求。
        response = self.client.post(ENDPOINT, json=request, headers=HEADERS)
        # 斷言：回 400。
        self.assertEqual(response.status_code, 400)
        # 斷言：錯誤碼指出格式不符。
        self.assertEqual(response.json()["code"], "IMAGE_TYPE_MISMATCH")

    def test_unsafe_dimensions(self):
        """太小或太大的影像回 400 INVALID_IMAGE_DIMENSIONS（每邊必須在 64～4096 px 之間）。

        為什麼重要：太小的影像看不清人臉；太大的影像會耗盡記憶體與運算資源。
        """
        # 10×10：短邊小於 64；4097×64：長邊超過 4096（短邊剛好 64，是合法的下限）。
        for size in ((10, 10), (4097, 64)):
            # 用該尺寸的 PNG 送出請求。
            response = self.client.post(ENDPOINT, json=image_request(size=size), headers=HEADERS)
            # 斷言：回 400。
            self.assertEqual(response.status_code, 400)
            # 斷言：錯誤碼指出尺寸不合格。
            self.assertEqual(response.json()["code"], "INVALID_IMAGE_DIMENSIONS")

    def test_truncated_pixels_rejected(self):
        """檔頭完整但像素資料被截斷的 JPEG 要回 400。

        為什麼重要：verify() 只檢查結構，verification.py 另外用 load() 完整解碼像素，
        確保交給 provider 的是能完整讀出的影像（目前會得到 INVALID_IMAGE）。
        """
        # 先產生一個合法的 JPEG 請求。
        request = image_request(format="JPEG")
        # 把 Base64 解回原始的 JPEG 位元組。
        raw = base64.b64decode(request["imageBase64"])
        # 砍掉最後 20 個位元組（[:-20] 是「除了最後 20 個以外」的切片），再編回 Base64。
        request["imageBase64"] = base64.b64encode(raw[:-20]).decode()
        # 斷言：截斷的影像回 400。
        self.assertEqual(self.client.post(ENDPOINT, json=request, headers=HEADERS).status_code, 400)

    def test_animated_image_rejected(self):
        """多影格的動畫圖片（這裡用 APNG）回 400 ANIMATED_IMAGE_NOT_ALLOWED。

        為什麼重要：只接受單張影像，避免用動畫夾帶多張不同的臉。
        """
        # 建立記憶體中的檔案。
        buffer = io.BytesIO()
        # 第一格是紅色 128×128 圖片。
        Image.new("RGB", (128, 128), "red").save(
            # save_all=True 加上 append_images：把後面的藍色圖片當第二格，存成有 2 格的動畫 PNG（APNG）。
            buffer, format="PNG", save_all=True, append_images=[Image.new("RGB", (128, 128), "blue")],
        )
        # 用這個動畫 PNG 取代請求的 imageBase64（宣告的 image/png 和實際格式一致）。
        request = {**image_request(), "imageBase64": base64.b64encode(buffer.getvalue()).decode()}
        # 送出請求。
        response = self.client.post(ENDPOINT, json=request, headers=HEADERS)
        # 斷言：回 400。
        self.assertEqual(response.status_code, 400)
        # 斷言：錯誤碼指出不允許動畫（格式與尺寸都合法，所以是影格數的檢查擋下的）。
        self.assertEqual(response.json()["code"], "ANIMATED_IMAGE_NOT_ALLOWED")

    def test_validation_does_not_echo_sensitive_input(self):
        """請求格式錯誤（這裡是多了未定義的欄位）回 422，而且回應不包含自拍或任何輸入內容。

        為什麼重要：FastAPI 預設的 422 會列出錯誤欄位的輸入值；main.py 換成固定訊息，避免自拍 Base64 外洩到日誌或呼叫端。
        """
        # 在合法請求中多加一個 schema 沒定義的欄位（schemas.py 設 extra="forbid"，會驗證失敗）。
        request = {**image_request(), "untrustedField": "secret-selfie-content"}
        # 送出請求。
        response = self.client.post(ENDPOINT, json=request, headers=HEADERS)
        # 斷言：格式錯誤回 422。
        self.assertEqual(response.status_code, 422)
        # 斷言：回應不含自拍的 Base64。
        self.assertNotIn(request["imageBase64"], response.text)
        # 斷言：回應也不含那個多出來的欄位值。
        self.assertNotIn("secret-selfie-content", response.text)

    def test_http_body_limit_even_without_content_length(self):
        """即使請求沒有 Content-Length（分段串流傳送），超過 MAX_BODY_BYTES 也要回 413。

        為什麼重要：只看 Content-Length 的話，攻擊者可以省略它來繞過限制；
        InternalBoundary 是邊讀邊累計大小，超過就立刻停止，不會把整個超大 body 讀進記憶體。
        """
        # 定義一個產生器（generator）：每次 yield 1 MiB 的 "a"，共 16 次（16 MiB）。
        # 用產生器當 content 時，httpx 不知道總長度，會用分段（chunked）方式傳送、不帶 Content-Length。
        def chunks():
            """逐段產出 16 段、每段 1 MiB 的資料（總共 16 MiB）；回傳產生器，不會一次把 16 MiB 放進記憶體。"""
            # 重複 16 次。
            for _ in range(16):
                # 產出一段 1 MiB（1024 × 1024 bytes）的資料。
                yield b"a" * (1024 * 1024)
        # 帶正確 token 送出這個串流 body。
        response = self.client.post(ENDPOINT, content=chunks(), headers=HEADERS)
        # 斷言（確認測試前提）：送出的 16 MiB 確實大於上限，否則這個測試沒有意義。
        self.assertGreater(16 * 1024 * 1024, MAX_BODY_BYTES)
        # 斷言：回 413 Payload Too Large。
        self.assertEqual(response.status_code, 413)

    def test_largest_legal_request_not_cut_by_body_limit(self):
        """每個欄位都用到最大長度、張數也用到上限的請求，不能在讀 body 時就被 413 擋掉。

        為什麼重要：MAX_BODY_BYTES 是用各影像的 Base64 上限加上預留空間算出來的；
        這個測試確認預留空間夠用，合法的請求一定能進到後續的驗證流程。
        """
        # 手動組出 JSON 字串再轉成 bytes（.encode() 預設 UTF-8），才能精確控制與量測 body 大小。
        body = json.dumps({
            # 自拍：長度剛好是上限的 "AAAA..."；"A" 是合法的 Base64 字元，能通過 schema 的長度檢查。
            "imageBase64": "A" * MAX_BASE64_LENGTH, "mimeType": "image/jpeg", "requestId": "test-request-1",
            # 參照照片：1 張（上限），長度也是上限。
            "referenceImages": [{"imageBase64": "A" * MAX_REFERENCE_BASE64_LENGTH, "mimeType": "image/jpeg"}],
            # 即時鏡頭資料。
            "liveCapture": {
                # 挑戰編號。
                "challengeId": "challenge-1",
                # 動作影格清單。
                "frames": [
                    # 每張影格的 Base64 都是上限長度。
                    {"action": action, "imageBase64": "A" * MAX_FRAME_BASE64_LENGTH, "mimeType": "image/jpeg"}
                    # 3 個動作 = MAX_ACTION_FRAMES（張數上限）。
                    for action in ("turn_left", "turn_right", "look_up")
                ],
            },
        }).encode()
        # 斷言（確認測試前提）：這個最大的合法請求沒有超過 body 上限。
        self.assertLessEqual(len(body), MAX_BODY_BYTES)
        # 用原始 bytes 送出，並明確標示 Content-Type 是 JSON。
        response = self.client.post(ENDPOINT, content=body, headers={**HEADERS, "Content-Type": "application/json"})
        # 內容不是合法影像所以回 400，但不能在讀 body 時就被 413 擋掉。
        # （實際上最先失敗的是自拍：6,990,508 個 "A" 解碼後是 5,242,881 bytes，比 5 MiB 多 1 byte，
        # 所以錯誤碼是 IMAGE_TOO_LARGE；重點是狀態碼是 400 而不是 413，代表請求已經通過 body 上限與 schema 檢查。）
        self.assertEqual(response.status_code, 400)

    def test_body_limit_matches_face_provider(self):
        """把 MAX_BODY_BYTES 鎖在 12,588,044，確保 AI 服務和 services/face 的上限一致。

        為什麼重要：AI 服務會把請求原樣轉給 provider；如果 provider 的上限比較小，
        AI 服務接受的合法請求到了 provider 會被擋掉。
        """
        # services/face/app/schemas.py 的 MAX_BODY_BYTES 必須相同；兩邊的測試各自鎖同一個數字。
        # （services/face/tests/test_api.py 也斷言同一個值；任何一邊改了上限，另一邊的測試就會提醒要一起改。）
        self.assertEqual(MAX_BODY_BYTES, 12_588_044)

    def test_reference_image_validated_with_prefixed_code(self):
        """參照照片不合格時，錯誤碼加上 REFERENCE_ 前綴（這裡是 REFERENCE_IMAGE_TYPE_MISMATCH）。

        為什麼重要：前端要能分辨是「自拍」還是「主照片」有問題，才能給使用者正確的提示。
        """
        # 附 1 張參照照片的請求。
        request = image_request(references=1)
        # 參照照片實際是 JPEG，改宣告成 image/png 製造格式不符。
        request["referenceImages"][0]["mimeType"] = "image/png"
        # 送出請求。
        response = self.client.post(ENDPOINT, json=request, headers=HEADERS)
        # 斷言：回 400。
        self.assertEqual(response.status_code, 400)
        # 斷言：錯誤碼有 REFERENCE_ 前綴。
        self.assertEqual(response.json()["code"], "REFERENCE_IMAGE_TYPE_MISMATCH")

    def test_action_frame_validated_with_prefixed_code(self):
        """動作影格不合格時，錯誤碼加上 FRAME_ 前綴（這裡是 FRAME_IMAGE_TYPE_MISMATCH）。

        為什麼重要：同上，要能分辨是哪一類影像出問題。
        """
        # 附 1 張參照照片與 1 個動作（turn_left）影格的請求。
        request = image_request(references=1, actions=["turn_left"])
        # 動作影格實際是 JPEG，改宣告成 image/png。
        request["liveCapture"]["frames"][0]["mimeType"] = "image/png"
        # 送出請求。
        response = self.client.post(ENDPOINT, json=request, headers=HEADERS)
        # 斷言：回 400。
        self.assertEqual(response.status_code, 400)
        # 斷言：錯誤碼有 FRAME_ 前綴。
        self.assertEqual(response.json()["code"], "FRAME_IMAGE_TYPE_MISMATCH")

    def test_live_capture_limits(self):
        """動作影格的 schema 限制：超過 3 張、未知動作、空清單都回 422。

        為什麼重要：限制張數才能控制請求大小；只接受 provider 認得的四種動作。
        """
        # 三組不合法的動作清單：4 張（超過上限 3）、"blink"（不在允許的動作中）、空清單（至少要 1 張）。
        for actions in (["turn_left", "turn_right", "look_up", "look_down"], ["blink"], []):
            # 每組各自報告結果。
            with self.subTest(actions=actions):
                # 送出含該動作清單的請求。
                response = self.client.post(ENDPOINT, json=image_request(references=1, actions=actions), headers=HEADERS)
                # 斷言：schema 驗證失敗回 422。
                self.assertEqual(response.status_code, 422)

    def test_too_many_reference_images(self):
        """參照照片超過 1 張回 422（MAX_REFERENCE_IMAGES = 1）。"""
        # 送出附 2 張參照照片的請求。
        response = self.client.post(ENDPOINT, json=image_request(references=2), headers=HEADERS)
        # 斷言：schema 驗證失敗回 422。
        self.assertEqual(response.status_code, 422)

    def test_request_with_reference_still_needs_model(self):
        """就算附了參照照片，沒有模型時仍然只回 MODEL_NOT_CONFIGURED。

        為什麼重要：有參照照片不代表已完成比對，比對只能由 provider 做。
        """
        # 送出附 1 張參照照片的合法請求。
        response = self.client.post(ENDPOINT, json=image_request(references=1), headers=HEADERS)
        # 斷言：影像都合法，回 200。
        self.assertEqual(response.status_code, 200)
        # 斷言：結果仍是模型未設定。
        self.assertEqual(response.json()["reasonCode"], "MODEL_NOT_CONFIGURED")


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    """測試 VerificationService 和 provider 之間的契約（用 httpx.MockTransport 假造 provider）。

    IsolatedAsyncioTestCase：可以寫 async def 的測試，每個測試有自己獨立的 event loop。
    httpx.MockTransport(handler)：所有 HTTP 請求都不會真的送出，而是交給 handler 函式，
    由 handler 回傳假的 httpx.Response。

    重點規則：
    - provider 的回應必須完整、型別正確、沒有多餘欄位，且判定前後一致，否則一律 unavailable / PROVIDER_INVALID_RESPONSE。
    - 服務不自訂門檻：provider 的 rejected 原樣保留，不會因為分數高就改判。
    - 逾時、連線錯誤、非 200、非 JSON、回應太大都「失敗即關閉」，而且不外洩上游的錯誤內容。
    """

    async def get_result(self, payload, status_code=200):
        """輔助函式：讓假 provider 回傳指定的 JSON 與狀態碼，回傳 VerificationService 的結果。

        參數：
            payload：假 provider 要回的 JSON 內容（dict）。
            status_code：假 provider 回的 HTTP 狀態碼，預設 200。

        回傳：
            VerificationResult（服務整理後要回給 NestJS 的結果）。

        注意：httpx.Response(json=...) 會自動設定 Content-Type: application/json，
        而且編碼時不允許 NaN／Infinity（有的話會直接丟 ValueError）。
        """
        # 建立使用完整 provider 設定的服務；MockTransport 的 handler 忽略請求內容（參數用 _），一律回傳指定的回應。
        service = VerificationService(PROVIDER_SETTINGS, httpx.MockTransport(lambda _: httpx.Response(status_code, json=payload)))
        # 用合法的自拍請求呼叫 verify 並回傳結果（VerificationRequest(**dict) 把 dict 的鍵當成具名參數建立 model）。
        return await service.verify(VerificationRequest(**image_request()))

    async def test_verified_requires_identity_and_liveness_evidence(self):
        """各種「證據不足或格式不對」的 provider 回應，都要變成 unavailable / PROVIDER_INVALID_RESPONSE。

        為什麼重要：只要 provider 的回應有一點不可信，就不能把使用者標成已驗證。
        """
        # 每一筆都是一種不合格的回應。
        cases = [
            # 只說 verified 與「偵測到臉」：缺少原因代碼、模型與兩項判定，還多了未定義的 faceDetected 欄位。
            {"status": "verified", "faceDetected": True},
            # 驗證類型只是人臉偵測，不是身分驗證。
            decision(verificationType="face_detection"),
            # verified 但身分比對沒通過；verified 但活體沒通過（兩者都違反「verified 必須兩項都通過」）。
            decision(identityVerified=False), decision(livenessVerified=False),
            # 布林值用字串 "true"（strict 模式不接受）；verified 卻沒有活體分數。
            decision(livenessVerified="true"), decision(livenessScore=None),
            # 缺少模型版本（無法追溯）；人臉比對分數超過 1。
            decision(modelVersion=None), decision(faceMatchScore=1.1),
            # 分數用字串 "0.99"（strict 模式不接受）；分數是無限大（Score 不允許 inf／NaN）。
            decision(faceMatchScore="0.99"), decision(livenessScore=float("inf")),
            # 未知的狀態 "pending"；多了未定義的 embedding 欄位（provider 不能把人臉特徵向量回傳過來）。
            decision(status="pending"), decision(embedding=[1, 2, 3]),
        ]
        # 逐一測試每種回應。
        for payload in cases:
            # 每種回應各自報告結果。
            with self.subTest(payload=payload):
                # 手動 JSON 允許非有限值，模擬不符合 JSON／schema 的上游。
                # （httpx.Response(json=...) 不允許 inf，所以改用 json.dumps 自己編碼：它會輸出非標準的 Infinity，
                # 再用 content= 送出並手動加上 Content-Type。Pydantic 解析得了 Infinity，但 Score 型別會拒絕它。）
                service = VerificationService(PROVIDER_SETTINGS, httpx.MockTransport(
                    # p=payload 是「預設參數綁定」：在建立 lambda 當下就把這一輪的 payload 存進 p，
                    # 避免 Python 閉包「晚綁定」拿到迴圈變數之後的值。
                    lambda _, p=payload: httpx.Response(200, content=json.dumps(p), headers={"Content-Type": "application/json"})
                ))
                # 呼叫 verify。
                result = await service.verify(VerificationRequest(**image_request()))
                # 斷言：狀態是 unavailable（不是 verified，也不是 rejected）。
                self.assertEqual(result.status, "unavailable")
                # 斷言：原因是 provider 回應不合格。
                self.assertEqual(result.reasonCode, "PROVIDER_INVALID_RESPONSE")

    async def test_complete_provider_decision_forwarded_without_extra_fields(self):
        """完整的通過回應會被轉交，但 provider 專用的證據欄位不會出現在結果中。

        為什麼重要：NestJS 只需要公開欄位；三個證據欄位（verificationType、livenessVerified、identityVerified）只在本服務內部檢查。
        """
        # 假 provider 回傳完整的通過回應。
        result = await self.get_result(decision())
        # 斷言：狀態原樣轉交為 verified。
        self.assertEqual(result.status, "verified")
        # 斷言：模型版本原樣保留（可追溯）。
        self.assertEqual(result.modelVersion, "test-version")
        # 斷言：分數原樣保留。
        self.assertEqual(result.faceMatchScore, 0.9)
        # 斷言：model_dump() 轉成的 dict 裡沒有 identityVerified，provider 專用欄位已被移除。
        self.assertNotIn("identityVerified", result.model_dump())

    async def test_provider_rejection_preserved_without_new_threshold(self):
        """provider 判 rejected 時原樣保留；即使活體分數高達 0.99，服務也不會自己改判成通過。

        為什麼重要：判定門檻只在 provider（services/face/app/policy.py），AI 服務不另訂門檻。
        """
        # 假 provider 回傳「活體檢查失敗」的拒絕，但附上很高的活體分數。
        result = await self.get_result(decision(
            # 拒絕、原因是活體失敗、活體判定為 False。
            status="rejected", reasonCode="LIVENESS_FAILED", livenessVerified=False,
            # 刻意給高分，確認服務不會拿分數自己比門檻。
            livenessScore=0.99,
        ))
        # 斷言：狀態維持 rejected。
        self.assertEqual(result.status, "rejected")
        # 斷言：原因代碼維持 provider 給的值。
        self.assertEqual(result.reasonCode, "LIVENESS_FAILED")

    async def test_provider_request_contract(self):
        """檢查送給 provider 的請求：POST、帶 Bearer token、body 和原始請求完全相同（含 referenceImages）。

        為什麼重要：provider（services/face）依這個契約解析請求；多送、少送欄位或沒帶認證都會壞掉。
        """
        # 假 provider 的 handler：在收到請求時直接檢查內容。
        # 如果這裡的斷言失敗，AssertionError 不在 verify 會攔截的例外清單裡，會一路往外丟，讓測試失敗。
        def respond(request):
            """檢查收到的請求並回傳通過的回應。參數 request 是 httpx.Request；回傳 httpx.Response。"""
            # 斷言：HTTP 方法是 POST。
            self.assertEqual(request.method, "POST")
            # 斷言：帶了 PROVIDER_SETTINGS 設定的 Bearer token。
            self.assertEqual(request.headers["Authorization"], "Bearer provider-test-secret")
            # 斷言：body 解析回 dict 後和原始請求完全相同；原始請求沒有 liveCapture，
            # 轉送時也不能多出 "liveCapture": null（verification.py 用 exclude_none=True 省略它）。
            self.assertEqual(json.loads(request.content), image_request(references=1))
            # 回傳完整的通過回應。
            return httpx.Response(200, json=decision())
        # 用這個 handler 建立服務並呼叫 verify（附 1 張參照照片）。
        result = await VerificationService(PROVIDER_SETTINGS, httpx.MockTransport(respond)).verify(
            # 請求內容和 respond 裡比對的一樣。
            VerificationRequest(**image_request(references=1))
        )
        # 斷言：流程走完、結果是 verified（也證明 respond 裡的斷言都通過了）。
        self.assertEqual(result.status, "verified")

    async def test_live_capture_forwarded_to_provider(self):
        """即時鏡頭的 liveCapture（挑戰編號與動作影格）要完整轉送給 provider。

        為什麼重要：provider 要靠動作影格判斷頭部角度與是否同一人；少了它只會得到 LIVE_CAPTURE_REQUIRED。
        """
        # 假 provider 的 handler。
        def respond(request):
            """檢查 body 含完整的 liveCapture，並回傳通過的回應。"""
            # 斷言：body 和含兩個動作（turn_left、look_up）的原始請求完全相同。
            self.assertEqual(json.loads(request.content), image_request(references=1, actions=["turn_left", "look_up"]))
            # 回傳完整的通過回應。
            return httpx.Response(200, json=decision())
        # 建立服務並呼叫 verify。
        result = await VerificationService(PROVIDER_SETTINGS, httpx.MockTransport(respond)).verify(
            # 附 1 張參照照片與兩張動作影格的請求。
            VerificationRequest(**image_request(references=1, actions=["turn_left", "look_up"]))
        )
        # 斷言：結果是 verified。
        self.assertEqual(result.status, "verified")

    async def test_self_hosted_provider_outcomes_accepted(self):
        """自架 provider（services/face）實際會回的非通過結果，都要被接受並原樣轉交。

        為什麼重要：schemas.py 的 ProviderResult 規則很嚴格，要確認它不會誤判自家 provider 的正常回應為不合格。
        """
        # services/face（pipeline.py）除了 verified 以外會回的其中三種形狀：全部通過但沒有即時鏡頭、疑似翻拍、找不到臉。
        cases = [
            # 只上傳自拍、全部通過：unavailable / LIVE_CAPTURE_REQUIRED；身分比對通過（decision 預設 True）但活體未確認。
            decision(status="unavailable", reasonCode="LIVE_CAPTURE_REQUIRED", livenessVerified=False),
            # 防偽判定為假臉：rejected / SPOOF_SUSPECTED，活體未通過、活體分數低。
            decision(status="rejected", reasonCode="SPOOF_SUSPECTED", livenessVerified=False, livenessScore=0.2),
            # 找不到臉：rejected / NO_FACE_DETECTED。
            decision(
                # 偵測階段就被拒絕，兩項判定都是 False、兩個分數都還沒算（None）。
                status="rejected", reasonCode="NO_FACE_DETECTED", livenessVerified=False,
                identityVerified=False, livenessScore=None, faceMatchScore=None,
            ),
        ]
        # 逐一測試。
        for payload in cases:
            # 以原因代碼區分每個子測試。
            with self.subTest(reason=payload["reasonCode"]):
                # 假 provider 回傳這個結果。
                result = await self.get_result(payload)
                # 斷言：狀態與原因代碼都原樣轉交（用 tuple 一次比對兩個值）。
                self.assertEqual((result.status, result.reasonCode), (payload["status"], payload["reasonCode"]))

    async def test_disabled_provider_has_no_network_call(self):
        """沒有設定 provider 時，服務完全不發出網路請求，直接回 MODEL_NOT_CONFIGURED。

        為什麼重要：未配置時不能把使用者的自拍送到任何地方。
        """
        # 假 transport 的 handler：一旦被呼叫就代表有網路請求，直接讓測試失敗。
        def unexpected(_request):
            """被呼叫就讓測試失敗（self.fail 會丟出 AssertionError）。"""
            # 標記測試失敗並附上原因。
            self.fail("Unconfigured model must not make a network call")
        # SETTINGS 沒有 provider_url；呼叫 verify。
        result = await VerificationService(SETTINGS, httpx.MockTransport(unexpected)).verify(VerificationRequest(**image_request()))
        # 斷言：原因是模型未設定。
        self.assertEqual(result.reasonCode, "MODEL_NOT_CONFIGURED")

    async def test_partial_provider_config_fails_closed(self):
        """只設定 provider 網址、沒設定 token 時，回 PROVIDER_CONFIGURATION_INVALID，而不是不帶認證就送出。

        為什麼重要：設定不完整要「失敗即關閉」。
        """
        # 有網址但沒有 provider_token，provider_configured 會是 False；會在發出任何請求之前就回傳。
        result = await VerificationService(Settings(provider_url="https://example.com")).verify(VerificationRequest(**image_request()))
        # 斷言：原因是 provider 設定無效。
        self.assertEqual(result.reasonCode, "PROVIDER_CONFIGURATION_INVALID")

    async def test_provider_timeout(self):
        """provider 逾時回 PROVIDER_TIMEOUT，而且結果不包含上游例外的訊息。

        為什麼重要：例外訊息可能含有敏感資訊（網址、內部錯誤），不能往外傳。
        """
        # 假 transport 的 handler：直接丟出讀取逾時，訊息裡放一段不該外洩的文字。
        def timeout(_request):
            """模擬 provider 讀取逾時（丟出 httpx.ReadTimeout）。"""
            # httpx.ReadTimeout 是 httpx.TimeoutException 的子類別。
            raise httpx.ReadTimeout("sensitive-provider-error-do-not-return")
        # 用這個 handler 建立服務並呼叫 verify。
        result = await VerificationService(PROVIDER_SETTINGS, httpx.MockTransport(timeout)).verify(VerificationRequest(**image_request()))
        # 斷言：原因是逾時。
        self.assertEqual(result.reasonCode, "PROVIDER_TIMEOUT")
        # 斷言：整個結果轉成 JSON 後不含例外訊息的任何部分。
        self.assertNotIn("sensitive", result.model_dump_json())

    async def test_provider_redirect_and_error_fail_closed(self):
        """provider 回重導向（302）、未授權（401）或伺服器錯誤（500）時，一律 unavailable。

        為什麼重要：服務不跟隨重導向（follow_redirects=False），避免自拍被轉送到其他網址；非 200 都不採信。
        """
        # 三種非 200 的狀態碼。
        for status in (302, 401, 500):
            # 假 provider 用該狀態碼回傳一個「內容看起來是通過」的回應。
            result = await self.get_result(decision(), status_code=status)
            # 斷言：只要不是 200，就算內容說通過也是 unavailable（目前原因代碼是 PROVIDER_UNAVAILABLE）。
            self.assertEqual(result.status, "unavailable")

    async def test_non_json_and_oversized_provider_body(self):
        """provider 回非 JSON，或回應超過 16 KiB（MAX_PROVIDER_BYTES）時，回 PROVIDER_INVALID_RESPONSE。

        為什麼重要：限制回應大小，避免異常的上游用超大回應耗盡記憶體。
        """
        # 兩種不合格回應：純文字 "OK"（Content-Type 是 text/plain）、合法 JSON 但約 20 KB（超過 16 KiB）。
        for response in (httpx.Response(200, text="OK"), httpx.Response(200, json={"huge": "x" * 20000})):
            # handler 一律回傳這一輪的 response；服務在同一輪就用掉，所以不受閉包晚綁定影響。
            service = VerificationService(PROVIDER_SETTINGS, httpx.MockTransport(lambda _: response))
            # 呼叫 verify。
            result = await service.verify(VerificationRequest(**image_request()))
            # 斷言：原因是 provider 回應不合格。
            self.assertEqual(result.reasonCode, "PROVIDER_INVALID_RESPONSE")


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    """測試 worker.py：BullMQ 工作處理函式（make_processor）與心跳檔（maintain_heartbeat）。

    這裡不連 Redis，而是用 SimpleNamespace 假造 BullMQ 的 job 與 worker、用 AsyncMock 假造 job.updateData。
    真實 Redis 的版本在 tests/integration_queue.py。
    """

    async def test_worker_consumes_same_contract_and_redacts(self):
        """worker 用和 HTTP API 相同的請求格式；一拿到工作就把 Redis 裡的影像換成 {"redacted": True}，結果不含分數。

        為什麼重要：自拍不能留在 Redis；佇列結果也不能留下生物特徵相關的分數。
        """
        # 假 job：名稱 "verify"、資料是合法的自拍請求、updateData 是可以 await 的假函式。
        job = SimpleNamespace(name="verify", data=image_request(), updateData=AsyncMock())
        # 建立處理函式（SETTINGS 沒有 provider）並執行；第二個參數是 BullMQ 傳入的 lock token，這裡用不到。
        result = await make_processor(VerificationService(SETTINGS))(job, "lock-token")
        # 斷言：updateData 剛好被 await 一次，參數是 {"redacted": True}（影像已從 job 資料中移除）。
        job.updateData.assert_awaited_once_with({"redacted": True})
        # 斷言：沒有模型，結果是 unavailable。
        self.assertEqual(result["status"], "unavailable")
        # 斷言：結果裡沒有影像內容。
        self.assertNotIn("imageBase64", result)
        # 斷言：結果裡沒有人臉比對分數（make_processor 回傳前排除了兩個分數）。
        self.assertNotIn("faceMatchScore", result)

    async def test_unknown_jobs_rejected(self):
        """名稱不是 "verify" 的工作一律拒絕（丟 ValueError UNSUPPORTED_JOB）。

        為什麼重要：這個 worker 只處理真人驗證，不能順手執行其他尚未啟用的工作（例如對話向量化）。
        """
        # 假 job：名稱是未支援的 "embed-conversation"。
        job = SimpleNamespace(name="embed-conversation", data={}, updateData=AsyncMock())
        # 斷言：執行時丟出 ValueError，且訊息符合 "UNSUPPORTED_JOB"（assertRaisesRegex 用正規表示式比對訊息）。
        with self.assertRaisesRegex(ValueError, "UNSUPPORTED_JOB"):
            # 執行處理函式。
            await make_processor(VerificationService(SETTINGS))(job, "lock-token")

    async def test_failed_jobs_never_include_input_in_error(self):
        """工作資料格式錯誤時，丟出的錯誤訊息只有固定代碼 INVALID_VERIFICATION_INPUT，不含輸入內容。

        為什麼重要：BullMQ 會把失敗原因存進 Redis 的失敗紀錄；Pydantic 的預設錯誤訊息會包含輸入值。
        """
        # 假 job：名稱正確，但資料只有一個不合法的欄位（含敏感文字）。
        job = SimpleNamespace(name="verify", data={"private": "sensitive-image"}, updateData=AsyncMock())
        # 斷言會丟出 ValueError；as caught 讓我們之後可以檢查例外內容。
        with self.assertRaises(ValueError) as caught:
            # 執行處理函式。
            await make_processor(VerificationService(SETTINGS))(job, "lock-token")
        # 斷言：錯誤訊息完全等於固定代碼，沒有夾帶 "sensitive-image" 或任何欄位資訊。
        self.assertEqual(str(caught.exception), "INVALID_VERIFICATION_INPUT")

    async def test_heartbeat_requires_redis_and_worker(self):
        """心跳檔只在「Redis 連得上」且「worker 正在執行」時才存在。

        為什麼重要：容器的 healthcheck（app/worker_health.py）看心跳檔判斷 worker 是否健康；
        Redis 斷線或 worker 停了卻還顯示健康，就不會被重啟。
        """
        # 三種組合：都正常、Redis 連不上、worker 沒在執行。
        for redis_available, running in ((True, True), (False, True), (True, False)):
            # 每種組合各自報告結果，並建立一個暫存資料夾（區塊結束自動刪除）。
            with self.subTest(redis_available=redis_available, running=running), tempfile.TemporaryDirectory() as tmp:
                # 心跳檔路徑放在暫存資料夾內，不會碰到正式的 /tmp/ai-worker-heartbeat。
                path = Path(tmp) / "heartbeat"
                # 停止訊號；set() 之後 maintain_heartbeat 的迴圈就會結束。
                stopped = asyncio.Event()

                # 假的 Redis ping。
                async def ping():
                    """模擬 Redis PING：先發出停止訊號，讓心跳迴圈只跑一輪；Redis 不可用時丟 OSError。"""
                    # 先設定停止訊號，確保 maintain_heartbeat 這一輪結束後就離開迴圈，測試不會卡住。
                    stopped.set()
                    # 模擬 Redis 連不上。
                    if not redis_available:
                        # maintain_heartbeat 會攔截 OSError 並刪除心跳檔。
                        raise OSError("not connected")

                # 假的 Redis client，只有 ping 方法。
                client = SimpleNamespace(ping=ping)
                # 假的 BullMQ worker：running 表示是否在執行，closing=False 表示沒有正在關閉。
                worker = SimpleNamespace(running=running, closing=False)
                # 執行心跳維護；因為 ping 已設定 stopped，跑完一輪就會回傳。
                await maintain_heartbeat(client, worker, stopped, path)
                # 斷言：只有 Redis 可用且 worker 在執行時，心跳檔才存在。
                self.assertEqual(path.exists(), redis_available and running)


# 直接執行這個檔案（python -m tests.test_verification）時，啟動 unittest 執行上面所有測試。
if __name__ == "__main__":
    # 找出本檔案中所有 TestCase 並執行。
    unittest.main()
