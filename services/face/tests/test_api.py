"""人臉驗證 provider 的 HTTP 邊界與回應契約測試（用假模型，不需要真的模型檔）。

這支測試檔鎖住的是「HTTP 這一層」的規則，而不是人臉辨識的判定細節（判定細節在 test_pipeline.py、test_live.py）：
- 回應格式：欄位必須和 services/ai 的 ProviderResult 一模一樣，大小不能超過 AI 服務願意讀的上限。
- 認證順序：先檢查 Bearer token，再讀 body、解析 JSON；沒帶對 token 的請求連 body 都不看。
- 大小上限：用實際收到的位元組數計算，不相信 Content-Length；但最大的合法請求不能被誤擋。
- 輸入驗證：多餘欄位、太多影格、未知動作、太多參照照片都回 422，而且錯誤訊息不回顯使用者的影像內容。
- 失敗時一律 fail closed：模型出錯或模型檔不存在時回 503，不能回傳「通過」。

測試方式：用 FastAPI 的 TestClient 直接在同一個行程內呼叫 app（不開真的網路埠），
並透過 create_app 的 verifier 參數注入 tests/fakes.py 的假模型，讓每個情境的結果可以精準控制。
"""

# json：把請求字典轉成 JSON 字串，以及確認回應可以用嚴格模式（不允許 NaN）序列化。
import json
# tempfile：建立暫時的空資料夾，模擬「模型檔不存在」的情況，測完會自動刪除。
import tempfile
# unittest：Python 內建的測試框架；每個 test_ 開頭的方法都是一個測試案例。
import unittest
# Path：用物件表示檔案路徑，Settings.model_dir 需要的型別就是 Path。
from pathlib import Path

# TestClient：FastAPI 提供的測試用 HTTP 客戶端，直接呼叫 ASGI app，不需要真的啟動伺服器。
from fastapi.testclient import TestClient

# Settings：provider 的設定（token 與模型資料夾）；測試直接建構，不讀環境變數。
from app.config import Settings
# VERIFY_PATH 是 "/verify"；create_app 建立 FastAPI app，可注入假的 verifier。
# 注意：匯入 app.main 時，模組最底下的 app = create_app() 也會執行一次（讀環境變數、嘗試載入真模型），
# 在沒有模型檔的環境只會印出 FACE_MODELS_UNAVAILABLE，不影響這裡的測試。
from app.main import VERIFY_PATH, create_app
# 各種 Base64 長度上限與整個 body 的位元組上限，測試用它們組出「剛好合法」與「超過上限」的請求。
from app.schemas import MAX_BASE64_LENGTH, MAX_BODY_BYTES, MAX_FRAME_BASE64_LENGTH, MAX_REFERENCE_BASE64_LENGTH

# FakeDetector：依呼叫順序回傳預先指定的偵測結果（或丟出例外）；face：產生一筆假的人臉偵測結果；
# verifier：用假模型組出 FaceVerifier；verify_request：產生一個格式正確的 /verify 請求字典。
from tests.fakes import FakeDetector, face, verifier, verify_request

# 測試用的 provider 設定：只給 token，model_dir 用預設值（測試會注入假模型，所以不會真的去讀模型檔）。
SETTINGS = Settings(token="provider-test-secret")
# 正確的認證標頭：格式必須是 "Bearer <token>"，和 ProviderBoundary 比對的字串完全相同。
HEADERS = {"Authorization": "Bearer provider-test-secret"}
# services/ai 的 ProviderResult 欄位；多一個或少一個，AI 服務都會判成 PROVIDER_INVALID_RESPONSE。
# 用 set（集合）是為了比較「欄位名稱的組合」而不在乎順序。
PROVIDER_RESULT_FIELDS = {
    # 判定狀態（verified／rejected／unavailable）、原因代碼、模型名稱、模型版本、驗證類型。
    "status", "reasonCode", "modelName", "modelVersion", "verificationType",
    # 是否通過活體檢查、是否通過身分比對、活體分數、比對分數（分數在被拒絕時可能是 null）。
    "livenessVerified", "identityVerified", "livenessScore", "faceMatchScore",
}
# AI 服務讀 provider 回應的大小上限（services/ai/app/verification.py 的 MAX_PROVIDER_BYTES，16 KiB）；
# 回應超過這個大小，AI 服務會直接判成 PROVIDER_INVALID_RESPONSE。
MAX_PROVIDER_BYTES = 16 * 1024


class ProviderApiTests(unittest.TestCase):
    """/verify 與認證邊界的測試：每個測試都用假模型，結果完全由 tests/fakes.py 的設定決定。"""

    def client(self, **kwargs):
        """建立一個全新的測試客戶端。

        參數：
            **kwargs：原樣轉給 tests.fakes.verifier，例如 detector=FakeDetector(...)，用來替換某個假模型。
        回傳：
            TestClient：包著一個新建的 FastAPI app，可以直接呼叫 .post()／.get()。
        設計理由：
            假模型會「依呼叫順序消耗」預先準備好的結果（例如 FakeDetector 每次 detect 會 pop 掉一筆），
            所以每個測試都要用新的 app 與新的假模型，避免上一個請求吃掉下一個請求要用的結果。
        """
        # 用測試設定和假模型建立 app，再包成 TestClient 回傳。
        return TestClient(create_app(SETTINGS, verifier(**kwargs)))

    def test_response_matches_provider_contract(self):
        """規則：成功的回應格式必須和 AI 服務的 ProviderResult 契約完全一致。

        為什麼重要：AI 服務會嚴格檢查 provider 的回應（Content-Type、大小、欄位），
        任何一點不符都會被當成 PROVIDER_INVALID_RESPONSE，使用者就永遠無法完成驗證。
        情境：預設的請求只有自拍與主照片、沒有 liveCapture（即時鏡頭的動作影格），
        假模型讓偵測、比對（相似度 0.8）、防偽都通過，依 policy 版本 3 應回 unavailable / LIVE_CAPTURE_REQUIRED。
        """
        # 帶正確 token 送出預設請求（一張自拍、一張主照片、沒有動作影格）。
        response = self.client().post(VERIFY_PATH, json=verify_request(), headers=HEADERS)
        # 檢查 HTTP 狀態碼是 200：判定結果（即使不是 verified）是正常回應，不是錯誤。
        self.assertEqual(response.status_code, 200)
        # 檢查 Content-Type 去掉 ";" 後面的參數（例如 charset）後是 application/json，AI 服務就是這樣比對的。
        self.assertEqual(response.headers["content-type"].split(";")[0], "application/json")
        # 檢查回應的原始位元組數不超過 AI 服務願意讀的 16 KiB。
        self.assertLessEqual(len(response.content), MAX_PROVIDER_BYTES)
        # 把回應內容解析成 Python 字典，後面檢查欄位。
        body = response.json()
        # set(body) 取出所有欄位名稱；必須和契約的欄位集合完全相同，不多也不少。
        self.assertEqual(set(body), PROVIDER_RESULT_FIELDS)
        # 驗證類型固定是 identity_verification（VerifyResponse 的預設值），AI 服務只接受這個值。
        self.assertEqual(body["verificationType"], "identity_verification")
        # 沒有即時鏡頭的動作挑戰：就算全部通過，也只能回 unavailable / LIVE_CAPTURE_REQUIRED。
        self.assertEqual((body["status"], body["reasonCode"]), ("unavailable", "LIVE_CAPTURE_REQUIRED"))

    def test_auth_checked_before_body(self):
        """規則：先驗 token，才讀 body 與解析 JSON。

        為什麼重要：沒有認證的人不應該能讓服務花力氣讀大檔、解析 JSON，
        也不應該能從 422（格式錯誤）與 401（未認證）的差異推測出 API 的格式。
        """
        # 建立一個客戶端，這個測試的所有請求都會在認證這關被擋下，不會消耗假模型的結果，所以可以共用。
        client = self.client()
        # 沒帶 Authorization、body 還不是 JSON：如果先解析 body 會回 422，先驗 token 才會回 401。
        self.assertEqual(client.post(VERIFY_PATH, content="not json").status_code, 401)
        # 兩種錯誤的標頭：token 錯誤，以及 token 正確但少了 "Bearer " 前綴。
        for header in ("Bearer wrong", "provider-test-secret"):
            # subTest 讓每個標頭各自回報成功或失敗，其中一個失敗不會讓另一個不被檢查。
            with self.subTest(header=header):
                # 帶格式正確的請求與錯誤的標頭送出。
                response = client.post(VERIFY_PATH, json=verify_request(), headers={"Authorization": header})
                # 兩種都必須回 401，標頭必須和 "Bearer <token>" 完全相同才算通過。
                self.assertEqual(response.status_code, 401)

    def test_missing_token_disables_verify(self):
        """規則：provider 沒設定 token 時，/verify 一律回 503 PROVIDER_AUTH_NOT_CONFIGURED。

        為什麼重要：忘記設定 token 時不能變成「不用認證就能呼叫」，而是整個 /verify 停用（fail closed）。
        """
        # Settings() 的 token 是空字串，代表沒有設定；模型用假模型，確保 503 不是因為缺模型。
        client = TestClient(create_app(Settings(), verifier()))
        # 就算帶了標頭也一樣，因為服務端根本沒有可以比對的 token。
        response = client.post(VERIFY_PATH, json=verify_request(), headers=HEADERS)
        # 503 表示服務目前無法使用（設定不完整），而不是 401（請求者的錯）。
        self.assertEqual(response.status_code, 503)
        # 錯誤代碼固定，方便部署時從回應看出是 token 沒設定。
        self.assertEqual(response.json()["code"], "PROVIDER_AUTH_NOT_CONFIGURED")

    def test_body_limit_even_without_content_length(self):
        """規則：body 大小上限用「實際收到的位元組」計算，不依賴 Content-Length 標頭。

        為什麼重要：攻擊者可以不帶或謊報 Content-Length；如果只看標頭，服務可能被灌爆記憶體。
        ProviderBoundary 會邊收邊累加，一超過 MAX_BODY_BYTES 就回 413。
        """
        def chunks():
            """產生 16 段、每段 1 MiB 的資料（總共 16 MiB）。

            回傳：
                generator：每次 yield 一段 bytes。用 generator 當 body，客戶端無法事先知道總長度，
                所以請求不會帶 Content-Length，伺服器只能一段一段收。
            """
            # 重複 16 次，每次送出一段。
            for _ in range(16):
                # 一段 1 MiB（1024 × 1024 個 "a" 位元組）。
                yield b"a" * (1024 * 1024)
        # 先確認 16 MiB 確實超過上限，否則這個測試就沒有意義。
        self.assertGreater(16 * 1024 * 1024, MAX_BODY_BYTES)
        # 帶正確 token 送出分段的 body（認證通過後才會開始讀 body）。
        response = self.client().post(VERIFY_PATH, content=chunks(), headers=HEADERS)
        # 413 Payload Too Large：讀到超過上限時就停止並回傳，不會把 16 MiB 全部收進記憶體。
        self.assertEqual(response.status_code, 413)

    def test_largest_legal_request_not_cut_by_body_limit(self):
        """規則：每個欄位都用到上限的「最大合法請求」，不能被 body 大小上限誤擋成 413。

        為什麼重要：MAX_BODY_BYTES 是由各欄位的上限加總再加上 JSON 額外開銷估出來的；
        如果估太小，真實使用者上傳最大張的照片時會莫名失敗。
        """
        # 組出每個 Base64 欄位都是最大長度的請求，再轉成 JSON 字串並編碼成 UTF-8 位元組。
        body = json.dumps({
            # 自拍：Base64 長度剛好等於上限；"A" 是合法的 Base64 字元，解碼後是一串 0 位元組。
            "imageBase64": "A" * MAX_BASE64_LENGTH, "mimeType": "image/jpeg", "requestId": "test-request-1",
            # 參照照片（主照片）最多 1 張，Base64 長度也用到上限。
            "referenceImages": [{"imageBase64": "A" * MAX_REFERENCE_BASE64_LENGTH, "mimeType": "image/jpeg"}],
            # 即時鏡頭的動作挑戰。
            "liveCapture": {
                # 挑戰 id，只能是英數字、底線、連字號。
                "challengeId": "challenge-1",
                # 動作影格：用到上限 3 張（MAX_ACTION_FRAMES），每張 Base64 長度也用到上限。
                "frames": [
                    # 每個動作一張影格。
                    {"action": action, "imageBase64": "A" * MAX_FRAME_BASE64_LENGTH, "mimeType": "image/jpeg"}
                    # 取三個不同的合法動作。
                    for action in ("turn_left", "turn_right", "look_up")
                ],
            },
        }).encode()
        # 先確認這個最大請求的總位元組數沒有超過上限（也就是上限有留足 JSON 鍵名、引號等額外開銷）。
        self.assertLessEqual(len(body), MAX_BODY_BYTES)
        # 用 content 直接送位元組，所以要自己加上 Content-Type: application/json，FastAPI 才會當成 JSON 解析。
        response = self.client().post(VERIFY_PATH, content=body, headers={**HEADERS, "Content-Type": "application/json"})
        # 通過了大小上限與欄位驗證，進到影像解碼才失敗，所以是 400，不能在讀 body 時就被 413 擋掉。
        # （自拍的 Base64 長度用到上限時解碼後是 5 MiB + 1 byte，會先被判成 IMAGE_TOO_LARGE；
        #  而且內容全是 0 位元組，本來也不是合法影像。兩種情況都回 400。）
        self.assertEqual(response.status_code, 400)

    def test_body_limit_matches_ai_service(self):
        """規則：provider 的 body 上限必須和 AI 服務的上限是同一個數字。

        為什麼重要：兩邊不同時，AI 服務允許的請求可能被 provider 擋掉（或反過來），造成難以追查的失敗。
        兩個服務各自在測試裡寫死同一個數字，任何一邊改了，自己那邊的測試就會失敗。
        """
        # services/ai/app/schemas.py 的 MAX_BODY_BYTES 必須相同；兩邊的測試各自鎖同一個數字。
        # 12_588_044 = 自拍 Base64 上限 6,990,508 + 1 張主照片（1,398,104 + 256）
        #            + 3 張動作影格（每張 1,398,104 + 256）+ 4,096。
        self.assertEqual(MAX_BODY_BYTES, 12_588_044)

    def test_validation_errors_do_not_echo_input(self):
        """規則：請求格式錯誤（422）時，錯誤訊息不能包含使用者送來的內容。

        為什麼重要：FastAPI 預設的 422 會把出錯的輸入值放進回應；自拍的 Base64 是生物特徵資料，
        不能出現在回應或 log 裡。main.py 的 request_error 會把錯誤換成固定訊息。
        """
        # 在合法請求之外多塞一個欄位；schemas 設了 extra="forbid"，多餘欄位會驗證失敗。
        request = {**verify_request(), "untrustedField": "secret-selfie-content"}
        # 帶正確 token 送出（認證通過，才會走到欄位驗證）。
        response = self.client().post(VERIFY_PATH, json=request, headers=HEADERS)
        # 欄位驗證失敗回 422。
        self.assertEqual(response.status_code, 422)
        # 回應文字裡不能有自拍的 Base64 內容。
        self.assertNotIn(request["imageBase64"], response.text)
        # 回應文字裡也不能有那個多餘欄位的值。
        self.assertNotIn("secret-selfie-content", response.text)

    def test_live_capture_limits(self):
        """規則：動作影格的數量必須是 1～3 張，而且動作只能是四個合法動作之一。

        為什麼重要：限制張數能控制運算量與 body 大小；限制動作名稱，才不會收到 pipeline 不認識、
        永遠判不出結果的動作。這些都在 schemas（pydantic）這層就擋掉，回 422。
        """
        # 這三個請求都會在欄位驗證就被擋下，不會用到假模型，所以共用同一個客戶端。
        client = self.client()
        # 4 張影格超過 MAX_ACTION_FRAMES（3 張）。
        too_many = verify_request(actions=["turn_left", "turn_right", "look_up", "look_down"])
        # 必須回 422。
        self.assertEqual(client.post(VERIFY_PATH, json=too_many, headers=HEADERS).status_code, 422)
        # "blink"（眨眼）不是合法動作（只接受 turn_left／turn_right／look_up／look_down）。
        unknown = verify_request(actions=["blink"])
        # 必須回 422。
        self.assertEqual(client.post(VERIFY_PATH, json=unknown, headers=HEADERS).status_code, 422)
        # 有 liveCapture 但一張影格都沒有（frames 至少要 1 張）。
        empty = verify_request(actions=[])
        # 必須回 422；不能讓「空的動作挑戰」被當成通過挑戰。
        self.assertEqual(client.post(VERIFY_PATH, json=empty, headers=HEADERS).status_code, 422)

    def test_too_many_references_rejected(self):
        """規則：參照照片（主照片）最多 1 張（MAX_REFERENCE_IMAGES），多了回 422。

        為什麼重要：pipeline 只拿第一張主照片比對，多送的照片只會浪費頻寬與解碼時間。
        """
        # 送出帶 2 張參照照片的請求。
        response = self.client().post(VERIFY_PATH, json=verify_request(references=2), headers=HEADERS)
        # 必須在欄位驗證就被擋下。
        self.assertEqual(response.status_code, 422)

    def test_bad_image_returns_fixed_code(self):
        """規則：影像無法解碼時回 400 與固定代碼 INVALID_IMAGE，而且不回顯輸入內容。

        為什麼重要：呼叫端（AI 服務）靠固定代碼判斷錯誤種類；回顯輸入則可能洩漏資料。
        """
        # "????" 長度 4，符合欄位的最小長度，但不是合法的 Base64 字元，解碼時才會失敗。
        request = {**verify_request(), "imageBase64": "????"}
        # 帶正確 token 送出。
        response = self.client().post(VERIFY_PATH, json=request, headers=HEADERS)
        # 影像問題回 400（請求本身格式正確，是內容不對）。
        self.assertEqual(response.status_code, 400)
        # 代碼固定為 INVALID_IMAGE（自拍沒有前綴；主照片與動作影格才會加 REFERENCE_／FRAME_）。
        self.assertEqual(response.json()["code"], "INVALID_IMAGE")
        # 回應裡不能出現送進來的內容。
        self.assertNotIn("????", response.text)

    def test_model_failure_fails_closed(self):
        """規則：模型執行時丟出任何例外，都回 503 MODEL_FAILED，且不洩漏例外內容。

        為什麼重要：模型出錯時絕不能回傳「通過」（fail closed）；例外訊息可能含內部細節，不能送出去。
        """
        # 假的偵測器：第一次呼叫（偵測自拍）就丟出帶有敏感字樣的例外，第二次的結果不會被用到。
        detector = FakeDetector(RuntimeError("sensitive-model-error"), [face()])
        # 用這個會出錯的偵測器建立客戶端並送出請求。
        response = self.client(detector=detector).post(VERIFY_PATH, json=verify_request(), headers=HEADERS)
        # 模型失敗回 503；AI 服務收到非 200 會記成 PROVIDER_UNAVAILABLE。
        self.assertEqual(response.status_code, 503)
        # 代碼固定為 MODEL_FAILED。
        self.assertEqual(response.json()["code"], "MODEL_FAILED")
        # 例外訊息裡的 "sensitive" 不能出現在回應中。
        self.assertNotIn("sensitive", response.text)

    def test_rejections_serialize_null_scores(self):
        """規則：在算出分數之前就被拒絕時，兩個分數欄位要是 JSON 的 null，而且回應是嚴格合法的 JSON。

        為什麼重要：AI 服務要求分數是 0～1 的數字或 null；如果出現 NaN、Infinity 之類的值，
        就不是標準 JSON，AI 服務會解析失敗。
        """
        # 假的偵測器：第一次（自拍）回傳空清單，代表沒偵測到臉；第二次的結果不會被用到。
        detector = FakeDetector([], [face()])
        # 送出請求並直接解析成字典。
        body = self.client(detector=detector).post(VERIFY_PATH, json=verify_request(), headers=HEADERS).json()
        # 沒偵測到臉回 rejected / NO_FACE_DETECTED。
        self.assertEqual((body["status"], body["reasonCode"]), ("rejected", "NO_FACE_DETECTED"))
        # 還沒做防偽，活體分數是 null（Python 的 None）。
        self.assertIsNone(body["livenessScore"])
        # 還沒做比對，比對分數也是 null。
        self.assertIsNone(body["faceMatchScore"])
        # allow_nan=False：遇到 NaN／Infinity 會丟例外；沒丟例外就代表內容全是標準 JSON 可表示的值。
        json.dumps(body, allow_nan=False)


class MissingModelTests(unittest.TestCase):
    """模型檔不存在時的行為，以及不需要認證的 /health。"""

    def test_missing_models_fail_closed(self):
        """規則：模型檔不存在時，服務仍會啟動，但 /health 回 503、/verify 回 503 MODELS_UNAVAILABLE。

        為什麼重要：缺模型時不能讓驗證「看起來成功」（fail closed）；/health 回 503，
        Docker 的 HEALTHCHECK 就會把容器標成 unhealthy，維運人員能馬上發現模型沒有載入。
        """
        # 建立一個暫時的空資料夾，離開 with 區塊時自動刪除。
        with tempfile.TemporaryDirectory() as empty:
            # 不注入假模型，讓 create_app 自己去空資料夾載入模型；載入失敗時 verifier 會是 None。
            client = TestClient(create_app(Settings(token=SETTINGS.token, model_dir=Path(empty))))
            # 健康檢查必須回 503，表示服務無法正常使用。
            self.assertEqual(client.get("/health").status_code, 503)
            # 帶正確 token 呼叫 /verify（token 正確，才能確認 503 是因為缺模型而不是認證）。
            response = client.post(VERIFY_PATH, json=verify_request(), headers=HEADERS)
            # 必須回 503。
            self.assertEqual(response.status_code, 503)
            # 代碼固定為 MODELS_UNAVAILABLE。
            self.assertEqual(response.json()["code"], "MODELS_UNAVAILABLE")

    def test_health_without_auth(self):
        """規則：/health 不需要認證，模型載入正常時回 200 與 models: loaded。

        為什麼重要：Docker 的 HEALTHCHECK 與 compose 在容器內直接打 /health，不會帶 token；
        ProviderBoundary 只保護 /verify，其他路徑直接放行。
        """
        # 用假模型建立 app，不帶任何標頭呼叫 /health。
        response = TestClient(create_app(SETTINGS, verifier())).get("/health")
        # 不需要 token 也回 200。
        self.assertEqual(response.status_code, 200)
        # 回應中標示模型已載入。
        self.assertEqual(response.json()["models"], "loaded")


# 直接執行這個檔案（python -m tests.test_api）時才跑測試；被 unittest discover 匯入時不會進來。
if __name__ == "__main__":
    # 找出這個模組裡所有 TestCase 並執行。
    unittest.main()
