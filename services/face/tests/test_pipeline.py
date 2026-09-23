"""app/pipeline.py（FaceVerifier）的單元測試：「只有上傳自拍檔、沒有即時鏡頭」的情境。

測試範圍：
- DecisionTests：各種情況下回傳的 status／reasonCode／分數是否符合判定政策（app/policy.py，版本 3）。
- GeometryTests：大張照片縮圖後，偵測、特徵比對、防偽各自拿到的影像與座標是否正確。
- InputTests：影像本身有問題（base64 壞掉、格式不符、尺寸太小）時丟出的錯誤碼，以及參照照片的前綴。

即時鏡頭（liveCapture，動作挑戰）的情境在 test_live.py。

所有模型都用 tests/fakes.py 的假模型，所以不需要模型檔，也能精準控制每個情境。
執行方式（在 services/face 目錄下）：python -m unittest tests.test_pipeline
"""

# unittest：Python 內建的測試框架；每個繼承 unittest.TestCase、名稱以 test_ 開頭的方法都是一個測試。
import unittest

# policy：判定政策與門檻（MATCH_THRESHOLD、MIN_FACE_SIDE、MODEL_NAME 等），測試直接引用，門檻改了測試也跟著走。
from app import policy
# InvalidImage：影像解碼失敗時 pipeline 丟出的例外，錯誤碼就是例外訊息（str(例外)）。
from app.imaging import InvalidImage
# VerifyRequest：/verify 請求的 pydantic 模型；測試用它把 dict 轉成 pipeline 需要的請求物件。
from app.schemas import VerifyRequest

# 共用的假模型與假資料工具，說明見 tests/fakes.py。
from tests.fakes import FakeDetector, FakeEmbedder, FakeSpoof, face, verifier, verify_request


def run(request=None, **kwargs):
    """用假模型跑一次完整的驗證流程，回傳 VerifyResponse。

    參數：
        request：請求 dict；沒給時用 verify_request() 的預設值
            （一張 128×128 自拍、一張主照片、沒有 liveCapture）。
        **kwargs：原封不動轉給 tests.fakes.verifier()，可指定 detector、similarity、spoof、embedder。

    回傳：
        VerifyResponse：pipeline 的判定結果。

    可能丟出的錯誤：
        - pydantic.ValidationError：request 不符合 VerifyRequest 的格式；
        - InvalidImage：影像解碼或檢查失敗（InputTests 就是在測這個）。
    """
    # 1. verifier(**kwargs) 建立使用假模型的 FaceVerifier；
    # 2. VerifyRequest(**dict) 做 pydantic 格式驗證並轉成請求物件（request 為 None 時改用預設請求）；
    # 3. 呼叫 verify() 跑完整流程並回傳結果。
    return verifier(**kwargs).verify(VerifyRequest(**(request or verify_request())))


class DecisionTests(unittest.TestCase):
    """判定結果的測試：確認每種情況回傳正確的 status、reasonCode、驗證旗標與分數。"""

    def test_all_checks_pass_is_never_verified_without_live_capture(self):
        """規則：只有上傳的自拍檔（沒有 liveCapture）時，就算全部檢查通過，也只能回 unavailable。

        為什麼重要：上傳的照片可能是事先拍好、或別人的照片，無法證明是活人「當下」拍的；
        只有即時鏡頭的動作挑戰通過才能回 verified（policy 版本 3 的核心規則）。
        """
        # 預設情境：自拍與主照片各一張臉、防偽判定真人、相似度 0.8（高於門檻 0.363）。
        result = run(similarity=0.8)
        # 狀態必須是 unavailable（無法完成驗證），不是 verified。
        self.assertEqual(result.status, "unavailable")
        # 原因碼告訴前端：需要改用即時鏡頭。
        self.assertEqual(result.reasonCode, "LIVE_CAPTURE_REQUIRED")
        # 身分比對本身是通過的（自拍和主照片是同一人），這個資訊仍然如實回報。
        self.assertTrue(result.identityVerified)
        # 但活體（liveness）沒有被證明，所以必須是 False。
        self.assertFalse(result.livenessVerified)
        # 比對分數 = 假 embedder 給的相似度 0.8（pipeline 會四捨五入到小數 4 位）。
        self.assertEqual(result.faceMatchScore, 0.8)
        # 活體分數 = FakeSpoof 預設的真人機率 0.9。
        self.assertEqual(result.livenessScore, 0.9)

    def test_missing_reference_unavailable_without_running_models(self):
        """規則：沒有主照片（參照照片）就直接回 unavailable，而且完全不執行模型。

        為什麼重要：沒有參照就無從做 1:1 比對；提早結束也能省下模型運算。
        回 unavailable 而不是 rejected：問題不在鏡頭前的人，NestJS 不會因此撤銷既有的驗證狀態。
        """
        # 建一個沒有準備任何結果的偵測器：如果 pipeline 呼叫了 detect，pop 會丟出 IndexError，測試就會失敗。
        detector = FakeDetector()
        # 送出沒有參照照片的請求（referenceImages 是空清單）。
        result = run(verify_request(references=0), detector=detector)
        # 必須回 unavailable／REFERENCE_PHOTO_REQUIRED。
        self.assertEqual((result.status, result.reasonCode), ("unavailable", "REFERENCE_PHOTO_REQUIRED"))
        # 偵測器完全沒被呼叫過（沒有記錄到任何影像 shape）。
        self.assertEqual(detector.shapes, [])

    def test_selfie_face_problems(self):
        """規則：自拍必須剛好一張夠大的臉，否則回對應的原因碼（沒有前綴）。

        為什麼重要：沒有臉無法比對；多張臉無法確定要比對哪一個人；臉太小時 SFace 對齊放大後細節不足，
        比對結果不可靠。這些都應該明確拒絕，並讓使用者知道原因以便重拍。
        """
        # 原因碼 → 自拍的偵測結果：
        cases = {
            # 空清單：偵測不到任何臉。
            "NO_FACE_DETECTED": [],
            # 兩張臉（第二張把 x 移到 120，只是讓兩張臉的位置不同）。
            "MULTIPLE_FACES_DETECTED": [face(), face(x=120)],
            # 臉框寬度比最小邊長少 1 像素（63 < 64），剛好不合格，測試邊界。
            "FACE_TOO_SMALL": [face(w=policy.MIN_FACE_SIDE - 1)],
        }
        # 逐一測試每個情境。
        for reason, faces in cases.items():
            # subTest：某個情境失敗時會標出是哪一個 reason，而且其他情境仍會繼續跑。
            with self.subTest(reason=reason):
                # 第 1 次偵測（自拍）回傳有問題的結果，第 2 次（主照片）正常。
                result = run(detector=FakeDetector(faces, [face()]))
                # 必須回 rejected 與對應的原因碼。
                self.assertEqual((result.status, result.reasonCode), ("rejected", reason))
                # 在比對之前就結束，所以沒有比對分數（None，序列化成 JSON 的 null）。
                self.assertIsNone(result.faceMatchScore)

    def test_reference_face_problems_are_prefixed(self):
        """規則：主照片的臉有問題時，原因碼加上 REFERENCE_ 前綴。

        為什麼重要：同樣是「沒有臉」，要讓前端分辨是自拍的問題（請重拍）還是主照片的問題
        （請換一張主照片），使用者才知道該修正哪一張。
        主照片的問題回 unavailable：NestJS 收到 rejected 會撤銷既有的驗證狀態，但錯的是照片，不是鏡頭前的人。
        """
        # 原因碼（含前綴）→ 主照片的偵測結果：
        cases = {
            # 主照片偵測不到臉。
            "REFERENCE_NO_FACE_DETECTED": [],
            # 主照片有兩張臉。
            "REFERENCE_MULTIPLE_FACES_DETECTED": [face(), face(x=120)],
            # 主照片的臉框高度比最小邊長少 1 像素（這次改測高度，確認寬、高都會檢查）。
            "REFERENCE_FACE_TOO_SMALL": [face(h=policy.MIN_FACE_SIDE - 1)],
        }
        # 逐一測試每個情境。
        for reason, faces in cases.items():
            # 用 subTest 分開標示每個情境。
            with self.subTest(reason=reason):
                # 第 1 次偵測（自拍）正常，第 2 次（主照片）回傳有問題的結果。
                result = run(detector=FakeDetector([face()], faces))
                # 必須回 unavailable 與帶有 REFERENCE_ 前綴的原因碼。
                self.assertEqual((result.status, result.reasonCode), ("unavailable", reason))

    def test_spoof_rejected_even_when_faces_match(self):
        """規則：被動防偽判定不是真人（例如翻拍螢幕、印出來的照片）時，就算臉比對吻合也要拒絕。

        為什麼重要：拿別人的照片對著鏡頭，臉當然會和主照片吻合；防偽檢查就是要擋這種冒用。
        """
        # 相似度 0.9（吻合），但防偽模型判定不是真人、真人機率只有 0.2。
        result = run(similarity=0.9, spoof=FakeSpoof(real_probability=0.2, is_real=False))
        # 必須回 rejected／SPOOF_SUSPECTED（防偽檢查在比對結果之前判定）。
        self.assertEqual((result.status, result.reasonCode), ("rejected", "SPOOF_SUSPECTED"))
        # 活體未通過。
        self.assertFalse(result.livenessVerified)
        # 身分比對結果仍如實回報為通過（pipeline 把 identity 帶進這個結果）。
        self.assertTrue(result.identityVerified)
        # 活體分數如實回報 0.2，方便事後稽核。
        self.assertEqual(result.livenessScore, 0.2)

    def test_mismatch_rejected(self):
        """規則：自拍和主照片的相似度低於門檻時，判定不是同一人並拒絕。

        為什麼重要：這是真人驗證的核心，確認「照片中的人」就是「正在驗證的人」。
        """
        # 相似度 0.2，低於 MATCH_THRESHOLD（0.363）。
        result = run(similarity=0.2)
        # 必須回 rejected／FACE_MISMATCH。
        self.assertEqual((result.status, result.reasonCode), ("rejected", "FACE_MISMATCH"))
        # 身分比對未通過。
        self.assertFalse(result.identityVerified)

    def test_match_threshold_boundary(self):
        """規則：相似度「大於或等於」門檻才算同一人（>=，不是 >）。

        為什麼重要：邊界值最容易寫錯（>= 與 > 之差）；fakes 的 _unit 讓相似度精確等於門檻，
        沒有浮點誤差，才能準確測到邊界。
        """
        # 相似度剛好等於門檻：必須判定為同一人。
        self.assertTrue(run(similarity=policy.MATCH_THRESHOLD).identityVerified)
        # 相似度比門檻少 0.0001：必須判定為不同人。
        self.assertEqual(run(similarity=policy.MATCH_THRESHOLD - 1e-4).reasonCode, "FACE_MISMATCH")

    def test_negative_similarity_clamped_to_zero(self):
        """規則：餘弦相似度可能是負數，但回傳的 faceMatchScore 要限制在 0～1。

        為什麼重要：回應的分數欄位規定在 0～1（schemas 的 Score，和 AI 服務的 ProviderResult 一致）；
        負數沒有被截成 0 的話，回應會驗證失敗，本來單純的「不吻合」就會變成錯誤。
        """
        # 相似度 -0.5 → 結果是 FACE_MISMATCH，但分數必須被截成 0.0。
        self.assertEqual(run(similarity=-0.5).faceMatchScore, 0.0)

    def test_results_carry_model_provenance(self):
        """規則：每個結果都要帶上模型名稱與版本（來源資訊，provenance）。

        為什麼重要：之後調整門檻或換模型時，能從紀錄知道每筆判定是用哪個模型、哪個政策版本做的；
        AI 服務的 ProviderResult 也要求 verified／rejected 一定要有這兩個欄位。
        """
        # 用預設情境跑一次。
        result = run()
        # 模型名稱等於 policy 定義的值。
        self.assertEqual(result.modelName, policy.MODEL_NAME)
        # 模型版本等於 policy 定義的值（包含各模型檔版本與 policy 版本）。
        self.assertEqual(result.modelVersion, policy.MODEL_VERSION)
        # 版本字串不超過 128 字元：這是回應 schema（以及 AI 服務 ProviderResult）的長度上限。
        self.assertLessEqual(len(result.modelVersion), 128)


class GeometryTests(unittest.TestCase):
    """座標與影像尺寸的測試：大張照片會先縮小再偵測，各步驟必須用對影像、用對座標。"""

    def test_large_selfie_detected_on_working_image_and_spoof_uses_original(self):
        """規則：
        - 偵測與 SFace 特徵在「縮小後的工作影像」上進行（長邊縮到 policy.WORKING_MAX_SIDE = 640）；
        - 防偽用「原圖」加上「換算回原圖座標的臉框」。

        為什麼重要：YuNet 適合偵測中等大小的臉，所以大圖要先縮小；但 MiniFASNet 訓練時是從原圖裁切，
        防偽要用原圖。座標系搞混的話，裁切到的位置會錯，判定就不可靠。
        """
        # 自拍偵測到的臉框（工作影像座標）：x=100、y=50、寬高 200；主照片偵測正常。
        detector = FakeDetector([face(x=100, y=50, w=200, h=200)], [face()])
        # 假防偽模型，用來檢查它收到的影像與框。
        spoof = FakeSpoof()
        # 假特徵模型（相似度 0.8），用來檢查它收到的影像與框。
        embedder = FakeEmbedder(0.8)
        # 自拍是 1280（寬）×960（高）的大圖。
        run(verify_request(size=(1280, 960)), detector=detector, spoof=spoof, embedder=embedder)
        # 長邊 1280 縮到 640，比例 0.5；防偽拿原圖與換算回原圖的框。
        # 偵測收到的影像是縮小後的 480（高）×640（寬）；numpy shape 是 (高, 寬, 通道)，所以取前兩個值比較。
        self.assertEqual(detector.shapes[0][:2], (480, 640))
        # 取出第一次（也是唯一一次）防偽呼叫收到的影像 shape 與臉框。
        image_shape, box = spoof.calls[0]
        # 防偽收到的是原圖 960（高）×1280（寬）。
        self.assertEqual(image_shape[:2], (960, 1280))
        # 臉框每個值都除以比例 0.5（即乘以 2）：(100, 50, 200, 200) → (200, 100, 400, 400)。
        self.assertEqual(box, (200.0, 100.0, 400.0, 400.0))
        # SFace 對齊用的 landmark 是在工作影像上偵測到的，影像也必須是同一張工作影像，座標才對得上。
        # 第 1 次 embed（自拍）：影像是工作影像 (480, 640, 3)，臉框是工作影像座標的原始值。
        self.assertEqual(embedder.calls[0], ((480, 640, 3), (100.0, 50.0, 200.0, 200.0)))
        # 第 2 次 embed（主照片，因為沒有動作影格）：影像 shape 必須和主照片偵測時的影像相同，
        # 代表特徵也是在主照片的工作影像上計算。
        self.assertEqual(embedder.calls[1][0], detector.shapes[1])


class InputTests(unittest.TestCase):
    """輸入影像本身有問題時的測試：pipeline 丟出 InvalidImage，錯誤碼由例外訊息表示。

    （在 app/main.py 裡，InvalidImage 會轉成 HTTP 400，body 的 code 就是這個錯誤碼。）
    """

    def test_invalid_reference_image_code_is_prefixed(self):
        """規則：主照片的 base64 壞掉時，錯誤碼是 REFERENCE_INVALID_IMAGE（加上前綴）。

        為什麼重要：讓呼叫端分得出是自拍壞掉還是主照片壞掉。
        """
        # 先組一個正常的請求。
        request = verify_request()
        # 把主照片內容換成 "????"：長度 4 能通過 schema 的最小長度，但不是合法的 base64。
        request["referenceImages"][0]["imageBase64"] = "????"
        # 預期會丟出 InvalidImage；caught 會保存丟出的例外，供之後檢查。
        with self.assertRaises(InvalidImage) as caught:
            # 執行驗證流程。
            run(request)
        # 例外訊息（錯誤碼）必須是帶有 REFERENCE_ 前綴的 INVALID_IMAGE。
        self.assertEqual(str(caught.exception), "REFERENCE_INVALID_IMAGE")

    def test_reference_mime_mismatch(self):
        """規則：主照片宣告的 mimeType 和實際檔案格式不符時，錯誤碼是 REFERENCE_IMAGE_TYPE_MISMATCH。

        為什麼重要：不能只相信呼叫端宣告的格式；實際格式和宣告不一致，可能是檔案被竄改或呼叫端有錯。
        """
        # 先組一個正常的請求（主照片實際上是 JPEG）。
        request = verify_request()
        # 故意把主照片宣告成 PNG。
        request["referenceImages"][0]["mimeType"] = "image/png"
        # 預期會丟出 InvalidImage。
        with self.assertRaises(InvalidImage) as caught:
            # 執行驗證流程。
            run(request)
        # 錯誤碼必須是帶前綴的 IMAGE_TYPE_MISMATCH。
        self.assertEqual(str(caught.exception), "REFERENCE_IMAGE_TYPE_MISMATCH")

    def test_selfie_too_small(self):
        """規則：自拍尺寸太小（短邊小於 app/imaging.py 的 MIN_DIMENSION 64）時，錯誤碼是
        INVALID_IMAGE_DIMENSIONS，而且自拍的錯誤碼「沒有」前綴。

        為什麼重要：太小的影像不可能有足夠細節做比對，應該在跑模型前就擋下。
        """
        # 預期會丟出 InvalidImage。
        with self.assertRaises(InvalidImage) as caught:
            # 送出 32×32 的自拍。
            run(verify_request(size=(32, 32)))
        # 錯誤碼是沒有前綴的 INVALID_IMAGE_DIMENSIONS。
        self.assertEqual(str(caught.exception), "INVALID_IMAGE_DIMENSIONS")


# 這個檔案被當成主程式執行時（例如在 services/face 目錄下執行 python -m tests.test_pipeline）才跑測試；
# 被 unittest 探索或被其他模組 import 時，__name__ 不是 "__main__"，不會重複執行。
if __name__ == "__main__":
    # 找出這個檔案中所有的測試並執行。
    unittest.main()
