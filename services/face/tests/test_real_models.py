"""用真的模型檔確認 OpenCV／onnxruntime 能載入並執行；只在容器內（有 /app/models）時執行。

依 SECURITY.md，repo 不放人臉照片，所以這裡只用合成影像，確認的是「模型接得上、輸出格式正確」，
不是辨識準確度。用真實照片的手動檢查見 tests/smoke_local_images.py。

為什麼需要這支測試：其他測試都用假模型（tests/fakes.py），測不到「真的模型檔能不能載入、
輸入格式對不對、輸出長什麼樣子」。模型檔是在 Docker build 時下載並轉換的（見 Dockerfile），
所以這支測試要在容器裡執行；在沒有模型檔的電腦上會自動略過（skip），不算失敗。
"""

# unittest：Python 內建的測試框架。
import unittest

# numpy：建立合成影像（純灰色、隨機雜訊）與假的偵測結果。
import numpy as np

# policy：判定政策，這裡只用到 YuNet 的偵測信心門檻 DETECTION_SCORE_THRESHOLD。
from app import policy
# Settings：讀環境變數取得模型資料夾（FACE_MODEL_DIR，沒設定時是 /app/models）。
from app.config import Settings
# Face：一筆人臉偵測結果；load_models：載入 YuNet、SFace、MiniFASNet 三個真模型。
from app.models import Face, load_models
# FaceVerifier：完整的判定流程，這裡用真模型跑一次。
from app.pipeline import FaceVerifier
# VerifyRequest：/verify 的請求格式（pydantic 模型），把字典轉成 pipeline 需要的物件。
from app.schemas import VerifyRequest

# verify_request：產生格式正確的請求字典，自拍是一張純色的 128×128 JPEG（裡面沒有臉）。
from tests.fakes import verify_request

# 模組載入時就決定模型資料夾，下面的 skipUnless 要用它判斷模型檔在不在。
MODEL_DIR = Settings.from_env().model_dir


# 只有 YuNet 模型檔存在時才執行整個類別；否則全部標記為略過，並顯示原因。
@unittest.skipUnless((MODEL_DIR / "face_detection_yunet_2023mar.onnx").is_file(), "模型檔不存在（只在容器內執行）")
class RealModelTests(unittest.TestCase):
    """用真的三個模型檢查輸入輸出格式；只用合成影像，不測準確度。"""

    @classmethod
    def setUpClass(cls):
        """整個類別開始前只執行一次：載入真模型，給所有測試共用。

        參數：
            cls：這個測試類別本身（classmethod 的第一個參數）。
        可能丟出的錯誤：
            ModelError("MODEL_FILES_MISSING")：YuNet 以外的模型檔缺少時（YuNet 缺少時整個類別已被略過）。
        設計理由：載入 ONNX 模型比較慢，每個測試各載一次會浪費時間；這些測試都不會改變模型的狀態。
        """
        # 用和正式執行相同的偵測門檻載入三個模型，存成類別屬性，每個測試用 self.models 取用。
        cls.models = load_models(MODEL_DIR, policy.DETECTION_SCORE_THRESHOLD)

    def test_blank_image_has_no_face(self):
        """規則：沒有臉的影像，偵測器要回傳空清單。

        為什麼重要：OpenCV 的 YuNet 在沒有臉時回傳 None 而不是空陣列，models.py 要把它轉成 []，
        detect 才會和假模型一樣永遠回傳清單（型別是 list[Face]）；pipeline 看到空清單就回 NO_FACE_DETECTED。
        這也確認真模型不會把純灰色的畫面誤判成臉。
        """
        # 建立一張高 480、寬 640、3 通道、每個像素值都是 128 的純灰色影像。
        image = np.full((480, 640, 3), 128, dtype=np.uint8)
        # 真的 YuNet 偵測結果必須是空清單。
        self.assertEqual(self.models.detector.detect(image), [])

    def test_pipeline_rejects_faceless_selfie_with_real_models(self):
        """規則：用真模型跑完整流程，沒有臉的自拍要回 rejected / NO_FACE_DETECTED。

        為什麼重要：確認真模型接進 FaceVerifier 後整條流程能跑（解碼、縮圖、偵測、判定），
        而不是在某個環節丟出例外。
        """
        # 把請求字典轉成 VerifyRequest，交給用真模型建立的 FaceVerifier 判定。
        result = FaceVerifier(self.models).verify(VerifyRequest(**verify_request()))
        # 純色自拍裡沒有臉，第一步就會被拒絕。
        self.assertEqual((result.status, result.reasonCode), ("rejected", "NO_FACE_DETECTED"))

    def test_embedding_is_unit_length_and_deterministic(self):
        """規則：SFace 的人臉特徵向量是 128 維、長度為 1，而且同樣的輸入每次都得到同樣的結果。

        為什麼重要：pipeline 用兩個向量的內積當作餘弦相似度（cosine similarity），
        向量長度必須是 1 內積才等於餘弦相似度；結果不穩定的話，同一張照片的判定會忽高忽低。
        """
        # 固定種子 0 的亂數產生器，每次執行都產生相同的雜訊。
        rng = np.random.default_rng(0)
        # 產生一張 240×240、3 通道、像素值 0～255 的隨機雜訊影像。
        image = rng.integers(0, 256, size=(240, 240, 3), dtype=np.uint8)
        # YuNet 每筆偵測結果是 15 個 float32：框 4 個、5 個臉部點各 (x, y) 共 10 個、信心分數 1 個。
        row = np.zeros(15, dtype=np.float32)
        # 前 4 個是人臉框 (x, y, w, h) = (60, 60, 120, 120)；接著 10 個是 5 個臉部點的座標。
        # 合成的 5 點位置（右眼、左眼、鼻尖、右嘴角、左嘴角），只為了讓 alignCrop 有東西可對齊。
        # 依序是：右眼 (95, 105)、左眼 (145, 105)、鼻尖 (120, 130)、右嘴角 (100, 155)、左嘴角 (140, 155)。
        row[:14] = (60, 60, 120, 120, 95, 105, 145, 105, 120, 130, 100, 155, 140, 155)
        # 第 15 個值是偵測信心分數。
        row[14] = 0.95
        # 組成 Face 物件；embed 只用 row 的前 14 個值做對齊。
        synthetic = Face(box=(60.0, 60.0, 120.0, 120.0), row=row, score=0.95)
        # 同一張影像、同一個臉，算第一次特徵向量。
        first = self.models.embedder.embed(image, synthetic)
        # 再算第二次，用來確認結果不會變。
        second = self.models.embedder.embed(image, synthetic)
        # SFace 的特徵是 128 維的一維陣列。
        self.assertEqual(first.shape, (128,))
        # 向量長度（L2 norm）要是 1（比對到小數點後 5 位），代表已經正規化。
        self.assertAlmostEqual(float(np.linalg.norm(first)), 1.0, places=5)
        # 兩次結果每個元素的差距都不超過 0.000001。
        np.testing.assert_allclose(first, second, atol=1e-6)

    def test_spoof_model_outputs_probability(self):
        """規則：MiniFASNet 防偽的輸出是 0～1 之間的真人機率，以及一個 bool 的判定。

        為什麼重要：回應裡的 livenessScore 必須在 0～1 之間（AI 服務會檢查）；
        is_real 必須是真正的 bool，pipeline 用它決定是否回 SPOOF_SUSPECTED。
        """
        # 固定種子 1 的亂數產生器。
        rng = np.random.default_rng(1)
        # 產生一張高 640、寬 480 的隨機雜訊影像（形狀是 (高, 寬, 通道)）。
        image = rng.integers(0, 256, size=(640, 480, 3), dtype=np.uint8)
        # 用框 (x=160, y=200, w=160, h=200) 跑兩個 MiniFASNet 模型並合併結果。
        result = self.models.spoof.assess(image, (160.0, 200.0, 160.0, 200.0))
        # 真人機率不小於 0。
        self.assertGreaterEqual(result.real_probability, 0.0)
        # 真人機率不大於 1。
        self.assertLessEqual(result.real_probability, 1.0)
        # is_real 必須是 Python 的 bool（不是 numpy 的 bool_ 或數字）。
        self.assertIsInstance(result.is_real, bool)


# 直接執行這個檔案時才跑測試；被 unittest discover 匯入時不會進來。
if __name__ == "__main__":
    # 找出這個模組裡所有 TestCase 並執行。
    unittest.main()
