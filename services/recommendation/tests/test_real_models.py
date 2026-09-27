"""用真的模型跑一次（只在映像裡、模型檔存在時執行）：載得起來、沒有臉回 None、壞影像丟 InvalidImage。
有臉的照片和 ml 匯入向量的一致性檢查見 README「已執行驗證」（種子照片不進版控，不放在這裡）。
"""
import io
import unittest
from pathlib import Path

from PIL import Image

from app.config import Settings
from appearance.pipeline import MODEL_VERSION, AppearanceEmbedder, InvalidImage

MODELS = Settings.from_env().models_dir


@unittest.skipUnless((MODELS / "selfie_multiclass_256x256.tflite").exists(), "模型檔不存在（只在 Docker 映像裡跑）")
class RealModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = AppearanceEmbedder.from_dir(Path(MODELS))

    def test_blank_image_has_no_face(self):
        buffer = io.BytesIO()
        Image.new("RGB", (320, 320), (180, 180, 180)).save(buffer, format="JPEG")
        self.assertIsNone(self.embedder.embed(buffer.getvalue()))

    def test_garbage_bytes_are_invalid(self):
        with self.assertRaises(InvalidImage):
            self.embedder.embed(b"not an image")

    def test_model_version(self):
        self.assertEqual(MODEL_VERSION, "clip-vit-b32@3d74acf+seg-face+seg3")


if __name__ == "__main__":
    unittest.main()
