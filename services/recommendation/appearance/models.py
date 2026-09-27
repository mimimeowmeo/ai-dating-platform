"""三個模型的薄包裝：YuNet 找臉、MediaPipe 分割、CLIP 算向量。

模型檔在 build 時由 tools/fetch_models.py 下載並驗證 sha256，執行期不連網。
這些物件都不是執行緒安全的，由 pipeline.AppearanceEmbedder 用鎖保護。
"""
from pathlib import Path

import cv2
import numpy as np

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SEGMENTER_FILE = "selfie_multiclass_256x256.tflite"
CLIP_DIR = "clip-vit-base-patch32"
# 和 ml 實驗相同的偵測門檻；多張臉時取面積最大的一張。
YUNET_SCORE = 0.6


class FaceDetector:
    def __init__(self, models_dir: Path):
        self._detector = cv2.FaceDetectorYN.create(str(models_dir / YUNET_FILE), "", (320, 320), YUNET_SCORE, 0.3, 5000)

    def largest(self, rgb: np.ndarray):
        """回傳最大那張臉的 (x, y, w, h)；找不到臉回傳 None。"""
        height, width = rgb.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if faces is None or len(faces) == 0:
            return None
        i = int(np.argmax(faces[:, 2] * faces[:, 3]))
        return tuple(float(v) for v in faces[i, :4])


class Segmenter:
    def __init__(self, models_dir: Path):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.ImageSegmenterOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(models_dir / SEGMENTER_FILE)),
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True,
            output_confidence_masks=False,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)

    def categories(self, rgb: np.ndarray) -> np.ndarray:
        """每個像素的類別（和原圖同尺寸的 uint8）。"""
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        mask = np.asarray(self._segmenter.segment(image).category_mask.numpy_view())
        return (mask[..., 0] if mask.ndim == 3 else mask).copy()


class ClipEncoder:
    def __init__(self, models_dir: Path):
        import torch
        from transformers import CLIPModel

        try:
            from transformers import CLIPImageProcessorPil as Processor  # 沒有 torchvision 時的 PIL 版前處理，和 ml 實驗相同
        except ImportError:
            from transformers import CLIPImageProcessor as Processor
        self._torch = torch
        self._processor = Processor.from_pretrained(models_dir / CLIP_DIR)
        self._model = CLIPModel.from_pretrained(models_dir / CLIP_DIR).eval()

    def encode(self, rgb: np.ndarray) -> np.ndarray:
        """L2 正規化後的 512 維影像向量。"""
        from PIL import Image

        pixel = self._processor(images=[Image.fromarray(rgb)], return_tensors="pt")["pixel_values"]
        with self._torch.no_grad():
            features = self._model.get_image_features(pixel_values=pixel)
        if not isinstance(features, self._torch.Tensor):  # transformers 5.x 回傳 BaseModelOutputWithPooling
            features = features.pooler_output
        features = self._torch.nn.functional.normalize(features.float(), dim=-1)
        return features[0].numpy().astype(np.float32)
