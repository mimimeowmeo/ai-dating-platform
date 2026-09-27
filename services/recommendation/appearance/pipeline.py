"""照片 bytes → 外貌向量。"""
import io
import threading
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .cutout import face_input

CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
# seg_cutout 規則版本；和 ml/experiments/appearance_face_clip.py 的 SEG_VERSION 一致。
SEG_VERSION = 3
# 寫進 appearance_embeddings.model_version；NestJS 用它對應同版本的「戴眼鏡」方向。
MODEL_VERSION = f"clip-vit-b32@{CLIP_REVISION[:7]}+seg-face+seg{SEG_VERSION}"
# NestJS 送來的是上傳時已縮到長邊 1600px 以內的 JPEG；再大就拒收，避免超大影像耗盡記憶體。
MAX_PIXELS = 4_000_000


class InvalidImage(Exception):
    """讀不出來或太大的影像。"""


class AppearanceEmbedder:
    def __init__(self, detector, segmenter, encoder):
        self._detector = detector
        self._segmenter = segmenter
        self._encoder = encoder
        self._lock = threading.Lock()

    @classmethod
    def from_dir(cls, models_dir: Path):
        from .models import ClipEncoder, FaceDetector, Segmenter

        return cls(FaceDetector(models_dir), Segmenter(models_dir), ClipEncoder(models_dir))

    def embed(self, image_bytes: bytes) -> list[float] | None:
        """找不到臉回傳 None；影像壞掉丟出 InvalidImage。"""
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                if image.width * image.height > MAX_PIXELS:
                    raise InvalidImage("IMAGE_TOO_LARGE")
                rgb = np.asarray(image.convert("RGB"))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise InvalidImage("INVALID_IMAGE") from error
        with self._lock:
            box = self._detector.largest(rgb)
            if box is None:
                return None
            vector = self._encoder.encode(face_input(rgb, box, self._segmenter.categories(rgb)))
        return [float(v) for v in vector]
