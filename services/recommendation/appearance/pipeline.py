"""照片 bytes → 外貌向量。"""
import io
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from .cutout import face_input, face_input_with_origin
from .explain import explain_pair, face_regions

CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
# seg_cutout 規則版本；和 ml/experiments/appearance_face_clip.py 的 SEG_VERSION 一致。
SEG_VERSION = 3
# 寫進 appearance_embeddings.model_version；NestJS 用它對應同版本的「戴眼鏡」方向。
MODEL_VERSION = f"clip-vit-b32@{CLIP_REVISION[:7]}+seg-face+seg{SEG_VERSION}"
# NestJS 送來的是上傳時已縮到長邊 1600px 以內的 JPEG；再大就拒收，避免超大影像耗盡記憶體。
MAX_PIXELS = 4_000_000
# 去背後的臉很小（種子照片約 100 像素寬），放大後臉部特徵點才找得穩。
LANDMARK_SIZE = 448


class InvalidImage(Exception):
    """讀不出來或太大的影像。"""


def _decode(image_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.width * image.height > MAX_PIXELS:
                raise InvalidImage("IMAGE_TOO_LARGE")
            return np.asarray(image.convert("RGB"))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise InvalidImage("INVALID_IMAGE") from error


class AppearanceEmbedder:
    def __init__(self, detector, segmenter, encoder, landmarker=None):
        self._detector = detector
        self._segmenter = segmenter
        self._encoder = encoder
        self._landmarker = landmarker
        self._lock = threading.Lock()

    @classmethod
    def from_dir(cls, models_dir: Path):
        from .models import ClipEncoder, FaceDetector, Landmarker, Segmenter

        return cls(FaceDetector(models_dir), Segmenter(models_dir), ClipEncoder(models_dir), Landmarker(models_dir))

    def embed(self, image_bytes: bytes) -> list[float] | None:
        """找不到臉回傳 None；影像壞掉丟出 InvalidImage。"""
        rgb = _decode(image_bytes)
        with self._lock:
            box = self._detector.largest(rgb)
            if box is None:
                return None
            vector = self._encoder.encode(face_input(rgb, box, self._segmenter.categories(rgb)))
        return [float(v) for v in vector]

    def explain(self, candidate: bytes, anchor: bytes, direction: list[float] | None) -> dict | None:
        """測試畫面「像在哪裡」（見 explain.py）。direction 是要扣掉的戴眼鏡方向，和存進資料庫的向量一致。
        任一張找不到臉或臉部特徵點就回傳 None；影像壞掉丟出 InvalidImage。"""
        images = [_decode(candidate), _decode(anchor)]
        d = np.asarray(direction, dtype=np.float64) if direction else None

        def embed(face: np.ndarray) -> np.ndarray:
            v = self._encoder.encode(face).astype(np.float64)
            if d is not None:
                v = v - (v @ d) * d
            return v / np.linalg.norm(v)

        with self._lock:
            faces = []
            for rgb in images:
                box = self._detector.largest(rgb)
                if box is None:
                    return None
                cutout, origin = face_input_with_origin(rgb, box, self._segmenter.categories(rgb))
                big = cv2.resize(cutout, (LANDMARK_SIZE, LANDMARK_SIZE), interpolation=cv2.INTER_CUBIC)
                points = self._landmarker.points(big)
                if points is None:
                    return None
                faces.append({
                    "cutout": cutout,
                    "origin": origin,
                    "size": (rgb.shape[1], rgb.shape[0]),
                    "regions": face_regions(points * (cutout.shape[0] / LANDMARK_SIZE), self._landmarker.connections),
                })
            return explain_pair(faces, embed)
