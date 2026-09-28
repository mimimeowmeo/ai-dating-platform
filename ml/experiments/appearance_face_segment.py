"""MediaPipe 人像分割：替 appearance_face_clip.py 的「精確去背」產生每張照片的類別遮罩。

【為什麼獨立一支、獨立環境】
mediapipe 會一併安裝 opencv-contrib-python，和 ml/.venv 裡的 opencv-python-headless 搶同一個 cv2 模組，
所以放在 ml/.venv-seg 跑，這支腳本也不 import heartlink_ml（那邊需要 pandas、scikit-learn）。

【模型】
MediaPipe selfie_multiclass_256x256（Apache-2.0，16,371,837 bytes，sha256 會檢查）。
類別：0 背景、1 頭髮、2 身體皮膚、3 臉部皮膚、4 衣服、5 其他（飾品）。實測「臉部皮膚」會連同眼睛、嘴巴一起涵蓋。

【怎麼跑】
cd ml
uv venv --python 3.12 .venv-seg && uv pip install --python .venv-seg/bin/python mediapipe==1.0.1 pillow
curl -o .models/selfie_multiclass_256x256.tflite <下方 MODEL_URL>
.venv/bin/python experiments/appearance_face_clip.py --source db --limit 2000        # 先抓照片
.venv-seg/bin/python experiments/appearance_face_segment.py --photo-dir outputs/appearance_face_clip/db_photos_2000
輸出：outputs/appearance_face_clip/segmentation_<照片資料夾名>.npz，每張照片一個 uint8 類別遮罩（和原圖同尺寸）。
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from PIL import Image

ML_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = ML_DIR / "outputs" / "appearance_face_clip"
MODEL = ML_DIR / ".models" / "selfie_multiclass_256x256.tflite"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
             "selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite")
MODEL_SHA256 = "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"


def main():
    ap = argparse.ArgumentParser(description="MediaPipe 人像分割，產生精確去背用的類別遮罩")
    ap.add_argument("--photo-dir", default=str(OUT_DIR / "db_photos_2000"))
    photo_dir = Path(ap.parse_args().photo_dir)

    if not MODEL.exists():
        sys.exit(f"找不到 {MODEL}，請先下載：curl -o {MODEL} {MODEL_URL}")
    if hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA256:
        sys.exit(f"{MODEL.name} 的 sha256 不符")
    files = sorted(photo_dir.glob("*.jpg"))
    if not files:
        sys.exit(f"{photo_dir} 沒有照片；先跑 appearance_face_clip.py --source db 抓照片")

    options = vision.ImageSegmenterOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.IMAGE, output_category_mask=True, output_confidence_masks=False)
    masks = {}
    t0 = time.time()
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        for f in files:
            rgb = np.ascontiguousarray(np.asarray(Image.open(f).convert("RGB")))
            result = segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            cat = np.asarray(result.category_mask.numpy_view())
            masks[f.name] = (cat[..., 0] if cat.ndim == 3 else cat).copy()
    out = OUT_DIR / f"segmentation_{photo_dir.name}.npz"
    np.savez_compressed(out, **masks)
    face_share = np.mean([(m == 3).mean() for m in masks.values()])
    print(f"[seg] {len(masks)} 張，{time.time() - t0:.1f}s；臉部皮膚平均佔畫面 {face_share:.1%} → {out}")


if __name__ == "__main__":
    main()
