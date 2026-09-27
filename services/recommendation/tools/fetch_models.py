"""下載推薦 worker 的模型檔並驗證大小與 sha256；只在 Docker build 的 models 階段執行。

任何一個檔案大小或 sha256 不符就 sys.exit，讓 build 失敗；驗證通過才寫檔。
網址都鎖在固定版本（commit、版本號或 revision），和 ml/experiments/appearance_face_clip.py 用的是同一批檔案。
用法：python tools/fetch_models.py <輸出資料夾>
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

OPENCV_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models"
MEDIAPIPE_MODELS = "https://storage.googleapis.com/mediapipe-models"
CLIP = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"

# (輸出路徑, 網址, 位元組數, sha256)
FILES = (
    (
        "face_detection_yunet_2023mar.onnx",
        f"{OPENCV_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        232_589,
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    (
        "selfie_multiclass_256x256.tflite",
        f"{MEDIAPIPE_MODELS}/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite",
        16_371_837,
        "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0",
    ),
    (
        "clip-vit-base-patch32/config.json",
        f"{CLIP}/config.json",
        4_186,
        "b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770",
    ),
    (
        "clip-vit-base-patch32/preprocessor_config.json",
        f"{CLIP}/preprocessor_config.json",
        316,
        "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd",
    ),
    (
        "clip-vit-base-patch32/pytorch_model.bin",
        f"{CLIP}/pytorch_model.bin",
        605_247_071,
        "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f",
    ),
)


def fetch(url: str, size: int) -> bytes:
    """只讀 size + 1 個位元組：回應比預期大時不會整個讀進記憶體。"""
    with urllib.request.urlopen(url, timeout=600) as response:
        data = response.read(size + 1)
    if len(data) != size:
        raise ValueError(f"size mismatch: expected {size}, got {len(data)}")
    return data


def main(out_dir: Path) -> None:
    for name, url, size, sha256 in FILES:
        data = fetch(url, size)
        digest = hashlib.sha256(data).hexdigest()
        if digest != sha256:
            sys.exit(f"{name}: sha256 mismatch ({digest})")
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"{name}: ok ({size} bytes)", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "models"))
