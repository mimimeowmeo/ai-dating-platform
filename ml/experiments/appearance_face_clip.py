"""裁臉 + CLIP：只用臉部區域的照片向量做外貌偏好推薦，並和「整張照片 CLIP」並列比較。

【這個方法是什麼】
先用 YuNet（services/face 真人驗證已在用的人臉偵測模型）找出照片裡最大的一張臉，把臉框放大成正方形裁下，
再丟進 CLIP ViT-B/32 算 512 維向量。衣服、背景大多被裁掉，只留下臉與臉周圍的頭髮、妝容、表情。
推薦邏輯沿用外貌核心（HeartLink_Appearance_Core_V1）的多錨點：a 最近按喜歡的 20 人各當一張參考臉，
候選人的基本分 = 和這些參考臉的最高 cosine；PASS V2 扣分沿用同一條公式，只是「附近」門檻要依向量空間重新校準。

【這支腳本量什麼】
1. 偵測率：10,000 張種子照片有幾張找得到臉。
2. 相似度分布：隨機兩人的 cosine 落在哪裡，原本的 0.45 門檻在兩種向量空間各代表多少比例的配對。
3. 臉以外區域的影響：「臉框以外的顏色相似度」和向量相似度的等級相關（Spearman）。
   CelebA 對齊照的臉以外區域很小，這個指標在種子照上兩種做法都接近 0，只能當參考。
4. 真實滑卡：用 dating 資料庫的 like / pass 紀錄，依時間順序重播，
   (a) 下一個 LIKE 的人能不能排進「它 + 99 位同性別隨機候選人」的前 10 名（隨機基準 0.10）；
   (b) 有 PASS 的使用者，分數能不能分辨 like 與 pass（AUC，隨機基準 0.5；以每位 seeker 自己的 AUC 加權平均為準）。
5. 近鄰對照圖：同一張查詢照片，裁臉版和整張版各自找出最像的 6 人。

【資料限制，讀結果前必看】
- 種子照片是 CelebA 對齊裁切的名人正臉（178×218），只限非商業研究用途。偵測率會被高估，
  「裁臉 vs 整張」的差異會被低估（整張照片本身幾乎就是臉加頭髮）。
- 滑卡紀錄很少，而且有程式批次寫入的假紀錄：
  - 同一 seeker、同一毫秒寫入多筆的事件標為批次，只當歷史、不拿來評估；
  - 扣掉批次後，兩次滑卡間隔中位數 < 1 秒的帳號歸為「程式產生」組。這組 27 個帳號按喜歡的對象幾乎一樣
    （兩兩 Jaccard 中位數 0.93），實際上接近一個樣本，不要拿來判斷好壞（2026-09-27 診斷）。
- pooled AUC 會被 seeker 之間的差異灌水（只用錨點數當分數就有 0.8），請看 within_seeker 那一欄。
- 裁臉 1.3 倍仍含頭髮、帽子、耳環、領口；要更接近「只有臉」請用 --crop-scale 1.1 之類較緊的值。
- 精確去背會保留眼鏡（算長相的一部分），頭帶、帽子、耳環不留；分割出來的輪廓形狀本身會影響相似度，
  所以用凸包補齊缺口（見 seg_cutout）。
- 2026-09-27 在 db 2,000 張上，「精確去背只留臉」的 within-seeker AUC 0.64、整張照片 0.57，6 位 seeker 全部變好。
  但同一批 6 人、295 筆資料前後試了七種做法，最後這版有「剛好適合這批資料」的風險；
  要下結論請先固定做法，再用新收集的真人滑卡資料測一次。

【怎麼跑】
cd ml && .venv/bin/python experiments/appearance_face_clip.py
第一次需要：
- CLIP 權重 openai/clip-vit-base-patch32（固定 revision，pytorch_model.bin 約 605 MB，放在 Hugging Face 快取）
- ml/.models/face_detection_yunet_2023mar.onnx（和 services/face 同一個檔，sha256 會檢查）
- opencv-python-headless（ml venv 另外安裝）
- 滑卡紀錄：第一次會用 docker exec 從 heartlink-pg 以唯讀交易匯出成 outputs/appearance_face_clip/swipes.csv
輸出依設定分檔：outputs/appearance_face_clip/{embeddings,metrics}_<設定>.*、outputs/figures/appearance_face_clip_*_<設定>.png
參數：--source tar|db、--limit 2000、--crop-scale 1.3、--mask none|ellipse|seg-face|seg-face-hair、
      --device auto|mps|cpu、--batch-size 64、--refresh-embeddings、--refresh-swipes、--refresh-photos

【照片來源】
- tar（預設）：backups/heartlink-media-images.tar.gz 的 10,000 張種子照。
- db：從 dating 資料庫列出種子使用者的主照片（storage_key 以 profiles/ 開頭），先放入所有被真人滑過的對象
  （評估才完整），再用固定種子隨機補到 --limit 張，從物件儲存（.env 的 S3_ENDPOINT，本機 SeaweedFS）下載到
  outputs/appearance_face_clip/db_photos_<張數>/。真實使用者自己上傳的照片不抓（隱私）。

【去背方式（--mask）】
- ellipse：簡易去背，只留臉框內接橢圓，外面填中性灰；不需要分割模型，但邊界是幾何形狀。
- seg-face / seg-face-hair：精確去背。先用 appearance_face_segment.py（MediaPipe selfie_multiclass_256x256，
  要在 ml/.venv-seg 跑）替每張照片產生類別遮罩，這裡取和 YuNet 臉框重疊最多的那塊「臉部皮膚」，補滿內部空洞
  （眼睛、嘴巴、眼鏡），seg-face-hair 再加上和臉相連的頭髮；其餘填中性灰，依遮罩外框裁成正方形。
  只支援 --source db。分割失敗的照片退回橢圓去背，metrics 會記錄張數。

【眼鏡（--glasses，只影響精確去背）】
- project（預設）：去背時保留眼鏡，再從裁臉向量扣掉「戴眼鏡 − 沒戴眼鏡」的平均方向（男女分開算再平均），
  方向存成 outputs/appearance_face_clip/glasses_direction_<設定>.npy。2026-09-27 在 db 2,000 張上，
  戴眼鏡的人前 10 名近鄰裡戴眼鏡的比例從 46% 降到 30%（隨機 9%），沒戴眼鏡的人前 20 名有 84% 不變。
  方向是用 CelebA 的 face_Eyeglasses 標註估的；真實使用者沒有這個標註，上線前要另外處理（例如改用 CLIP 文字描述估）。
- keep：保留眼鏡，不做處理。
- gray：眼鏡塗成中性灰。會留下眼鏡形狀的灰洞，實測戴眼鏡的人反而更容易互相排在一起（51%），不建議。
"""
import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from heartlink_ml import plotting  # noqa: F401  匯入即設定 Agg 後端與中文字型
from heartlink_ml.config import FIG_DIR, ML_DIR, OUTPUT_DIR, PROJECT_ROOT, SEED, ensure_dirs
from heartlink_ml.data import load_seed
from heartlink_ml.evaluation import save_json

import matplotlib.pyplot as plt  # noqa: E402  要在 heartlink_ml.plotting 之後匯入

MODULE = "appearance_face_clip"
OUT_DIR = OUTPUT_DIR / MODULE
PHOTO_TAR = PROJECT_ROOT / "backups" / "heartlink-media-images.tar.gz"
TAR_DIR = "minio-heartlink-media/profiles/"
SEG_FACE, SEG_HAIR, SEG_ACCESSORY = 3, 1, 5  # selfie_multiclass_256x256：0 背景、1 頭髮、2 身體皮膚、3 臉部皮膚、4 衣服、5 飾品
SEG_MARGIN = 0.06
GLASSES_COL = "face_Eyeglasses"  # CelebA 標註：估「戴眼鏡」方向（--glasses project），以及量眼鏡的影響
SEG_VERSION = 3  # seg_cutout 規則有改就加 1，讓快取失效（2：凸包；3：飾品只收眼睛高度、異常形狀退回橢圓）
MASKS = ("none", "ellipse", "seg-face", "seg-face-hair")
MASK_LABEL = {"none": "裁臉", "ellipse": "裁臉＋橢圓去背", "seg-face": "精確去背（只留臉）", "seg-face-hair": "精確去背（臉＋頭髮）"}
YUNET_PATH = ML_DIR / ".models" / "face_detection_yunet_2023mar.onnx"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"  # 同 services/face/tools/fetch_models.py
YUNET_SCORE = 0.6
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"

# 外貌核心 V1 的參數（HeartLink_Appearance_Core_V1/heartlink/config.py）
ACTIVE_LIKES = 20
PASS_NEAR_V1 = 0.45
PASS_WEIGHT, MIN_PASS, MAX_PENALTY, PASS_RATE, LIKE_RATE = 0.25, 5, 0.20, 0.45, 0.80

MIN_PRIOR_LIKES = 3        # 至少要有幾個先前的 LIKE 才開始預測
N_NEGATIVES = 99           # 命中率評估：正例 + 99 位同性別隨機候選人
HIT_K = 10
SCRIPTED_GAP_SEC = 1.0     # 兩次滑卡間隔中位數低於這個值，視為程式產生
N_PAIRS = 100_000
N_BOOT = 1000
SPACES = ("face", "whole")
SPACE_LABEL = {"face": "裁臉 + CLIP", "whole": "整張照片 CLIP"}

SWIPES_SQL = """
COPY (
  SELECT left(md5(l.from_user_id::text), 10) AS seeker,
         (su.email LIKE '%@heartlink.local') AS seeker_is_seed,
         l.action,
         to_char(l.created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.MS') AS created_at,
         ph.storage_key AS target_photo
  FROM likes l
  JOIN users su ON su.id = l.from_user_id
  LEFT JOIN LATERAL (
    SELECT p.storage_key FROM user_photos p
    WHERE p.user_id = l.to_user_id AND p.deleted_at IS NULL
    ORDER BY p.is_avatar DESC, p.display_order, p.created_at LIMIT 1
  ) ph ON true
  ORDER BY 1, l.created_at
) TO STDOUT WITH CSV HEADER
"""


PHOTOS_SQL = """
COPY (
  SELECT DISTINCT ON (p.user_id) p.storage_key, pr.gender
  FROM user_photos p JOIN profiles pr ON pr.user_id = p.user_id
  WHERE p.deleted_at IS NULL AND p.storage_key LIKE 'profiles/%'
  ORDER BY p.user_id, p.is_avatar DESC, p.display_order, p.created_at
) TO STDOUT WITH CSV HEADER
"""


def parse_args():
    ap = argparse.ArgumentParser(description="裁臉 + CLIP 的外貌向量實驗（和整張照片 CLIP 並列比較）")
    ap.add_argument("--crop-scale", type=float, default=1.3, help="臉框放大倍數（正方形邊長 = 臉框長邊 × 倍數）")
    ap.add_argument("--mask", default="none", choices=MASKS, help="去背方式，見檔頭說明")
    ap.add_argument("--glasses", default="project", choices=("project", "keep", "gray"),
                    help="精確去背時的眼鏡處理：project（預設）保留並從向量扣掉戴眼鏡方向；keep 保留；gray 塗灰")
    ap.add_argument("--source", default="tar", choices=("tar", "db"), help="照片來源，見檔頭說明")
    ap.add_argument("--limit", type=int, default=2000, help="--source db 時抓幾張")
    ap.add_argument("--refresh-photos", action="store_true")
    ap.add_argument("--fetch-only", action="store_true", help="只從資料庫抓照片就結束（給分割腳本先用）")
    ap.add_argument("--export-vectors", action="store_true",
                    help="另外輸出 appearance_vectors_<設定>.jsonl，給 apps/api 的匯入腳本寫進 appearance_embeddings")
    ap.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--refresh-embeddings", action="store_true")
    ap.add_argument("--refresh-swipes", action="store_true")
    return ap.parse_args()


# ---------------------------------------------------------------- 照片、偵測、裁切
def psql_copy(sql: str) -> str:
    cmd = ["docker", "exec", "-e", "PGOPTIONS=-c default_transaction_read_only=on", "heartlink-pg",
           "psql", "-U", "heartlink", "-d", "dating", "-v", "ON_ERROR_STOP=1", "-c", " ".join(sql.split())]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def read_env(path: Path) -> dict[str, str]:
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def db_photo_dir(limit: int) -> Path:
    return OUT_DIR / f"db_photos_{limit}"


def fetch_db_photos(limit: int, swipes: pd.DataFrame) -> None:
    """列出種子使用者的主照片 → 被真人滑過的全收，其餘隨機補到 limit 張 → 從物件儲存下載。"""
    listing = pd.read_csv(io.StringIO(psql_copy(PHOTOS_SQL)))
    listing["image_file"] = listing.storage_key.str.replace("profiles/", "", regex=False)
    listing["gender"] = listing.gender.map({"woman": "Female", "man": "Male"})
    listing = listing.dropna(subset=["gender"])
    swiped = listing.storage_key.isin(set(swipes.target_photo.dropna()))
    rest = listing[~swiped]
    rest = rest.sample(n=min(len(rest), max(0, limit - int(swiped.sum()))), random_state=SEED)
    chosen = pd.concat([listing[swiped].assign(reason="swiped"), rest.assign(reason="random")]).head(limit)

    env = read_env(PROJECT_ROOT / ".env")
    endpoint, bucket = env["S3_ENDPOINT"].rstrip("/"), env["S3_BUCKET"]
    out_dir = db_photo_dir(limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = [f'user = "{env["S3_ACCESS_KEY"]}:{env["S3_SECRET_KEY"]}"', 'aws-sigv4 = "aws:amz:us-east-1:s3"', "fail", "silent"]
    for r in chosen.itertuples():
        config += [f'url = "{endpoint}/{bucket}/{r.storage_key}"', f'output = "{out_dir / r.image_file}"']
    subprocess.run(["curl", "-K", "-"], input="\n".join(config), text=True, check=True)
    missing = [f for f in chosen.image_file if not (out_dir / f).is_file() or (out_dir / f).stat().st_size == 0]
    if missing:
        sys.exit(f"有 {len(missing)} 張照片沒下載成功，例如 {missing[:3]}")
    chosen[["image_file", "storage_key", "gender", "reason"]].sort_values("image_file").to_csv(out_dir / "manifest.csv", index=False)
    print(f"[photos] 從資料庫列出 {len(listing)} 張種子主照片，下載 {len(chosen)} 張（被滑過 {int(swiped.sum())}）→ {out_dir}")


def load_photos(args, swipes: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """回傳 ({image_file: RGB ndarray}, {image_file: gender})。"""
    if args.source == "tar":
        seed = load_seed()[["image_file", "gender"]]
        return load_tar_photos(), dict(zip(seed.image_file, seed.gender))
    out_dir = db_photo_dir(args.limit)
    if args.refresh_photos or not (out_dir / "manifest.csv").exists():
        fetch_db_photos(args.limit, swipes)
    manifest = pd.read_csv(out_dir / "manifest.csv")
    photos = {f: np.asarray(Image.open(out_dir / f).convert("RGB")) for f in manifest.image_file}
    return photos, dict(zip(manifest.image_file, manifest.gender))


def load_tar_photos() -> dict[str, np.ndarray]:
    """回傳 {image_file: RGB ndarray}，直接從備份 tar 讀進記憶體，不解壓到磁碟。"""
    photos = {}
    with tarfile.open(PHOTO_TAR, "r:gz") as tar:
        for m in tar:
            name = m.name
            if not (m.isfile() and name.startswith(TAR_DIR) and name.endswith(".jpg")):
                continue
            data = tar.extractfile(m).read()
            photos[name[len(TAR_DIR):]] = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    return photos


def check_yunet() -> None:
    if not YUNET_PATH.exists():
        sys.exit(f"找不到 {YUNET_PATH}。可從 face 容器複製：docker cp ai-dating-platform-face-1:/app/models/"
                 f"face_detection_yunet_2023mar.onnx {YUNET_PATH}")
    digest = hashlib.sha256(YUNET_PATH.read_bytes()).hexdigest()
    if digest != YUNET_SHA256:
        sys.exit(f"YuNet sha256 不符：{digest}")


def detect_largest_face(detector, rgb: np.ndarray):
    """回傳 (x, y, w, h, score, 偵測到的臉數)；沒有臉回傳 None。"""
    h, w = rgb.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if faces is None or len(faces) == 0:
        return None
    i = int(np.argmax(faces[:, 2] * faces[:, 3]))
    x, y, bw, bh = (float(v) for v in faces[i, :4])
    return x, y, bw, bh, float(faces[i, 14]), len(faces)


def square_crop(rgb: np.ndarray, box, scale: float) -> np.ndarray:
    x, y, w, h = box
    side = int(round(max(w, h) * scale))
    x0 = int(round(x + w / 2 - side / 2))
    y0 = int(round(y + h / 2 - side / 2))
    H, W = rgb.shape[:2]
    pad = max(0, -x0, -y0, x0 + side - W, y0 + side - H)
    if pad:
        rgb = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        x0, y0 = x0 + pad, y0 + pad
    return rgb[y0:y0 + side, x0:x0 + side]


MASK_FILL = (123, 117, 104)  # CLIP 前處理的 mean × 255，正規化後約為 0，對向量影響最小


def ellipse_mask(crop: np.ndarray, box) -> np.ndarray:
    """square_crop 的結果以臉框中心為中心；保留臉框內接橢圓（高度多留 10% 給額頭），邊緣羽化避免硬邊。"""
    side = crop.shape[0]
    w, h = box[2], box[3]
    mask = np.zeros((side, side), np.float32)
    cv2.ellipse(mask, (side // 2, side // 2), (int(w / 2), int(h * 0.55)), 0, 0, 360, 1.0, -1)
    k = max(3, int(side * 0.04) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)[..., None]
    fill = np.array(MASK_FILL, np.float32)
    return (crop * mask + fill * (1 - mask)).astype(np.uint8)


def seg_cutout(rgb: np.ndarray, box, cat: np.ndarray, keep_hair: bool, glasses: str = "keep"):
    """精確去背：取和臉框重疊最多的「臉部皮膚」區塊，加上臉框內的眼鏡等飾品，再取凸包補齊被瀏海、帽子、
    眼鏡切掉的缺口；keep_hair 時加上和臉相連的頭髮。依保留區域的外框裁成正方形，其餘填中性灰。
    不取凸包的話，缺口形狀本身會變成相似度訊號（戴眼鏡的人只剩下半張臉，會和其他只剩下半張臉的人互相排第一）。
    分割結果和臉框對不上時回傳 None。"""
    x, y, w, h = box
    H, W = cat.shape
    face = (cat == SEG_FACE).astype(np.uint8)
    n, labels, _, _ = cv2.connectedComponentsWithStats(face, connectivity=8)
    x0, y0, x1, y1 = max(0, int(x)), max(0, int(y)), min(W, int(x + w)), min(H, int(y + h))
    overlap = np.bincount(labels[y0:y1, x0:x1].ravel(), minlength=n)
    overlap[0] = 0
    if n < 2 or overlap.max() < 0.2 * w * h:
        return None
    face_region = (labels == overlap.argmax()).astype(np.uint8)
    acc = (cat == SEG_ACCESSORY).astype(np.uint8)
    n_acc, acc_labels, _, centroids = cv2.connectedComponentsWithStats(acc, connectivity=8)
    eyewear = np.zeros((H, W), np.uint8)
    for j in range(1, n_acc):
        cx, cy = centroids[j]
        # 只收臉框內、眼睛高度那一帶的飾品（眼鏡）；頭帶、帽子在上緣，耳環在框外，都不收
        if x0 <= cx < x1 and y + 0.15 * h <= cy <= y + 0.65 * h:
            eyewear[acc_labels == j] = 1
    face_region |= eyewear
    keep = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(keep, cv2.convexHull(cv2.findNonZero(face_region)), 1)
    if glasses == "gray":  # 輪廓不變，只把眼鏡像素塗灰
        keep[eyewear > 0] = 0
    if keep.sum() > 1.8 * w * h:  # 凸包比臉框大很多 = 分割把手臂、脖子等當成臉，改用橢圓
        return None
    if keep_hair:
        hair = (cat == SEG_HAIR).astype(np.uint8)
        _, hair_labels = cv2.connectedComponents(hair, connectivity=8)
        near = cv2.dilate(keep, np.ones((7, 7), np.uint8)).astype(bool)
        touching = np.unique(hair_labels[near & (hair > 0)])
        keep |= np.isin(hair_labels, touching[touching > 0]).astype(np.uint8)
    ys, xs = np.nonzero(keep)
    bw, bh = xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min()
    side = max(bw, bh) * (1 + 2 * SEG_MARGIN)
    square = (xs.min() + bw / 2 - side / 2, ys.min() + bh / 2 - side / 2, side, side)
    crop = square_crop(rgb, square, 1.0)
    m = square_crop(keep.astype(np.float32), square, 1.0)
    m = cv2.GaussianBlur(m, (3, 3), 0)[..., None]
    return (crop * m + np.array(MASK_FILL, np.float32) * (1 - m)).astype(np.uint8)


def face_input(rgb: np.ndarray, box, scale: float, mask: str, cat: np.ndarray | None = None,
               glasses: str = "keep") -> np.ndarray:
    if mask.startswith("seg"):
        out = seg_cutout(rgb, box, cat, keep_hair=mask == "seg-face-hair", glasses=glasses) if cat is not None else None
        if out is not None:
            return out
        mask = "ellipse"  # 分割失敗時退回橢圓去背
    crop = square_crop(rgb, box, scale)
    return ellipse_mask(crop, box) if mask == "ellipse" else crop


def load_segmentation(args) -> dict | None:
    if not args.mask.startswith("seg"):
        return None
    if args.source != "db":
        sys.exit("精確去背目前只支援 --source db")
    path = OUT_DIR / f"segmentation_{db_photo_dir(args.limit).name}.npz"
    if not path.exists():
        sys.exit(f"找不到 {path}。先在分割環境跑：.venv-seg/bin/python experiments/appearance_face_segment.py "
                 f"--photo-dir {db_photo_dir(args.limit)}")
    z = np.load(path)
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- CLIP
def pick_device(name: str) -> str:
    if name == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return name


def load_clip(device: str):
    from transformers import CLIPModel
    try:
        from transformers import CLIPImageProcessorPil as Processor  # 沒有 torchvision 時的 PIL 版前處理
    except ImportError:
        from transformers import CLIPImageProcessor as Processor
    processor = Processor.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION)
    model = CLIPModel.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION).eval().to(device)
    return processor, model


def clip_embed(images: list[np.ndarray], processor, model, device: str, batch_size: int) -> np.ndarray:
    out = []
    for i in range(0, len(images), batch_size):
        pixel = processor(images=[Image.fromarray(a) for a in images[i:i + batch_size]],
                          return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            feats = model.get_image_features(pixel_values=pixel)
        if not isinstance(feats, torch.Tensor):  # transformers 5.x 回傳 BaseModelOutputWithPooling
            feats = feats.pooler_output
        out.append(torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def build_embeddings(args, photos: dict[str, np.ndarray], seg: dict | None) -> dict:
    cache = OUT_DIR / f"embeddings_{embed_tag(args)}.npz"
    meta = {"clip_model": CLIP_MODEL, "clip_revision": CLIP_REVISION, "detector": YUNET_PATH.name,
            "yunet_score": YUNET_SCORE, "crop_scale": args.crop_scale, "n_photos": len(photos)}
    if args.mask != "none":
        meta["mask"] = args.mask
    if args.mask.startswith("seg"):
        meta["seg_version"] = SEG_VERSION
        if cutout_glasses(args) != "keep":
            meta["glasses"] = cutout_glasses(args)
    if args.source != "tar":
        meta["source"] = f"db{args.limit}"
    if cache.exists() and not args.refresh_embeddings:
        z = np.load(cache, allow_pickle=False)
        if str(z["meta"]) == repr(sorted(meta.items())):
            print(f"[embed] 使用快取 {cache}")
            return {k: z[k] for k in z.files if k != "meta"} | {"meta": meta, "timing": None}

    check_yunet()
    ids = sorted(photos)
    detector = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), YUNET_SCORE, 0.3, 5000)
    boxes = np.full((len(ids), 4), np.nan, dtype=np.float32)
    scores = np.full(len(ids), np.nan, dtype=np.float32)
    n_faces = np.zeros(len(ids), dtype=np.int16)
    crops, crop_idx, n_seg_fallback = [], [], 0
    t0 = time.time()
    for i, name in enumerate(ids):
        hit = detect_largest_face(detector, photos[name])
        if hit is None:
            continue
        boxes[i], scores[i], n_faces[i] = hit[:4], hit[4], hit[5]
        cat = seg.get(name) if seg else None
        if seg and (cat is None or seg_cutout(photos[name], hit[:4], cat, args.mask == "seg-face-hair") is None):
            n_seg_fallback += 1
        crops.append(face_input(photos[name], hit[:4], args.crop_scale, args.mask, cat, cutout_glasses(args)))
        crop_idx.append(i)
    t_detect = time.time() - t0

    device = pick_device(args.device)
    processor, model = load_clip(device)
    t0 = time.time()
    face = np.full((len(ids), 512), np.nan, dtype=np.float32)
    face[crop_idx] = clip_embed(crops, processor, model, device, args.batch_size)
    t_face = time.time() - t0
    t0 = time.time()
    whole_cache = OUT_DIR / f"embeddings_whole_{'tar' if args.source == 'tar' else f'db{args.limit}'}.npz"
    wz = np.load(whole_cache) if whole_cache.exists() else None
    if wz is not None and str(wz["revision"]) == CLIP_REVISION and wz["ids"].tolist() == ids:
        whole = wz["whole"]
    else:
        whole = clip_embed([photos[n] for n in ids], processor, model, device, args.batch_size)
        np.savez_compressed(whole_cache, ids=np.array(ids), whole=whole, revision=np.array(CLIP_REVISION))
    t_whole = time.time() - t0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, ids=np.array(ids), face=face, whole=whole, boxes=boxes, scores=scores,
                        n_faces=n_faces, n_seg_fallback=np.array(n_seg_fallback), meta=np.array(repr(sorted(meta.items()))))
    timing = {"device": device, "detect_sec": round(t_detect, 1), "clip_face_sec": round(t_face, 1),
              "clip_whole_sec": round(t_whole, 1), "detect_ms_per_photo": round(1000 * t_detect / len(ids), 2)}
    print(f"[embed] 偵測 {t_detect:.1f}s、裁臉 CLIP {t_face:.1f}s、整張 CLIP {t_whole:.1f}s（{device}）")
    return {"ids": np.array(ids), "face": face, "whole": whole, "boxes": boxes, "scores": scores,
            "n_faces": n_faces, "n_seg_fallback": np.array(n_seg_fallback), "meta": meta, "timing": timing}


# ---------------------------------------------------------------- 分析
def detection_stats(emb: dict, photos: dict[str, np.ndarray]) -> dict:
    found = ~np.isnan(emb["scores"])
    h, w = next(iter(photos.values())).shape[:2]
    area = emb["boxes"][found, 2] * emb["boxes"][found, 3] / (h * w)
    return {"n_photos": int(len(found)), "n_face_found": int(found.sum()),
            "detection_rate": float(found.mean()),
            "n_multi_face": int((emb["n_faces"] > 1).sum()),
            "face_area_fraction_p5_p50_p95": np.percentile(area, [5, 50, 95]).round(3).tolist(),
            "detector_score_p5_p50": np.percentile(emb["scores"][found], [5, 50]).round(3).tolist()}


def random_pairs(idx_a, idx_b, n, rng, same=True):
    a = rng.choice(idx_a, n)
    b = rng.choice(idx_b, n)
    keep = a != b if same else np.ones(n, bool)
    return a[keep], b[keep]


def similarity_stats(vecs: dict, gender: np.ndarray, valid: np.ndarray, rng) -> dict:
    """隨機配對的 cosine 分布；same_gender 才是推薦時真正會比較的母體。"""
    out = {}
    fem = np.where((gender == "Female") & valid)[0]
    mal = np.where((gender == "Male") & valid)[0]
    for space in SPACES:
        v = vecs[space]
        res = {}
        for label, (ia, ib, same) in {"same_gender": (None, None, True), "cross_gender": (fem, mal, False)}.items():
            if label == "same_gender":
                a1, b1 = random_pairs(fem, fem, N_PAIRS // 2, rng)
                a2, b2 = random_pairs(mal, mal, N_PAIRS // 2, rng)
                a, b = np.concatenate([a1, a2]), np.concatenate([b1, b2])
            else:
                a, b = random_pairs(ia, ib, N_PAIRS // 2, rng, same=False)
            cos = np.einsum("ij,ij->i", v[a], v[b])
            res[label] = {"percentiles_1_5_25_50_75_95_99": np.percentile(cos, [1, 5, 25, 50, 75, 95, 99]).round(3).tolist(),
                          "share_ge_0.45": float((cos >= PASS_NEAR_V1).mean())}
        same_cos = np.einsum("ij,ij->i", v[a1], v[b1]).tolist() + np.einsum("ij,ij->i", v[a2], v[b2]).tolist()
        res["suggested_near_threshold_p95_same_gender"] = float(np.percentile(same_cos, 95).round(3))
        res["near_threshold_p95_by_gender"] = {
            "Female": float(np.percentile(np.einsum("ij,ij->i", v[a1], v[b1]), 95).round(3)),
            "Male": float(np.percentile(np.einsum("ij,ij->i", v[a2], v[b2]), 95).round(3))}
        out[space] = res
    return out


def outside_face_hist(rgb: np.ndarray, box, scale=1.6, bins=4) -> np.ndarray:
    """臉框放大 scale 倍以外區域（背景、衣服）的 RGB 色彩直方圖。"""
    H, W = rgb.shape[:2]
    x, y, w, h = box
    cx, cy, half = x + w / 2, y + h / 2, max(w, h) * scale / 2
    mask = np.ones((H, W), bool)
    mask[max(0, int(cy - half)):min(H, int(cy + half)), max(0, int(cx - half)):min(W, int(cx + half))] = False
    px = rgb[mask]
    if len(px) < 50:
        return None
    q = (px // (256 // bins)).astype(np.int32)
    hist = np.bincount(q[:, 0] * bins * bins + q[:, 1] * bins + q[:, 2], minlength=bins ** 3).astype(np.float32)
    return hist / hist.sum()


def outside_face_influence(vecs, emb, photos, gender, valid, rng) -> dict:
    ids = emb["ids"]
    hists = {}
    for i in np.where(valid)[0]:
        h = outside_face_hist(photos[ids[i]], emb["boxes"][i])
        if h is not None:
            hists[i] = h
    pool = np.array(sorted(hists))
    res = {"n_photos_with_outside_region": int(len(pool))}
    for g in ("Female", "Male"):
        idx = pool[gender[pool] == g]
        a, b = random_pairs(idx, idx, 20_000, rng)
        bg = np.array([np.minimum(hists[i], hists[j]).sum() for i, j in zip(a, b)])  # 直方圖交集
        for space in SPACES:
            cos = np.einsum("ij,ij->i", vecs[space][a], vecs[space][b])
            res.setdefault(space, {})[g] = round(float(spearmanr(bg, cos).statistic), 3)
    return res


def neighbor_overlap(vecs, gender, valid, rng, n_query=300, k=20) -> dict:
    jac = []
    for q in rng.choice(np.where(valid)[0], n_query, replace=False):
        pool = np.where(valid & (gender == gender[q]))[0]
        pool = pool[pool != q]
        tops = [set(pool[np.argsort(-(vecs[s][pool] @ vecs[s][q]))[:k]]) for s in SPACES]
        jac.append(len(tops[0] & tops[1]) / len(tops[0] | tops[1]))
    return {"k": k, "n_query": n_query, "mean_jaccard_face_vs_whole": round(float(np.mean(jac)), 3)}


# ---------------------------------------------------------------- 真實滑卡
def load_swipes(refresh: bool) -> pd.DataFrame:
    path = OUT_DIR / "swipes.csv"
    if refresh or not path.exists():
        cmd = ["docker", "exec", "-e", "PGOPTIONS=-c default_transaction_read_only=on", "heartlink-pg",
               "psql", "-U", "heartlink", "-d", "dating", "-v", "ON_ERROR_STOP=1", "-c", " ".join(SWIPES_SQL.split())]
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(out, encoding="utf-8")
        print(f"[swipes] 從 heartlink-pg 唯讀匯出 → {path}")
    df = pd.read_csv(path, parse_dates=["created_at"], dtype={"seeker": str})
    df["target_image"] = df["target_photo"].fillna("").str.replace("profiles/", "", regex=False)
    df["batch"] = df.duplicated(["seeker", "created_at"], keep=False)  # 同一毫秒寫入多筆 = 批次產生，不是人滑的
    return df.sort_values(["seeker", "created_at"]).reset_index(drop=True)


def pass_penalty(cand: np.ndarray, likes: np.ndarray, passes: np.ndarray, near: float) -> float:
    """外貌核心 V1 的 PASS V2（HeartLink_Appearance_Core_V1/heartlink/appearance/pass_suppression.py）。"""
    n_pass = int((passes @ cand >= near).sum()) if len(passes) else 0
    if n_pass < MIN_PASS:
        return 0.0
    n_like = int((likes @ cand >= near).sum()) if len(likes) else 0
    p = 1 - np.exp(-PASS_RATE * n_pass)
    lp = 1 - np.exp(-LIKE_RATE * n_like)
    return float(min(MAX_PENALTY, PASS_WEIGHT * p * (1 - lp)))


def replay(swipes, vecs, index, gender, valid, near, rng_seed) -> tuple[pd.DataFrame, pd.DataFrame]:
    """依時間重播每位 seeker 的滑卡；同一組事件、同一組負例同時餵給兩個向量空間。"""
    hits, pairs = [], []
    for seeker, g in swipes.groupby("seeker", sort=True):
        rng = np.random.default_rng(rng_seed + int(seeker, 16) % 100_000)
        rows = [(r.action, index.get(r.target_image), r.batch) for r in g.itertuples()]
        seen = {i for _, i, _ in rows if i is not None}
        like_hist, pass_hist = [], []
        for action, t, batch in rows:
            if t is None or not valid[t]:
                continue
            if not batch and len(like_hist) >= MIN_PRIOR_LIKES:
                anchors = like_hist[-ACTIVE_LIKES:]
                if action == "like":
                    pool = np.where(valid & (gender == gender[t]))[0]
                    pool = np.setdiff1d(pool, list(seen))
                    negs = rng.choice(pool, min(N_NEGATIVES, len(pool)), replace=False)
                    cand = np.concatenate([[t], negs])
                    rec = {"seeker": seeker}
                    for s in SPACES:
                        base = (vecs[s][cand] @ vecs[s][anchors].T).max(axis=1)
                        rec[f"rank_{s}"] = int((base[1:] > base[0]).sum() + 1)
                    hits.append(rec)
                rec = {"seeker": seeker, "label": int(action == "like"), "n_anchors": len(anchors)}
                for s in SPACES:
                    v = vecs[s]
                    base = float((v[anchors] @ v[t]).max())
                    rec[f"base_{s}"] = base
                    rec[f"pv2_{s}"] = max(0.0, base - pass_penalty(v[t], v[anchors], v[pass_hist], near[s][gender[t]]))
                pairs.append(rec)
            (like_hist if action == "like" else pass_hist).append(t)
    return pd.DataFrame(hits), pd.DataFrame(pairs)


def seeker_groups(swipes: pd.DataFrame) -> pd.Series:
    human = swipes[~swipes.batch]
    gap = human.groupby("seeker")["created_at"].apply(lambda s: s.diff().dt.total_seconds().median())
    gap = gap.reindex(swipes.seeker.unique())
    only_batch = gap.index.difference(human.seeker.unique())
    groups = gap.fillna(np.inf).lt(SCRIPTED_GAP_SEC).map({True: "scripted", False: "other"})
    groups[only_batch] = "scripted"
    return groups


def bootstrap_mean(values_by_seeker: np.ndarray, rng) -> list[float]:
    if len(values_by_seeker) == 0:
        return [None, None]
    boots = [rng.choice(values_by_seeker, len(values_by_seeker)).mean() for _ in range(N_BOOT)]
    return np.percentile(boots, [2.5, 97.5]).round(3).tolist()


def within_seeker_auc(p: pd.DataFrame, col: str) -> float:
    """每位 seeker 各自算 AUC，再依 (like 數 × pass 數) 加權平均，排除 seeker 之間的差異。"""
    aucs, weights = [], []
    for _, g in p.groupby("seeker"):
        n_pos, n_neg = int(g.label.sum()), int((g.label == 0).sum())
        if n_pos and n_neg:
            aucs.append(roc_auc_score(g.label, g[col]))
            weights.append(n_pos * n_neg)
    return round(float(np.average(aucs, weights=weights)), 3) if aucs else None


def within_seeker_diff_ci(p: pd.DataFrame, rng) -> list[float]:
    """以 seeker 為單位重抽樣，算「裁臉 − 整張」的 within-seeker AUC 差的 95% 區間。seeker 很少時區間會偏窄。"""
    seekers = p.seeker.unique()
    diffs = []
    for _ in range(N_BOOT):
        pick = rng.choice(seekers, len(seekers))
        b = pd.concat([p[p.seeker == s].assign(seeker=f"{s}_{i}") for i, s in enumerate(pick)])
        a_face, a_whole = within_seeker_auc(b, "base_face"), within_seeker_auc(b, "base_whole")
        if a_face is not None and a_whole is not None:
            diffs.append(a_face - a_whole)
    return np.percentile(diffs, [2.5, 97.5]).round(3).tolist() if diffs else [None, None]


def evaluate_swipes(hits: pd.DataFrame, pairs: pd.DataFrame, groups: pd.Series, rng) -> dict:
    out = {}
    for grp in ("other", "scripted", "all"):
        sel = groups.index if grp == "all" else groups.index[groups == grp]
        h = hits[hits.seeker.isin(sel)] if len(hits) else hits
        res = {"n_seekers": int(len(sel))}
        if len(h):
            per = h.groupby("seeker").agg(**{f"hit_{s}": (f"rank_{s}", lambda r: float((r <= HIT_K).mean())) for s in SPACES})
            res["hit_at_10"] = {"n_events": int(len(h)), "n_seekers_evaluated": int(len(per)), "random_baseline": HIT_K / (N_NEGATIVES + 1)}
            for s in SPACES:
                res["hit_at_10"][s] = {"mean_over_seekers": round(float(per[f"hit_{s}"].mean()), 3),
                                        "ci95": bootstrap_mean(per[f"hit_{s}"].to_numpy(), rng),
                                        "mean_rank_of_100": round(float(h[f"rank_{s}"].mean()), 1)}
            res["hit_at_10"]["face_minus_whole_ci95"] = bootstrap_mean((per.hit_face - per.hit_whole).to_numpy(), rng)
        p = pairs[pairs.seeker.isin(sel)] if len(pairs) else pairs
        both = p.groupby("seeker").label.nunique() if len(p) else pd.Series(dtype=int)
        p = p[p.seeker.isin(both.index[both == 2])] if len(p) else p
        if len(p) and p.label.nunique() == 2:
            res["like_vs_pass_auc"] = {"n_events": int(len(p)), "n_seekers": int(p.seeker.nunique()),
                                       "n_pass_events": int((p.label == 0).sum())}
            for s in SPACES:
                res["like_vs_pass_auc"][s] = {"pooled_base": round(float(roc_auc_score(p.label, p[f"base_{s}"])), 3),
                                               "pooled_with_pass_v2": round(float(roc_auc_score(p.label, p[f"pv2_{s}"])), 3),
                                               "within_seeker": within_seeker_auc(p, f"base_{s}")}
            res["like_vs_pass_auc"]["within_seeker_face_minus_whole_ci95"] = within_seeker_diff_ci(p, rng)
            # 對照：只用「錨點數」當分數。pooled AUC 若接近或高於模型，代表 pooled 數字被 seeker 間差異灌水
            res["like_vs_pass_auc"]["pooled_anchor_count_only"] = round(float(roc_auc_score(p.label, p.n_anchors)), 3)
        out[grp] = res
    return out


# ---------------------------------------------------------------- 眼鏡
def glasses_labels(ids) -> np.ndarray:
    seed = load_seed()
    if GLASSES_COL not in seed.columns:
        return np.full(len(ids), np.nan)
    by_file = dict(zip(seed.image_file, seed[GLASSES_COL].astype(float)))
    return np.array([by_file.get(str(i), np.nan) for i in ids])


def glasses_direction(v: np.ndarray, labels: np.ndarray, gender: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """「戴眼鏡 − 沒戴眼鏡」的平均向量差，男女分開算再平均，避免把性別差異也算進去。"""
    diffs = []
    for g in ("Female", "Male"):
        m = idx[gender[idx] == g]
        pos, neg = m[labels[m] == 1], m[labels[m] == 0]
        if len(pos) and len(neg):
            diffs.append(v[pos].mean(axis=0) - v[neg].mean(axis=0))
    d = np.mean(diffs, axis=0)
    return d / np.linalg.norm(d)


def project_out(v: np.ndarray, d: np.ndarray) -> np.ndarray:
    out = v - np.outer(v @ d, d)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def glasses_neighbor_rates(v, labels, gender, known, folds, k=10) -> dict:
    """在同一折、同性別內找前 k 名近鄰：戴眼鏡的查詢，近鄰裡戴眼鏡的比例；隨機基準 = 候選池的戴眼鏡比例。"""
    rate, base, rate_non = [], [], []
    for f in np.unique(folds):
        for q in np.where(known & (folds == f))[0]:
            pool = np.where(known & (folds == f) & (gender == gender[q]))[0]
            pool = pool[pool != q]
            top = pool[np.argsort(-(v[pool] @ v[q]))[:k]]
            if labels[q] == 1:
                rate.append(labels[top].mean())
                base.append(labels[pool].mean())
            else:
                rate_non.append(labels[top].mean())
    return {"glasses_query_top10_glasses_share": round(float(np.mean(rate)), 3),
            "random_baseline": round(float(np.mean(base)), 3),
            "non_glasses_query_top10_glasses_share": round(float(np.mean(rate_non)), 3),
            "n_glasses_queries": len(rate)}


def glasses_analysis(vecs, labels, gender, valid, swipes, index, groups) -> dict:
    """保留眼鏡時，戴眼鏡的人會不會互相排前面；再試「從向量扣掉戴眼鏡方向」（兩折交叉：一半估方向、另一半量）。"""
    known = valid & ~np.isnan(labels)
    if known.sum() == 0 or np.nansum(labels) == 0:
        return {}
    folds = np.arange(len(labels)) % 2
    out = {s: glasses_neighbor_rates(vecs[s], labels, gender, known, folds) for s in SPACES}
    proj = vecs["face"].copy()
    for f in (0, 1):
        d = glasses_direction(vecs["face"], labels, gender, np.where(known & (folds != f))[0])
        proj[folds == f] = project_out(vecs["face"][folds == f], d)
    out["face_glasses_projected_out"] = glasses_neighbor_rates(proj, labels, gender, known, folds)
    non = np.where(known & (labels == 0))[0]
    jac = []
    for q in non[:: max(1, len(non) // 300)]:
        pool = np.where(known & (gender == gender[q]))[0]
        pool = pool[pool != q]
        a = set(pool[np.argsort(-(vecs["face"][pool] @ vecs["face"][q]))[:20]])
        b = set(pool[np.argsort(-(proj[pool] @ proj[q]))[:20]])
        jac.append(len(a & b) / len(a | b))
    out["face_glasses_projected_out"]["non_glasses_top20_overlap_with_unprojected"] = round(float(np.mean(jac)), 3)
    near = {s: {"Female": 0.99, "Male": 0.99} for s in SPACES}
    _, pairs = replay(swipes, {"face": proj, "whole": vecs["whole"]}, index, gender, valid, near, SEED)
    p = pairs[pairs.seeker.isin(groups.index[groups == "other"])]
    out["face_glasses_projected_out"]["within_seeker_auc"] = within_seeker_auc(p, "base_face")
    return out


def glasses_preview(photos, emb, seg, labels, path, rng, n=6) -> None:
    """戴眼鏡的人：原圖、精確去背保留眼鏡、精確去背眼鏡塗灰。"""
    ids = emb["ids"]
    picks = rng.choice(np.where((labels == 1) & ~np.isnan(emb["scores"]))[0], n, replace=False)
    cols = [("原圖", None), ("精確去背：保留眼鏡", "keep"), ("精確去背：眼鏡塗灰", "gray")]
    fig, axes = plt.subplots(n, len(cols), figsize=(1.9 * len(cols), 1.9 * n))
    for r, i in enumerate(picks):
        name = ids[i]
        for c, (title, mode) in enumerate(cols):
            img = photos[name] if mode is None else face_input(photos[name], emb["boxes"][i], 1.1, "seg-face", seg.get(name), mode)
            axes[r, c].imshow(img)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- 近鄰對照圖
def neighbor_figure(vecs, emb, photos, gender, valid, rng, path, crop_scale, mask="none", seg=None, glasses="keep",
                    n_query=4, k=6) -> list:
    ids = emb["ids"]
    queries = [rng.choice(np.where(valid & (gender == g))[0]) for g in ("Female", "Male") for _ in range(n_query // 2)]
    fig, axes = plt.subplots(2 * len(queries), k + 1, figsize=(1.6 * (k + 1), 2.1 * 2 * len(queries)))
    shown = []
    for qi, q in enumerate(queries):
        pool = np.where(valid & (gender == gender[q]))[0]
        pool = pool[pool != q]
        for si, space in enumerate(SPACES):
            row = axes[2 * qi + si]
            sims = vecs[space][pool] @ vecs[space][q]
            top = pool[np.argsort(-sims)[:k]]
            show = ((lambda i: face_input(photos[ids[i]], emb["boxes"][i], crop_scale, mask,
                                          seg.get(ids[i]) if seg else None, glasses))
                    if space == "face" else (lambda i: photos[ids[i]]))
            row[0].imshow(show(q))
            row[0].set_title(f"查詢（{SPACE_LABEL[space]}）", fontsize=7)
            for j, t in enumerate(top, start=1):
                row[j].imshow(show(t))
                row[j].set_title(f"{float(vecs[space][t] @ vecs[space][q]):.2f}", fontsize=7)
            for ax in row:
                ax.axis("off")
            shown.append({"query": str(ids[q]), "space": space, "neighbors": [str(ids[t]) for t in top]})
    label = MASK_LABEL[mask] if mask.startswith("seg") else f"{MASK_LABEL[mask]}（{crop_scale} 倍）"
    fig.suptitle(f"同一張查詢照片：上排{label} + CLIP、下排整張照片 CLIP 找出的最像 6 人（數字是 cosine）", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return shown


def cutout_preview(photos, emb, seg, path, rng, n=8) -> None:
    """同一張照片的幾種輸入並排：原圖、裁臉 1.1、橢圓去背、精確去背（只留臉）、精確去背（臉＋頭髮）。"""
    ids = emb["ids"]
    picks = rng.choice(np.where(~np.isnan(emb["scores"]))[0], n, replace=False)
    cols = [("原圖", None), ("裁臉 1.1 倍", "none"), ("橢圓去背", "ellipse"),
            ("精確去背：只留臉", "seg-face"), ("精確去背：臉＋頭髮", "seg-face-hair")]
    fig, axes = plt.subplots(n, len(cols), figsize=(1.7 * len(cols), 1.9 * n))
    for r, i in enumerate(picks):
        name, box = ids[i], emb["boxes"][i]
        for c, (title, mask) in enumerate(cols):
            img = photos[name] if mask is None else face_input(photos[name], box, 1.1, mask, seg.get(name))
            axes[r, c].imshow(img)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def export_vectors(args, ids, face: np.ndarray, valid: np.ndarray) -> None:
    """一行一張照片：storage_key、model_version、512 維向量。只有 --source db 才知道 storage_key。"""
    if args.source != "db":
        sys.exit("--export-vectors 需要 --source db")
    manifest = pd.read_csv(db_photo_dir(args.limit) / "manifest.csv")
    key_by_file = dict(zip(manifest.image_file, manifest.storage_key))
    version = f"clip-vit-b32@{CLIP_REVISION[:7]}+{embed_tag(args).split('_', 1)[-1]}"
    if args.mask.startswith("seg"):
        version += f"+seg{SEG_VERSION}+glasses-{args.glasses}"
    path = OUT_DIR / f"appearance_vectors_{run_tag(args)}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i in np.where(valid)[0]:
            f.write(json.dumps({"storage_key": key_by_file[str(ids[i])], "model_version": version,
                                "embedding": [round(float(x), 6) for x in face[i]]}) + "\n")
    print(f"[export] {int(valid.sum())} 筆向量（{version}）→ {path}")


# ---------------------------------------------------------------- main
def cutout_glasses(args) -> str:
    """去背時眼鏡要不要塗灰；project 和 keep 的裁切一樣，差別只在之後的向量處理。"""
    return "gray" if args.glasses == "gray" else "keep"


def embed_tag(args) -> str:
    """裁臉向量快取的檔名：只跟裁切方式有關，所以 project 和 keep 共用同一份。"""
    prefix = "" if args.source == "tar" else f"db{args.limit}_"
    if args.mask.startswith("seg"):
        return prefix + args.mask + ("_glasses-gray" if cutout_glasses(args) == "gray" else "")
    return prefix + f"s{args.crop_scale}" + ("" if args.mask == "none" else f"_{args.mask}")


def run_tag(args) -> str:
    """metrics 和圖的檔名：精確去背的預設（--glasses project）不加後綴。"""
    tag = embed_tag(args)
    return tag + "_glasses-keep" if args.mask.startswith("seg") and args.glasses == "keep" else tag


def apply_glasses_mode(args, raw: dict, labels: np.ndarray, gender: np.ndarray, valid: np.ndarray) -> tuple[dict, dict]:
    """--glasses project：用有標註的照片估「戴眼鏡」方向，從所有裁臉向量扣掉，方向另存成檔案。"""
    if not (args.mask.startswith("seg") and args.glasses == "project"):
        return raw, {}
    known = np.where(valid & ~np.isnan(labels))[0]
    if len(known) == 0 or np.nansum(labels[known]) == 0:
        print("[glasses] 沒有戴眼鏡標註，無法估方向，這次改用保留眼鏡")
        return raw, {"projected": False}
    d = glasses_direction(raw["face"], labels, gender, known)
    face = raw["face"].copy()
    face[valid] = project_out(raw["face"][valid], d)
    path = OUT_DIR / f"glasses_direction_{embed_tag(args)}.npy"
    np.save(path, d.astype(np.float32))
    # 給 apps/api 的 pnpm db:import-appearance 第二個參數；model_version 要和推薦 worker 的 MODEL_VERSION 一致。
    raw_version = f"clip-vit-b32@{CLIP_REVISION[:7]}+{args.mask}+seg{SEG_VERSION}"
    path.with_suffix(".json").write_text(json.dumps(
        {"name": "glasses", "model_version": raw_version, "direction": [round(float(x), 8) for x in d]}), encoding="utf-8")
    n_glasses = {g: int(((labels[known] == 1) & (gender[known] == g)).sum()) for g in ("Female", "Male")}
    print(f"[glasses] 已從裁臉向量扣掉戴眼鏡方向（估計用戴眼鏡人數 {n_glasses}）→ {path.name}")
    return {"face": face, "whole": raw["whole"]}, {"projected": True, "direction_file": path.name, "n_glasses_used": n_glasses}


def main():
    args = parse_args()
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    t0 = time.time()
    swipes = load_swipes(args.refresh_swipes)
    photos, gender_by_file = load_photos(args, swipes)
    print(f"[photos] 讀入 {len(photos)} 張（{time.time() - t0:.1f}s）")
    if args.fetch_only:
        return
    seg = load_segmentation(args)
    emb = build_embeddings(args, photos, seg)
    ids = emb["ids"]
    index = {str(n): i for i, n in enumerate(ids)}
    gender = np.array([gender_by_file.get(str(n), "unknown") for n in ids])
    if seg:
        print(f"[seg] 分割失敗退回橢圓：{int(emb['n_seg_fallback'])} 張")
    valid = ~np.isnan(emb["face"][:, 0])
    raw = {"face": np.nan_to_num(emb["face"]), "whole": emb["whole"]}
    labels = glasses_labels(ids)
    vecs, glasses_direction_info = apply_glasses_mode(args, raw, labels, gender, valid)
    if args.export_vectors:
        export_vectors(args, ids, vecs["face"], valid)

    det = detection_stats(emb, photos)
    sims = similarity_stats(vecs, gender, valid, rng)
    outside = outside_face_influence(vecs, emb, photos, gender, valid, rng)
    overlap = neighbor_overlap(vecs, gender, valid, rng)
    print(f"[detect] 偵測率 {det['detection_rate']:.4f}（{det['n_face_found']}/{det['n_photos']}），多臉 {det['n_multi_face']}")
    for s in SPACES:
        print(f"[sim] {SPACE_LABEL[s]}：同性別中位數 {sims[s]['same_gender']['percentiles_1_5_25_50_75_95_99'][3]}，"
              f"≥0.45 比例 {sims[s]['same_gender']['share_ge_0.45']:.3f}，建議門檻（p95）{sims[s]['suggested_near_threshold_p95_same_gender']}")
    print(f"[outside] 臉以外區域色彩相似度 vs 向量相似度（Spearman）：{outside}")
    print(f"[overlap] 前 20 名重疊（Jaccard）：{overlap['mean_jaccard_face_vs_whole']}")

    groups = seeker_groups(swipes)
    near = {s: sims[s]["near_threshold_p95_by_gender"] for s in SPACES}
    hits, pairs = replay(swipes, vecs, index, gender, valid, near, SEED)
    swipe_eval = evaluate_swipes(hits, pairs, groups, rng)
    for grp, res in swipe_eval.items():
        if "hit_at_10" in res:
            hr = res["hit_at_10"]
            print(f"[swipes:{grp}] 命中率@10 裁臉 {hr['face']['mean_over_seekers']} {hr['face']['ci95']}、"
                  f"整張 {hr['whole']['mean_over_seekers']} {hr['whole']['ci95']}（隨機 {hr['random_baseline']}），"
                  f"差 {hr['face_minus_whole_ci95']}，{hr['n_seekers_evaluated']} 人 {hr['n_events']} 筆")
        if "like_vs_pass_auc" in res:
            au = res["like_vs_pass_auc"]
            print(f"[swipes:{grp}] like vs pass AUC 裁臉 {au['face']}、整張 {au['whole']}、"
                  f"差（每人分開算）{au['within_seeker_face_minus_whole_ci95']}、"
                  f"只用錨點數 {au['pooled_anchor_count_only']}（{au['n_seekers']} 人、{au['n_events']} 筆）")

    tag = run_tag(args)
    fig_path = FIG_DIR / f"{MODULE}_neighbors_{tag}.png"
    shown = neighbor_figure(vecs, emb, photos, gender, valid, rng, fig_path, args.crop_scale, args.mask, seg,
                            cutout_glasses(args))
    glasses = glasses_analysis(raw, labels, gender, valid, swipes, index, groups)  # 用未扣方向的向量，才能比較處理前後
    for key, res in glasses.items():
        print(f"[glasses] {key}：戴眼鏡的人前 10 名近鄰裡戴眼鏡的比例 {res['glasses_query_top10_glasses_share']}"
              f"（隨機 {res['random_baseline']}）" + (f"、每人分開算的 AUC {res['within_seeker_auc']}"
                                                   f"、沒戴眼鏡的人前 20 名和原本重疊 {res['non_glasses_top20_overlap_with_unprojected']}"
                                                   if "within_seeker_auc" in res else ""))
    if seg and not np.isnan(labels).all():
        glasses_preview(photos, emb, seg, labels, FIG_DIR / f"{MODULE}_glasses_{tag}.png", np.random.default_rng(SEED))
    if seg:
        cutout_preview(photos, emb, seg, FIG_DIR / f"{MODULE}_cutouts_{tag}.png", np.random.default_rng(SEED))
    batch = swipes.groupby(swipes.seeker.map(groups)).batch.sum().astype(int).to_dict()
    save_json({"meta": emb["meta"], "timing": emb["timing"], "detection": det,
               "n_seg_fallback_to_ellipse": int(emb["n_seg_fallback"]), "similarity": sims,
               "outside_face_influence_spearman": outside, "neighbor_overlap": overlap,
               "pass_near_thresholds_used_in_replay": near,
               "seeker_groups": groups.value_counts().to_dict(), "batch_events_not_evaluated": batch,
               "swipes": swipe_eval, "glasses_mode": args.glasses if args.mask.startswith("seg") else None,
               "glasses_direction": glasses_direction_info, "glasses": glasses, "neighbor_figure": shown},
              OUT_DIR / f"metrics_{tag}.json")
    print(f"[done] {OUT_DIR / f'metrics_{tag}.json'}、{fig_path}（共 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
