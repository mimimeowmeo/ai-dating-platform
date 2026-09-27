"""把照片裁成「只有臉」的輸入：精確去背，分割失敗時退回橢圓去背。

只用 numpy 與 OpenCV，不載入任何模型，方便單元測試。
數值與 ml/experiments/appearance_face_clip.py 的 seg_cutout（SEG_VERSION 3）一致。
"""
import cv2
import numpy as np

# selfie_multiclass_256x256 的類別：0 背景、1 頭髮、2 身體皮膚、3 臉部皮膚、4 衣服、5 飾品
SEG_FACE, SEG_ACCESSORY = 3, 5
SEG_MARGIN = 0.06
# CLIP 前處理的 mean × 255，正規化後約為 0，對向量影響最小。
MASK_FILL = (123, 117, 104)
# 分割失敗時退回橢圓去背用的裁切倍數；ml 匯入的 1 萬筆向量就是用這個值。
FALLBACK_CROP_SCALE = 1.3


def square_crop(image: np.ndarray, box, scale: float) -> np.ndarray:
    """以 box 中心裁出邊長 max(w, h) × scale 的正方形，超出畫面的部分補 0。"""
    x, y, w, h = box
    side = int(round(max(w, h) * scale))
    x0 = int(round(x + w / 2 - side / 2))
    y0 = int(round(y + h / 2 - side / 2))
    height, width = image.shape[:2]
    pad = max(0, -x0, -y0, x0 + side - width, y0 + side - height)
    if pad:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        x0, y0 = x0 + pad, y0 + pad
    return image[y0:y0 + side, x0:x0 + side]


def ellipse_cutout(rgb: np.ndarray, box, scale: float = FALLBACK_CROP_SCALE) -> np.ndarray:
    """簡易去背：只留臉框內接橢圓（高度多留 10% 給額頭），邊緣羽化，外面填中性灰。"""
    crop = square_crop(rgb, box, scale)
    side = crop.shape[0]
    mask = np.zeros((side, side), np.float32)
    cv2.ellipse(mask, (side // 2, side // 2), (int(box[2] / 2), int(box[3] * 0.55)), 0, 0, 360, 1.0, -1)
    k = max(3, int(side * 0.04) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)[..., None]
    return (crop * mask + np.array(MASK_FILL, np.float32) * (1 - mask)).astype(np.uint8)


def seg_cutout(rgb: np.ndarray, box, categories: np.ndarray) -> np.ndarray | None:
    """精確去背只留臉。

    取和臉框重疊最多的「臉部皮膚」區塊，加上臉框內、眼睛高度的飾品（眼鏡），再取凸包補齊被瀏海、
    帽子、眼鏡切掉的缺口；頭帶、帽子、耳環不收。依保留區域外框裁成正方形，其餘填中性灰。
    不取凸包的話，缺口形狀本身會變成相似度訊號（戴眼鏡的人只剩下半張臉，會和同樣缺口的人互相排第一）。
    分割結果和臉框對不上，或凸包比臉框大很多（把手臂、脖子當成臉）時回傳 None。
    """
    x, y, w, h = box
    height, width = categories.shape
    face = (categories == SEG_FACE).astype(np.uint8)
    n, labels, _, _ = cv2.connectedComponentsWithStats(face, connectivity=8)
    x0, y0, x1, y1 = max(0, int(x)), max(0, int(y)), min(width, int(x + w)), min(height, int(y + h))
    overlap = np.bincount(labels[y0:y1, x0:x1].ravel(), minlength=n)
    overlap[0] = 0
    if n < 2 or overlap.max() < 0.2 * w * h:
        return None
    region = (labels == overlap.argmax()).astype(np.uint8)
    accessories = (categories == SEG_ACCESSORY).astype(np.uint8)
    n_acc, acc_labels, _, centroids = cv2.connectedComponentsWithStats(accessories, connectivity=8)
    for j in range(1, n_acc):
        cx, cy = centroids[j]
        if x0 <= cx < x1 and y + 0.15 * h <= cy <= y + 0.65 * h:
            region[acc_labels == j] = 1
    keep = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(keep, cv2.convexHull(cv2.findNonZero(region)), 1)
    if keep.sum() > 1.8 * w * h:
        return None
    ys, xs = np.nonzero(keep)
    bw, bh = xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min()
    side = max(bw, bh) * (1 + 2 * SEG_MARGIN)
    square = (xs.min() + bw / 2 - side / 2, ys.min() + bh / 2 - side / 2, side, side)
    crop = square_crop(rgb, square, 1.0)
    mask = square_crop(keep.astype(np.float32), square, 1.0)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)[..., None]
    return (crop * mask + np.array(MASK_FILL, np.float32) * (1 - mask)).astype(np.uint8)


def face_input(rgb: np.ndarray, box, categories: np.ndarray) -> np.ndarray:
    """給 CLIP 的輸入：精確去背成功就用它，否則退回橢圓去背。"""
    cutout = seg_cutout(rgb, box, categories)
    return cutout if cutout is not None else ellipse_cutout(rgb, box)
