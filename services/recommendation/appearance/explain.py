"""測試畫面的「像在哪裡」：用遮蔽法（occlusion）找出兩張臉最像的部位。

兩張照片照線上流程去背、算向量（再扣掉戴眼鏡方向），先得到整體相似度；再用臉部特徵點圍出
眉毛、眼睛、鼻子、嘴唇、左右臉頰（顴骨）、下巴，每一區各自塗成灰色重算。相似度下降越多，代表模型越依賴
這一塊判斷「像」。兩張臉各遮一次取平均：只遮一邊時，數字會因為遮哪一邊而不同。
框的座標換回原始照片的像素，前端疊在照片上。只給測試畫面用，結果不存資料庫。
"""
import cv2
import numpy as np

from .cutout import MASK_FILL

# 左右照畫面來分（MediaPipe 的 LEFT 是本人的左邊，在畫面右側）。
REGIONS = ("eyebrows", "eyes", "nose", "lips", "left_cheek", "right_cheek", "chin")


def _indices(connections) -> list[int]:
    return sorted({i for c in connections for i in (c.start, c.end)})


def face_regions(points: np.ndarray, connections) -> dict[str, list[np.ndarray]]:
    """478 個點（像素座標）→ 各區域的多邊形。臉頰沒有現成的點集合，用眼睛下緣、鼻子、上唇與臉部輪廓圍出來。"""
    pick = lambda group: points[_indices(group)]  # noqa: E731
    left_eye, right_eye = sorted(
        (pick(connections.FACE_LANDMARKS_LEFT_EYE), pick(connections.FACE_LANDMARKS_RIGHT_EYE)),
        key=lambda eye: eye[:, 0].mean(),
    )
    nose = pick(connections.FACE_LANDMARKS_NOSE)
    lips = pick(connections.FACE_LANDMARKS_LIPS)
    oval = pick(connections.FACE_LANDMARKS_FACE_OVAL)
    lips_top, lips_bottom = lips[:, 1].min(), lips[:, 1].max()

    def cheek(eye: np.ndarray, outward: int) -> np.ndarray:
        top = eye[:, 1].max()
        band = oval[(oval[:, 1] >= top) & (oval[:, 1] <= lips_top)]
        band = band if len(band) else oval
        edge = band[:, 0].min() if outward < 0 else band[:, 0].max()
        inner = nose[:, 0].min() if outward < 0 else nose[:, 0].max()
        x0, x1 = sorted((edge, inner))
        return np.array([[x0, top], [x1, top], [x1, lips_top], [x0, lips_top]])

    chin = np.vstack([
        oval[oval[:, 1] >= lips_bottom],
        [[lips[:, 0].min(), lips_bottom], [lips[:, 0].max(), lips_bottom]],
    ])
    return {
        "eyebrows": [
            pick(connections.FACE_LANDMARKS_LEFT_EYEBROW),
            pick(connections.FACE_LANDMARKS_RIGHT_EYEBROW),
        ],
        "eyes": [left_eye, right_eye],
        "nose": [nose],
        "lips": [lips],
        "left_cheek": [cheek(left_eye, -1)],
        "right_cheek": [cheek(right_eye, 1)],
        "chin": [chin],
    }


def occlude(image: np.ndarray, polygons: list[np.ndarray]) -> np.ndarray:
    """把區域（凸包再往外擴一點）塗成去背用的中性灰。"""
    mask = np.zeros(image.shape[:2], np.uint8)
    for polygon in polygons:
        cv2.fillConvexPoly(mask, cv2.convexHull(polygon.astype(np.int32)), 1)
    pad = max(2, int(image.shape[0] * 0.03))
    mask = cv2.dilate(mask, np.ones((pad * 2 + 1, pad * 2 + 1), np.uint8))
    out = image.copy()
    out[mask > 0] = MASK_FILL
    return out


def boxes(polygons: list[np.ndarray], origin: tuple[int, int], size: tuple[int, int]) -> list[list[int]]:
    """區域的外框 [x, y, w, h]，換成原始照片的像素並裁在照片範圍內。"""
    width, height = size
    result = []
    for polygon in polygons:
        x, y, w, h = cv2.boundingRect(polygon.astype(np.int32))
        x0, y0 = max(0, x + origin[0]), max(0, y + origin[1])
        x1, y1 = min(width, x + w + origin[0]), min(height, y + h + origin[1])
        if x1 > x0 and y1 > y0:
            result.append([int(x0), int(y0), int(x1 - x0), int(y1 - y0)])
    return result


def explain_pair(faces: list[dict], embed) -> dict:
    """faces：卡片上的人、喜歡過的人各一份（cutout、origin、size、regions）；embed：去背後的臉 → 單位向量。"""
    candidate, anchor = faces
    e_candidate, e_anchor = embed(candidate["cutout"]), embed(anchor["cutout"])
    similarity = float(e_candidate @ e_anchor)
    regions = []
    for name in REGIONS:
        drop_candidate = similarity - float(embed(occlude(candidate["cutout"], candidate["regions"][name])) @ e_anchor)
        drop_anchor = similarity - float(e_candidate @ embed(occlude(anchor["cutout"], anchor["regions"][name])))
        regions.append({
            "name": name,
            "drop": round((drop_candidate + drop_anchor) / 2, 5),
            "dropCandidate": round(drop_candidate, 5),
            "dropAnchor": round(drop_anchor, 5),
            "candidate": boxes(candidate["regions"][name], candidate["origin"], candidate["size"]),
            "anchor": boxes(anchor["regions"][name], anchor["origin"], anchor["size"]),
        })
    regions.sort(key=lambda region: -region["drop"])
    return {
        "similarity": round(similarity, 5),
        "candidate": {"width": candidate["size"][0], "height": candidate["size"][1]},
        "anchor": {"width": anchor["size"][0], "height": anchor["size"][1]},
        "regions": regions,
    }
