"""explain.py：臉部區域怎麼切、遮住後相似度怎麼算、框怎麼換回原圖座標。用合成的點與影像。"""
import unittest
from types import SimpleNamespace

import numpy as np

from appearance.cutout import MASK_FILL
from appearance.explain import REGIONS, boxes, explain_pair, face_regions, occlude


def group(start, count):
    return [SimpleNamespace(start=i, end=i + 1 if i + 1 < start + count else start) for i in range(start, start + count)]


# 合成的臉（100×100）：每個部位 4 個點圍成方形，臉部輪廓 8 個點。
CONNECTIONS = SimpleNamespace(
    FACE_LANDMARKS_LEFT_EYE=group(0, 4),
    FACE_LANDMARKS_RIGHT_EYE=group(4, 4),
    FACE_LANDMARKS_NOSE=group(8, 4),
    FACE_LANDMARKS_LIPS=group(12, 4),
    FACE_LANDMARKS_FACE_OVAL=group(16, 8),
    FACE_LANDMARKS_LEFT_EYEBROW=group(24, 4),
    FACE_LANDMARKS_RIGHT_EYEBROW=group(28, 4),
)


def square(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


POINTS = np.array(
    # MediaPipe 的 LEFT 是本人的左邊（在畫面右側），故意放在右邊，確認左右是照畫面來分。
    square(60, 35, 75, 42)  # 本人左眼
    + square(25, 35, 40, 42)  # 本人右眼
    + square(45, 40, 55, 60)  # 鼻子
    + square(38, 68, 62, 76)  # 嘴唇
    + [[50, 5], [85, 20], [92, 50], [80, 85], [50, 95], [20, 85], [8, 50], [15, 20]]  # 臉部輪廓
    + square(58, 26, 78, 30)
    + square(22, 26, 42, 30),
    dtype=float,
)


class RegionsTest(unittest.TestCase):
    def test_left_right_follow_the_image_and_cheeks_sit_between_eye_and_lips(self):
        regions = face_regions(POINTS, CONNECTIONS)
        self.assertEqual(set(regions), set(REGIONS))
        left, right = regions["eyes"]
        self.assertLess(left[:, 0].mean(), right[:, 0].mean())
        cheek = regions["left_cheek"][0]
        self.assertEqual(cheek[:, 1].min(), 42)  # 眼睛下緣
        self.assertEqual(cheek[:, 1].max(), 68)  # 上唇
        self.assertEqual(cheek[:, 0].min(), 8)  # 臉部輪廓左緣
        self.assertEqual(cheek[:, 0].max(), 45)  # 鼻子左緣
        self.assertGreaterEqual(regions["chin"][0][:, 1].min(), 76)

    def test_occlude_paints_region_gray(self):
        image = np.full((100, 100, 3), 200, np.uint8)
        out = occlude(image, face_regions(POINTS, CONNECTIONS)["lips"])
        self.assertTrue(np.array_equal(out[72, 50], MASK_FILL))
        self.assertTrue(np.array_equal(out[10, 10], (200, 200, 200)))

    def test_boxes_shift_to_original_and_clip(self):
        polygon = np.array(square(10, 10, 30, 20), dtype=float)
        self.assertEqual(boxes([polygon], (100, 50), (1000, 1000)), [[110, 60, 21, 11]])
        self.assertEqual(boxes([polygon], (-15, -15), (1000, 1000)), [[0, 0, 16, 6]])


class ExplainPairTest(unittest.TestCase):
    def test_identical_faces_start_at_one_and_every_region_lowers_it(self):
        rng = np.random.default_rng(0)
        cutout = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)

        def embed(image):
            v = image.astype(np.float64).ravel() - 127
            return v / np.linalg.norm(v)

        face = {"cutout": cutout, "origin": (20, 30), "size": (300, 400), "regions": face_regions(POINTS, CONNECTIONS)}
        result = explain_pair([face, dict(face)], embed)
        self.assertAlmostEqual(result["similarity"], 1.0, places=5)
        drops = [r["drop"] for r in result["regions"]]
        self.assertEqual(drops, sorted(drops, reverse=True))
        self.assertTrue(all(d > 0 for d in drops))
        lips = next(r for r in result["regions"] if r["name"] == "lips")
        self.assertEqual(lips["candidate"], [[58, 98, 25, 9]])
        self.assertEqual(result["candidate"], {"width": 300, "height": 400})


if __name__ == "__main__":
    unittest.main()
