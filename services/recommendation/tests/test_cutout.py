"""cutout.py 的規則：只留臉、眼鏡保留、頭帶不留、分割對不上時退回橢圓。用合成的影像與類別遮罩。"""
import unittest

import cv2
import numpy as np

from appearance.cutout import (
    FALLBACK_CROP_SCALE,
    MASK_FILL,
    SEG_ACCESSORY,
    SEG_FACE,
    face_input,
    seg_cutout,
)

GLASSES = (0, 0, 255)
HEADBAND = (255, 0, 0)
BOX = (60, 50, 80, 100)  # 臉框 (x, y, w, h)


def scene():
    """200×200：臉是橢圓（類別 3），眼睛高度有一副眼鏡、頭頂有一條頭帶（都是類別 5）。"""
    rgb = np.full((200, 200, 3), 30, np.uint8)
    categories = np.zeros((200, 200), np.uint8)
    cv2.ellipse(categories, (100, 100), (40, 50), 0, 0, 360, SEG_FACE, -1)
    rgb[categories == SEG_FACE] = (210, 170, 150)
    categories[85:95, 70:130] = SEG_ACCESSORY  # 眼鏡：中心 y=90，在臉框 15%～65% 高度內
    rgb[85:95, 70:130] = GLASSES
    categories[40:48, 70:130] = SEG_ACCESSORY  # 頭帶：中心 y=44，在臉框上緣之外
    rgb[40:48, 70:130] = HEADBAND
    return rgb, categories


def has_color(image, color):
    return bool(np.all(image == np.array(color, np.uint8), axis=-1).any())


class SegCutoutTest(unittest.TestCase):
    def test_keeps_face_and_glasses_but_not_headband(self):
        rgb, categories = scene()
        out = seg_cutout(rgb, BOX, categories)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape[0], out.shape[1])
        self.assertTrue(has_color(out, GLASSES))
        self.assertFalse(has_color(out, HEADBAND))
        self.assertTrue(np.array_equal(out[0, 0], np.array(MASK_FILL, np.uint8)))

    def test_background_is_filled_with_neutral_gray(self):
        rgb, categories = scene()
        out = seg_cutout(rgb, BOX, categories)
        self.assertFalse(has_color(out, (30, 30, 30)))

    def test_returns_none_when_face_does_not_overlap_box(self):
        rgb, categories = scene()
        self.assertIsNone(seg_cutout(rgb, (0, 0, 20, 20), categories))

    def test_returns_none_when_hull_is_much_larger_than_box(self):
        rgb, _ = scene()
        categories = np.full((200, 200), SEG_FACE, np.uint8)  # 整張都被當成臉（例如手臂、脖子）
        self.assertIsNone(seg_cutout(rgb, BOX, categories))


class FaceInputTest(unittest.TestCase):
    def test_falls_back_to_ellipse_when_segmentation_fails(self):
        rgb, _ = scene()
        out = face_input(rgb, BOX, np.zeros((200, 200), np.uint8))
        side = int(round(max(BOX[2], BOX[3]) * FALLBACK_CROP_SCALE))
        self.assertEqual(out.shape, (side, side, 3))
        self.assertTrue(np.array_equal(out[0, 0], np.array(MASK_FILL, np.uint8)))

    def test_uses_segmentation_when_available(self):
        rgb, categories = scene()
        np.testing.assert_array_equal(face_input(rgb, BOX, categories), seg_cutout(rgb, BOX, categories))


if __name__ == "__main__":
    unittest.main()
