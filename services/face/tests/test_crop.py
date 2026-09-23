"""被動防偽（MiniFASNet）前的人臉裁切測試：app/crop.py。

MiniFASNet 不是直接看整張照片，而是看「以人臉框為中心、放大某個倍率後裁出來、再縮成 80×80」的小圖。
兩個模型用不同倍率：minifasnet_v2 用 2.7 倍、minifasnet_v1se 用 4.0 倍（見 app/models.py）。
模型是用 Silent-Face-Anti-Spoofing 的 CropImage 裁法訓練的，裁切範圍或長寬比只要和訓練時不同，
分數就會系統性偏移，而且不會有任何錯誤訊息，所以這裡用固定的數字把計算結果鎖住。

座標慣例：人臉框是 (x, y, w, h)，也就是左上角 x、左上角 y、寬、高；
_get_new_box 回傳的是 (左上 x, 左上 y, 右下 x, 右下 y)，右下角是「包含」的像素位置，最大到 寬−1／高−1。
"""

# unittest：Python 內建的測試框架。
import unittest

# numpy：產生隨機數與建立假影像（影像在 OpenCV 裡就是 numpy 陣列）。
import numpy as np

# _get_new_box：計算放大後的裁切範圍；crop_patch：實際裁切並縮放；int_box：把浮點數的人臉框取整。
from app.crop import _get_new_box, crop_patch, int_box


class CropTests(unittest.TestCase):
    """裁切必須和 Silent-Face 的 CropImage 一致，MiniFASNet 是用這種裁法訓練的。"""

    def test_centered_box_scaled_around_center(self):
        """規則：人臉框在圖中間時，以框的中心為基準等比例放大。

        計算：影像 200×200，框 (80, 80, 40, 40) 的中心是 (100, 100)；放大 2.7 倍後寬高 108，
        左上 = 100 − 54 = 46，右下 = 100 + 54 = 154。倍率沒有被影像大小限制（199 ÷ 40 大於 2.7）。
        """
        # 寬 40 放大 2.7 倍 = 108，以中心 100 展開。
        self.assertEqual(_get_new_box(200, 200, (80, 80, 40, 40), 2.7), (46, 46, 154, 154))

    def test_box_at_edge_shifted_back_inside(self):
        """規則：放大後超出影像邊界時，整塊裁切範圍平移回圖內，大小不變。

        為什麼重要：如果改成把超出的部分截掉，裁出來的範圍會變小、長寬比會變，和模型訓練時看到的不一樣。
        計算：框 (0, 80, 40, 40) 放大 4 倍變 160×160，中心 (20, 100)，左上 x = 20 − 80 = −60，
        所以整塊往右移 60：x 範圍變成 0～160；y 範圍 20～180 本來就在圖內，不動。
        """
        # 左邊超出 60 px，整塊往右平移，而不是截掉。
        self.assertEqual(_get_new_box(200, 200, (0, 80, 40, 40), 4.0), (0, 20, 160, 180))

    def test_scale_limited_by_image_size(self):
        """規則：放大後會比整張影像還大時，倍率會被降到剛好塞進影像。

        計算：框寬高 150，影像 200×200，最大倍率是 (200 − 1) ÷ 150 ≈ 1.327（小於要求的 4.0），
        放大後剛好是 199，再平移回圖內，就是整張影像 (0, 0, 199, 199)。
        """
        # 放大 4 倍會遠超過影像，所以倍率被限制，結果是整張影像的範圍。
        self.assertEqual(_get_new_box(200, 200, (10, 10, 150, 150), 4.0), (0, 0, 199, 199))

    def test_int_box_only_truncates_like_upstream(self):
        """規則：int_box 只把座標轉成整數，不會把框限制在影像範圍內。

        為什麼重要：越界的框要原樣交給 _get_new_box，由它先限制倍率、再整塊平移；
        如果在這裡先截短，裁切範圍會和上游不同。
        """
        # 上游 get_bbox 只取整、不夾限；越界的框原樣交給 _get_new_box。
        # Python 的 int() 是「往 0 的方向捨去」：−5.7 變 −5（不是 −6），50.9 變 50；負的 x 保留，不會被改成 0。
        self.assertEqual(int_box((-5.7, 10.2, 50.9, 40.1)), (-5, 10, 50, 40))
        # 框超出影像右邊與下邊（假設影像更小）也原樣保留，只是轉成整數。
        self.assertEqual(int_box((190.0, 95.0, 50.0, 50.0)), (190, 95, 50, 50))
        # 寬高不到 1 時會變成 1（int_box 的 max(1, ...)），避免 _get_new_box 計算倍率時除以 0。
        self.assertEqual(int_box((10.0, 10.0, 0.4, 0.2)), (10, 10, 1, 1))

    def test_box_past_right_and_bottom_matches_upstream(self):
        """規則：框超出右邊與下邊時，結果要和上游的算法一致。

        計算：影像 640×480，框取整後是 (500, 300, 200, 220)。倍率受高度限制：479 ÷ 220 ≈ 2.177（小於 2.7），
        放大後約 435×479，中心 (600, 410)；右邊超出，往左平移到右下 x = 639，左上 x ≈ 203；
        下邊超出，往上平移到 y 範圍 0～479。
        """
        # 臉靠近右下角的自拍：寬高不先截短，裁切範圍與長寬比才會和上游一樣。
        self.assertEqual(_get_new_box(640, 480, int_box((500.4, 300.9, 200.2, 220.7)), 2.7), (203, 0, 639, 479))

    def test_crop_always_inside_image(self):
        """規則：不管人臉框在哪裡、多大，算出來的裁切範圍一定在影像內，而且左上不會超過右下。

        為什麼重要：numpy 切片遇到負數索引會從另一端算起，範圍顛倒時會切出空陣列；
        兩種情況都會讓防偽模型收到錯誤的影像，甚至讓 cv2.resize 出錯。
        做法：用固定種子的亂數產生 2000 組各種尺寸與位置（包含超出影像）的框，兩種倍率都測。
        """
        # 固定種子 0 的亂數產生器：每次執行都產生同樣的數字，失敗時可以重現。
        rng = np.random.default_rng(0)
        # 隨機測 2000 次。
        for _ in range(2000):
            # 隨機產生影像寬、高，範圍 64～1599（integers 不含上限 1600），轉成 Python 的 int。
            src_w, src_h = (int(v) for v in rng.integers(64, 1600, size=2))
            # 隨機產生人臉框：x、y 可以是負的（最多超出 200 px）或在影像右／下緣；寬高介於 1 和影像尺寸之間。
            box = (rng.uniform(-200, src_w), rng.uniform(-200, src_h), rng.uniform(1, src_w), rng.uniform(1, src_h))
            # 兩個 MiniFASNet 模型使用的倍率都要測。
            for scale in (2.7, 4.0):
                # 和 crop_patch 一樣：先 int_box 取整，再算裁切範圍。
                x1, y1, x2, y2 = _get_new_box(src_w, src_h, int_box(box), scale)
                # 檢查 0 ≤ 左上 ≤ 右下 ≤ 最後一個像素；失敗時把這組輸入印出來，方便除錯。
                self.assertTrue(0 <= x1 <= x2 <= src_w - 1 and 0 <= y1 <= y2 <= src_h - 1, (src_w, src_h, box, scale))

    def test_patch_size(self):
        """規則：crop_patch 的輸出一定是指定的 80×80×3，不管框在哪裡。

        為什麼重要：MiniFASNet 的輸入固定是 80×80 的 3 通道影像，尺寸不對模型就無法執行。
        """
        # 建立一張全黑的假影像：高 300、寬 200、3 個色彩通道，每個值是 0～255 的 uint8。
        image = np.zeros((300, 200, 3), dtype=np.uint8)
        # 框在影像內，用 2.7 倍裁切，輸出形狀必須是 (高 80, 寬 80, 3 通道)。
        self.assertEqual(crop_patch(image, (50.0, 80.0, 60.0, 70.0), 2.7, 80, 80).shape, (80, 80, 3))
        # 框超出影像右邊與下邊（150 + 120 > 200、250 + 90 > 300），用 4.0 倍裁切，輸出形狀仍然正確。
        self.assertEqual(crop_patch(image, (150.0, 250.0, 120.0, 90.0), 4.0, 80, 80).shape, (80, 80, 3))


# 直接執行這個檔案時才跑測試；被 unittest discover 匯入時不會進來。
if __name__ == "__main__":
    # 找出這個模組裡所有 TestCase 並執行。
    unittest.main()
