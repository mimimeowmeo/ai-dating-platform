"""影像解碼測試：app/imaging.py 的 decode_bgr。

decode_bgr 把 Base64 字串解碼成 OpenCV 使用的 numpy 陣列，這裡鎖住兩件「錯了也不會丟例外」的事：
1. 色彩通道順序必須是 BGR（藍、綠、紅），不是一般常見的 RGB。
2. 手機照片的 EXIF 方向（Orientation）必須套用，照片要先轉正。
這兩件事錯了，模型照樣會輸出分數，只是分數會偏掉，所以要用測試明確檢查像素值。
"""

# base64：把影像的位元組編碼成 Base64 字串，模擬 API 傳進來的 imageBase64。
import base64
# io：BytesIO 是放在記憶體裡的「假檔案」，讓 PIL 把影像存進去而不用寫到硬碟。
import io
# unittest：Python 內建的測試框架。
import unittest

# PIL（Pillow）的 Image：用程式產生測試用的影像，並存成 PNG／JPEG。
from PIL import Image

# 被測的函式：把 Base64 影像解碼成 BGR 的 numpy 陣列。
from app.imaging import decode_bgr

# 解碼時允許的最大位元組數（10 MiB）；測試影像很小，只是需要一個夠大的上限。
LIMIT = 10 * 1024 * 1024
# PIL 使用 RGB 順序：紅色是 (255, 0, 0)、藍色是 (0, 0, 255)。
RED, BLUE = (255, 0, 0), (0, 0, 255)


def left_red_right_blue():
    """產生一張寬 128、高 64 的測試圖：左半邊紅色、右半邊藍色。

    回傳：
        PIL.Image.Image：RGB 模式的影像。
    設計理由：
        左右兩色很好檢查，既能看出通道順序（紅、藍會不會對調），也能看出影像有沒有被旋轉。
    """
    # 建立一張 128×64、整張藍色的 RGB 影像。
    image = Image.new("RGB", (128, 64), BLUE)
    # 再把一張 64×64 的紅色影像貼在左上角 (0, 0)，蓋住左半邊。
    image.paste(Image.new("RGB", (64, 64), RED), (0, 0))
    # 回傳左紅右藍的影像。
    return image


def encode(image, format, **options):
    """把 PIL 影像存成指定格式，再轉成 Base64 字串。

    參數：
        image：要編碼的 PIL 影像。
        format：檔案格式，例如 "PNG"、"JPEG"。
        **options：原樣轉給 PIL 的 save，例如 JPEG 的 quality（品質）與 exif（EXIF 資料）。
    回傳：
        str：ASCII 的 Base64 字串，和 API 請求裡的 imageBase64 格式相同。
    """
    # 建立記憶體中的暫存檔。
    buffer = io.BytesIO()
    # 把影像用指定格式與選項寫進暫存檔。
    image.save(buffer, format=format, **options)
    # 取出全部位元組，Base64 編碼後（仍是 bytes）再轉成字串。
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class DecodeTests(unittest.TestCase):
    """三個模型都吃 BGR；通道或方向錯了不會丟例外，只會讓分數系統性偏移，所以要鎖住。"""

    def test_output_is_bgr(self):
        """規則：解碼結果的形狀是 (高, 寬, 3)，而且通道順序是 BGR。

        用 PNG 是因為 PNG 無損壓縮，像素值會和原圖完全相同，可以用等號精確比對。
        """
        # 把左紅右藍的圖存成 PNG、轉 Base64，再用 decode_bgr 解回來。
        out = decode_bgr(encode(left_red_right_blue(), "PNG"), "image/png", LIMIT)
        # numpy 影像的形狀是 (高, 寬, 通道數)：高 64、寬 128、3 個通道。
        self.assertEqual(out.shape, (64, 128, 3))
        # 第 32 列、第 10 行在左半邊（紅色）；BGR 順序下紅色是 [藍 0, 綠 0, 紅 255]。
        self.assertEqual(out[32, 10].tolist(), [0, 0, 255])  # 紅色在 BGR 是 (0, 0, 255)
        # 第 100 行在右半邊（藍色）；BGR 順序下藍色是 [藍 255, 綠 0, 紅 0]。
        self.assertEqual(out[32, 100].tolist(), [255, 0, 0])

    def test_exif_orientation_applied(self):
        """規則：照片帶有 EXIF 方向標記時，解碼後要依標記轉正。

        為什麼重要：手機拍的照片常常是「像素橫著存、再用 EXIF 標記要轉 90 度」；
        如果沒轉正，臉會是橫躺的，人臉偵測就找不到臉或位置錯誤。
        用 JPEG 是因為手機照片的 EXIF 通常在 JPEG 裡；JPEG 有損壓縮，所以用範圍比對而不是等號。
        """
        # 建立一個空的 EXIF 資料。
        exif = Image.Exif()
        # 0x0112 是 EXIF 的 Orientation 標籤；值 6 代表顯示時要順時針轉 90 度。
        exif[0x0112] = 6  # Orientation：需要順時針轉 90 度才是正的
        # 存成品質 95 的 JPEG 並附上 EXIF，再解碼。
        out = decode_bgr(encode(left_red_right_blue(), "JPEG", quality=95, exif=exif.tobytes()), "image/jpeg", LIMIT)
        # 轉正後變成直的，原本左半的紅色轉到上半部。
        # 原圖寬 128、高 64，順時針轉 90 度後變成高 128、寬 64。
        self.assertEqual(out.shape[:2], (128, 64))
        # 上半部（第 10 列）應該是紅色：BGR 的第 2 個通道（紅）要很高。
        self.assertGreater(int(out[10, 32, 2]), 200)
        # 同一點的第 0 個通道（藍）要很低，確認是紅色而不是其他顏色。
        self.assertLess(int(out[10, 32, 0]), 60)
        # 下半部（第 118 列）應該是藍色：BGR 的第 0 個通道（藍）要很高。
        self.assertGreater(int(out[118, 32, 0]), 200)


# 直接執行這個檔案時才跑測試；被 unittest discover 匯入時不會進來。
if __name__ == "__main__":
    # 找出這個模組裡所有 TestCase 並執行。
    unittest.main()
