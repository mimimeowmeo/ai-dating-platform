"""影像解碼與縮放：把 AI 服務轉送來的 Base64 影像變成 OpenCV 可以直接用的陣列。

這個模組是 provider 收到影像後的「第一道關卡」，pipeline.py 會對三種影像都呼叫它：
- 正面影格或上傳的自拍（imageBase64，上限 schemas.MAX_IMAGE_BYTES）
- 使用者的第一張主照片（referenceImages，上限 MAX_REFERENCE_BYTES，錯誤碼會被加上 REFERENCE_ 前綴）
- 即時鏡頭的動作影格（liveCapture.frames，上限 MAX_FRAME_BYTES，錯誤碼會被加上 FRAME_ 前綴）

設計重點：
1. 全部在記憶體處理，不寫到磁碟，避免自拍這類敏感個資留在伺服器上。
2. 任何不合格的輸入都丟 InvalidImage（帶一個錯誤碼字串），main.py 會把它轉成 HTTP 400，
   回應只含錯誤碼與固定訊息，不會回顯影像內容。
3. 先用檔頭資訊（格式、寬高、影格數）檢查，通過了才真正解碼像素，
   以免惡意的超大圖（decompression bomb，解壓縮炸彈）把記憶體吃光。
"""

# base64：把前端／AI 服務傳來的 Base64 文字還原成原始的影像位元組。
import base64
# binascii：base64 解碼失敗時丟出的 binascii.Error 定義在這裡，要 import 才能 except 它。
import binascii
# io：用 io.BytesIO 把 bytes 包成「像檔案一樣可以讀」的物件，Pillow 就能直接從記憶體開檔，不必寫到磁碟。
import io
# warnings：Pillow 對「疑似解壓縮炸彈」的大圖只發警告（warning）而不是丟錯，要用 warnings 模組把它升級成例外。
import warnings

# cv2：OpenCV 的 Python 套件，這裡只用來縮放影像（cv2.resize）。
import cv2
# numpy：影像在 OpenCV 裡就是 numpy 陣列（高 × 寬 × 通道），np 是慣用的簡稱。
import numpy as np
# Pillow（PIL）：負責解析 JPEG／PNG／WebP 檔案。
#   Image：開檔與影像物件；ImageOps：exif_transpose（依 EXIF 轉正）；
#   UnidentifiedImageError：Pillow 認不出檔案格式時丟出的例外。
from PIL import Image, ImageOps, UnidentifiedImageError

# 允許的影像格式：key 是 Pillow 判讀出來的格式名稱（image.format），value 是對應的 MIME type。
# 用途是比對「呼叫端宣稱的 mimeType」和「檔案實際的格式」是否一致；不在這個表裡的格式（例如 GIF）
# 查不到會得到 None，必然和宣稱的 mimeType 不同，因此一律被拒絕。
# 這三種格式和 schemas.py 的 MimeType 允許值相同。
FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
# 影像短邊至少 64 px：再小就不可能有夠大的臉（policy.MIN_FACE_SIDE 也是 64），直接拒絕省下偵測成本。
MIN_DIMENSION = 64
# 影像長邊最多 4096 px：擋掉異常大的圖，限制解碼後佔用的記憶體。
MAX_DIMENSION = 4096
# 總像素上限 1600 萬（寫成 16_000_000，底線只是方便閱讀的千分位，數值不變）。
# 注意 4096 × 4096 ≈ 1677 萬，會超過這個上限，所以長寬都接近 4096 的圖仍會被這條擋下。
MAX_PIXELS = 16_000_000


class InvalidImage(ValueError):
    """影像不合格時丟出的例外；例外訊息（str(error)）就是錯誤碼，例如 "INVALID_IMAGE"。

    繼承 ValueError：語意上是「輸入值不合法」。
    main.py 有專門的 exception handler 把它轉成 HTTP 400，回應內容是 {"code": 錯誤碼, "message": 固定中文訊息}；
    pipeline.py 的 _decode_prefixed 會攔下它，替主照片／動作影格的錯誤碼加上 REFERENCE_／FRAME_ 前綴後再丟出。
    """

    # 不需要額外的屬性或方法，錯誤碼直接放在例外訊息裡；pass 表示「這個 class 本體是空的」。
    pass


def decode_bgr(image_base64: str, mime_type: str, max_bytes: int) -> np.ndarray:
    """解碼成 OpenCV 使用的 BGR uint8 陣列；只在記憶體處理，不落地、不讀 EXIF 以外的中繼資料。

    做什麼：
        Base64 文字 → bytes → 用 Pillow 開檔並檢查 → 依 EXIF 轉正 → 轉成 RGB → 翻成 BGR 的 numpy 陣列。

    參數：
        image_base64: 影像檔的 Base64 字串（不含 "data:image/...;base64," 這種前綴）。
        mime_type: 呼叫端宣稱的格式，"image/jpeg"、"image/png" 或 "image/webp"。
        max_bytes: 解碼後原始 bytes 的上限；自拍、主照片、動作影格各自不同（見 schemas.py 的 MAX_*_BYTES）。

    回傳：
        形狀為 (高, 寬, 3)、型別 uint8、通道順序 B、G、R 的 numpy 陣列，而且記憶體是連續的（contiguous）。

    可能丟出的錯誤（都是 InvalidImage，訊息為錯誤碼）：
        INVALID_IMAGE：Base64 格式錯誤、檔案無法辨識或已損壞、疑似解壓縮炸彈。
        IMAGE_TOO_LARGE：解碼後是空的，或超過 max_bytes。
        IMAGE_TYPE_MISMATCH：實際格式和 mime_type 不符，或不是允許的三種格式。
        INVALID_IMAGE_DIMENSIONS：短邊小於 64、長邊大於 4096，或總像素超過 1600 萬。
        ANIMATED_IMAGE_NOT_ALLOWED：多影格的動畫圖（例如動態 PNG／WebP）。

    設計理由：
        先檢查、後解碼；錯誤一律轉成少數幾個固定錯誤碼，並用 `from None` 切斷原始例外，
        避免 Pillow 的內部錯誤細節外洩到日誌或回應。
    """
    # 第一步：Base64 文字 → 原始 bytes。放在 try 裡是因為格式錯誤會丟例外。
    try:
        # validate=True：遇到 Base64 字母表以外的字元（空白、換行、奇怪符號）就直接丟錯，
        # 而不是預設行為那樣默默略過；這樣格式不正確的輸入不會被「猜」成一張圖。
        data = base64.b64decode(image_base64, validate=True)
    # binascii.Error：非法字元或 padding（結尾的 =）不正確；ValueError：字串含非 ASCII 字元等情況。
    except (binascii.Error, ValueError):
        # 統一回報成 INVALID_IMAGE。`from None` 讓 Python 不附帶原本的例外鏈，錯誤訊息保持乾淨。
        raise InvalidImage("INVALID_IMAGE") from None
    # 解碼結果是空的，或 bytes 數超過這類影像的上限，就拒絕（兩種情況共用 IMAGE_TOO_LARGE 這個錯誤碼）。
    if not data or len(data) > max_bytes:
        # 拒絕過大的檔案，避免後面解析時耗用過多記憶體與 CPU。
        raise InvalidImage("IMAGE_TOO_LARGE")
    # 第二步：用 Pillow 開檔、檢查、解碼。這整段可能丟出各種例外，外層 try 會統一整理。
    try:
        # catch_warnings()：在這個 with 區塊內暫時改變警告的處理方式，離開區塊就恢復原狀，不影響其他程式。
        with warnings.catch_warnings():
            # 把 Pillow 的 DecompressionBombWarning（像素數超過 Pillow 預設安全值時發出的警告）升級成例外，
            # 讓它被下面的 except 接住並拒絕；否則只是警告，程式會繼續去解碼那張巨大的圖。
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            # Image.open 是「懶惰」的：這時只讀檔頭（格式、尺寸），還沒解碼像素，所以先檢查很便宜。
            # io.BytesIO(data) 讓 Pillow 從記憶體讀；with 區塊結束時會自動關閉影像、釋放資源。
            with Image.open(io.BytesIO(data)) as image:
                # 從檔頭取得寬與高（像素）。Pillow 的 size 順序是 (寬, 高)。
                width, height = image.size
                # image.format 是 Pillow 依檔案內容判斷的真正格式（例如 "JPEG"），查表換成 MIME type 後
                # 和呼叫端宣稱的 mime_type 比較；不一致代表副檔名／宣稱有誤或檔案被偽裝，直接拒絕。
                if FORMATS.get(image.format) != mime_type:
                    # 格式不符或不支援的格式。
                    raise InvalidImage("IMAGE_TYPE_MISMATCH")
                # 尺寸檢查：三個條件任一成立就拒絕。
                if (
                    # 短邊太小：不可能有夠大的臉。
                    min(width, height) < MIN_DIMENSION
                    # 長邊太大：限制記憶體用量。
                    or max(width, height) > MAX_DIMENSION
                    # 總像素太多：即使單邊沒超過 4096，面積也可能太大。
                    or width * height > MAX_PIXELS
                ):
                    # 尺寸不符合要求。
                    raise InvalidImage("INVALID_IMAGE_DIMENSIONS")
                # 動畫圖（動態 PNG、動態 WebP）有 n_frames 屬性且大於 1；沒有這個屬性的格式視為 1 張。
                # 驗證只接受單張靜態影像，避免「一個檔案塞多張臉部畫面」造成判讀不明確。
                if getattr(image, "n_frames", 1) != 1:
                    # 拒絕動畫圖。
                    raise InvalidImage("ANIMATED_IMAGE_NOT_ALLOWED")
                # NestJS 已經依 EXIF 轉正；這裡再做一次，直接呼叫 provider 時臉才不會是橫的。
                # 詳細步驟：
                #   ImageOps.exif_transpose(image)：讀 EXIF 的 Orientation 標籤，把手機直拍但存成橫向的照片轉正；
                #   .convert("RGB")：統一成 3 通道 RGB（去掉 PNG 的透明通道、灰階轉彩色），這一步才真正解碼像素；
                #   np.asarray(...)：轉成 numpy 陣列，形狀 (高, 寬, 3)、型別 uint8（0–255）。
                rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    # 上面自己丟的 InvalidImage 要原樣往外丟，保留具體的錯誤碼。
    # 這個 except 必須放在下一個之前：InvalidImage 是 ValueError 的子類別，
    # 若先被下面的 ValueError 接住，就會被改寫成籠統的 INVALID_IMAGE。
    except InvalidImage:
        # 不做任何處理，直接重新丟出同一個例外。
        raise
    # 其餘 Pillow 在解析／解碼檔案時可能丟出的例外，全部視為「壞掉或可疑的影像」。
    except (
        # UnidentifiedImageError：認不出格式；OSError：檔案截斷、解碼失敗等 I/O 類錯誤；
        # SyntaxError、ValueError：部分格式解析器遇到不合規格的內容時丟出的錯誤。
        UnidentifiedImageError, OSError, SyntaxError, ValueError,
        # DecompressionBombError：像素數遠超 Pillow 安全值時直接丟的錯；
        # DecompressionBombWarning：上面用 simplefilter 升級成例外的警告。
        Image.DecompressionBombError, Image.DecompressionBombWarning,
    ):
        # 統一回報成 INVALID_IMAGE，並用 from None 隱藏內部細節。
        raise InvalidImage("INVALID_IMAGE") from None
    # PIL 是 RGB，OpenCV 與 MiniFASNet 的訓練資料都是 BGR。
    # rgb[:, :, ::-1]：所有列、所有行保持不變，只把最後一維（通道）反轉，RGB 變成 BGR。
    # 反轉後的陣列只是原資料的「檢視」（view），記憶體不連續；OpenCV 的函式需要連續記憶體，
    # 所以用 np.ascontiguousarray 複製成連續排列的新陣列再回傳。
    return np.ascontiguousarray(rgb[:, :, ::-1])


def fit_within(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """等比例縮到長邊不超過 max_side，回傳（縮小後的影像, 縮放比例）；本來就夠小時比例為 1。

    參數：
        image: BGR 影像陣列，形狀 (高, 寬, 3)。
        max_side: 長邊上限（pipeline 傳入 policy.WORKING_MAX_SIDE = 640）。

    回傳：
        (工作影像, ratio)。ratio = 新尺寸 ÷ 原尺寸，介於 0 與 1 之間；
        影像本來就夠小時，回傳「同一個」陣列物件（沒有複製）與 1.0。

    設計理由：
        YuNet 適合偵測約 10–300 px 的臉，大照片先縮小再偵測，速度快、結果也穩定；只縮小、不放大。
        pipeline 會用 ratio 把工作影像上的臉框除回去，換算成原圖座標，給 MiniFASNet 在原圖上裁切。
    """
    # numpy 影像的 shape 是 (高, 寬, 通道)，[:2] 只取前兩個：高與寬。
    height, width = image.shape[:2]
    # 縮放比例 = 上限 ÷ 長邊；再和 1.0 取最小值，確保只會縮小（長邊已經 ≤ max_side 時比例就是 1）。
    ratio = min(1.0, max_side / max(height, width))
    # 不需要縮放：直接回傳原影像，省下一次 resize 的運算。
    if ratio == 1.0:
        # 回傳原陣列與比例 1.0。
        return image, 1.0
    # 計算新尺寸。注意 OpenCV 的 cv2.resize 要的尺寸順序是 (寬, 高)，和 numpy shape 的 (高, 寬) 相反。
    # round 取最接近的整數像素（Python 3 剛好 .5 時取偶數，即 banker's rounding），max(1, ...) 保證至少 1 px，避免極端長寬比時算出 0 導致 resize 失敗。
    size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
    # INTER_AREA（區域平均插值）是 OpenCV 建議用來「縮小」影像的方法，能減少鋸齒與摩爾紋。
    # 回傳縮小後的影像與縮放比例。
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), ratio
