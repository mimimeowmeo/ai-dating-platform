# 改寫自 minivision-ai/Silent-Face-Anti-Spoofing 的 src/generate_patches.py（commit b6d5f04）。
# Copyright 2020 Minivision，依 Apache License 2.0 授權，全文見 third_party/licenses/。
# 本專案的修改：把 CropImage 類別改成模組函式、移除不裁切（crop=False）的分支，
# 並加上 bbox 取整（原本由 RetinaFace 的 get_bbox 負責，這裡改用 YuNet 的框）。
# 裁切的計算方式與原始碼相同，MiniFASNet 是用這種裁切方式訓練的，不要改。
# 另外加上中文註解說明；程式邏輯未變動。

"""MiniFASNet（被動防偽模型）輸入影像的裁切。

做什麼：
    以 YuNet 偵測到的臉框為中心，把框放大 scale 倍（例如 2.7 倍或 4.0 倍），
    從原圖切出一塊包含臉與周圍背景的區域，再縮成模型要的 80×80。

為什麼要放大、而且帶背景：
    上游的兩個 MiniFASNet 權重分別是用「臉框放大 2.7 倍」與「放大 4.0 倍」裁出來的影像訓練的，
    讓模型除了臉本身，也看得到臉周圍不同範圍的畫面；推論時必須用同樣的倍率與裁法，模型才看到它熟悉的輸入。
    models.py 的 MINIFASNET_FILES 記錄了這兩個倍率，數值來自原始權重檔名。

為什麼不能改計算方式：
    模型只在「和訓練時相同的裁切方式」下才準；任何看似無害的調整（例如先把框夾在圖內）
    都會改變裁切範圍與長寬比，讓防偽分數系統性偏移，而且不會有任何錯誤訊息提醒。
"""

# cv2：OpenCV，這裡只用 cv2.resize 把裁下來的區塊縮放成模型輸入大小。
import cv2
# numpy：影像是 numpy 陣列；這裡只用於型別標註（np.ndarray）。
import numpy as np


def _get_new_box(src_w, src_h, bbox, scale):
    """把臉框以中心放大 scale 倍，並確保結果完全落在影像內（上游原始演算法）。

    參數：
        src_w, src_h: 原圖的寬與高（像素）。
        bbox: 已取整的臉框 (x, y, w, h)：左上角座標與寬高，見 int_box。
        scale: 想要的放大倍率（2.7 或 4.0）。

    回傳：
        (左上 x, 左上 y, 右下 x, 右下 y) 四個整數，座標是「包含端點」的像素索引，
        保證 0 ≤ 左上 ≤ 右下 ≤ 影像邊長 − 1。

    演算法：
        1. 倍率不能大到讓放大後的框比整張圖還大，所以先把倍率限制住。
        2. 以原框中心為中心，算出放大後框的四個角。
        3. 若框超出影像某一邊，就把整個框「平移」回圖內，而不是把超出的部分截掉，
           這樣裁切的長寬比維持不變，和訓練時一致。

    函式名稱開頭的底線 _ 是 Python 慣例，表示「模組內部使用」，外部不應直接呼叫（測試除外）。
    """
    # 取出臉框左上角的 x 座標。
    x = bbox[0]
    # 取出臉框左上角的 y 座標。
    y = bbox[1]
    # 取出臉框寬度。
    box_w = bbox[2]
    # 取出臉框高度。
    box_h = bbox[3]

    # 限制倍率：(src_h - 1) / box_h 是「放大後高度不超過影像高度」允許的最大倍率，
    # (src_w - 1) / box_w 是寬度方向的最大倍率；三者取最小，放大後的框就一定塞得進影像。
    # box_w、box_h 至少為 1（int_box 保證），所以不會除以零。
    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))

    # 放大後的框寬。
    new_width = box_w * scale
    # 放大後的框高。
    new_height = box_h * scale
    # 原框的中心點：左上角加上一半的寬／高。放大是「以中心為基準」往四周擴張。
    center_x, center_y = box_w / 2 + x, box_h / 2 + y

    # 放大後框的左上角 x = 中心 − 半寬。
    left_top_x = center_x - new_width / 2
    # 放大後框的左上角 y = 中心 − 半高。
    left_top_y = center_y - new_height / 2
    # 放大後框的右下角 x = 中心 + 半寬。
    right_bottom_x = center_x + new_width / 2
    # 放大後框的右下角 y = 中心 + 半高。
    right_bottom_y = center_y + new_height / 2

    # 左邊超出影像（x 為負）：整個框往右平移超出的量。
    if left_top_x < 0:
        # left_top_x 是負數，減掉它等於右邊界往右加上超出的距離。
        right_bottom_x -= left_top_x
        # 左邊界貼齊影像左緣。
        left_top_x = 0

    # 上方超出影像（y 為負）：整個框往下平移超出的量。
    if left_top_y < 0:
        # 下邊界往下加上超出的距離。
        right_bottom_y -= left_top_y
        # 上邊界貼齊影像上緣。
        left_top_y = 0

    # 右邊超出影像（最大合法 x 是 src_w - 1）：整個框往左平移超出的量。
    if right_bottom_x > src_w - 1:
        # 超出量 = right_bottom_x - (src_w - 1)；左邊界往左移同樣距離。
        left_top_x -= right_bottom_x - src_w + 1
        # 右邊界貼齊影像右緣。
        right_bottom_x = src_w - 1

    # 下方超出影像（最大合法 y 是 src_h - 1）：整個框往上平移超出的量。
    if right_bottom_y > src_h - 1:
        # 超出量 = right_bottom_y - (src_h - 1)；上邊界往上移同樣距離。
        left_top_y -= right_bottom_y - src_h + 1
        # 下邊界貼齊影像下緣。
        right_bottom_y = src_h - 1

    # 用 int() 取整（直接捨去小數部分）後回傳四個角的座標，方便當作陣列索引。
    return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)


def int_box(box):
    """把 YuNet 的浮點框（x, y, w, h）取整，同上游 get_bbox 的寫法。

    不夾在影像範圍內：上游的框也可能超出影像，越界由 _get_new_box 先限制倍率、再整塊平移回圖內；
    自己先截短寬高會改變裁切範圍與長寬比，和模型訓練時看到的不一樣。

    參數：
        box: (x, y, w, h) 浮點數框，座標系是原圖（pipeline 已把工作影像上的框除以縮放比例換算回來）。

    回傳：
        (x, y, w, h) 四個整數。int() 會捨去小數（往 0 的方向取整，例如 -5.7 → -5）；
        寬高至少為 1，避免 _get_new_box 計算倍率時除以零。
    """
    # x、y 直接捨去小數；w、h 捨去小數後再和 1 取最大值，保證不會是 0。
    return int(box[0]), int(box[1]), max(1, int(box[2])), max(1, int(box[3]))


def crop_patch(org_img: np.ndarray, box, scale: float, out_w: int, out_h: int) -> np.ndarray:
    """從原圖裁出放大 scale 倍的臉部區塊，縮放成 out_w × out_h，作為 MiniFASNet 的輸入。

    參數：
        org_img: 原圖（BGR，形狀 (高, 寬, 3)），不是縮小後的工作影像。
        box: 原圖座標的臉框 (x, y, w, h)，可以是浮點數，也可以超出影像範圍。
        scale: 放大倍率（models.py 對兩個模型分別傳 2.7 與 4.0）。
        out_w, out_h: 輸出大小（models.py 傳 MINIFASNET_INPUT = 80）。

    回傳：
        形狀 (out_h, out_w, 3) 的 BGR 影像陣列。

    呼叫端：models.MiniFASNetSpoofDetector.assess。
    """
    # 取原圖的高與寬（numpy shape 順序是 高, 寬, 通道）。
    src_h, src_w = org_img.shape[:2]
    # 先把框取整，再算出放大並平移回圖內後的四個角座標。
    left_top_x, left_top_y, right_bottom_x, right_bottom_y = _get_new_box(src_w, src_h, int_box(box), scale)
    # 用 numpy 切片裁出區塊：第一維是列（y），第二維是行（x）。
    # 切片的結束索引不包含在內，而 _get_new_box 回傳的右下角是「包含」的，所以要 +1 才會切到右下角那個像素。
    img = org_img[left_top_y: right_bottom_y + 1, left_top_x: right_bottom_x + 1]
    # 縮放成模型輸入大小；cv2.resize 的尺寸參數順序是 (寬, 高)，未指定插值法時用預設的雙線性插值（INTER_LINEAR）。
    return cv2.resize(img, (out_w, out_h))
