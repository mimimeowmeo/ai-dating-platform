"""下載模型檔並驗證 sha256；只在 Docker build 的 models 階段執行，任何一個檔案不符就讓 build 失敗。

網址都鎖定在特定 commit。opencv_zoo 的 .onnx 放在 Git LFS，raw.githubusercontent.com 只會拿到
LFS pointer，要用 media.githubusercontent.com 才是真正的二進位檔。

這支腳本在整個人臉驗證服務裡的位置：
- services/face 執行期需要三種模型：YuNet（人臉偵測，.onnx）、SFace（人臉特徵，.onnx）、
  MiniFASNet（被動防偽，上游只提供 PyTorch 的 .pth 權重）。
- Dockerfile 的第一階段（`FROM python:3.12-slim AS models`）執行
  `python tools/fetch_models.py downloads`，把四個檔案下載到 downloads/；
  接著把兩個 .onnx 搬到 models/，再由 export_minifasnet.py 把兩個 .pth 轉成 ONNX 放進 models/。
- 第二階段（執行期映像）只複製 models/，所以這支腳本與下載的原始檔都不會出現在正式執行的容器裡。

為什麼要鎖 commit、檢查大小與 sha256：
- 鎖 commit：上游 repo 之後就算改了檔案，網址內容也不會變，build 結果可重現。
- 檢查位元組數與 sha256（一種密碼學雜湊）：確保下載到的就是當初驗證過的那個檔案，
  避免網路傳輸出錯、拿到 LFS pointer 這種小文字檔，或檔案被人替換（供應鏈攻擊）。

用法：`python fetch_models.py [輸出資料夾]`，沒給資料夾時預設是 "downloads"。

可能的失敗（都會讓程式以非 0 狀態結束，因此 Docker build 失敗）：
- 網路錯誤或 HTTP 錯誤：urllib 丟出的例外（例如 urllib.error.HTTPError、URLError、逾時）不處理，直接中止。
- 大小不符：fetch() 丟出 ValueError。
- sha256 不符：main() 呼叫 sys.exit(訊息)，訊息印到 stderr、結束代碼為 1。
"""

# hashlib：Python 標準庫的雜湊模組，這裡用它算 sha256。
import hashlib
# sys：用來讀命令列參數（sys.argv）以及用 sys.exit 帶著錯誤訊息結束程式。
import sys
# urllib.request：Python 標準庫的 HTTP 客戶端；build 階段不想多裝 requests 之類的套件，用標準庫就夠。
import urllib.request
# Path：物件導向的檔案路徑工具，可以用 `/` 組路徑、直接 write_bytes 寫檔。
from pathlib import Path

# opencv_zoo（OpenCV 官方模型庫）的下載根網址。
# 路徑中的 47534e27… 是鎖定的 commit hash；網域用 media.githubusercontent.com 才能拿到 Git LFS 的真正檔案內容。
OPENCV_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models"
# Silent-Face-Anti-Spoofing（Minivision 開源的被動防偽專案）存放預訓練權重的資料夾網址。
# 用括號把兩段字串寫在兩行：Python 會把相鄰的字串常值自動接成一個字串，只是為了避免單行過長。
SILENT_FACE = (
    # 第一段：raw.githubusercontent.com + repo 路徑（這些 .pth 不是放在 LFS，用 raw 網域就能拿到二進位檔）。
    "https://raw.githubusercontent.com/minivision-ai/Silent-Face-Anti-Spoofing/"
    # 第二段：鎖定的 commit hash（b6d5f04…）加上權重資料夾路徑。
    "b6d5f04ad78778917853b25c778acef6d5626d15/resources/anti_spoof_models"
)

# (檔名, 網址, 位元組數, sha256)
# 要下載的檔案清單。用 tuple（不可變的序列）而不是 list，表示這是固定設定、執行中不會被改動。
# 位元組數與 sha256 是事先下載並確認過的正確值；數字中的底線（例如 232_589）只是千分位分隔，方便閱讀，值不變。
FILES = (
    # YuNet 人臉偵測模型（2023 年 3 月版）：執行期用 cv2.FaceDetectorYN 載入，找出人臉框與 5 個臉部點。
    (
        # 存檔用的檔名，也是 app/models.py 載入時使用的檔名。
        "face_detection_yunet_2023mar.onnx",
        # 下載網址：opencv_zoo 根網址 + 模型子資料夾 + 檔名（f-string 會把 {OPENCV_ZOO} 代換成上面的常數）。
        f"{OPENCV_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        # 預期的檔案大小（位元組）。
        232_589,
        # 預期的 sha256（64 個十六進位字元）。
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    # SFace 人臉特徵模型（2021 年 12 月版）：執行期用 cv2.FaceRecognizerSF 算人臉特徵做 1:1 比對。
    (
        # 存檔用的檔名。
        "face_recognition_sface_2021dec.onnx",
        # 下載網址。
        f"{OPENCV_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        # 預期的檔案大小（約 38.7 MB，是四個檔案中最大的）。
        38_696_353,
        # 預期的 sha256。
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
    # MiniFASNetV2 的 PyTorch 權重；檔名開頭的 2.7 是裁切倍率、80x80 是模型輸入大小（見 app/models.py 的 MINIFASNET_FILES）。
    (
        # 存檔用的檔名；export_minifasnet.py 會用同一個檔名去讀它。
        "2.7_80x80_MiniFASNetV2.pth",
        # 下載網址：Silent-Face 權重資料夾 + 檔名。
        f"{SILENT_FACE}/2.7_80x80_MiniFASNetV2.pth",
        # 預期的檔案大小（位元組）。
        1_849_453,
        # 預期的 sha256。
        "a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0",
    ),
    # MiniFASNetV1SE 的 PyTorch 權重；檔名開頭的 4_0_0 代表裁切倍率 4.0，輸入同樣是 80x80。
    (
        # 存檔用的檔名；export_minifasnet.py 會用同一個檔名去讀它。
        "4_0_0_80x80_MiniFASNetV1SE.pth",
        # 下載網址。
        f"{SILENT_FACE}/4_0_0_80x80_MiniFASNetV1SE.pth",
        # 預期的檔案大小（位元組）。
        1_856_130,
        # 預期的 sha256。
        "84ee1d37d96894d5e82de5a57df044ef80a58be2b218b5ed7cdfd875ec2f5990",
    ),
)


# 下載單一檔案並確認大小正確。
def fetch(url: str, size: int) -> bytes:
    """下載 `url` 的內容，並確認長度剛好是 `size` 個位元組。

    參數：
        url：要下載的網址（FILES 裡鎖定 commit 的網址）。
        size：預期的檔案大小（位元組）。

    回傳：
        下載到的完整檔案內容（bytes），交給呼叫端算 sha256 與寫檔。

    可能丟出的錯誤：
        ValueError：實際長度和 `size` 不同（太短或太長）。
        urllib.error.HTTPError／urllib.error.URLError／逾時相關例外：網路或伺服器出錯時由 urllib 丟出，
        這裡不攔截，讓 build 直接失敗。

    設計理由：
        只讀 `size + 1` 個位元組，而不是無上限地讀到結束：如果伺服器回的內容比預期大，
        最多也只多讀 1 個位元組就足以判斷「太長」，不會把一個異常巨大的回應整個讀進記憶體。
        先比大小是很便宜的第一道檢查（例如拿到只有一百多位元組的 LFS pointer 會立刻被擋下），
        真正確認內容正確的是 main() 裡的 sha256 比對。
    """
    # 開啟 HTTP 連線；timeout=120 是每次等待網路回應的秒數上限，避免 build 卡住不動。
    # `with` 區塊結束時會自動關閉連線（即使中途發生例外也一樣）。
    with urllib.request.urlopen(url, timeout=120) as response:
        # 最多讀 size + 1 個位元組：多讀的 1 個位元組用來偵測「檔案比預期大」。
        data = response.read(size + 1)
    # 長度不等於預期（少了代表下載不完整或拿錯檔，多了代表內容和預期不同）就丟出錯誤。
    if len(data) != size:
        # 錯誤訊息寫出預期與實際長度，方便從 build log 判斷原因。
        raise ValueError(f"size mismatch: expected {size}, got {len(data)}")
    # 大小正確，把內容交回呼叫端。
    return data


# 腳本主流程：逐一下載、驗證、寫檔。
def main(out_dir: Path) -> None:
    """依序下載 FILES 列出的所有模型檔，驗證 sha256 後寫進 `out_dir`。

    參數：
        out_dir：輸出資料夾；不存在時會自動建立（包含中間各層資料夾）。

    回傳：
        None。每成功一個檔案就在標準輸出印一行 "<檔名>: ok (<大小> bytes)"。

    可能丟出的錯誤／結束方式：
        SystemExit：某個檔案的 sha256 不符時呼叫 sys.exit(訊息)，訊息印到 stderr、結束代碼 1。
        ValueError 與網路例外：由 fetch() 傳上來，不攔截。

    設計理由：
        先驗證、再寫檔，所以驗證失敗的內容永遠不會被寫進輸出資料夾。
        任何一個檔案失敗就整個中止，因為少一個模型服務就無法運作，寧可讓 build 失敗也不要產出不完整的映像。
    """
    # 建立輸出資料夾；parents=True 會一併建立缺少的上層資料夾，exist_ok=True 表示資料夾已存在也不報錯。
    out_dir.mkdir(parents=True, exist_ok=True)
    # 逐一處理 FILES 裡的每個檔案；每個元素是 4 個值的 tuple，這裡直接拆成 4 個變數。
    for name, url, size, sha256 in FILES:
        # 下載並確認大小（大小不符會在 fetch 裡丟出 ValueError）。
        data = fetch(url, size)
        # 計算下載內容的 sha256，hexdigest() 轉成 64 個小寫十六進位字元的字串，方便和預期值比較。
        digest = hashlib.sha256(data).hexdigest()
        # 和 FILES 裡記錄的預期 sha256 比較。
        if digest != sha256:
            # 不符就立刻結束程式：sys.exit 收到字串時會把它印到 stderr 並以代碼 1 結束，Docker build 因此失敗。
            # 訊息中附上實際算出的雜湊值，方便查是哪個版本的檔案。
            sys.exit(f"{name}: sha256 mismatch ({digest})")
        # 驗證通過才寫檔；`out_dir / name` 用 Path 的 `/` 運算子組出完整路徑，write_bytes 以二進位寫入（已存在會覆蓋）。
        (out_dir / name).write_bytes(data)
        # 印出進度，讓 build log 看得到每個檔案都驗證成功。
        print(f"{name}: ok ({size} bytes)")


# 只有「直接執行這個檔案」時才跑下面這段；被其他程式 import 時不會自動下載。
if __name__ == "__main__":
    # sys.argv[0] 是腳本本身，sys.argv[1] 是第一個命令列參數（輸出資料夾）。
    # 有給參數就用它，沒給就用預設的 "downloads"，再包成 Path 物件交給 main。
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "downloads"))
