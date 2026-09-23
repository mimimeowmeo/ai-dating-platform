"""手動檢查：用本機兩張照片跑真模型，只印出判定結果（不印影像、不印特徵向量）。不會被自動執行。

從專案根目錄執行（照片目錄唯讀掛進容器，照片不要放進 repo，見 SECURITY.md）：
  docker compose --profile face run --rm --no-deps \
    -v "$PWD/services/face/tests:/app/tests:ro" -v "<照片目錄>:/data:ro" \
    face python -m tests.smoke_local_images /data/selfie.jpg /data/avatar.jpg

補充說明：
- 檔名沒有 test_ 開頭，所以 `python -m unittest discover` 不會把它當成測試執行；只能像上面那樣手動跑。
- 模型檔只在 face 容器裡（Docker build 時下載與轉換），所以要在容器內執行；
  映像檔本身不含 tests 資料夾，要用 -v 把本機的 tests 唯讀掛進 /app/tests。
- 這裡只送自拍與主照片、沒有 liveCapture（即時鏡頭的動作影格），依 policy 版本 3，
  全部通過時的結果是 unavailable / LIVE_CAPTURE_REQUIRED，不會是 verified；
  要看的是 reasonCode 以及 livenessScore（防偽分數）、faceMatchScore（比對分數）是否合理。
"""

# base64：把處理好的 JPEG 位元組轉成 Base64 字串，和 API 請求的 imageBase64 格式相同。
import base64
# io：BytesIO 是記憶體裡的「假檔案」，JPEG 直接存在記憶體，不寫到硬碟。
import io
# sys：讀取命令列參數（sys.argv），以及參數錯誤時結束程式（sys.exit）。
import sys
# Path：用物件表示照片的檔案路徑。
from pathlib import Path

# Image：開啟與存檔影像；ImageOps.exif_transpose：依 EXIF 的方向標記把照片轉正。
from PIL import Image, ImageOps

# policy：判定政策，這裡用到 YuNet 的偵測信心門檻。
from app import policy
# Settings：從環境變數取得模型資料夾（FACE_MODEL_DIR，預設 /app/models）。
from app.config import Settings
# load_models：載入 YuNet、SFace、MiniFASNet 三個真模型。
from app.models import load_models
# FaceVerifier：完整的判定流程。
from app.pipeline import FaceVerifier
# VerifyRequest：/verify 的請求格式，會做和正式 API 相同的欄位驗證（長度上限等）。
from app.schemas import VerifyRequest


def like_api(path: Path, max_side: int, quality: int) -> dict:
    """照 NestJS 的做法轉正、縮圖、轉 JPEG：自拍 1600px／q88，主照片 800px／q85。

    參數：
        path：本機照片的路徑。
        max_side：長邊的上限（像素）；照片比這個小時維持原尺寸，不會放大。
        quality：JPEG 品質（數字越大畫質越好、檔案越大）。
    回傳：
        dict：{"imageBase64": ..., "mimeType": "image/jpeg"}，可以直接展開放進 VerifyRequest。
    可能丟出的錯誤：
        FileNotFoundError（檔案不存在）、PIL.UnidentifiedImageError（不是影像檔）等 PIL 的讀檔錯誤。
    設計理由：
        正式流程裡，照片是先經過 NestJS API（apps/api/src/profiles.ts 用 sharp）處理才送到 AI 服務與 provider；
        手動檢查時先做同樣的處理，模型看到的影像才會和正式上線時一致，結果才有參考價值。
    """
    # 開啟照片；with 區塊結束時會關閉檔案。
    with Image.open(path) as image:
        # 依 EXIF 方向轉正，再轉成 RGB（去掉透明通道、統一色彩模式）；結果是新的影像物件，關檔後仍可使用。
        upright = ImageOps.exif_transpose(image).convert("RGB")
    # 就地（in-place）等比例縮小，讓寬、高都不超過 max_side；thumbnail 只會縮小、不會放大。
    upright.thumbnail((max_side, max_side))
    # 建立記憶體中的暫存檔。
    buffer = io.BytesIO()
    # 用指定品質存成 JPEG。
    upright.save(buffer, format="JPEG", quality=quality)
    # 取出 JPEG 位元組做 Base64 編碼，連同 MIME 類型組成請求需要的格式。
    return {"imageBase64": base64.b64encode(buffer.getvalue()).decode("ascii"), "mimeType": "image/jpeg"}


def main(selfie: Path, reference: Path) -> None:
    """載入真模型，用一張自拍與一張主照片跑一次判定，並印出結果 JSON。

    參數：
        selfie：自拍照片的路徑。
        reference：主照片（個人檔案的第一張照片）的路徑，當作身分比對的參照。
    回傳：
        None；結果直接印在終端機，只包含 VerifyResponse 的欄位（狀態、代碼、分數），不含影像或特徵向量。
    可能丟出的錯誤：
        ModelError：模型檔不存在或模型執行失敗；InvalidImage：處理後的影像不符合 provider 的要求；
        pydantic.ValidationError：處理後的影像超過請求欄位的長度上限。
    """
    # 從環境變數設定的資料夾載入三個模型，建立判定流程。
    verifier = FaceVerifier(load_models(Settings.from_env().model_dir, policy.DETECTION_SCORE_THRESHOLD))
    # 組出和正式流程相同格式的請求（沒有 liveCapture）。
    request = VerifyRequest(
        # 自拍照 NestJS 的規則處理：長邊 1600px、JPEG 品質 88；requestId 隨便給一個合法的值。
        **like_api(selfie, 1600, 88), requestId="smoke",
        # 主照片照 NestJS 的規則處理：長邊 800px、JPEG 品質 85；最多只能有 1 張。
        referenceImages=[like_api(reference, 800, 85)],
    )
    # 執行判定，把結果轉成縮排 2 格的 JSON 字串印出。
    print(verifier.verify(request).model_dump_json(indent=2))


# 用 python -m tests.smoke_local_images 直接執行時才會進來。
if __name__ == "__main__":
    # sys.argv[0] 是程式本身，後面必須剛好兩個參數：自拍與主照片。
    if len(sys.argv) != 3:
        # 參數數量不對：印出用法到標準錯誤輸出，並以結束碼 1 結束程式。
        sys.exit("用法：python -m tests.smoke_local_images <自拍> <主照片>")
    # 把兩個參數轉成 Path 後執行檢查。
    main(Path(sys.argv[1]), Path(sys.argv[2]))
