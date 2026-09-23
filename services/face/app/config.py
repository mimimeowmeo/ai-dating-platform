"""人臉驗證 provider 的執行期設定。

只放「跟部署環境有關」的設定：和 AI 服務共用的認證 token、模型檔所在的資料夾。
正式執行時由 main.py 的 create_app 呼叫 Settings.from_env() 從環境變數讀取；
測試（例如 tests/test_api.py）則直接 Settings(token=...) 建構，不必設定環境變數。

刻意不放在這裡的東西：人臉偵測、比對、動作挑戰的門檻。門檻屬於判定政策，
集中寫死在 policy.py，改動要有測試、升版本並經人工核准，不開放用環境變數調整。
"""

# os：讀取環境變數（os.getenv）。
import os
# dataclass：自動產生 __init__ 等方法的裝飾器，寫法接近 TypeScript 只有欄位的 class。
from dataclasses import dataclass
# Path：跨平台的檔案路徑物件，可以用 / 串接路徑（例如 model_dir / "xxx.onnx"），models.py 就是這樣用的。
from pathlib import Path


# frozen=True：建立後欄位不能再修改（再指定值會丟 dataclasses.FrozenInstanceError），
# 避免執行途中有程式不小心改掉 token 或模型路徑。
@dataclass(frozen=True)
class Settings:
    """provider 的設定（不可變）；正式執行用 from_env，測試直接建構。辨識門檻不在這裡，見 policy.py。

    欄位：
        token：和 AI 服務共用的密鑰。main.py 的 ProviderBoundary 會要求請求帶
            「Authorization: Bearer <token>」；token 是空字串時，/verify 一律回 503
            PROVIDER_AUTH_NOT_CONFIGURED（fail closed：沒設定好就拒絕服務，而不是不驗證就放行）。
        model_dir：YuNet、SFace、MiniFASNet 模型檔所在的資料夾；main.py 會把它交給 models.load_models 載入。
            預設 /app/models，也就是 Docker 映像裡放模型的位置（Dockerfile 的 WORKDIR 是 /app，
            模型被 COPY 到 ./models）。
    """

    # 認證 token，預設空字串（代表尚未設定，/verify 會拒絕所有請求）。
    token: str = ""
    # 模型資料夾，預設是容器內的 /app/models。
    model_dir: Path = Path("/app/models")

    # @classmethod：這個方法屬於 class 本身，用 Settings.from_env() 呼叫，cls 參數就是 Settings 這個 class。
    # 常見用途是當作「另一種建構方式」（工廠方法），類似 TypeScript 的 static create()。
    @classmethod
    def from_env(cls) -> "Settings":
        """從環境變數建立設定。

        讀取的環境變數：
            FACE_PROVIDER_TOKEN：認證 token；沒設定時為空字串。
            FACE_MODEL_DIR：模型資料夾；沒設定、或只有空白時，退回預設的 /app/models。

        參數：
            cls：Settings class 本身（由 @classmethod 自動傳入）。

        回傳：
            新的 Settings 物件。

        錯誤：
            不會主動丟錯；缺少的設定都以預設值處理。token 是空字串的後果由 main.py 處理（/verify 回 503）；
            模型資料夾不存在或缺檔則在 main.py 載入模型時失敗，服務仍會啟動但 /health 回 503。
        """
        # 呼叫 cls(...) 就等於呼叫 Settings(...)，用關鍵字參數指定每個欄位。
        return cls(
            # 與 AI 服務的 AI_VERIFICATION_PROVIDER_TOKEN 是同一個值（compose 會帶同一個變數）。
            # os.getenv(名稱, 預設值)：環境變數不存在時回傳預設值（這裡是空字串）。
            token=os.getenv("FACE_PROVIDER_TOKEN", ""),
            # 讀 FACE_MODEL_DIR 後用 .strip() 去掉前後空白；結果是空字串時，「or」會改用右邊的預設路徑
            # （Python 的空字串是 falsy，類似 JS 的 ""||"預設值"），最後再包成 Path 物件。
            model_dir=Path(os.getenv("FACE_MODEL_DIR", "").strip() or "/app/models"),
        )
