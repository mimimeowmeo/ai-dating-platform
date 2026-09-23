"""provider 的 HTTP 契約：請求同 services/ai 的 VerificationRequest，回應同 ProviderResult。

這個檔案定義 /verify 端點「收什麼、回什麼」的資料格式（schema），用 Pydantic 的 BaseModel 描述。
對前端工程師來說，可以把它想成 TypeScript 的 interface／type 加上 zod 的執行期驗證：
FastAPI 收到 JSON 後會自動用這裡的 class 驗證，不合格就直接回 422，根本不會進到判定流程。

資料流向：
- AI 服務（services/ai）把 VerificationRequest 以 JSON POST 到本服務的 /verify，
  本服務用 VerifyRequest 接收；兩邊的欄位與上限刻意保持一致。
- 本服務回傳 VerifyResponse，AI 服務用 ProviderResult 嚴格驗證（extra="forbid"），
  所以兩邊的欄位必須完全一致：多出任何欄位、或缺少 ProviderResult 的必填欄位，
  都會被 AI 服務判成 PROVIDER_INVALID_RESPONSE。

AI 服務已經驗過一次請求；這裡照同樣的上限再驗一次，避免有人繞過 AI 服務直接呼叫。
（縱深防禦：每一層都自己把關，不假設上一層一定可靠。）
"""

# typing 模組提供型別標註工具：
# - Annotated：在型別上附加額外的驗證條件（例如分數必須在 0～1）。
# - Literal：限定只能是幾個固定值之一，效果類似 TypeScript 的字串聯集型別 "a" | "b"。
from typing import Annotated, Literal

# Pydantic 是 Python 的資料驗證函式庫，FastAPI 用它來解析與驗證 JSON：
# - BaseModel：所有 schema class 的基底類別，繼承它就能自動驗證欄位。
# - ConfigDict：設定整個 model 的驗證行為（例如禁止多餘欄位、嚴格型別）。
# - Field：替單一欄位加上限制（長度、正規表示式、預設值等）。
# - model_validator：定義「整個 model 驗完之後」再做的跨欄位檢查。
from pydantic import BaseModel, ConfigDict, Field, model_validator

# 自拍（或即時鏡頭的正面影格）解碼後的原始檔案大小上限：5 MiB（5 × 1024 × 1024 bytes）。
# pipeline.py 解碼 imageBase64 時也會用這個值再檢查一次實際位元組數。
MAX_IMAGE_BYTES = 5 * 1024 * 1024
# 上面那個大小轉成 Base64 字串後的最大長度。
# Base64 每 3 個位元組編成 4 個字元，不足 3 個也會補齊成 4 個（用 = 補位），
# 所以長度是 4 × ceil(n ÷ 3)；(n + 2) // 3 就是用整數除法算 ceil(n ÷ 3) 的寫法（// 是無條件捨去的整數除法）。
# 在 schema 階段先用字串長度擋掉過大的輸入，不必真的解碼就能拒絕。
MAX_BASE64_LENGTH = 4 * ((MAX_IMAGE_BYTES + 2) // 3)
# 參照照片（主照片）由 NestJS 先縮到 800px 內再送，上限比自拍小。
# 參照照片解碼後的大小上限：1 MiB。
MAX_REFERENCE_BYTES = 1024 * 1024
# 參照照片 Base64 字串的最大長度，算法同 MAX_BASE64_LENGTH。
MAX_REFERENCE_BASE64_LENGTH = 4 * ((MAX_REFERENCE_BYTES + 2) // 3)
# 每次請求最多帶幾張參照照片：目前只拿使用者的第一張主照片來比對，所以是 1。
MAX_REFERENCE_IMAGES = 1
# 即時鏡頭的動作影格：NestJS 先縮到 1024px 內，每張 1 MiB 以下，最多 3 張（每個動作一張）。
# 每張動作影格解碼後的大小上限：1 MiB。
MAX_FRAME_BYTES = 1024 * 1024
# 每張動作影格 Base64 字串的最大長度，算法同 MAX_BASE64_LENGTH。
MAX_FRAME_BASE64_LENGTH = 4 * ((MAX_FRAME_BYTES + 2) // 3)
# 一次請求最多幾張動作影格。動作種類有 4 種；目前 NestJS API 每次挑戰抽 2 個
# （apps/api/src/profiles.ts 的 CHALLENGE_ACTION_COUNT），這裡的上限 3 比實際用量寬一點。
MAX_ACTION_FRAMES = 3
# 整個 HTTP 請求本體（body）的位元組上限，由 main.py 的 ProviderBoundary 在解析 JSON 之前就檢查，
# 避免對方送超大請求把記憶體吃光。算法是把每個欄位的最大可能長度加總：
MAX_BODY_BYTES = (
    # 自拍／正面影格的 Base64 字串最大長度。
    MAX_BASE64_LENGTH
    # 每張參照照片的 Base64 最大長度，外加 256 bytes 預留給 JSON 的鍵名、引號、mimeType 等額外字元。
    + MAX_REFERENCE_IMAGES * (MAX_REFERENCE_BASE64_LENGTH + 256)
    # 每張動作影格的 Base64 最大長度，同樣外加 256 bytes 給 action、mimeType 等欄位與 JSON 符號。
    + MAX_ACTION_FRAMES * (MAX_FRAME_BASE64_LENGTH + 256)
    # 最後再預留 4096 bytes 給其他短欄位（requestId、challengeId、外層括號等）。
    + 4096
)

# 允許的影像格式（MIME type）：只接受 JPEG、PNG、WebP 三種。
# 定義成型別別名（type alias），讓下面多個 class 共用，避免各自寫一份而不一致。
MimeType = Literal["image/jpeg", "image/png", "image/webp"]
# 動作挑戰允許的動作：往左轉、往右轉、抬頭、低頭。
# 左右是以「使用者自己」的左右為準；各動作怎麼判定見 pose.py 與 policy.py。
Action = Literal["turn_left", "turn_right", "look_up", "look_down"]


class ReferenceImage(BaseModel):
    """一張參照照片（使用者的主照片），用來和自拍做 1:1 人臉比對。

    欄位：
        imageBase64：照片檔案內容的 Base64 字串（不含 data: 前綴），長度 4 到 MAX_REFERENCE_BASE64_LENGTH。
        mimeType：照片格式，只能是 MimeType 列出的三種。

    驗證失敗時 Pydantic 會丟出 ValidationError，FastAPI 會轉成 422（見 main.py 的 request_error）。
    """

    # extra="forbid"：JSON 裡出現沒定義的欄位就驗證失敗，避免對方夾帶不明資料。
    # strict=True：嚴格型別，不做寬鬆的自動轉型（例如字串 "true" 不會被自動當成布林值 True），
    # 型別不對就直接驗證失敗。
    model_config = ConfigDict(extra="forbid", strict=True)
    # 照片的 Base64 字串；min_length=4 是因為最短的合法 Base64 就是 4 個字元，
    # max_length 讓過大的照片在 schema 階段就被擋下。
    imageBase64: str = Field(min_length=4, max_length=MAX_REFERENCE_BASE64_LENGTH)
    # 照片格式；解碼時會拿來確認檔案內容真的是這個格式。
    mimeType: MimeType


class ActionFrame(BaseModel):
    """即時鏡頭拍到的一張動作影格：使用者做某個動作（例如往左轉）時的畫面。

    欄位：
        action：這張影格對應的動作，只能是 Action 列出的四種之一。
        imageBase64：影格的 Base64 字串，長度 4 到 MAX_FRAME_BASE64_LENGTH。
        mimeType：影格格式，只能是 MimeType 列出的三種。

    pipeline 會拿這張影格的頭部角度和正面影格比較，確認動作真的做到，而且是同一個人。
    """

    # 同 ReferenceImage：禁止多餘欄位、嚴格型別。
    model_config = ConfigDict(extra="forbid", strict=True)
    # 這張影格要證明的動作。
    action: Action
    # 影格的 Base64 字串，長度限制理由同 ReferenceImage。
    imageBase64: str = Field(min_length=4, max_length=MAX_FRAME_BASE64_LENGTH)
    # 影格格式。
    mimeType: MimeType


class LiveCapture(BaseModel):
    """即時鏡頭的動作挑戰：imageBase64（正面影格）之外，每個動作各一張影格。

    只有請求帶了 liveCapture，policy 才可能回 verified；只上傳自拍檔（沒有 liveCapture）
    即使全部通過也只回 unavailable / LIVE_CAPTURE_REQUIRED，因為無法證明是活人當下拍攝。

    欄位：
        challengeId：挑戰的識別碼（由 NestJS API 發給前端、只能用一次），只允許英數字、底線與連字號。
        frames：動作影格清單，至少 1 張、最多 MAX_ACTION_FRAMES 張。
    """

    # 同上：禁止多餘欄位、嚴格型別。
    model_config = ConfigDict(extra="forbid", strict=True)
    # 挑戰 ID：長度 1～128；pattern 是正規表示式，^ 與 $ 表示整個字串都必須符合，
    # [A-Za-z0-9_-]+ 表示只能由英文字母、數字、底線、連字號組成，避免奇怪字元混進日誌或其他地方。
    challengeId: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    # 動作影格清單：list[ActionFrame] 表示每個元素都要符合 ActionFrame；
    # min_length=1 確保至少有一個動作，max_length 限制張數以控制請求大小與運算量。
    frames: list[ActionFrame] = Field(min_length=1, max_length=MAX_ACTION_FRAMES)


class VerifyRequest(BaseModel):
    """/verify 的請求本體，欄位與上限和 services/ai 的 VerificationRequest 一致。

    欄位：
        imageBase64：自拍或即時鏡頭的正面影格（Base64），長度 4 到 MAX_BASE64_LENGTH。
        mimeType：上面那張影像的格式。
        requestId：這次請求的識別碼，只允許英數字、底線與連字號。
        referenceImages：參照照片清單，最多 MAX_REFERENCE_IMAGES 張；沒帶時預設為空清單，
            pipeline 遇到空清單會回 rejected / REFERENCE_PHOTO_REQUIRED。
        liveCapture：即時鏡頭的動作挑戰；沒帶時為 None，代表只是上傳的自拍檔。
            （AI 服務轉送時用 model_dump(exclude_none=True)，None 的欄位根本不會出現在 JSON 裡，
            所以這裡需要預設值 None。）
    """

    # 同上：禁止多餘欄位、嚴格型別。
    model_config = ConfigDict(extra="forbid", strict=True)
    # 自拍／正面影格的 Base64 字串，長度上限對應 5 MiB 的原始檔案。
    imageBase64: str = Field(min_length=4, max_length=MAX_BASE64_LENGTH)
    # 自拍／正面影格的格式。
    mimeType: MimeType
    # 請求 ID：規則同 LiveCapture.challengeId（長度 1～128、只允許安全字元）。
    requestId: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    # 參照照片清單。default_factory=list 表示沒給時每次都建立一個「新的」空清單
    # （不直接寫 default=[]，是 Python 避免多個物件共用同一個可變預設值的慣例寫法）。
    referenceImages: list[ReferenceImage] = Field(default_factory=list, max_length=MAX_REFERENCE_IMAGES)
    # 即時鏡頭資料；型別 LiveCapture | None 表示可以是 LiveCapture 或 None，預設 None。
    liveCapture: LiveCapture | None = None


# 分數型別：必須是 0.0～1.0 之間的浮點數（ge = 大於等於、le = 小於等於），
# allow_inf_nan=False 拒絕 NaN 與無限大，避免不合理的數值被送出去。
# 用 Annotated 把限制附加在 float 上，下面的 livenessScore 與 faceMatchScore 共用。
Score = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class VerifyResponse(BaseModel):
    """欄位必須和 ProviderResult 完全一致；多一個欄位，AI 服務就會判成 PROVIDER_INVALID_RESPONSE。

    pipeline.py 的 _result 用這個 class 組出每一種判定結果；建立物件時就會執行下面的驗證，
    所以不合規則的結果根本組不出來（會丟出錯誤，main.py 的 verify 會把它當成模型失敗、回 503）。

    欄位：
        status：判定狀態，只能是 verified（通過）、rejected（不通過）、unavailable（無法判定）。
        reasonCode：原因代碼，例如 VERIFICATION_PASSED、FACE_MISMATCH；只允許大寫英文、數字、底線，
            且必須以大寫英文字母開頭，長度 1～80。
        modelName：使用的模型組合名稱（見 policy.MODEL_NAME）。
        modelVersion：模型與判定政策的版本（見 policy.MODEL_VERSION），方便事後追查是哪一版做的判定。
        verificationType：固定為 "identity_verification"，有預設值，pipeline 不用自己填。
        livenessVerified：是否確認為活人（真人當下拍攝）。
        identityVerified：是否確認和參照照片是同一人。
        livenessScore：防偽模型判斷為真人的機率（0～1），沒有算到時為 None。
        faceMatchScore：自拍與參照照片的人臉相似度（0～1），沒有算到時為 None。
    """

    # 同上：禁止多餘欄位、嚴格型別。
    model_config = ConfigDict(extra="forbid", strict=True)
    # 判定狀態，三選一。
    status: Literal["verified", "rejected", "unavailable"]
    # 原因代碼；pattern 規則：第一個字元是 A～Z，後面是 A～Z、0～9 或底線；和 AI 服務的限制相同。
    reasonCode: str = Field(min_length=1, max_length=80, pattern=r"^[A-Z][A-Z0-9_]*$")
    # 模型名稱，長度 1～128。這裡是必填；AI 服務在 verified／rejected 時也要求一定要有。
    modelName: str = Field(min_length=1, max_length=128)
    # 模型版本，長度 1～128，必填。
    modelVersion: str = Field(min_length=1, max_length=128)
    # 驗證類型，只能是這個固定字串，並給預設值讓呼叫端不必傳入。
    verificationType: Literal["identity_verification"] = "identity_verification"
    # 活體（真人）判定結果，必填。
    livenessVerified: bool
    # 身分（同一人）判定結果，必填。
    identityVerified: bool
    # 活體分數，可以是 Score 或 None，預設 None（例如偵測階段就被拒絕時還沒有分數）。
    livenessScore: Score | None = None
    # 人臉比對分數，可以是 Score 或 None，預設 None。
    faceMatchScore: Score | None = None

    # mode="after"：在每個欄位各自驗證完成、物件建立好之後，才執行這個跨欄位的一致性檢查。
    @model_validator(mode="after")
    def consistent_decision(self) -> "VerifyResponse":
        """確保判定狀態和各項結果不互相矛盾。

        規則：
            1. status 是 verified 時，活體與身分都必須為 True，而且兩個分數都必須有值。
            2. status 不是 verified 時，不可以同時「活體通過」又「身分通過」。

        參數：
            self：已經通過欄位驗證的 VerifyResponse 物件。

        回傳：
            規則都符合時回傳 self（Pydantic 的 after validator 規定要回傳物件本身）。

        錯誤：
            ValueError：違反任一條規則時丟出，Pydantic 會把它包成 ValidationError。

        設計理由：AI 服務的 ProviderResult 也做同樣的檢查；在 provider 這端先擋，
        矛盾的判定就不會送出去，問題能在產生的地方被發現。
        """
        # 和 ProviderResult 的檢查相同：在這裡先擋，矛盾的判定不會送出去。
        # 規則 1：宣稱 verified，就必須同時具備活體通過、身分通過、兩個分數都有值；缺任何一項都不合理。
        if self.status == "verified" and not (
            # 活體必須通過。
            self.livenessVerified
            # 身分比對必須通過。
            and self.identityVerified
            # 必須有活體分數作為證據。
            and self.livenessScore is not None
            # 必須有人臉比對分數作為證據。
            and self.faceMatchScore is not None
        ):
            # 缺少證據就不能宣稱 verified，丟錯拒絕建立這個結果。
            raise ValueError("Identity and liveness decisions are required")
        # 規則 2：活體與身分都通過了卻不是 verified，前後矛盾，同樣拒絕。
        if self.status != "verified" and self.livenessVerified and self.identityVerified:
            # 丟錯拒絕建立這個結果。
            raise ValueError("Contradictory provider decision")
        # 兩條規則都通過，回傳物件本身。
        return self
