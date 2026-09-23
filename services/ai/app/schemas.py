"""真人驗證的資料格式（schema）與大小上限。

這個檔案定義 AI 私有服務在「真人驗證」流程中收發的所有 JSON 形狀，全部用 Pydantic v2 的
BaseModel 描述。對前端工程師來說，可以把它想成「TypeScript 型別 + zod 驗證」合在一起：
欄位型別寫在 class 上，Pydantic 會在建立物件時自動檢查，不符合就丟出 ValidationError。

資料流向：
- NestJS API（apps/api/src/profiles.ts）→ 本服務：`VerificationRequest`
  （自拍或即時鏡頭的正面影格、使用者的主照片 referenceImages、可選的 liveCapture 動作影格）。
- 本服務 → 人臉驗證 provider（services/face 的 /verify）：原樣轉送 `VerificationRequest`。
- provider → 本服務：`ProviderResult`（比對外公開的結果多三個「決策證據」欄位）。
- 本服務 → NestJS API：`VerificationResult`（去掉 provider 專用欄位後的公開結果）。

設計重點：
- 所有 model 都設 `extra="forbid"`（多一個不認得的欄位就拒絕）與 `strict=True`（不做型別轉換，
  例如字串 "0.9" 不會被自動轉成數字）。這樣上游送錯格式時會直接失敗，而不是被默默「修正」。
- 大小上限同時用在兩個地方：Pydantic 欄位的 `max_length`，以及 main.py 的 InternalBoundary
  在解析 JSON 之前就用 `MAX_BODY_BYTES` 擋掉過大的請求（避免先把超大 body 讀進記憶體）。
"""

# Annotated：替型別附加額外的驗證資訊（下面的 Score 用它把「0～1 的有限浮點數」包成一個可重複使用的型別）。
# Literal：限定只能是列出的幾個固定值，類似 TypeScript 的 "a" | "b" 字串聯集型別。
from typing import Annotated, Literal

# BaseModel：所有資料格式的父類別；ConfigDict：設定 model 的行為（extra、strict）；
# Field：替欄位加上長度、範圍、正規表示式等限制；model_validator：寫「跨欄位」的自訂驗證規則。
from pydantic import BaseModel, ConfigDict, Field, model_validator


# 自拍（或即時鏡頭的正面影格）解碼後的最大位元組數：5 MiB（5 × 1024 × 1024 = 5,242,880 bytes）。
# verification.py 解碼 Base64 後會再用這個數字檢查一次實際大小。
MAX_IMAGE_BYTES = 5 * 1024 * 1024
# 5 MiB 的檔案編成 Base64 後最長會有幾個字元：Base64 每 3 個位元組變成 4 個字元，
# 不足 3 個時補 "=" 湊滿 4 個，所以長度是 4 × ceil(位元組數 ÷ 3)；
# (n + 2) // 3 是「無條件進位的整數除法」寫法（// 是整數除法）。結果是 6,990,508 個字元。
MAX_BASE64_LENGTH = 4 * ((MAX_IMAGE_BYTES + 2) // 3)
# 身分參照（使用者的主照片）由 NestJS 先縮到 800px 內再送，上限比自拍小。
# （NestJS 用 sharp 等比例縮到長邊 800px 內、轉成 JPEG，實際大小遠小於 1 MiB。）
# 參照照片解碼後的最大位元組數：1 MiB（1,048,576 bytes）。
MAX_REFERENCE_BYTES = 1024 * 1024
# 參照照片 Base64 字串的最大長度，算法同上：4 × ceil(1,048,576 ÷ 3) = 1,398,104 個字元。
MAX_REFERENCE_BASE64_LENGTH = 4 * ((MAX_REFERENCE_BYTES + 2) // 3)
# 一次請求最多只能帶 1 張參照照片（目前只拿使用者的第一張主照片來比對）。
MAX_REFERENCE_IMAGES = 1
# 即時鏡頭的動作影格：NestJS 先縮到 1024px 內，每張 1 MiB 以下，最多 3 張（每個動作一張）。
# （NestJS 目前每次挑戰只出 2 個動作，見 profiles.ts 的 CHALLENGE_ACTION_COUNT；3 是這裡允許的上限。）
# 每張動作影格解碼後的最大位元組數：1 MiB。
MAX_FRAME_BYTES = 1024 * 1024
# 每張動作影格 Base64 字串的最大長度：同樣是 1,398,104 個字元。
MAX_FRAME_BASE64_LENGTH = 4 * ((MAX_FRAME_BYTES + 2) // 3)
# liveCapture.frames 最多幾張動作影格。
MAX_ACTION_FRAMES = 3
# 所有 /internal/* 共用這個上限；要增加參照或影格張數，先確認 reply API 也能接受更大的上限。
# （原因：main.py 的 InternalBoundary 對所有 /internal/ 路徑都用這一個數字，AI 推薦回覆的 API
# 也在 /internal/ai/ 底下，所以加大這裡等於連 reply API 可接受的請求大小一起放寬。）
# 這個數字也必須和 services/face/app/schemas.py 的 MAX_BODY_BYTES 相同，
# tests/test_verification.py 的 test_body_limit_matches_face_provider 把它鎖在 12,588,044。
# HTTP 請求 body 的總上限（位元組）＝「所有影像的 Base64 最大長度」加上 JSON 其餘部分的預留空間：
MAX_BODY_BYTES = (
    # 自拍（正面影格）的 Base64 最大長度。
    MAX_BASE64_LENGTH
    # 每張參照照片的 Base64 最大長度，再各預留 256 bytes 給 JSON 的鍵名、引號、mimeType 等。
    + MAX_REFERENCE_IMAGES * (MAX_REFERENCE_BASE64_LENGTH + 256)
    # 每張動作影格的 Base64 最大長度，再各預留 256 bytes 給 action、mimeType、鍵名與引號。
    + MAX_ACTION_FRAMES * (MAX_FRAME_BASE64_LENGTH + 256)
    # 最後預留 4 KiB 給 requestId、challengeId、外層大括號與其他欄位。
    + 4096
)


class ReferenceImage(BaseModel):
    """一張身分參照照片（使用者的主照片），provider 會拿自拍的人臉和它做 1:1 比對。

    欄位：
        imageBase64：照片檔案內容的 Base64 字串（不含 `data:image/...;base64,` 前綴）。
        mimeType：照片格式，只接受 JPEG、PNG、WebP。

    驗證失敗時 Pydantic 會丟出 ValidationError（FastAPI 會轉成 HTTP 422）。
    這裡只檢查字串長度與宣告的格式；「內容真的是那種格式的合法影像」由 verification.py 解碼後再檢查。
    """

    # extra="forbid"：多送任何未定義的欄位就拒絕；strict=True：不做自動型別轉換。
    model_config = ConfigDict(extra="forbid", strict=True)
    # Base64 字串：至少 4 個字元（Base64 最短的完整單位），最多是 1 MiB 影像編碼後的長度。
    imageBase64: str = Field(min_length=4, max_length=MAX_REFERENCE_BASE64_LENGTH)
    # 宣告的 MIME 類型，只能是這三種之一；verification.py 會比對它和實際檔案格式是否一致。
    mimeType: Literal["image/jpeg", "image/png", "image/webp"]


class ActionFrame(BaseModel):
    """即時鏡頭動作挑戰中的「一個動作」拍下的一張影格。

    欄位：
        action：這張影格對應的動作；方向以使用者自己為準（turn_left 是往自己的左邊轉頭）。
            只能是 turn_left（向左轉頭）、turn_right（向右轉頭）、look_up（抬頭）、look_down（低頭）。
        imageBase64：影格的 Base64 字串，最多是 1 MiB 影像編碼後的長度。
        mimeType：影格格式，只接受 JPEG、PNG、WebP。

    provider 會用這張影格和正面影格比較頭部角度，確認使用者真的做了指定動作，並確認是同一個人。
    """

    # 同上：拒絕多餘欄位、不做型別轉換。
    model_config = ConfigDict(extra="forbid", strict=True)
    # 動作名稱只能是這四個之一；送 "blink" 之類不認得的動作會驗證失敗。
    action: Literal["turn_left", "turn_right", "look_up", "look_down"]
    # 動作影格的 Base64 字串長度限制。
    imageBase64: str = Field(min_length=4, max_length=MAX_FRAME_BASE64_LENGTH)
    # 動作影格宣告的 MIME 類型。
    mimeType: Literal["image/jpeg", "image/png", "image/webp"]


class LiveCapture(BaseModel):
    """即時鏡頭的動作挑戰：imageBase64（正面影格）之外，每個動作各一張影格。

    欄位：
        challengeId：NestJS 發給前端的挑戰編號（NestJS 存在 Redis、120 秒內有效且只能用一次）。
            只允許英數字、底線、連字號，最長 128 字元。
        frames：動作影格清單，至少 1 張、最多 MAX_ACTION_FRAMES（3）張。

    正面影格不在這裡，而是放在 VerificationRequest.imageBase64，和「只上傳自拍」時同一個欄位，
    讓 provider 用同一條流程處理正面照。
    """

    # 同上：拒絕多餘欄位、不做型別轉換。
    model_config = ConfigDict(extra="forbid", strict=True)
    # 挑戰編號：1～128 字元；pattern 限制只能是英文字母、數字、底線、連字號，避免奇怪字元混進日誌或路徑。
    challengeId: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    # 動作影格清單：空清單或超過 3 張都會驗證失敗（FastAPI 回 422）。
    frames: list[ActionFrame] = Field(min_length=1, max_length=MAX_ACTION_FRAMES)


class VerificationRequest(BaseModel):
    """一次真人驗證請求：NestJS 送給本服務，本服務再原樣轉送給 provider。

    欄位：
        imageBase64：使用者上傳的自拍，或即時鏡頭的正面影格（Base64）。
        mimeType：上面那張影像的格式。
        requestId：這次請求的編號（英數字、底線、連字號，1～128 字元）。
        referenceImages：身分參照照片清單，預設空清單、最多 1 張。
        liveCapture：即時鏡頭的動作挑戰；只上傳自拍時是 None（轉送時會被省略，不送 null）。

    判定規則在 provider（services/face/app/policy.py）：沒有 liveCapture 時，就算全部檢查通過也只會回
    unavailable / LIVE_CAPTURE_REQUIRED；有 liveCapture 且全部通過才會回 verified。
    """

    # 同上：拒絕多餘欄位、不做型別轉換。
    model_config = ConfigDict(extra="forbid", strict=True)
    # 自拍（或正面影格）的 Base64 字串：最多是 5 MiB 影像編碼後的長度。
    imageBase64: str = Field(min_length=4, max_length=MAX_BASE64_LENGTH)
    # 自拍宣告的 MIME 類型。
    mimeType: Literal["image/jpeg", "image/png", "image/webp"]
    # 請求編號，格式限制同 challengeId。
    requestId: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    # 參照照片清單；default_factory=list 表示沒送這個欄位時，每次都建立一個新的空清單當預設值
    # （Python 不建議直接用 [] 當預設值，因為可變物件會被共用）。超過 1 張會驗證失敗。
    referenceImages: list[ReferenceImage] = Field(default_factory=list, max_length=MAX_REFERENCE_IMAGES)
    # 即時鏡頭資料；`LiveCapture | None = None` 代表可以不送，不送時值是 None。
    liveCapture: LiveCapture | None = None


# 分數型別：0.0～1.0 之間（ge＝大於等於、le＝小於等於）的浮點數，而且不能是 NaN 或無限大
# （allow_inf_nan=False）。定義成型別別名，讓 livenessScore 與 faceMatchScore 共用同一組限制。
Score = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class VerificationResult(BaseModel):
    """本服務回給 NestJS 的驗證結果（公開欄位）。

    欄位：
        status：verified（通過）、rejected（不通過）、unavailable（無法判定，例如沒設定 provider、
            provider 逾時，或只上傳自拍、沒有即時鏡頭）。
        reasonCode：原因代碼，全大寫英數字與底線、以英文字母開頭，最長 80 字元
            （例如 VERIFICATION_PASSED、LIVE_CAPTURE_REQUIRED、PROVIDER_TIMEOUT）。
        modelName／modelVersion：做出判定的模型名稱與版本；本服務自己回的 unavailable 沒有這兩個值（None）。
        livenessScore：防偽（活體）分數，0～1，可為 None。
        faceMatchScore：自拍和參照照片的人臉相似度分數，0～1，可為 None。

    也是 main.py 驗證路由的 response_model，所以 FastAPI 回應只會有這些欄位。
    """

    # 同上：拒絕多餘欄位、不做型別轉換。
    model_config = ConfigDict(extra="forbid", strict=True)
    # 驗證狀態只能是這三種之一。
    status: Literal["verified", "rejected", "unavailable"]
    # 原因代碼：1～80 字元，pattern 規定第一個字是大寫英文字母，後面只能是大寫英文字母、數字或底線。
    reasonCode: str = Field(min_length=1, max_length=80, pattern=r"^[A-Z][A-Z0-9_]*$")
    # 模型名稱：可以是 None（預設）；有值時長度 1～128。
    modelName: str | None = Field(default=None, min_length=1, max_length=128)
    # 模型版本：可以是 None（預設）；有值時長度 1～128。
    modelVersion: str | None = Field(default=None, min_length=1, max_length=128)
    # 防偽分數：符合 Score 限制的浮點數或 None。
    livenessScore: Score | None = None
    # 人臉比對分數：符合 Score 限制的浮點數或 None。
    faceMatchScore: Score | None = None


class ProviderResult(VerificationResult):
    """Provider 的決策須明確表明活體與身分比對；本服務不自行訂門檻。

    繼承 VerificationResult 的所有欄位與設定（extra="forbid"、strict=True 一樣生效），
    另外多三個 provider 必須回的欄位：
        verificationType：固定是 "identity_verification"，表示這是「身分驗證」而不只是人臉偵測。
        livenessVerified：provider 判定是否為真人（防偽通過）。
        identityVerified：provider 判定自拍和參照照片是否為同一人。

    「本服務不自行訂門檻」的意思：本服務不會拿 livenessScore／faceMatchScore 自己比門檻，
    而是只相信 provider 明確給出的布林決策，並用下面的 require_verified_evidence 檢查決策是否自相矛盾。
    驗證通過後，verification.py 會移除這三個欄位，轉成 VerificationResult 回給 NestJS。

    可能丟出：ValidationError（欄位不合格，或下面的自訂規則不成立）。
    """

    # 驗證類型：只能是這個固定字串；provider 回其他值（例如 "face_detection"）會驗證失敗。
    verificationType: Literal["identity_verification"]
    # 活體（防偽）判定結果；strict 模式下必須是真正的 JSON 布林值，字串 "true" 不會被接受。
    livenessVerified: bool
    # 身分比對判定結果；同樣必須是布林值。
    identityVerified: bool

    # mode="after"：在所有欄位各自驗證完成、物件建立好之後才執行，所以可以直接讀 self 的欄位。
    @model_validator(mode="after")
    def require_verified_evidence(self) -> "ProviderResult":
        """檢查 provider 的決策是否完整且一致，任何一條不成立都視為不合格的回應。

        規則：
        1. status 是 verified 或 rejected（真的做出判定）時，必須附上 modelName 與 modelVersion，
           才知道是哪個模型、哪個版本做的決定（可追溯）。unavailable 可以沒有。
        2. status 是 verified 時，livenessVerified 與 identityVerified 必須都是 True，
           而且 livenessScore 與 faceMatchScore 都要有值；缺任何一項證據都不能算通過。
        3. status 不是 verified 時，不能同時宣稱活體與身分都通過，否則決策自相矛盾。

        參數：無（檢查的是 self，也就是剛建立好的 ProviderResult）。
        回傳：通過檢查時回傳 self（Pydantic 的 after 驗證器必須回傳物件本身）。
        可能丟出：ValueError；Pydantic 會把它包成 ValidationError，
            verification.py 接到後回 unavailable / PROVIDER_INVALID_RESPONSE。
        """
        # 規則 1：有判定（verified／rejected）卻缺少模型名稱或版本。
        if self.status in {"verified", "rejected"} and (
            # 名稱或版本任一個是 None（或空值）就不合格。
            not self.modelName or not self.modelVersion
        ):
            # 丟出錯誤，讓整個回應被判為不合格。
            raise ValueError("Missing model provenance")
        # 規則 2：宣稱通過，但下面四項證據沒有全部齊全。
        if self.status == "verified" and not (
            # 活體判定必須通過。
            self.livenessVerified
            # 身分比對必須通過。
            and self.identityVerified
            # 防偽分數必須有值。
            and self.livenessScore is not None
            # 人臉比對分數必須有值。
            and self.faceMatchScore is not None
        ):
            # 缺少證據就不能接受這個「通過」。
            raise ValueError("Identity and liveness decisions are required")
        # 規則 3：沒有通過，卻同時宣稱活體與身分都通過，這是矛盾的決策。
        # （只有其中一項通過是允許的，例如 LIVE_CAPTURE_REQUIRED：身分比對通過但沒有即時鏡頭。）
        if self.status != "verified" and self.livenessVerified and self.identityVerified:
            # 矛盾的決策一律拒絕，避免把可疑結果往下傳。
            raise ValueError("Contradictory provider decision")
        # 全部規則都成立，回傳物件本身。
        return self
