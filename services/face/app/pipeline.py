"""真人驗證的判定流程（pipeline）：把請求裡的影像交給三個模型，依 policy 決定回應。

輸入（VerifyRequest，由 AI 服務轉送）：
- imageBase64：正面影格（即時鏡頭）或使用者上傳的自拍檔。
- referenceImages：使用者的第一張主照片，用來比對「自拍的人是不是照片裡的人」。
- liveCapture（可有可無）：即時鏡頭的動作影格，每個動作一張（turn_left／turn_right／look_up／look_down）。

流程（FaceVerifier._decide）：
1. 每張影像都要剛好偵測到一張夠大的臉。
2. 有 liveCapture 時：用 pose.py 算頭部角度，確認正面影格大致正對鏡頭、每個動作都做到，而且動作影格和正面影格是同一個人。
3. 被動防偽（MiniFASNet，每張影格都做）與大頭貼比對（SFace）。
4. 依 policy 版本 3 決定狀態：只有上傳的自拍檔（沒有 liveCapture）全部通過也只回
   unavailable／LIVE_CAPTURE_REQUIRED；即時鏡頭全部通過才回 verified／VERIFICATION_PASSED。

回應（VerifyResponse）的欄位必須和 services/ai 的 ProviderResult 完全一致。
"""

# threading：Python 內建的執行緒工具；這裡只用它的 Lock（鎖），讓模型一次只被一個請求使用。
import threading

# NumPy：影像陣列與向量內積（np.dot）。
import numpy as np

# policy：所有判定門檻與模型名稱／版本（見 policy.py）。
from . import policy
# InvalidImage：影像不合格時丟出的錯誤；decode_bgr：Base64 → BGR 影像；fit_within：等比例縮小影像。
from .imaging import InvalidImage, decode_bgr, fit_within
# Face：YuNet 偵測結果；Models：三個模型的組合（測試可換成假模型）。
from .models import Face, Models
# action_performed：判斷動作有沒有做到；head_pose：用 5 個臉部點算頭部角度。
from .pose import action_performed, facing_camera, head_pose
# 三種影像的大小上限，以及請求／回應的資料格式。
from .schemas import MAX_FRAME_BYTES, MAX_IMAGE_BYTES, MAX_REFERENCE_BYTES, VerifyRequest, VerifyResponse


def _face_problem(faces: list[Face]) -> str | None:
    """檢查一張影像的偵測結果是否「剛好一張夠大的臉」。

    參數：
        faces：YuNet 在這張影像偵測到的臉。

    回傳：
        有問題時回傳原因代碼（呼叫端會視影像種類加上 REFERENCE_／ACTION_ 前綴）；沒問題回傳 None。
        檢查順序固定：沒有臉 → 多張臉 → 臉太小。

    設計理由：
        多張臉時無法確定要驗證的是哪一個人，所以直接拒絕；臉太小時 SFace 對齊放大後細節不足，
        比對結果不可靠（門檻見 policy.MIN_FACE_SIDE）。
    """
    # 空清單代表沒有偵測到臉。
    if not faces:
        # 回傳「沒有偵測到人臉」。
        return "NO_FACE_DETECTED"
    # 超過一張臉：畫面裡有其他人（或照片、海報上的臉），無法確定要驗證誰。
    if len(faces) > 1:
        # 回傳「偵測到多張人臉」。
        return "MULTIPLE_FACES_DETECTED"
    # 臉框的寬（box[2]）與高（box[3]）取較短的一邊，小於門檻就算太小。
    # 座標是縮小後工作影像的座標（pipeline 都先用 fit_within 縮到長邊 640 以內再偵測）。
    if min(faces[0].box[2], faces[0].box[3]) < policy.MIN_FACE_SIDE:
        # 回傳「臉太小」。
        return "FACE_TOO_SMALL"
    # 三項都沒問題。
    return None


def _score(value: float) -> float:
    """把分數限制在 0–1 並四捨五入到小數第 4 位。

    參數：
        value：原始分數（防偽的真人機率，或 SFace 的餘弦相似度）。

    回傳：
        0 到 1 之間、最多 4 位小數的分數。

    設計理由：
        回應的 livenessScore／faceMatchScore 在 schema 裡限制為 0–1（strict 驗證），
        但餘弦相似度的範圍是 -1 到 1，不同人時可能是負數，所以先夾在 0–1 之間，否則建立回應會失敗。
    """
    # max(0.0, value) 把負數變成 0；min(1.0, ...) 把超過 1 的值變成 1；round(..., 4) 四捨五入到小數第 4 位。
    return round(min(1.0, max(0.0, value)), 4)


def _result(status, reason, *, liveness=False, identity=False, liveness_score=None, match_score=None):
    """組出 VerifyResponse（provider 的回應）。

    參數：
        status：判定狀態，"verified"／"rejected"／"unavailable" 三選一。
        reason：原因代碼（reasonCode），例如 "VERIFICATION_PASSED"、"FACE_MISMATCH"。
        *：後面的參數只能用「名稱=值」的方式傳（keyword-only），例如 identity=True，讀程式時一看就知道是哪個欄位。
        liveness：是否確認為活人（livenessVerified），預設 False。
        identity：是否確認和主照片是同一人（identityVerified），預設 False。
        liveness_score：防偽的真人分數（0–1），沒算到時為 None。
        match_score：和主照片的相似度分數（0–1），沒算到時為 None。

    回傳：
        VerifyResponse。模型名稱與版本固定從 policy 帶入，讓呼叫端知道是哪一版模型與政策做出的判定。

    可能丟出：
        pydantic 的 ValidationError：參數組合矛盾時（例如 status 是 verified 但沒有分數），
        VerifyResponse 的 consistent_decision 檢查會擋下；main.py 會把它當成模型失敗回 503。
    """
    # 建立回應物件；欄位名稱用 camelCase，是因為要和 AI 服務的 ProviderResult（JSON 契約）完全一致。
    return VerifyResponse(
        # 判定狀態與原因代碼
        status=status, reasonCode=reason,
        # 模型組合名稱與版本（版本字串包含 policy 版本）
        modelName=policy.MODEL_NAME, modelVersion=policy.MODEL_VERSION,
        # 活體與身分兩項判定結果
        livenessVerified=liveness, identityVerified=identity,
        # 兩個分數（可能是 None）
        livenessScore=liveness_score, faceMatchScore=match_score,
    )


def _decode_prefixed(image_base64: str, mime_type: str, max_bytes: int, prefix: str) -> np.ndarray:
    """解碼影像；失敗時在錯誤代碼前加上前綴，讓呼叫端知道是哪一種影像出問題。

    參數：
        image_base64：Base64 編碼的影像。
        mime_type：宣告的影像格式（image/jpeg、image/png、image/webp），必須和實際內容相符。
        max_bytes：解碼後的位元組上限。
        prefix：錯誤代碼前綴，主照片用 "REFERENCE_"，動作影格用 "FRAME_"。

    回傳：
        OpenCV 使用的 BGR uint8 影像陣列。

    可能丟出：
        InvalidImage：錯誤代碼加上前綴，例如 "REFERENCE_INVALID_IMAGE"、"FRAME_IMAGE_TOO_LARGE"；
        main.py 會回 400，code 就是這個代碼。
    """
    # try：嘗試解碼，失敗時進入 except 區塊（類似 JS 的 try/catch）。
    try:
        # 解碼成功就直接回傳影像。
        return decode_bgr(image_base64, mime_type, max_bytes)
    # 只攔截影像不合格的錯誤，並把它存成 error 變數。
    except InvalidImage as error:
        # 用原本的代碼加上前綴，丟出新的 InvalidImage（f-string 類似 JS 的樣板字串 `${prefix}${error}`）。
        # from None：不附帶原本的錯誤鏈（exception chaining），錯誤資訊只保留新的代碼。
        raise InvalidImage(f"{prefix}{error}") from None


class FaceVerifier:
    """判定流程：偵測 →（即時鏡頭）動作挑戰與同一人檢查 → 被動防偽 → 1:1 比對 → 依 policy 決定狀態。

    main.py 在啟動時建立一個實例，所有請求共用；模型透過 Models 注入，所以測試可以換成假模型。
    main.py 用 asyncio.to_thread 在背景執行緒呼叫 verify，因此可能有多個請求同時進來。
    """

    def __init__(self, models: Models):
        """保存模型並建立鎖。

        參數：
            models：YuNet、SFace、MiniFASNet 三個模型（或測試用的假模型）。
        """
        # 保存模型組合，_decide 會用到。
        self._models = models
        # FaceDetectorYN.setInputSize 會改內部狀態，同一組模型不能多執行緒同時使用。
        # 所以用一把鎖（Lock）讓同一時間只有一個請求在跑模型，其他請求排隊等待。
        self._lock = threading.Lock()

    def verify(self, request: VerifyRequest) -> VerifyResponse:
        """驗證一次請求：先解碼所有影像，再（在鎖內）交給 _decide 判定。

        參數：
            request：已通過 schema 驗證的請求（欄位格式、長度、動作名稱都已檢查過）。

        回傳：
            VerifyResponse：判定結果。

        可能丟出：
            InvalidImage：影像不合格。自拍用原本的代碼（例如 "INVALID_IMAGE"），
                主照片加 "REFERENCE_" 前綴，動作影格加 "FRAME_" 前綴；main.py 回 400。
            ModelError 或其他錯誤：模型執行失敗；main.py 回 503 MODEL_FAILED（fail closed）。

        設計理由：
            解碼不需要用到模型，所以放在鎖外面；鎖只包住真正使用模型的 _decide。
            所有影像先解碼完才開始判定，任何一張影像不合格就整個請求回 400，不會跑到一半。
        """
        # 解碼正面影格／自拍；錯誤代碼不加前綴。上限是 MAX_IMAGE_BYTES（5 MiB）。
        selfie = decode_bgr(request.imageBase64, request.mimeType, MAX_IMAGE_BYTES)
        # 解碼所有參照照片（主照片），錯誤代碼加 "REFERENCE_" 前綴；schema 限制最多 1 張。
        references = [
            # 每張主照片的上限是 MAX_REFERENCE_BYTES（1 MiB，NestJS 會先縮小再送）
            _decode_prefixed(reference.imageBase64, reference.mimeType, MAX_REFERENCE_BYTES, "REFERENCE_")
            # 逐張走訪請求裡的參照照片（沒有時是空清單）
            for reference in request.referenceImages
        ]
        # 預設沒有動作影格（None 代表「沒有即時鏡頭」，和「空清單」意義不同；schema 也要求至少 1 張）。
        frames = None
        # 有 liveCapture 才代表這是即時鏡頭拍的（challengeId 只在 schema 驗格式，這裡不使用）。
        if request.liveCapture is not None:
            # 把每張動作影格解碼，和動作名稱配成 (動作, 影像) 的 tuple。
            frames = [
                # 每張影格上限 MAX_FRAME_BYTES（1 MiB），錯誤代碼加 "FRAME_" 前綴
                (frame.action, _decode_prefixed(frame.imageBase64, frame.mimeType, MAX_FRAME_BYTES, "FRAME_"))
                # 逐張走訪動作影格（schema 限制 1–3 張）
                for frame in request.liveCapture.frames
            ]
        # with 區塊：進入時取得鎖、離開時（包含 return 或丟錯）自動釋放，確保模型一次只被一個請求使用。
        with self._lock:
            # 交給 _decide 做實際判定。
            return self._decide(selfie, references, frames)

    def _decide(self, selfie: np.ndarray, references: list[np.ndarray], frames) -> VerifyResponse:
        """依序執行所有檢查，第一個不通過的檢查就決定回應（提早 return）。

        參數：
            selfie：正面影格或上傳自拍，原始大小的 BGR 影像。
            references：主照片（BGR 影像）清單，只使用第一張。
            frames：即時鏡頭的 [(動作名稱, BGR 影像), ...]；上傳自拍（沒有 liveCapture）時為 None。

        回傳：
            VerifyResponse。可能的結果：
            - rejected：NO_FACE_DETECTED／MULTIPLE_FACES_DETECTED／FACE_TOO_SMALL（動作影格加 ACTION_ 前綴）、
              CHALLENGE_FAILED、FACE_CHANGED_DURING_CAPTURE、SPOOF_SUSPECTED、FACE_MISMATCH。
            - unavailable／REFERENCE_PHOTO_REQUIRED、REFERENCE_NO_FACE_DETECTED 等：主照片不能用，問題不在鏡頭前的人。
            - unavailable／LIVE_CAPTURE_REQUIRED：上傳的自拍全部通過，但沒有即時鏡頭。
            - verified／VERIFICATION_PASSED：即時鏡頭全部通過。

        可能丟出：
            ModelError 等模型錯誤（由 main.py 轉成 503）。
        """
        # 沒有主照片就沒有身分參照，無從比對。
        # 這項檢查放在最前面，不必跑任何模型就能回應。
        if not references:
            # 無法判定：需要主照片（NestJS 一定會附上，正常不會發生）。
            return _result("unavailable", "REFERENCE_PHOTO_REQUIRED")
        # 取成區域變數，後面寫起來比較短。
        models = self._models

        # 自拍先等比例縮到長邊 640 以內（YuNet 適合偵測較小的臉）；ratio 是縮放比例，後面要換算回原圖座標。
        selfie_work, ratio = fit_within(selfie, policy.WORKING_MAX_SIDE)
        # 在縮小後的工作影像上偵測人臉。
        selfie_faces = models.detector.detect(selfie_work)
        # 檢查是否剛好一張夠大的臉。
        problem = _face_problem(selfie_faces)
        # 有問題（回傳的代碼不是 None）就拒絕。
        if problem:
            # 自拍的問題不加前綴，例如 "NO_FACE_DETECTED"。
            return _result("rejected", problem)

        # 主照片也縮到長邊 640 以內；縮放比例用不到（主照片不做防偽），用 _ 忽略。只取第一張。
        reference_work, _ = fit_within(references[0], policy.WORKING_MAX_SIDE)
        # 在主照片上偵測人臉。
        reference_faces = models.detector.detect(reference_work)
        # 主照片同樣要剛好一張夠大的臉，否則不知道要和誰比對。
        problem = _face_problem(reference_faces)
        # 有問題就回 unavailable（不是 rejected）：問題出在主照片，不是鏡頭前的人，
        # NestJS 收到 unavailable 不會撤銷既有的驗證狀態，和它自己發現主照片太小、讀不到時的處理一致。
        if problem:
            # 加上 "REFERENCE_" 前綴，例如 "REFERENCE_NO_FACE_DETECTED"，讓前端知道是主照片的問題。
            return _result("unavailable", f"REFERENCE_{problem}")

        # 已確認自拍剛好一張臉，取出來。
        face = selfie_faces[0]
        # 算出自拍這張臉的特徵向量（長度 1），後面和動作影格、主照片比對都會用到。
        selfie_embedding = models.embedder.embed(selfie_work, face)
        # 要做被動防偽的影像與臉框（原圖座標）：先放正面影格／自拍，有動作影格時再加進來。
        # 臉框是在縮小後的影像上偵測的，每個值除以縮放比例就是原圖座標（沒縮放時 ratio 是 1）。
        spoof_targets = [(selfie, tuple(value / ratio for value in face.box))]

        # 即時鏡頭：每張動作影格都要剛好一張臉、做到指定的動作，而且和正面影格是同一個人。
        # frames 是 None 時代表上傳的自拍檔，跳過這整段。
        if frames is not None:
            # 正面影格的頭部角度當基準（neutral），動作影格要和它比較變化量。
            neutral = head_pose(face)
            # 臉部點不合理（兩眼距離或眼睛到嘴角的距離小於 1 像素，包含嘴角在眼睛上方）時算不出角度；
            # 正面影格本身沒有大致正對鏡頭（左右偏或歪頭太多）也不行，否則偏頭的「正面」能讓正臉照冒充反方向的動作。
            if neutral is None or not facing_camera(neutral):
                # 無法判斷動作，視為挑戰失敗。
                return _result("rejected", "CHALLENGE_FAILED")
            # 逐張檢查動作影格；action 是動作名稱，image 是解碼後的影像。
            for action, image in frames:
                # 同樣縮到長邊 640 以內；frame_ratio 是縮放比例，防偽時要把臉框換算回原圖座標。
                frame_work, frame_ratio = fit_within(image, policy.WORKING_MAX_SIDE)
                # 偵測這張影格的人臉。
                frame_faces = models.detector.detect(frame_work)
                # 同樣要剛好一張夠大的臉。
                problem = _face_problem(frame_faces)
                # 有問題就拒絕。
                if problem:
                    # 加上 "ACTION_" 前綴，例如 "ACTION_MULTIPLE_FACES_DETECTED"。
                    return _result("rejected", f"ACTION_{problem}")
                # 算這張影格的頭部角度。
                pose = head_pose(frame_faces[0])
                # 算不出角度，或和正面影格相比的變化量沒有超過門檻（方向也要對），就算沒做到動作。
                if pose is None or not action_performed(action, neutral, pose):
                    # 挑戰失敗。
                    return _result("rejected", "CHALLENGE_FAILED")
                # 動作影格和正面影格的餘弦相似度：兩個向量長度都是 1，內積就是相似度；轉成 Python float。
                same_person = float(np.dot(selfie_embedding, models.embedder.embed(frame_work, frame_faces[0])))
                # 低於同一人門檻：拍攝途中換了人（例如正面是本人，轉頭時換成別人），拒絕。
                if same_person < policy.MATCH_THRESHOLD:
                    # 拒絕：拍攝途中臉變了。
                    return _result("rejected", "FACE_CHANGED_DURING_CAPTURE")
                # 這張動作影格也要做被動防偽：轉動照片或螢幕時，邊框、反光、透視變形最容易出現在動作影格裡。
                spoof_targets.append((image, tuple(value / frame_ratio for value in frame_faces[0].box)))

        # 防偽用原圖裁切（MiniFASNet 訓練時也是從原圖裁），所以用換算回原圖座標的框。
        # 逐張做被動防偽：上傳自拍只有 1 張，即時鏡頭是正面影格加每張動作影格。
        spoofs = [models.spoof.assess(image, box) for image, box in spoof_targets]
        # 任何一張被判成假臉就算假臉。
        spoof_real = all(result.is_real for result in spoofs)
        # 回報最低的真人機率：整組影格的可信度取決於最可疑的那一張。
        spoof_probability = min(result.real_probability for result in spoofs)
        # 自拍和主照片的餘弦相似度（1:1 比對），轉成 Python float。
        similarity = float(np.dot(selfie_embedding, models.embedder.embed(reference_work, reference_faces[0])))
        # 相似度達到門檻（>=）才算和主照片是同一人。
        identity = similarity >= policy.MATCH_THRESHOLD
        # 兩個分數先整理成字典，之後用 **scores 展開成 _result 的具名參數。
        scores = {
            # 所有影格中最低的防偽真人機率（每張是兩個模型的平均）→ livenessScore
            "liveness_score": _score(spoof_probability),
            # 和主照片的相似度（負數會被夾成 0）→ faceMatchScore
            "match_score": _score(similarity),
        }

        # 防偽判定為假臉：不論是否和主照片相符都拒絕；仍回報身分比對結果（identity）與兩個分數。
        if not spoof_real:
            # 拒絕：疑似假臉（翻拍的照片、螢幕等）。
            return _result("rejected", "SPOOF_SUSPECTED", identity=identity, **scores)
        # 防偽通過但和主照片不是同一人。
        if not identity:
            # 拒絕：臉和主照片不符。
            return _result("rejected", "FACE_MISMATCH", **scores)
        # 只有上傳的自拍檔、沒有即時鏡頭的動作挑戰：無法證明是活人當下拍攝，全部通過也不回 verified。
        if frames is None:
            # 回 unavailable：身分相符，但活體沒有確認（livenessVerified 維持 False），要求改用即時鏡頭。
            return _result("unavailable", "LIVE_CAPTURE_REQUIRED", identity=True, **scores)
        # 即時鏡頭的所有檢查都通過：活體與身分都確認，回 verified。
        return _result("verified", "VERIFICATION_PASSED", liveness=True, identity=True, **scores)
