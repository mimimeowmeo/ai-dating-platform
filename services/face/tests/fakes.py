"""測試共用的假資料與假模型（fake）。

這個檔案不是測試本身，而是給 test_pipeline.py、test_live.py、test_api.py 共用的工具：

1. 產生「合法格式」的請求 payload（真的 JPEG／PNG／WEBP 影像再轉 base64），
   讓請求能通過 app/schemas.py 的 pydantic 驗證與 app/imaging.py 的解碼檢查。
2. 用假的模型取代真正的 YuNet（人臉偵測）、SFace（人臉特徵）、MiniFASNet（被動防偽）。

為什麼要用假模型：
- 真模型需要下載 ONNX 模型檔與安裝 OpenCV／onnxruntime，單元測試不應依賴這些大型檔案。
- 真模型對「純色影像」根本偵測不到臉，無法精準製造「剛好一張臉」「兩張臉」「臉太小」等情境。
- 假模型可以「指定」每次呼叫要回傳什麼，所以能精準測試 app/pipeline.py 的判定邏輯與邊界值。

app/pipeline.py 只透過 app/models.py 的 Models（detector／embedder／spoof 三個欄位）呼叫
detect()、embed()、assess() 三個方法；Python 是鴨子型別（duck typing），只要物件有同名方法、
回傳值形狀相同，就能直接替換，不需要繼承真正的模型類別。
"""

# base64：把影像的二進位內容編成文字，因為 HTTP JSON 請求的 imageBase64 欄位是字串。
import base64
# io：提供 BytesIO（記憶體中的假檔案），讓 PIL 可以把影像「存檔」到記憶體而不寫進硬碟。
import io

# numpy：數值陣列函式庫；YuNet 的偵測結果與 SFace 的特徵向量都是 numpy 陣列。
import numpy as np
# PIL（Pillow）：Python 的影像處理函式庫，這裡用來憑空產生一張純色的測試影像。
from PIL import Image

# Face：一張臉的偵測結果；Models：三個模型的容器；SpoofResult：防偽判定結果。
from app.models import Face, Models, SpoofResult
# FaceVerifier：真正要被測試的判定流程（pipeline），測試只替換掉它底下的模型。
from app.pipeline import FaceVerifier


def image_payload(size=(128, 128), format="JPEG"):
    """產生一張純色影像，回傳請求裡「一張影像」的欄位（imageBase64 與 mimeType）。

    參數：
        size：影像尺寸 (寬, 高)，單位是像素；注意 PIL 的順序是「寬在前」，
            和 numpy 陣列 shape 的 (高, 寬, 通道數) 相反。
            預設 128×128，大於 app/imaging.py 的最小邊長 MIN_DIMENSION（64），能通過尺寸檢查。
        format：PIL 存檔格式，只能是 "JPEG"、"PNG"、"WEBP" 三者之一（對應 schemas 允許的三種 MIME）。

    回傳：
        dict，包含
        - "imageBase64"：影像檔案內容的 base64 字串；
        - "mimeType"：和實際格式一致的 MIME 類型（例如 "image/jpeg"）。
        這個 dict 可以直接展開（**）放進 VerifyRequest、ReferenceImage 或 ActionFrame。

    可能丟出的錯誤：
        format 不是上面三種時，最後查表會丟出 KeyError（測試只會傳合法值）。

    設計理由：
        影像內容是純色、沒有真的人臉，因為偵測結果由 FakeDetector 決定；
        但影像必須是「真的、能解碼的」圖檔，才能走過 app/imaging.py 的 decode_bgr 檢查
        （base64 格式、檔案大小、實際格式和 mimeType 一致、尺寸範圍、非動畫）。
    """
    # 建立一個記憶體中的二進位緩衝區，當作影像要寫入的「檔案」。
    buffer = io.BytesIO()
    # 產生一張 RGB 純色影像（顏色 R=50、G=80、B=100，數值本身沒有意義），並用指定格式存進緩衝區。
    Image.new("RGB", size, (50, 80, 100)).save(buffer, format=format)
    # 回傳請求欄位：
    return {
        # getvalue() 取出整個影像檔的 bytes → base64 編碼成 bytes → decode("ascii") 轉成一般字串，
        # 因為 JSON 與 pydantic 的 str 欄位（strict 模式）需要字串，不接受 bytes。
        "imageBase64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        # 用 PIL 格式名稱查出對應的 MIME 類型，確保 mimeType 和實際檔案格式一致
        # （不一致時 decode_bgr 會丟出 IMAGE_TYPE_MISMATCH，那是另外的測試要故意製造的情境）。
        "mimeType": {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[format],
    }


def verify_request(references=1, size=(128, 128), actions=None):
    """組出一個合法的 /verify 請求內容（dict），格式對應 app/schemas.py 的 VerifyRequest。

    參數：
        references：要附幾張參照照片（使用者的主照片）。預設 1 張；
            傳 0 可測試「沒有主照片」的情境（pipeline 會回 REFERENCE_PHOTO_REQUIRED）。
            schemas 的上限 MAX_REFERENCE_IMAGES 是 1，所以大於 1 會在建立 VerifyRequest 時驗證失敗。
        size：自拍（或即時鏡頭的正面影格）的尺寸 (寬, 高)；參照照片與動作影格固定用預設 128×128。
        actions：動作清單（例如 ["turn_left"]）。給值時會加上即時鏡頭的 liveCapture，
            每個動作一張影格；None（預設）代表「只有上傳的自拍檔」，沒有 liveCapture。

    回傳：
        dict，可以用 VerifyRequest(**request) 轉成 pydantic 模型，
        或直接當 JSON 送給 FastAPI 的 /verify（test_api.py 的用法）。

    設計理由：
        回傳 dict 而不是直接回傳 VerifyRequest，是為了讓個別測試能在送出前「故意改壞」某個欄位
        （例如把 imageBase64 換成 "????"），測試錯誤處理。
    """
    # 先組出基本請求：
    request = {
        # 展開 image_payload 的結果，得到自拍的 imageBase64 與 mimeType 兩個欄位；
        # requestId 必須符合 schemas 的格式（英數字、底線、連字號），這裡給固定的測試值。
        **image_payload(size=size), "requestId": "test-request-1",
        # 參照照片清單：依 references 的數量，各產生一張預設尺寸的影像（_ 代表不需要用到迴圈變數）。
        "referenceImages": [image_payload() for _ in range(references)],
    }
    # 只有指定了動作清單才加上 liveCapture；用 "is not None" 而不是直接判斷真假，
    # 是為了區分「沒給（None）」和「給了空清單（[]）」兩種不同意思。
    if actions is not None:
        # liveCapture 的格式對應 schemas 的 LiveCapture：
        request["liveCapture"] = {
            # challengeId：挑戰編號（真實流程由 NestJS 從 Redis 領取），這裡給固定的測試值。
            "challengeId": "challenge-1",
            # frames：每個動作一張影格；每張影格包含 action 名稱，以及展開後的 imageBase64、mimeType。
            "frames": [{"action": action, **image_payload()} for action in actions],
        }
    # 回傳組好的請求 dict。
    return request


def face(x=10.0, y=10.0, w=100.0, h=100.0, yaw=0.0, pitch=0.5, roll=0.0):
    """假的 YuNet 偵測結果；yaw、pitch 決定 5 個臉部點的位置，算法和 app/pose.py 相反。

    兩眼在 (40, 50)、(60, 50)（距離 20），嘴角在 y=80（和兩眼中點相距 30），
    所以鼻尖放在 (50 + yaw×20, 50 + pitch×30)，pose.head_pose 就會算回同樣的 yaw、pitch。

    參數：
        x, y, w, h：臉框（bounding box）的左上角座標與寬、高，單位是像素。
            pipeline 用 min(w, h) 和 policy.MIN_FACE_SIDE（64）比較，判斷臉是否太小；
            測試用 w 或 h = MIN_FACE_SIDE - 1 製造 FACE_TOO_SMALL。
            預設 100×100，比 64 大，能通過尺寸檢查。
        yaw：想要的左右轉頭量（見 app/policy.py 的定義）。
            yaw =（鼻尖 x − 兩眼中點 x）÷ 兩眼距離；0 代表鼻尖正好在兩眼中間（正面）。
        pitch：想要的抬頭／低頭量。
            pitch =（鼻尖 y − 兩眼中點 y）÷（嘴角中點 y − 兩眼中點 y）；
            預設 0.5 代表鼻尖在眼睛和嘴巴正中間，接近正常正臉的比例。
        roll：側傾角度（度）。把 5 個臉部點以兩眼中點 (50, 50) 為圓心旋轉這個角度，
            模擬歪頭或把照片在畫面平面內轉動；head_pose 算出的 yaw、pitch 不變，roll 等於這個值。

    回傳：
        app.models.Face，其中 row 是 YuNet 格式的 15 個 float32 數值，score 是 0.95。

    設計理由：
        臉部點座標固定在 (40~60, 50~80) 附近，和 x、y、w、h 無關；因為 head_pose 只看
        點與點之間的「相對」位置，臉框放在哪裡都不影響算出的角度，測試可以分開控制兩者。
    """
    # YuNet 每張臉輸出 15 個數值，先建立一個全為 0 的 float32 陣列
    # （真正的 YuNetDetector 也是轉成 float32，保持和真實資料相同的型別與精度）。
    row = np.zeros(15, dtype=np.float32)
    # 第 0～3 個值：臉框 x、y、寬、高。
    row[:4] = (x, y, w, h)
    # 第 4～13 個值：5 個臉部點，每點 (x, y) 兩個數值，順序和 YuNet 相同：
    #   row[4:6]   右眼（相機畫面左側的那隻眼睛）= (40, 50)
    #   row[6:8]   左眼（相機畫面右側的那隻眼睛）= (60, 50)
    #   row[8:10]  鼻尖 = (50 + yaw×20, 50 + pitch×30)：
    #              兩眼中點是 (50, 50)、兩眼距離 20，所以 x 偏移 yaw×20 會讓 head_pose 算出 yaw；
    #              兩眼中點到嘴角中點的垂直距離是 80 − 50 = 30，所以 y 偏移 pitch×30 會算出 pitch。
    #   row[10:12] 右嘴角 = (42, 80)
    #   row[12:14] 左嘴角 = (58, 80)
    row[4:14] = (40, 50, 60, 50, 50 + yaw * 20, 50 + pitch * 30, 42, 80, 58, 80)
    # 有側傾時，把 5 個點以兩眼中點 (50, 50) 為圓心旋轉 roll 度（影像 y 軸向下，正角度是順時針）。
    if roll:
        # 角度換成弧度，算出 cos、sin。
        cos, sin = np.cos(np.radians(roll)), np.sin(np.radians(roll))
        # 5 個點排成 (5, 2) 的陣列，減掉圓心，變成相對座標。
        points = row[4:14].reshape(5, 2) - 50
        # 套用旋轉矩陣 [[cos, −sin], [sin, cos]]，再加回圓心，寫回 row。
        row[4:14] = (points @ np.array([[cos, sin], [-sin, cos]]) + 50).ravel()
    # 第 14 個值：偵測信心分數，0.95 高於 policy.DETECTION_SCORE_THRESHOLD（0.9），代表一張可信的臉。
    row[14] = 0.95
    # 包成 Face 物件回傳：box 用傳入的原始數值（Python float），row 是完整的 15 個值，score 同 row[14]。
    return Face(box=(x, y, w, h), row=row, score=0.95)


class FakeDetector:
    """假的人臉偵測器（取代 YuNetDetector），依呼叫順序回傳預先指定的結果。

    pipeline 呼叫 detect() 的順序固定是：
        第 1 次：自拍（或即時鏡頭的正面影格）
        第 2 次：主照片（參照照片）
        第 3 次起：即時鏡頭的每張動作影格（依 liveCapture.frames 的順序，只有即時鏡頭才會呼叫）

    建構參數：
        *results：每次呼叫要回傳的結果，依序排列。每個結果可以是：
            - list[Face]：這次偵測到的臉（空清單代表沒有臉，兩個以上代表多張臉）；
            - Exception 物件：這次呼叫改為丟出這個例外，用來模擬模型執行失敗
              （test_api.py 用它測試「模型出錯要 fail closed、不洩漏錯誤內容」）。

    屬性：
        results：還沒用掉的結果清單。
        shapes：每次呼叫時收到的影像 shape（numpy 陣列的形狀），讓測試檢查傳進來的是哪一張影像、
            有沒有先縮圖，或檢查「根本沒有呼叫偵測」（shapes 為空）。

    設計理由：
        用「依順序彈出」而不是依影像內容判斷，因為測試影像都是純色、內容一樣，
        只能靠呼叫順序區分是哪一張。呼叫次數超過準備的結果時，pop 會丟出 IndexError，
        也能順便抓出 pipeline 多呼叫了偵測的錯誤。
    """

    def __init__(self, *results):
        """保存要依序回傳的結果，並建立空的呼叫紀錄。

        參數：
            *results：見類別說明；可以一個都不給（代表預期 detect 完全不會被呼叫）。
        """
        # *results 收到的是 tuple，轉成 list 才能用 pop 逐一取出。
        self.results = list(results)
        # 記錄每次呼叫時的影像 shape，一開始是空的。
        self.shapes = []

    def detect(self, image):
        """模擬 YuNetDetector.detect：記錄影像 shape，回傳（或丟出）下一個預先指定的結果。

        參數：
            image：pipeline 傳入的 BGR 影像（numpy 陣列，shape 是 (高, 寬, 3)）。

        回傳：
            list[Face]：這次要回傳的偵測結果。

        可能丟出的錯誤：
            - 下一個結果是 Exception 時，丟出該例外；
            - 結果已經用完時，list.pop 丟出 IndexError。
        """
        # 記下這次收到的影像形狀，例如 (480, 640, 3)。
        self.shapes.append(image.shape)
        # 取出（並移除）清單最前面的結果，確保每個結果只用一次、依順序使用。
        result = self.results.pop(0)
        # 如果測試準備的是例外物件，代表這次要模擬模型出錯。
        if isinstance(result, Exception):
            # 直接丟出該例外，交給 pipeline／API 的錯誤處理。
            raise result
        # 一般情況：回傳這次的臉清單。
        return result


def _unit(similarity):
    """產生一個 2 維單位向量，使它和 [1, 0] 的內積（dot product）剛好等於 similarity。

    參數：
        similarity：想要的餘弦相似度（cosine similarity），應介於 -1 到 1。

    回傳：
        numpy 陣列 [similarity, √(1 − similarity²)]；長度的平方是
        similarity² + (1 − similarity²) = 1，所以是單位向量，和真正 SFaceEmbedder
        回傳「已正規化」向量的性質相同。

    設計理由：
        pipeline 用 np.dot(自拍向量, 另一張的向量) 當相似度；自拍向量固定是 [1, 0]，
        內積就是 similarity × 1 + √(…) × 0，精確等於 similarity。
        這樣測試「剛好等於門檻」的邊界值時不會有浮點誤差。
    """
    # 和 [1, 0] 的內積剛好等於 similarity，邊界值測試才不會有浮點誤差。
    return np.array([similarity, np.sqrt(1.0 - similarity**2)])


class FakeEmbedder:
    """依呼叫順序回傳向量：自拍（[1, 0]）→ 每張動作影格 → 主照片。

    similarity 是自拍和主照片的相似度；frame_similarities 是每張動作影格和自拍的相似度。

    取代真正的 SFaceEmbedder。pipeline 呼叫 embed() 的順序固定是：
        第 1 次：自拍（正面影格）
        接著：每張動作影格各 1 次（只有即時鏡頭、而且該影格通過偵測與動作檢查才會呼叫）
        最後：主照片（參照照片）

    建構參數：
        similarity：自拍和主照片的相似度，決定 1:1 比對結果（和 policy.MATCH_THRESHOLD 比較）。
        frame_similarities：每張動作影格和自拍的相似度，決定是否判成
            FACE_CHANGED_DURING_CAPTURE（拍攝中途換人）；預設空 tuple（沒有動作影格）。

    屬性：
        vectors：尚未回傳的向量，依呼叫順序排列。
        calls：每次呼叫收到的 (影像 shape, 臉框)，讓測試檢查 embed 用的是哪張影像與哪個框。
    """

    def __init__(self, similarity, frame_similarities=()):
        """依呼叫順序準備好所有要回傳的向量。

        參數：
            similarity：自拍和主照片的相似度。
            frame_similarities：每張動作影格和自拍的相似度（可迭代物件）。
        """
        # 向量清單的順序必須和 pipeline 呼叫 embed 的順序一致：
        #   自拍固定是 [1, 0]（當作比較基準）→ 每張動作影格的向量 → 主照片的向量（最後一個）。
        # *(... for ...) 把產生器展開成清單中的多個元素。
        self.vectors = [np.array([1.0, 0.0]), *(_unit(s) for s in frame_similarities), _unit(similarity)]
        # 呼叫紀錄一開始是空的。
        self.calls = []

    def embed(self, image, face):
        """模擬 SFaceEmbedder.embed：記錄呼叫參數，回傳下一個預先準備的向量。

        參數：
            image：pipeline 傳入的 BGR 影像（numpy 陣列）。
            face：要算特徵的那張臉（Face 物件）。

        回傳：
            numpy 陣列：下一個特徵向量（已是單位向量）。

        可能丟出的錯誤：
            向量已經用完時，list.pop 丟出 IndexError。
        """
        # 記下影像 shape 與臉框；真正的 SFace 會用臉部點在「這張影像」上對齊裁切，
        # 所以測試要確認影像和臉框來自同一張影像（座標系一致）。
        self.calls.append((image.shape, face.box))
        # 依順序取出並回傳下一個向量。
        return self.vectors.pop(0)


class FakeSpoof:
    """假的被動防偽模型（取代 MiniFASNetSpoofDetector），回傳預先指定的結果。

    建構參數：
        real_probability：真人機率（0～1），預設 0.9；pipeline 會把它四捨五入後放進 livenessScore。
        is_real：是否判定為真人，預設 True；False 時 pipeline 會回 SPOOF_SUSPECTED。
        sequence：依呼叫順序回傳的 (real_probability, is_real) 清單（正面影格、各動作影格）；
            用完之後回到 real_probability／is_real。用來測「只有某張動作影格是翻拍」。

    屬性：
        result：固定回傳的 SpoofResult。
        calls：每次呼叫收到的 (影像 shape, 臉框)，讓測試檢查防偽用的是原圖與換算回原圖的框。

    設計理由：
        real_probability 和 is_real 分開給，是因為真實模型的 is_real 由兩個模型機率相加後的
        argmax 決定，不是單純拿 real_probability 和固定門檻比；測試可以自由組合兩者。
    """

    def __init__(self, real_probability=0.9, is_real=True, sequence=()):
        """建立固定回傳的防偽結果與空的呼叫紀錄。

        參數：
            real_probability：真人機率。
            is_real：是否判定為真人。
        """
        # 事先建好要回傳的結果物件（SpoofResult 是 frozen dataclass，建立後不能修改）。
        self.result = SpoofResult(real_probability=real_probability, is_real=is_real)
        # 依順序回傳的結果，轉成 SpoofResult 清單。
        self.sequence = [SpoofResult(real_probability=p, is_real=r) for p, r in sequence]
        # 呼叫紀錄一開始是空的。
        self.calls = []

    def assess(self, image, box):
        """模擬 MiniFASNetSpoofDetector.assess：記錄呼叫參數，回傳固定結果。

        參數：
            image：pipeline 傳入的影像（應該是未縮小的原圖）。
            box：臉框 (x, y, w, h)（應該是換算回原圖座標的框）。

        回傳：
            SpoofResult：建構時指定的結果。
        """
        # 記下影像 shape 與臉框，供測試檢查。
        self.calls.append((image.shape, box))
        # 還有依順序指定的結果就取出第一個，否則回傳固定結果。
        return self.sequence.pop(0) if self.sequence else self.result


def verifier(detector=None, similarity=0.8, spoof=None, embedder=None):
    """建立一個使用假模型的 FaceVerifier（真正的判定流程），方便每個測試只替換需要的部分。

    參數：
        detector：假偵測器；沒給時預設「自拍一張臉、主照片一張臉」，都是 face() 的預設正常臉。
        similarity：沒給 embedder 時，預設 FakeEmbedder 的「自拍和主照片」相似度，
            預設 0.8，高於 policy.MATCH_THRESHOLD（0.363），代表同一人。
        spoof：假防偽模型；沒給時預設真人（real_probability 0.9、is_real True）。
        embedder：假特徵模型；有給時 similarity 參數不會被使用。

    回傳：
        FaceVerifier：模型全是假的，但判定邏輯是真的，可以直接呼叫 verify()。

    設計理由：
        預設值組合起來是「所有檢查都通過」的情境；每個測試只改一個條件，就能確認
        「單一條件不符」會得到對應的 reasonCode。
    """
    # 用 Models 容器包住三個假模型，交給 FaceVerifier；
    # Models 的型別註記寫的是真模型類別，但 Python 執行時不檢查型別，假物件只要方法相同就能用。
    return FaceVerifier(Models(
        # `a or b`：a 是 None 時使用 b。預設偵測器準備兩次結果：第 1 次（自拍）與第 2 次（主照片）各一張正常的臉。
        detector=detector or FakeDetector([face()], [face()]),
        # 預設特徵模型：自拍和主照片的相似度是 similarity，沒有動作影格。
        embedder=embedder or FakeEmbedder(similarity),
        # 預設防偽模型：判定為真人。
        spoof=spoof or FakeSpoof(),
    ))
