"""三個模型的包裝：YuNet（人臉偵測）、SFace（人臉特徵）、MiniFASNet（被動防偽）。

pipeline 只依賴這裡的介面，測試可以換成假模型，不需要真的模型檔。

各模型在真人驗證流程中的角色：
- YuNet（cv2.FaceDetectorYN）：在影像裡找出人臉，每張臉給一個框（x, y, w, h）、
  5 個臉部點（右眼、左眼、鼻尖、右嘴角、左嘴角）與一個信心分數。
- SFace（cv2.FaceRecognizerSF）：依 YuNet 的 5 個點把臉對齊裁切，再算出一個特徵向量（embedding）；
  兩張臉的向量越相近，越可能是同一個人。這裡把向量正規化成長度 1，所以兩個向量的內積（dot product）
  就等於餘弦相似度（cosine similarity）。
- MiniFASNet（Silent-Face-Anti-Spoofing）：看臉部周圍的裁切影像，判斷是真人還是翻拍的照片、螢幕等假臉
  （被動防偽，passive anti-spoofing：不需要使用者做動作）。build 時已用 torch 轉成 ONNX，
  執行期只用 onnxruntime 推論，映像不必安裝 torch。

模型檔由 tools/fetch_models.py 與 tools/export_minifasnet.py 在 Docker build 時準備好，
放在 FACE_MODEL_DIR（預設 /app/models）。
"""

# dataclass：用類別宣告欄位就自動產生建構函式等方法，類似 TypeScript 裡只放資料的 interface + 物件。
from dataclasses import dataclass
# Path：跨平台的檔案路徑物件，可以用 / 串接路徑（model_dir / name）。
from pathlib import Path

# OpenCV 的 Python 套件；YuNet 與 SFace 都是用 OpenCV 內建的 API 執行。
import cv2
# NumPy：多維陣列與數值運算，影像在 OpenCV 裡就是 NumPy 陣列（高 × 寬 × 色彩通道）。
import numpy as np
# onnxruntime：執行 ONNX 模型的推論引擎，用來跑轉好的 MiniFASNet。
import onnxruntime as ort

# crop_patch：依 Silent-Face 原始的裁切方式，以臉框為中心放大後裁切並縮放（見 crop.py）。
from .crop import crop_patch

# YuNet 模型檔名（opencv_zoo 發布的 2023 年 3 月版本）；load_models 會檢查它存在。
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
# SFace 模型檔名（opencv_zoo 發布的 2021 年 12 月版本）。
SFACE_FILE = "face_recognition_sface_2021dec.onnx"
# (檔名, 裁切倍率)：倍率來自原始權重檔名 2.7_80x80_MiniFASNetV2.pth 與 4_0_0_80x80_MiniFASNetV1SE.pth。
# 倍率代表以臉框為中心把裁切範圍放大幾倍：兩個模型分別看「臉 + 少量周圍」與「臉 + 較大範圍背景」，
# 上游 Silent-Face 的 test.py 就是把這兩個模型的結果合併判定。檔名要和 tools/export_minifasnet.py 輸出的檔名一致。
MINIFASNET_FILES = (("minifasnet_v2.onnx", 2.7), ("minifasnet_v1se.onnx", 4.0))
# MiniFASNet 的輸入尺寸：裁切後縮放成 80×80 像素（權重是用 80x80 訓練的，見權重檔名）。
MINIFASNET_INPUT = 80
# MiniFASNet 三個類別中，index 1 是真人（Silent-Face test.py：label == 1 為 real face）。
# 其他兩個類別都算假臉。
REAL_CLASS = 1


class ModelError(RuntimeError):
    """模型檔缺漏或模型輸出異常時丟出的錯誤。

    錯誤訊息是固定的代碼（例如 "MODEL_FILES_MISSING"、"EMBEDDING_FAILED"、"SPOOF_MODEL_FAILED"），
    不帶影像內容。main.py 的處理方式：
    - 啟動時 load_models 丟錯：服務照常啟動，但 /health 回 503、/verify 一律回 503（fail closed）。
    - 驗證途中丟錯：/verify 回 503 MODEL_FAILED，AI 服務會記成 PROVIDER_UNAVAILABLE。
    繼承 RuntimeError 是因為這是執行期的狀況，不是呼叫端傳錯參數。
    """

    # 類別本身不需要額外內容，pass 代表「這裡什麼都不做」（Python 的區塊不能是空的）。
    pass


# frozen=True：建立後欄位不能再修改（不可變），避免 pipeline 的某一步不小心改到偵測結果。
@dataclass(frozen=True)
class Face:
    """YuNet 偵測到的一張臉。

    欄位：
        box：臉框 (x, y, w, h)，x、y 是左上角，w、h 是寬高；座標屬於「偵測時那張影像」
            （pipeline 會先縮小影像再偵測，所以要換算回原圖時得除以縮放比例）。
        row：YuNet 對這張臉輸出的原始 15 個值（float32），順序是
            0–3 臉框 x, y, w, h；4–5 右眼；6–7 左眼；8–9 鼻尖；10–11 右嘴角；12–13 左嘴角；14 信心分數。
            SFace 對齊（alignCrop）要用前 14 個；pose.py 用 4–13 的 5 個點算頭部角度。
        score：YuNet 的信心分數（0–1），就是 row[14]。
    """

    box: tuple[float, float, float, float]  # x, y, w, h（偵測時那張影像的座標）
    row: np.ndarray  # YuNet 原始的 15 個值（float32），SFace 對齊要用前 14 個
    # 偵測信心分數，已經過 score_threshold 過濾（低於門檻的臉 YuNet 不會回傳）
    score: float


# 同樣設成不可變：防偽結果算出來之後不應再被修改。
@dataclass(frozen=True)
class SpoofResult:
    """MiniFASNet 被動防偽的結果。

    欄位：
        real_probability：0–1，兩個模型各自算出的「真人」機率取平均；會回報成 livenessScore。
        is_real：是否判定為真人。沿用 Silent-Face 原始程式的做法：兩個模型的三類機率各自相加，
            加總後最大的類別是真人（index 1）才算真人。這和「real_probability > 0.5」不完全相同，
            因為要和另外兩個假臉類別分別比大小。
    """

    real_probability: float  # 0–1，兩個模型真人機率的平均
    is_real: bool  # 沿用原始程式的判定：兩個模型機率相加後 argmax 為真人


class YuNetDetector:
    """用 OpenCV 的 FaceDetectorYN 執行 YuNet 人臉偵測。

    注意：detect 會呼叫 setInputSize 改變內部狀態，所以同一個實例不能被多個執行緒同時使用；
    pipeline.FaceVerifier 用一把鎖（lock）保證一次只有一個請求在用模型。
    """

    def __init__(self, path: Path, score_threshold: float):
        """載入 YuNet 模型。

        參數：
            path：YuNet 的 ONNX 模型檔路徑。
            score_threshold：偵測信心門檻，低於這個分數的臉不會回傳（由 policy.DETECTION_SCORE_THRESHOLD 傳入）。

        可能丟出：模型檔無法讀取或格式錯誤時，OpenCV 會丟出 cv2.error。
        """
        # input_size 在 C++ 宣告裡沒有預設值，先給一個，detect 前會依實際影像重設。
        # create 的參數依序是：模型路徑（OpenCV 要字串，所以 str(path)）、設定檔（ONNX 不需要，給空字串）、
        # 輸入尺寸 (320, 320)（暫定值）、信心門檻、NMS 門檻 0.3（重疊度超過 0.3 的框只留分數最高的，
        # 避免同一張臉被框兩次）、top_k 5000（NMS 前最多保留的候選框數量）。
        self._detector = cv2.FaceDetectorYN.create(str(path), "", (320, 320), score_threshold, 0.3, 5000)

    def detect(self, image: np.ndarray) -> list[Face]:
        """偵測影像中的所有人臉。

        參數：
            image：BGR 色彩順序的 uint8 影像陣列，形狀是 (高, 寬, 3)。

        回傳：
            Face 的清單；沒有偵測到臉時回傳空清單。pipeline 會再檢查是否剛好一張、是否夠大。
        """
        # image.shape 是 (高, 寬, 通道數)，[:2] 只取前兩個：高與寬。
        height, width = image.shape[:2]
        # 尺寸不一致時 detect 會直接丟錯，每張圖都要重設。
        # 注意 OpenCV 的尺寸參數順序是 (寬, 高)，和 NumPy 的 shape (高, 寬) 相反。
        self._detector.setInputSize((width, height))
        # detect 回傳 (retval, faces)；retval 用不到，用 _ 表示忽略。
        # faces 是形狀 (臉的數量, 15) 的陣列，每一列是一張臉（欄位見 Face 的說明）。
        _, faces = self._detector.detect(image)
        # 沒有臉時 OpenCV 回傳 None，而不是空陣列，所以要自己轉成空清單，呼叫端才能統一處理。
        if faces is None:  # 沒有臉時回傳 None，不是空陣列
            return []
        # 串列生成式（list comprehension）：對 faces 的每一列建立一個 Face，類似 JS 的 faces.map(...)。
        return [
            # box：取前 4 個值（x, y, w, h）轉成 Python float 的 tuple，方便之後做一般的數學運算；
            # row：保留完整 15 個值並確保是 float32（SFace alignCrop 需要 float32）；
            # score：第 15 個值（index 14）是信心分數。
            Face(box=tuple(float(v) for v in row[:4]), row=np.asarray(row, dtype=np.float32), score=float(row[14]))
            # 逐列走訪偵測結果
            for row in faces
        ]


class SFaceEmbedder:
    """用 OpenCV 的 FaceRecognizerSF 執行 SFace，把一張臉轉成長度為 1 的特徵向量。

    兩張臉的特徵向量做內積就是餘弦相似度，pipeline 拿它和 policy.MATCH_THRESHOLD 比較，
    判斷是不是同一個人。
    """

    def __init__(self, path: Path):
        """載入 SFace 模型。

        參數：
            path：SFace 的 ONNX 模型檔路徑。

        可能丟出：模型檔無法讀取或格式錯誤時，OpenCV 會丟出 cv2.error。
        """
        # create 的參數：模型路徑（轉成字串）、設定檔（ONNX 不需要，給空字串）。
        self._recognizer = cv2.FaceRecognizerSF.create(str(path), "")

    def embed(self, image: np.ndarray, face: Face) -> np.ndarray:
        """算出某張臉的正規化特徵向量。

        參數：
            image：偵測這張臉時用的那張 BGR 影像（臉框與臉部點的座標要對得上這張影像）。
            face：YuNet 偵測到的臉。

        回傳：
            float32 的一維向量，長度（L2 norm）為 1。

        可能丟出：
            ModelError("EMBEDDING_FAILED")：模型輸出的向量長度是 0 或不是有限數值（NaN／無限大），
            這種向量無法正規化，也無法拿來比對，所以直接視為模型失敗（fail closed）。
        """
        # alignCrop 在 release build 用 at<float> 讀值，傳 float64 不會報錯但會算錯，一定要 float32。
        # face.row[:-1] 去掉最後一個值（信心分數），留下前 14 個（臉框 + 5 個臉部點）；
        # ascontiguousarray 確保記憶體是連續排列，OpenCV 的 C++ 端才能正確讀取。
        landmarks = np.ascontiguousarray(face.row[:-1], dtype=np.float32)
        # 依 5 個臉部點把臉旋轉、縮放、裁切成 SFace 需要的標準姿勢，讓不同照片的臉可以比較。
        aligned = self._recognizer.alignCrop(image, landmarks)
        # feature 算出特徵向量；轉成 float32 的 NumPy 陣列，再用 reshape(-1) 攤平成一維。
        feature = np.asarray(self._recognizer.feature(aligned), dtype=np.float32).reshape(-1)
        # 算向量長度（L2 norm，所有元素平方和再開根號），轉成 Python float。
        norm = float(np.linalg.norm(feature))
        # 長度不是有限數值或剛好是 0 時無法正規化（會除以 0 或得到 NaN），視為模型失敗。
        if not np.isfinite(norm) or norm == 0.0:
            # 丟出固定代碼的錯誤；main.py 會回 503，不會產生錯誤的比對結果。
            raise ModelError("EMBEDDING_FAILED")
        # 自己正規化再做內積，不用 FaceRecognizerSF.match()：已發布的版本會就地改寫傳入的陣列。
        # 除以長度後向量長度為 1，之後兩個向量的內積就等於餘弦相似度（-1 到 1）。
        return feature / norm


class MiniFASNetSpoofDetector:
    """用兩個 MiniFASNet（V2 與 V1SE）做被動防偽，結果合併成一個 SpoofResult。

    合併方式照 Silent-Face 原始的 test.py：每個模型各自以自己的倍率裁切、推論、算 softmax 機率，
    再把兩個模型的機率相加。
    """

    def __init__(self, model_dir: Path):
        """載入兩個 MiniFASNet 的 ONNX 模型。

        參數：
            model_dir：放模型檔的資料夾，檔名見 MINIFASNET_FILES。

        可能丟出：模型檔不存在或無法載入時，onnxruntime 會丟出錯誤（load_models 會先檢查檔案存在）。
        """
        # onnxruntime 的 session 設定物件。
        options = ort.SessionOptions()
        # log 等級 3 = 只輸出 error（0 verbose、1 info、2 warning、3 error、4 fatal），避免啟動時印出大量警告。
        options.log_severity_level = 3
        # 存放 (session, 輸入名稱, 裁切倍率) 的清單，assess 時依序使用。
        self._models = []
        # 依序載入兩個模型，同時取出各自的裁切倍率。
        for name, scale in MINIFASNET_FILES:
            # 建立推論 session；明確指定只用 CPU（CPUExecutionProvider），服務不依賴 GPU。
            session = ort.InferenceSession(str(model_dir / name), options, providers=["CPUExecutionProvider"])
            # 記下模型第一個（也是唯一一個）輸入的名稱，推論時要用這個名稱餵資料。
            self._models.append((session, session.get_inputs()[0].name, scale))

    def assess(self, image: np.ndarray, box) -> SpoofResult:
        """判斷影像中這張臉是真人還是假臉。

        參數：
            image：原始大小的 BGR 影像（MiniFASNet 訓練時是從原圖裁切，所以不用縮小後的工作影像）。
            box：臉框 (x, y, w, h)，必須是 image 這張影像的座標。

        回傳：
            SpoofResult：兩個模型真人機率的平均，以及是否判定為真人。

        可能丟出：
            ModelError("SPOOF_MODEL_FAILED")：機率裡出現 NaN 或無限大，代表模型輸出異常，視為失敗。
        """
        # 三個類別的機率加總，初始為 [0, 0, 0]；用 float64 減少加總時的精度誤差。
        total = np.zeros(3, dtype=np.float64)
        # 對每個模型：用它自己的倍率裁切、推論、轉成機率，再累加。
        for session, input_name, scale in self._models:
            # 以臉框中心放大 scale 倍裁切，再縮放成 80×80（和訓練時的前處理相同）。
            patch = crop_patch(image, box, scale, MINIFASNET_INPUT, MINIFASNET_INPUT)
            # 原始程式的 to_tensor 只做 HWC→CHW，不除以 255、不減平均值。
            # 轉成 float32；transpose(2, 0, 1) 把 (高, 寬, 通道) 換成 (通道, 高, 寬)，這是 PyTorch 模型要的順序；
            # [np.newaxis] 在最前面加一個 batch 維度，形狀變成 (1, 3, 80, 80)。
            tensor = patch.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
            # run(None, ...)：None 表示取回所有輸出；[0] 是第一個輸出（logits，形狀 (1, 3)），
            # 再 [0] 取出 batch 裡唯一一筆，得到 3 個類別的原始分數，轉成 float64 再計算。
            logits = session.run(None, {input_name: tensor})[0][0].astype(np.float64)
            # softmax 的第一步：先減掉最大值再取指數，數學結果不變，但可以避免數值太大溢位（overflow）。
            exp = np.exp(logits - logits.max())
            # softmax 的第二步：除以總和，得到三個加總為 1 的機率，累加到 total。
            total += exp / exp.sum()
        # 任何一個值是 NaN 或無限大，代表模型輸出異常；不能拿來判定，直接視為失敗（fail closed）。
        if not np.all(np.isfinite(total)):
            # 丟出固定代碼的錯誤；main.py 會回 503。
            raise ModelError("SPOOF_MODEL_FAILED")
        # 組出防偽結果。
        return SpoofResult(
            # 真人類別的機率加總除以模型數量（2），得到平均的真人機率（0–1）。
            real_probability=float(total[REAL_CLASS] / len(self._models)),
            # argmax 找出加總後機率最大的類別；是真人類別才判定為真人（和 Silent-Face 原始程式相同）。
            is_real=int(np.argmax(total)) == REAL_CLASS,
        )


# 不可變：模型組合建立後不應被替換。
@dataclass(frozen=True)
class Models:
    """驗證流程需要的三個模型，打包成一個物件傳給 pipeline.FaceVerifier。

    pipeline 只呼叫 detector.detect、embedder.embed、spoof.assess 這三個方法，
    所以測試（tests/fakes.py）可以放入同樣介面的假物件，不需要真的模型檔。
    """

    # YuNet 人臉偵測
    detector: YuNetDetector
    # SFace 人臉特徵
    embedder: SFaceEmbedder
    # MiniFASNet 被動防偽
    spoof: MiniFASNetSpoofDetector


def load_models(model_dir: Path, detection_score_threshold: float) -> Models:
    """從資料夾載入所有模型。

    參數：
        model_dir：放模型檔的資料夾（設定 FACE_MODEL_DIR，預設 /app/models）。
        detection_score_threshold：YuNet 的偵測信心門檻（policy.DETECTION_SCORE_THRESHOLD）。

    回傳：
        Models：載入好的三個模型。

    可能丟出：
        ModelError("MODEL_FILES_MISSING")：任何一個模型檔不存在。先檢查再載入，錯誤代碼比較明確；
            錯誤不列出缺了哪些檔名。
        其他 OpenCV／onnxruntime 的錯誤：模型檔存在但無法載入。
        main.py 捕捉所有錯誤後讓服務以「模型未載入」的狀態啟動（/health 回 503、/verify fail closed）。
    """
    # 需要的所有檔名：YuNet、SFace，再加上兩個 MiniFASNet 的檔名（* 把產生器展開放進清單）。
    required = [YUNET_FILE, SFACE_FILE, *(name for name, _ in MINIFASNET_FILES)]
    # 找出資料夾中不存在（或不是一般檔案）的檔名。
    missing = [name for name in required if not (model_dir / name).is_file()]
    # 只要有缺，就不載入任何模型。
    if missing:
        # 丟出固定代碼的錯誤。
        raise ModelError("MODEL_FILES_MISSING")
    # 所有檔案都在：逐一建立三個模型並打包回傳。
    return Models(
        # YuNet：模型路徑 + 偵測信心門檻
        detector=YuNetDetector(model_dir / YUNET_FILE, detection_score_threshold),
        # SFace：模型路徑
        embedder=SFaceEmbedder(model_dir / SFACE_FILE),
        # MiniFASNet：傳資料夾，兩個模型檔由類別自己依 MINIFASNET_FILES 載入
        spoof=MiniFASNetSpoofDetector(model_dir),
    )
