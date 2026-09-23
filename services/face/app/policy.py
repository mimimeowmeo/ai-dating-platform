"""判定政策：所有門檻集中在這裡。

依 docs/ai/AI-SPEC.md，改門檻要有測試、要升 POLICY_VERSION，並經人工核准；所以門檻寫在程式裡，
不開放用環境變數調整。以下數值都還沒用台灣使用者的資料校準（FAR／FRR），只是上游的起始值。
（FAR = False Acceptance Rate，錯誤接受率：不同人被當成同一人的比例；
FRR = False Rejection Rate，錯誤拒絕率：同一人被判成不同人的比例。門檻調高會降低 FAR、提高 FRR，反之亦然。）

這個模組只有常數、沒有函式，使用的地方：
- main.py：DETECTION_SCORE_THRESHOLD（載入 YuNet 時設定）、MODEL_VERSION（/health 回傳）。
- pipeline.py：WORKING_MAX_SIDE、MIN_FACE_SIDE、MATCH_THRESHOLD，以及回應裡的 MODEL_NAME／MODEL_VERSION。
- pose.py：YAW_CHANGE_THRESHOLD、PITCH_CHANGE_THRESHOLD、MAX_ROLL_CHANGE_DEGREES、NEUTRAL_MAX_YAW、NEUTRAL_MAX_ROLL_DEGREES（判斷動作挑戰有沒有做到、正面影格是否正對鏡頭）。

判定結果（pipeline.py 依這些門檻決定）：
- 只有上傳的自拍檔（請求沒有 liveCapture），即使全部通過也只回 unavailable / LIVE_CAPTURE_REQUIRED。
- 即時鏡頭（有 liveCapture）且動作挑戰、防偽、1:1 比對全部通過，才回 verified / VERIFICATION_PASSED。
"""

# 版本 2：加入即時鏡頭的動作挑戰；挑戰通過才可能回 verified。
# 版本 3：頭部角度改在兩眼連線座標系計算（不受側傾影響）、限制側傾變化與正面影格的角度、
#         每張動作影格都做被動防偽、主照片不能用改回 unavailable（不撤銷既有的驗證狀態）。
# 判定政策的版本號；任何門檻或判定規則改變都要升這個號碼，它會被併進下面的 MODEL_VERSION。
POLICY_VERSION = "3"
# 模型組合的名稱，會放進每一個 VerifyResponse 的 modelName 欄位：
# yunet（人臉偵測）+ sface（人臉特徵與比對）+ minifasnet（被動防偽）。
MODEL_NAME = "yunet+sface+minifasnet"
# 模型檔版本：YuNet／SFace 取自 opencv_zoo commit 47534e2，MiniFASNet 取自 Silent-Face commit b6d5f04。
# f"..." 是 Python 的格式化字串（f-string），{POLICY_VERSION} 會被代入 "2"，效果類似 JS 的模板字串 `${}`。
# 結果是 "yunet-2023mar/sface-2021dec/minifasnet-b6d5f04/policy-3"，放進回應的 modelVersion 欄位，
# 讓每一筆驗證紀錄都能追查當時用的是哪一版模型與政策。
MODEL_VERSION = f"yunet-2023mar/sface-2021dec/minifasnet-b6d5f04/policy-{POLICY_VERSION}"

# YuNet 適合偵測約 10–300 px 的臉（opencv_zoo README），大張照片先縮到長邊 640 再偵測與對齊。
# 「工作影像」長邊的上限（px）：pipeline.py 用 imaging.fit_within 把影像等比例縮到長邊不超過這個值，
# 本來就比較小的影像不會放大。縮小也能減少偵測的運算量。
WORKING_MAX_SIDE = 640
# YuNet 的偵測信心門檻，沿用 opencv_zoo demo 的預設值。
# 偵測結果的信心分數（0～1）低於 0.9 的候選臉會被 YuNet 直接丟掉，不算偵測到。
DETECTION_SCORE_THRESHOLD = 0.9
# 工作影像中臉框短邊的下限。SFace 會把臉對齊放大成 112×112，太小的臉細節不足；64 是起始值。
# 單位是 px（在縮放後的工作影像上量）；臉框寬或高任一邊小於 64，pipeline 會回 FACE_TOO_SMALL
# （參照照片回 REFERENCE_FACE_TOO_SMALL、動作影格回 ACTION_FACE_TOO_SMALL）。
MIN_FACE_SIDE = 64
# SFace 餘弦相似度（cosine）的同一人門檻：OpenCV 官方教學在 LFW 上的值。
# pipeline 把兩張臉的特徵向量各自正規化成長度 1 後做內積，得到餘弦相似度；
# 大於等於 0.363 視為同一人。用在兩個地方：
#   1. 自拍（正面影格）和主照片比對，未達門檻回 FACE_MISMATCH。
#   2. 每張動作影格和正面影格比對，未達門檻回 FACE_CHANGED_DURING_CAPTURE（防止拍到一半換人）。
# （LFW = Labeled Faces in the Wild，常用的人臉辨識公開測試資料集。）
MATCH_THRESHOLD = 0.363

# 動作挑戰：拿「動作影格」和「正面影格」比較頭部角度的變化量，超過門檻才算做到動作。
# 角度用 YuNet 的 5 個點換算（見 pose.py），座標是相機拍到的原始畫面（沒有左右鏡像）：
#   yaw   = (鼻尖 x − 兩眼中點 x) ÷ 兩眼距離。使用者往自己的左邊轉頭時，鼻尖在畫面上往右移，數值變大。
#   pitch = (鼻尖 y − 兩眼中點 y) ÷ (嘴角中點 y − 兩眼中點 y)。抬頭時鼻尖相對往上，數值變小。
# （yaw 是左右轉頭的角度，pitch 是上下抬頭／低頭的角度；這裡都是沒有單位的比例值，不是真正的度數。
#  除以兩眼距離或眼到嘴的距離，是為了讓數值不受臉在畫面中的大小影響。）
# 門檻依一般臉部比例估算：yaw 0.12 約轉 15–20 度，pitch 0.08 約抬／低 10–15 度；都還沒用真實資料校準。
# 前端（apps/web/lib/head-pose.ts）用同樣的算法，但要求 1.5 倍的變化量才拍下影格，留出誤差空間。
# 挑戰可以出的動作清單（tuple，不可變的序列）。目前 Python 程式裡沒有直接引用它；
# NestJS API（apps/api/src/profiles.ts）的動作清單註明必須和這裡相同，
# provider 實際接受的動作值則由 schemas.py 的 Action 型別限制。
CHALLENGE_ACTIONS = ("turn_left", "turn_right", "look_up", "look_down")
# 左右轉頭的門檻：turn_left 要求 yaw 變化量 ≥ +0.12，turn_right 要求 ≤ −0.12（判斷邏輯在 pose.action_performed）。
YAW_CHANGE_THRESHOLD = 0.12
# 抬頭／低頭的門檻：look_up 要求 pitch 變化量 ≤ −0.08，look_down 要求 ≥ +0.08（判斷邏輯在 pose.action_performed）。
PITCH_CHANGE_THRESHOLD = 0.08
# 動作影格和正面影格相比，頭部側傾（roll，兩眼連線的角度）最多只能差幾度。
# 在畫面平面內旋轉整張照片只會改變 roll，這個上限讓「轉照片」不能冒充轉頭。
MAX_ROLL_CHANGE_DEGREES = 15.0
# 正面影格本身要大致正對鏡頭：|yaw| 的上限（前端要求 0.08，這裡放寬給兩種臉部點模型的差異）。
# 沒有這個限制時，可以先拍一張偏頭的「正面」，再用正臉照冒充反方向的動作。
NEUTRAL_MAX_YAW = 0.15
# 正面影格的側傾上限（度）。
NEUTRAL_MAX_ROLL_DEGREES = 20.0
