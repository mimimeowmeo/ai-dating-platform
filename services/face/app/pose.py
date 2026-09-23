"""用 YuNet 的 5 個臉部點估算頭部轉動，判斷動作挑戰有沒有做到。

背景：
    即時鏡頭驗證時，前端會要求使用者做指定動作（turn_left／turn_right／look_up／look_down），
    每個動作拍一張「動作影格」。pipeline.py 會拿每張動作影格的頭部角度，
    和「正面影格」的頭部角度相比，變化量超過 policy.py 的門檻才算做到動作。
    這可以擋掉「拿一張靜態照片對著鏡頭」的攻擊：靜態照片無法依指示轉頭。

算法（和前端 apps/web/lib/head-pose.ts 相同，兩邊的數字才能互相比較）：
    不是真正的 3D 角度，而是用 5 個 2D 臉部點算出的「相對比例」，不受臉在畫面中大小影響。
    先把座標轉到「兩眼連線」座標系，再算比例，所以頭部側傾（roll）或整張照片在畫面平面內旋轉，都不會改變 yaw／pitch：
      u（橫軸）= 兩眼連線方向的單位向量，固定指向畫面右側（x 分量 ≥ 0）
      n（縱軸）= 和 u 垂直、指向畫面下方的單位向量
      d        = 鼻尖 − 兩眼中點
      yaw（左右轉頭）  = (d · u) ÷ 兩眼距離
      pitch（抬頭低頭）= (d · n) ÷ ((嘴角中點 − 兩眼中點) · n)
      roll（側傾）     = 兩眼連線和水平線的夾角（度）
    座標是相機拍到的原始畫面（沒有左右鏡像），影像的 y 軸向下為正。
    使用者往自己的左邊轉頭時鼻尖在畫面上往右移，yaw 變大；抬頭時鼻尖相對往上，pitch 變小。
    頭沒有側傾時（兩眼連線水平），u = (1, 0)、n = (0, 1)，結果和舊版只看 x、y 的公式完全相同。

限制：
    這只能擋住「拿照片或螢幕對著鏡頭」的呈現攻擊（presentation attack）。
    伺服器無法證明影格是鏡頭當下拍的；直接呼叫 API 送出事先準備好的影格（注入攻擊，injection attack）擋不住。
"""

# math：標準函式庫的數學工具，這裡用 math.hypot 算兩點間的直線距離。
import math
# dataclass：自動產生 __init__ 等方法的裝飾器，用來定義只裝資料的小類別（類似 TypeScript 的 type／interface 加上建構子）。
from dataclasses import dataclass

# policy：判定政策模組，所有門檻（YAW_CHANGE_THRESHOLD、PITCH_CHANGE_THRESHOLD）集中在那裡。
# 前面的「.」代表從同一個套件（services/face/app）匯入。
from . import policy
# Face：models.py 定義的偵測結果，裡面的 row 是 YuNet 原始輸出的 15 個數值。
from .models import Face


# frozen=True：建立後欄位不能再修改（不可變），避免計算結果在傳遞過程中被意外改寫。
@dataclass(frozen=True)
class HeadPose:
    """一張影像中頭部角度的估計值（兩個無單位的比例，不是角度的度數）。

    欄位：
        yaw: 左右轉頭的程度。正面時接近 0；使用者往自己的左邊轉頭，數值變大；往右轉，數值變小。
        pitch: 抬頭低頭的程度。鼻尖到兩眼中點的垂直距離佔「眼到嘴」距離的比例；抬頭變小、低頭變大。
        roll: 側傾角度（度）。只拿來檢查「有沒有歪頭或轉照片」，不參與動作判斷。

    只有和同一個人的正面影格相減（看變化量）才有意義，單獨一個值不代表絕對角度，
    因為每個人的五官比例不同。
    """

    # 左右轉頭的比例值（float）。
    yaw: float
    # 抬頭低頭的比例值（float）。
    pitch: float
    # 側傾角度（度）：兩眼連線和水平線的夾角，範圍 −90～90；往畫面右下傾斜為正。
    # 預設 0（沒有側傾），只給 yaw、pitch 建立時就是正的頭。
    roll: float = 0.0


def head_pose(face: Face) -> HeadPose | None:
    """用 YuNet 的 5 個臉部點算出 yaw、pitch 與 roll。

    參數：
        face: YuNet 的偵測結果。face.row 是 15 個 float32，依 OpenCV FaceDetectorYN 文件的順序：
            [0–3] 臉框 x, y, w, h；[4–5] 右眼 x, y；[6–7] 左眼 x, y；[8–9] 鼻尖 x, y；
            [10–11] 右嘴角 x, y；[12–13] 左嘴角 x, y；[14] 偵測信心分數。

    回傳：
        HeadPose；若臉部點的位置異常（兩眼幾乎重疊，或嘴角中點沿縱軸沒有低於兩眼中點至少 1 px），
        無法可靠計算，回傳 None。呼叫端（pipeline.py）收到 None 會判定 CHALLENGE_FAILED。

    設計理由：
        除以兩眼距離與眼到嘴的距離做「正規化」，讓結果和臉在畫面中的大小、與鏡頭的距離無關。
        先轉到兩眼連線座標系，頭部側傾或照片在畫面平面內旋轉都不會改變 yaw／pitch，
        否則把一張靜態照片轉約 20° 就能讓 yaw 變化超過門檻。橫軸固定指向畫面右側，左右眼互換也不影響結果。
    """
    # 取出 YuNet 對這張臉的原始 15 個輸出值（numpy 陣列）。
    row = face.row
    # 依索引取出右眼、左眼、鼻尖的 (x, y) 座標，各組成一個 tuple。
    right_eye, left_eye, nose = (row[4], row[5]), (row[6], row[7]), (row[8], row[9])
    # 取出右嘴角與左嘴角的 (x, y) 座標。
    mouth_right, mouth_left = (row[10], row[11]), (row[12], row[13])
    # 兩眼中點 = 兩眼 x 的平均、y 的平均；作為 yaw 與 pitch 的共同基準點。
    eye_mid = ((right_eye[0] + left_eye[0]) / 2, (right_eye[1] + left_eye[1]) / 2)
    # 兩個嘴角的中點。
    mouth_mid = ((mouth_right[0] + mouth_left[0]) / 2, (mouth_right[1] + mouth_left[1]) / 2)
    # 兩眼連線的向量（右眼 → 左眼）。
    axis_x, axis_y = left_eye[0] - right_eye[0], left_eye[1] - right_eye[1]
    # 兩眼之間的直線距離（像素）：math.hypot(dx, dy) = √(dx² + dy²)。作為 yaw 的分母。
    eye_distance = math.hypot(axis_x, axis_y)
    # 兩眼幾乎重疊：無法定出座標軸，直接放棄。
    if eye_distance < 1:
        # 無法估算頭部角度。
        return None
    # 讓橫軸固定指向畫面右側：不管 YuNet 把哪一隻眼標成「右眼」，結果都一樣（和舊公式一樣不受左右眼互換影響）。
    if axis_x < 0:
        # 反轉向量方向。
        axis_x, axis_y = -axis_x, -axis_y
    # 橫軸單位向量 u。
    ux, uy = axis_x / eye_distance, axis_y / eye_distance
    # 縱軸單位向量 n：u 逆時針轉 90°（影像 y 軸向下，所以 u = (1, 0) 時 n = (0, 1)，指向畫面下方）。
    nx, ny = -uy, ux
    # 鼻尖相對兩眼中點的向量 d。
    dx, dy = nose[0] - eye_mid[0], nose[1] - eye_mid[1]
    # 嘴角中點到兩眼中點的距離，投影到縱軸上；正常的臉嘴在眼睛下方，這個值為正。作為 pitch 的分母。
    eye_to_mouth = (mouth_mid[0] - eye_mid[0]) * nx + (mouth_mid[1] - eye_mid[1]) * ny
    # 分母小於 1 px：臉部點重疊、臉倒過來或偵測異常。除以這麼小的數會讓比例暴衝（或除以零），所以直接放棄。
    if eye_to_mouth < 1:
        # 無法估算頭部角度。
        return None
    # 建立並回傳 HeadPose。float(...) 把 numpy 的數值型別轉成 Python 原生 float，方便後續比較與序列化。
    return HeadPose(
        # yaw：d 在橫軸上的分量，除以兩眼距離做正規化。
        yaw=float((dx * ux + dy * uy) / eye_distance),
        # pitch：d 在縱軸上的分量，除以眼到嘴的距離做正規化。
        pitch=float((dx * nx + dy * ny) / eye_to_mouth),
        # roll：橫軸和水平線的夾角，atan2 回傳弧度，換成度。
        roll=float(math.degrees(math.atan2(uy, ux))),
    )


def facing_camera(pose: HeadPose) -> bool:
    """正面影格是否大致正對鏡頭（沒有明顯左右轉頭或歪頭）。

    參數：
        pose: 正面影格的頭部角度。

    回傳：
        |yaw| 與 |roll| 都在 policy 的上限內時回傳 True。

    設計理由：
        動作只比較「和正面影格的變化量」。正面影格如果已經偏向一側，
        拿一張正臉照就能冒充反方向的動作，所以正面影格本身也要檢查。
        pitch 的正常範圍因人而異、還沒校準，暫時不檢查。
    """
    # 左右轉頭與側傾都要在上限內。
    return abs(pose.yaw) <= policy.NEUTRAL_MAX_YAW and abs(pose.roll) <= policy.NEUTRAL_MAX_ROLL_DEGREES


def action_performed(action: str, neutral: HeadPose, pose: HeadPose) -> bool:
    """判斷某張動作影格是否做到指定動作。

    參數：
        action: 動作名稱，"turn_left"、"turn_right"、"look_up" 或 "look_down"（和 policy.CHALLENGE_ACTIONS 相同）。
        neutral: 正面影格（有即時鏡頭時 imageBase64 就是正面影格）的頭部角度，當作比較基準。
        pose: 這張動作影格的頭部角度。

    回傳：
        側傾變化在 MAX_ROLL_CHANGE_DEGREES 以內、變化量朝正確方向、且大小達到 policy 門檻時回傳 True；否則 False。
        不認得的動作名稱一律回傳 False（fail closed：有疑問就不通過）。

    設計理由：
        比較「變化量」而不是絕對值，抵銷每個人五官比例與拍攝角度的差異。
        門檻（yaw 0.12、pitch 0.08）集中在 policy.py，調整要升 POLICY_VERSION。
    """
    # 側傾變化太大：可能是在畫面平面內轉動照片，而不是轉頭，一律不算做到。
    if abs(pose.roll - neutral.roll) > policy.MAX_ROLL_CHANGE_DEGREES:
        # 沒做到。
        return False
    # 左右轉頭的變化量：動作影格減正面影格。正值代表鼻尖在畫面上往右移（使用者往自己的左邊轉）。
    yaw_change = pose.yaw - neutral.yaw
    # 抬頭低頭的變化量：負值代表鼻尖相對往上（抬頭），正值代表往下（低頭）。
    pitch_change = pose.pitch - neutral.pitch
    # 往自己的左邊轉頭。
    if action == "turn_left":
        # yaw 至少要增加門檻值。
        return yaw_change >= policy.YAW_CHANGE_THRESHOLD
    # 往自己的右邊轉頭。
    if action == "turn_right":
        # yaw 至少要減少門檻值（變化量 ≤ 負的門檻）。
        return yaw_change <= -policy.YAW_CHANGE_THRESHOLD
    # 抬頭。
    if action == "look_up":
        # pitch 至少要減少門檻值。
        return pitch_change <= -policy.PITCH_CHANGE_THRESHOLD
    # 低頭。
    if action == "look_down":
        # pitch 至少要增加門檻值。
        return pitch_change >= policy.PITCH_CHANGE_THRESHOLD
    # 未知的動作：保守地判定為沒做到。實際上 schemas.py 的 Action 型別已限制只能是上面四種，這裡是額外保險。
    return False
