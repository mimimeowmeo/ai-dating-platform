"""即時鏡頭（liveCapture，動作挑戰）的單元測試。

測試範圍：
- PoseTests：app/pose.py 用 YuNet 的 5 個臉部點算頭部角度（yaw 左右轉、pitch 抬低頭），
  以及判斷某個動作有沒有做到（和正面影格相比的變化量是否超過 policy 的門檻）。
- LiveDecisionTests：app/pipeline.py 在有 liveCapture 時的判定：
  每張動作影格都要剛好一張夠大的臉、做到指定動作、和正面影格是同一人，
  之後仍要通過被動防偽與大頭貼比對，全部通過才回 verified／VERIFICATION_PASSED。

為什麼這些規則重要：上傳的照片無法證明是活人當下拍的；要求使用者即時做出隨機指定的轉頭／抬頭動作，
並確認整段拍攝都是同一個人，才能降低拿照片或別人代拍冒用的風險。

所有模型都用 tests/fakes.py 的假模型；頭部角度由 fakes.face(yaw=..., pitch=...) 精準控制。
執行方式（在 services/face 目錄下）：python -m unittest tests.test_live
"""

# unittest：Python 內建的測試框架。
import unittest

# policy：判定門檻（YAW_CHANGE_THRESHOLD、PITCH_CHANGE_THRESHOLD、MIN_FACE_SIDE 等）。
from app import policy
# InvalidImage：影像解碼失敗時丟出的例外，錯誤碼就是例外訊息。
from app.imaging import InvalidImage
# HeadPose：頭部角度（yaw、pitch）；action_performed：判斷動作是否做到；head_pose：從臉部點算出角度。
from app.pose import HeadPose, action_performed, head_pose
# VerifyRequest：/verify 請求的 pydantic 模型。
from app.schemas import VerifyRequest

# 共用的假模型與假資料工具，說明見 tests/fakes.py。
from tests.fakes import FakeDetector, FakeEmbedder, FakeSpoof, face, verifier, verify_request

# 左右轉頭的門檻（yaw 變化量，目前是 0.12），取短名稱讓下面的測試比較好讀。
YAW = policy.YAW_CHANGE_THRESHOLD
# 抬頭／低頭的門檻（pitch 變化量，目前是 0.08）。
PITCH = policy.PITCH_CHANGE_THRESHOLD


def run_live(actions, frame_faces, frame_similarities=None, similarity=0.8, spoof=None):
    """跑一次即時鏡頭驗證：正面影格的臉是 face()，每張動作影格的偵測結果由 frame_faces 指定。

    主照片的臉也是 face()（正常、夠大的一張臉）。

    參數：
        actions：動作清單，例如 ["turn_left", "look_up"]；每個動作會產生一張動作影格。
        frame_faces：每張動作影格的偵測結果（list[list[Face]]），順序和 actions 對應。
        frame_similarities：每張動作影格和正面影格的相似度；沒給時每張都是 0.9（同一人）。
        similarity：正面影格和主照片的相似度，預設 0.8（高於門檻，同一人）。
        spoof：假防偽模型；None 時由 verifier() 使用預設的「真人」FakeSpoof。

    回傳：
        VerifyResponse：pipeline 的判定結果。
    """
    # 偵測器依呼叫順序回傳：第 1 次正面影格一張正常臉、第 2 次主照片一張正常臉，
    # 之後每張動作影格依序回傳 frame_faces 的元素（*frame_faces 把清單展開成多個參數）。
    detector = FakeDetector([face()], [face()], *frame_faces)
    # 沒指定動作影格相似度時，預設每張都是 0.9（明顯高於門檻，代表同一人）。
    frame_similarities = frame_similarities or [0.9] * len(frame_faces)
    # 特徵模型依呼叫順序回傳：正面影格 → 每張動作影格 → 主照片（見 FakeEmbedder 說明）。
    embedder = FakeEmbedder(similarity, frame_similarities)
    # 組出帶有 liveCapture 的請求，並用 pydantic 驗證、轉成請求物件。
    request = VerifyRequest(**verify_request(actions=actions))
    # 用假模型跑完整流程並回傳結果。
    return verifier(detector=detector, embedder=embedder, spoof=spoof).verify(request)


class PoseTests(unittest.TestCase):
    """app/pose.py 的測試：頭部角度的計算，以及動作是否做到的判斷。"""

    def test_head_pose_reads_yunet_landmarks(self):
        """規則：head_pose 從 YuNet 的 5 個臉部點算回正確的 yaw、pitch。

        為什麼重要：動作挑戰完全依賴這個角度；fakes.face() 用和 pose.py 相反的算法放置臉部點，
        這裡確認兩邊的定義一致（也間接確認臉部點在 YuNet 輸出中的位置沒讀錯）。
        """
        # 建一張 yaw=0.2、pitch=0.4 的假臉，再用 head_pose 算回角度。
        pose = head_pose(face(yaw=0.2, pitch=0.4))
        # yaw 要算回 0.2；臉部點存成 float32，精度有限，所以比到小數第 5 位即可。
        self.assertAlmostEqual(pose.yaw, 0.2, places=5)
        # pitch 要算回 0.4，同樣比到小數第 5 位。
        self.assertAlmostEqual(pose.pitch, 0.4, places=5)

    def test_in_plane_rotation_does_not_change_pose(self):
        """規則：臉部點整體在畫面平面內旋轉（側傾）時，yaw、pitch 不變，roll 等於旋轉角度。

        為什麼重要：yaw／pitch 要只反映「轉頭、抬頭」，不能被歪頭或轉照片影響，否則靜態照片轉一下就能冒充轉頭。
        """
        # 同一張 yaw=0.2、pitch=0.4 的假臉，旋轉 25°。
        pose = head_pose(face(yaw=0.2, pitch=0.4, roll=25))
        # yaw、pitch 和沒旋轉時相同（float32 精度，比到小數第 4 位）。
        self.assertAlmostEqual(pose.yaw, 0.2, places=4)
        self.assertAlmostEqual(pose.pitch, 0.4, places=4)
        # roll 等於旋轉角度。
        self.assertAlmostEqual(pose.roll, 25, places=3)

    def test_roll_change_limits_actions(self):
        """規則：動作影格和正面影格的側傾差超過 MAX_ROLL_CHANGE_DEGREES 時，一律不算做到動作。"""
        # 正面：沒有側傾。
        neutral = HeadPose(yaw=0.0, pitch=0.5, roll=0.0)
        # 轉頭量足夠、側傾差剛好在上限內：算做到。
        self.assertTrue(action_performed("turn_left", neutral, HeadPose(YAW * 1.5, 0.5, policy.MAX_ROLL_CHANGE_DEGREES)))
        # 側傾差超過上限：不算做到。
        self.assertFalse(action_performed("turn_left", neutral, HeadPose(YAW * 1.5, 0.5, policy.MAX_ROLL_CHANGE_DEGREES + 1)))

    def test_degenerate_landmarks_have_no_pose(self):
        """規則：臉部點退化（例如全部擠在同一點）時，head_pose 回傳 None，而不是算出奇怪的數字。

        為什麼重要：兩眼距離或眼到嘴的距離接近 0 時，除法會得到極大值或除以 0；
        回傳 None 讓 pipeline 判成 CHALLENGE_FAILED，而不是誤判動作做到了。
        """
        # 先建一張正常的假臉。
        flat = face()
        # 把 5 個臉部點（第 4～13 個值）全部設成 0：兩眼距離變成 0。
        # （Face 是 frozen dataclass，不能換掉 row 欄位本身，但 numpy 陣列的內容仍可以修改。）
        flat.row[4:14] = 0
        # 退化的臉部點必須得到 None。
        self.assertIsNone(head_pose(flat))

    def test_action_directions(self):
        """規則：四種動作各自對應正確的角度變化方向。

        - turn_left（往自己的左邊轉）：yaw 變大；
        - turn_right（往自己的右邊轉）：yaw 變小；
        - look_up（抬頭）：pitch 變小；
        - look_down（低頭）：pitch 變大。
        而且反方向或完全沒動都不算做到。

        為什麼重要：方向寫反的話，使用者照指示做會失敗、做反方向反而通過。
        """
        # 原始（沒有鏡像）的相機畫面：往自己左邊轉，鼻尖往畫面右邊移 → yaw 變大；抬頭 → pitch 變小。
        # 用門檻的 1.01 倍：剛好等於門檻的浮點數相減會有誤差（0.58 − 0.5 = 0.0799…），沒有意義。
        # 正面影格的角度：正面、pitch 0.5（fakes.face() 的預設值）。
        neutral = HeadPose(yaw=0.0, pitch=0.5)
        # 比門檻多 1% 的變化量，確保「應該通過」的情況確實超過門檻。
        yaw, pitch = YAW * 1.01, PITCH * 1.01
        # 動作 → (做到該動作的角度, 反方向的角度)：
        cases = {
            # 往左轉：yaw 往正方向變化才算做到；往負方向是反方向。
            "turn_left": (HeadPose(yaw, 0.5), HeadPose(-yaw, 0.5)),
            # 往右轉：yaw 往負方向變化才算做到。
            "turn_right": (HeadPose(-yaw, 0.5), HeadPose(yaw, 0.5)),
            # 抬頭：pitch 變小才算做到。
            "look_up": (HeadPose(0.0, 0.5 - pitch), HeadPose(0.0, 0.5 + pitch)),
            # 低頭：pitch 變大才算做到。
            "look_down": (HeadPose(0.0, 0.5 + pitch), HeadPose(0.0, 0.5 - pitch)),
        }
        # 逐一測試每個動作。
        for action, (done, opposite) in cases.items():
            # 用 subTest 分開標示每個動作，某個失敗時其他仍會繼續跑。
            with self.subTest(action=action):
                # 做到動作：必須回 True。
                self.assertTrue(action_performed(action, neutral, done))
                # 做反方向：必須回 False。
                self.assertFalse(action_performed(action, neutral, opposite))
                # 完全沒動（和正面影格一樣）：必須回 False。
                self.assertFalse(action_performed(action, neutral, neutral))

    def test_threshold_is_relative_to_neutral(self):
        """規則：門檻比的是「和正面影格相比的變化量」，不是動作影格的絕對角度。

        為什麼重要：每個人的臉型、拍攝角度不同，正面影格本來就可能有點偏；
        用變化量才能公平判斷，也避免「本來就偏一邊」的人不用轉頭就通過。
        """
        # 正面影格本來就偏一點，只看變化量。
        # 正面影格的 yaw 是 0.1（本來就偏一點）。
        neutral = HeadPose(yaw=0.1, pitch=0.5)
        # 變化量只有門檻的 0.9 倍：雖然絕對值 0.1 + 0.108 已超過門檻 0.12，仍然不算做到。
        self.assertFalse(action_performed("turn_left", neutral, HeadPose(0.1 + YAW * 0.9, 0.5)))
        # 變化量是門檻的 1.1 倍：算做到。
        self.assertTrue(action_performed("turn_left", neutral, HeadPose(0.1 + YAW * 1.1, 0.5)))

    def test_unknown_action_never_passes(self):
        """規則：不認識的動作名稱一律不算做到。

        為什麼重要：失敗時要偏向安全（fail closed）；請求格式雖然已經限制只能是四種動作，
        判斷函式本身也不應該因為未知的輸入而放行。
        """
        # "blink"（眨眼）不是支援的動作；就算角度變化很大，也必須回 False。
        self.assertFalse(action_performed("blink", HeadPose(0, 0.5), HeadPose(1, 1)))


class LiveDecisionTests(unittest.TestCase):
    """有 liveCapture 時 pipeline 的判定結果測試。"""

    def test_all_checks_pass_is_verified(self):
        """規則：即時鏡頭的所有檢查都通過時，才回 verified／VERIFICATION_PASSED。

        為什麼重要：這是唯一能讓使用者得到「已驗證」的路徑，必須確認它真的走得通，
        而且兩個驗證旗標與兩個分數都有填（AI 服務的 ProviderResult 要求 verified 一定要有這些）。
        """
        # 兩個動作：turn_left 的影格 yaw 是門檻的 1.5 倍；look_up 的影格 pitch 比正面的 0.5 少了門檻的 1.5 倍。
        result = run_live(["turn_left", "look_up"], [[face(yaw=YAW * 1.5)], [face(pitch=0.5 - PITCH * 1.5)]])
        # 狀態與原因碼：驗證通過。
        self.assertEqual((result.status, result.reasonCode), ("verified", "VERIFICATION_PASSED"))
        # 活體驗證通過。
        self.assertTrue(result.livenessVerified)
        # 身分驗證通過。
        self.assertTrue(result.identityVerified)
        # 比對分數 = 正面影格和主照片的相似度 0.8。
        self.assertEqual(result.faceMatchScore, 0.8)
        # 活體分數 = FakeSpoof 預設的真人機率 0.9。
        self.assertEqual(result.livenessScore, 0.9)

    def test_action_not_performed(self):
        """規則：動作影格的頭部角度和正面影格一樣（沒做動作）時，回 CHALLENGE_FAILED。

        為什麼重要：沒做動作代表可能是靜態照片或預錄畫面，不能通過。
        """
        # 動作影格的臉就是預設的 face()，角度和正面影格完全相同。
        result = run_live(["turn_left"], [[face()]])
        # 必須回 rejected／CHALLENGE_FAILED。
        self.assertEqual((result.status, result.reasonCode), ("rejected", "CHALLENGE_FAILED"))
        # 在比對主照片之前就結束，所以沒有比對分數。
        self.assertIsNone(result.faceMatchScore)

    def test_wrong_direction_fails(self):
        """規則：要求往左轉卻往右轉（方向相反）時，回 CHALLENGE_FAILED。

        為什麼重要：動作是隨機指定的，做錯方向代表沒有依照當下的指示，可能是預錄的影片。
        """
        # 要求 turn_left，但影格的 yaw 往負方向變化（等於往右轉）。
        result = run_live(["turn_left"], [[face(yaw=-YAW * 1.5)]])
        # 必須回 CHALLENGE_FAILED。
        self.assertEqual(result.reasonCode, "CHALLENGE_FAILED")

    def test_every_frame_must_pass(self):
        """規則：每一張動作影格都要做到，只要有一張沒做到就失敗。

        為什麼重要：避免「只做對一個動作就過關」的漏洞。
        """
        # 第一個動作 turn_left 做到了；第二個動作 look_down 的影格沒有動（face() 預設角度）。
        result = run_live(["turn_left", "look_down"], [[face(yaw=YAW * 1.5)], [face()]])
        # 必須回 CHALLENGE_FAILED。
        self.assertEqual(result.reasonCode, "CHALLENGE_FAILED")

    def test_action_frame_face_problems_are_prefixed(self):
        """規則：動作影格的臉有問題時，原因碼加上 ACTION_ 前綴。

        為什麼重要：讓前端分得出是動作影格的問題（例如轉頭時臉離開畫面），
        而不是自拍或主照片的問題，才能給使用者正確的提示。
        （注意：影像「解碼」失敗的前綴是 FRAME_，見最後一個測試；臉的偵測問題是 ACTION_。）
        """
        # 原因碼（含前綴）→ 動作影格的偵測結果：
        cases = {
            # 動作影格偵測不到臉。
            "ACTION_NO_FACE_DETECTED": [],
            # 動作影格有兩張臉（第一張有做到動作，但多了一張臉仍要拒絕）。
            "ACTION_MULTIPLE_FACES_DETECTED": [face(yaw=YAW * 1.5), face(x=120)],
            # 動作影格的臉有做到動作，但臉框寬度比最小邊長少 1 像素。
            "ACTION_FACE_TOO_SMALL": [face(yaw=YAW * 1.5, w=policy.MIN_FACE_SIDE - 1)],
        }
        # 逐一測試每個情境。
        for reason, faces in cases.items():
            # 用 subTest 分開標示每個情境。
            with self.subTest(reason=reason):
                # 一個動作 turn_left，其影格的偵測結果是 faces；明確給一張影格的相似度 0.9。
                result = run_live(["turn_left"], [faces], frame_similarities=[0.9])
                # 必須回 rejected 與帶有 ACTION_ 前綴的原因碼。
                self.assertEqual((result.status, result.reasonCode), ("rejected", reason))

    def test_different_person_during_capture(self):
        """規則：動作影格和正面影格的相似度低於門檻（像是換了一個人）時，回 FACE_CHANGED_DURING_CAPTURE。

        為什麼重要：避免「本人拍正面、再換別人做動作」或反過來的冒用方式；整段拍攝必須是同一個人。
        """
        # 動作有做到，但這張影格和正面影格的相似度只有 0.1（低於門檻 0.363）。
        result = run_live(["turn_left"], [[face(yaw=YAW * 1.5)]], frame_similarities=[0.1])
        # 必須回 rejected／FACE_CHANGED_DURING_CAPTURE。
        self.assertEqual((result.status, result.reasonCode), ("rejected", "FACE_CHANGED_DURING_CAPTURE"))

    def test_live_capture_still_needs_spoof_and_match(self):
        """規則：動作挑戰通過之後，仍然必須通過被動防偽與大頭貼（主照片）比對。

        為什麼重要：動作挑戰只證明「有人當下在鏡頭前動」，不證明畫面不是翻拍、也不證明是主照片本人；
        三道檢查缺一不可。
        """
        # 情境一：動作做到，但防偽判定不是真人（真人機率 0.2）。
        spoofed = run_live(["turn_left"], [[face(yaw=YAW * 1.5)]], spoof=FakeSpoof(0.2, is_real=False))
        # 必須回 SPOOF_SUSPECTED。
        self.assertEqual(spoofed.reasonCode, "SPOOF_SUSPECTED")
        # 活體驗證未通過。
        self.assertFalse(spoofed.livenessVerified)
        # 情境二：動作做到、防偽通過，但正面影格和主照片的相似度只有 0.1。
        mismatch = run_live(["turn_left"], [[face(yaw=YAW * 1.5)]], similarity=0.1)
        # 必須回 FACE_MISMATCH。
        self.assertEqual(mismatch.reasonCode, "FACE_MISMATCH")
        # 身分驗證未通過。
        self.assertFalse(mismatch.identityVerified)

    def test_invalid_frame_image_code_is_prefixed(self):
        """規則：動作影格的 base64 壞掉時，丟出 InvalidImage，錯誤碼是 FRAME_INVALID_IMAGE。

        為什麼重要：讓呼叫端分得出壞掉的是動作影格，而不是自拍或主照片。
        """
        # 組一個有一張 turn_left 動作影格的請求。
        request = verify_request(actions=["turn_left"])
        # 把動作影格內容換成 "????"：長度能通過 schema 的最小長度，但不是合法的 base64。
        request["liveCapture"]["frames"][0]["imageBase64"] = "????"
        # 預期會丟出 InvalidImage。
        with self.assertRaises(InvalidImage) as caught:
            # 用預設的假模型跑；解碼在執行模型之前就失敗，所以模型的設定不影響結果。
            verifier().verify(VerifyRequest(**request))
        # 錯誤碼必須是帶有 FRAME_ 前綴的 INVALID_IMAGE。
        self.assertEqual(str(caught.exception), "FRAME_INVALID_IMAGE")

    def test_policy_version_marks_live_capture(self):
        """規則：模型版本字串要以 policy-3 結尾，標示這是目前的判定政策版本。

        為什麼重要：判定規則改變時，事後要能從紀錄看出每筆結果是依哪一版政策判定的。
        """
        # 跑一次會通過的即時鏡頭驗證。
        result = run_live(["turn_left"], [[face(yaw=YAW * 1.5)]])
        # modelVersion 的結尾必須是 "policy-3"。
        self.assertTrue(result.modelVersion.endswith("policy-3"))

    def test_rotated_static_photo_fails(self):
        """規則：同一張正臉只在畫面平面內轉動（側傾），不能冒充左右轉頭。

        為什麼重要：舊的公式只看影像 x 軸，把照片轉約 20° 鼻尖就會水平位移，yaw 變化超過門檻；
        現在改在兩眼連線座標系計算，並限制側傾變化，靜態照片轉來轉去都做不到動作。
        """
        # 兩張動作影格都是正臉，只是分別側傾 +20° 與 −20°（超過 MAX_ROLL_CHANGE_DEGREES 的 15°）。
        rotated = run_live(["turn_left", "turn_right"], [[face(roll=20)], [face(roll=-20)]])
        # 必須回 CHALLENGE_FAILED。
        self.assertEqual((rotated.status, rotated.reasonCode), ("rejected", "CHALLENGE_FAILED"))
        # 側傾在上限內（10°）也一樣做不到，因為 yaw 本身沒有改變。
        slight = run_live(["turn_left"], [[face(roll=10)]])
        # 必須回 CHALLENGE_FAILED。
        self.assertEqual(slight.reasonCode, "CHALLENGE_FAILED")

    def test_real_turn_with_some_roll_still_passes(self):
        """規則：真的轉頭時順便歪一點頭（側傾在上限內），仍然算做到動作。

        為什麼重要：真人轉頭常伴隨輕微側傾；舊公式會讓側傾抵銷 yaw，造成真人被誤判失敗。
        """
        # 動作影格：yaw 是門檻的 1.5 倍，同時側傾 −10°。
        result = run_live(["turn_left"], [[face(yaw=YAW * 1.5, roll=-10)]])
        # 必須通過。
        self.assertEqual((result.status, result.reasonCode), ("verified", "VERIFICATION_PASSED"))

    def test_neutral_frame_must_face_camera(self):
        """規則：正面影格本身要大致正對鏡頭，偏頭或歪頭太多就回 CHALLENGE_FAILED。

        為什麼重要：動作只比較和正面影格的變化量。正面影格先偏向右邊，再送一張正臉當 turn_left，
        變化量一樣會超過門檻；限制正面影格的角度，才不能用「偏頭的正面＋正臉照」冒充動作。
        """
        # 正面影格 → 預期結果：
        cases = {
            # 往右偏超過 NEUTRAL_MAX_YAW，動作影格是正臉（變化量 +0.2，本來會算做到 turn_left）。
            "yaw": face(yaw=-(policy.NEUTRAL_MAX_YAW + 0.05)),
            # 側傾超過 NEUTRAL_MAX_ROLL_DEGREES。
            "roll": face(roll=policy.NEUTRAL_MAX_ROLL_DEGREES + 5),
        }
        # 逐一測試。
        for label, neutral in cases.items():
            # 用 subTest 分開標示。
            with self.subTest(label=label):
                # 偵測器：正面影格用指定的臉、主照片正常、動作影格是正臉。
                detector = FakeDetector([neutral], [face()], [face()])
                # 跑一個 turn_left 的即時鏡頭請求。
                result = verifier(detector=detector, embedder=FakeEmbedder(0.8, [0.9])).verify(
                    VerifyRequest(**verify_request(actions=["turn_left"]))
                )
                # 必須回 CHALLENGE_FAILED。
                self.assertEqual((result.status, result.reasonCode), ("rejected", "CHALLENGE_FAILED"))

    def test_every_action_frame_gets_spoof_check(self):
        """規則：每張動作影格都要做被動防偽，任何一張被判成假臉就回 SPOOF_SUSPECTED；
        livenessScore 是所有影格中最低的真人機率。

        為什麼重要：轉動照片或螢幕時，邊框、反光最容易出現在動作影格；只檢查正面影格會漏掉。
        """
        # 正面影格真人、第一張動作影格被判成翻拍。
        spoofed = run_live(
            ["turn_left", "look_up"],
            [[face(yaw=YAW * 1.5)], [face(pitch=0.5 - PITCH * 1.5)]],
            spoof=FakeSpoof(sequence=[(0.95, True), (0.2, False), (0.9, True)]),
        )
        # 必須回 SPOOF_SUSPECTED。
        self.assertEqual((spoofed.status, spoofed.reasonCode), ("rejected", "SPOOF_SUSPECTED"))
        # 分數是最低的那張 0.2。
        self.assertEqual(spoofed.livenessScore, 0.2)
        # 全部是真人時通過，分數仍取最低值。
        passed = run_live(
            ["turn_left"],
            [[face(yaw=YAW * 1.5)]],
            spoof=FakeSpoof(sequence=[(0.95, True), (0.7, True)]),
        )
        # 通過，livenessScore 是 0.7。
        self.assertEqual((passed.status, passed.livenessScore), ("verified", 0.7))


# 這個檔案被當成主程式執行時（例如在 services/face 目錄下執行 python -m tests.test_live）才跑測試；
# 被 unittest 探索或被其他模組 import 時，__name__ 不是 "__main__"，不會重複執行。
if __name__ == "__main__":
    # 找出這個檔案中所有的測試並執行。
    unittest.main()
