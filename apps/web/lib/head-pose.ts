// 即時框臉用的頭部角度計算。算法必須和 provider（services/face/app/pose.py）相同：
// 伺服器用 YuNet 的 5 個點重算，這裡用 MediaPipe 對應位置的點，讓兩邊的數字可以互相比較。
// 先轉到「兩眼連線」座標系（u＝兩眼連線方向、固定指向畫面右側；n＝和 u 垂直、指向畫面下方；d＝鼻尖 − 兩眼中點）：
//   yaw   = (d · u) ÷ 兩眼距離；往自己的左邊轉頭，數值變大。
//   pitch = (d · n) ÷ ((嘴角中點 − 兩眼中點) · n)；抬頭時數值變小。
//   roll  = 兩眼連線和水平線的夾角（度）；歪頭或轉動照片只會改變 roll，不會改變 yaw、pitch。
// 座標是相機原始畫面（沒有鏡像），並換算成像素，寬高比例才正確。

export type ChallengeAction =
  "turn_left" | "turn_right" | "look_up" | "look_down";
export type Point = { x: number; y: number };
export type HeadPose = { yaw: number; pitch: number; roll: number };

// 伺服器的門檻（services/face/app/policy.py）。前端要求 1.5 倍才拍下影格，留出兩邊算法的誤差。
const SERVER_YAW = 0.12;
const SERVER_PITCH = 0.08;
const SERVER_MAX_ROLL_CHANGE = 15;
export const CAPTURE_MARGIN = 1.5;
// 前端的側傾上限比伺服器嚴格（15 ÷ 1.5 = 10°）。
const MAX_ROLL_CHANGE = SERVER_MAX_ROLL_CHANGE / CAPTURE_MARGIN;

// MediaPipe Face Landmarker（478 點）的索引：兩眼各取內外眼角的中點（不受眼珠轉動影響）、鼻尖、兩個嘴角。
const RIGHT_EYE = [33, 133];
const LEFT_EYE = [362, 263];
const NOSE_TIP = 1;
const MOUTH = [61, 291];

const mid = (a: Point, b: Point): Point => ({
  x: (a.x + b.x) / 2,
  y: (a.y + b.y) / 2,
});

/** landmarks 是 MediaPipe 的正規化座標（0–1），width／height 是影像像素尺寸。 */
export function headPose(
  landmarks: { x: number; y: number }[],
  width: number,
  height: number,
): HeadPose | null {
  const px = (i: number): Point => ({
    x: landmarks[i].x * width,
    y: landmarks[i].y * height,
  });
  const rightEye = mid(px(RIGHT_EYE[0]), px(RIGHT_EYE[1]));
  const leftEye = mid(px(LEFT_EYE[0]), px(LEFT_EYE[1]));
  const eyeMid = mid(rightEye, leftEye);
  const mouthMid = mid(px(MOUTH[0]), px(MOUTH[1]));
  const nose = px(NOSE_TIP);
  let axisX = leftEye.x - rightEye.x;
  let axisY = leftEye.y - rightEye.y;
  const eyeDistance = Math.hypot(axisX, axisY);
  if (eyeDistance < 1) return null;
  if (axisX < 0) {
    axisX = -axisX;
    axisY = -axisY;
  }
  const ux = axisX / eyeDistance;
  const uy = axisY / eyeDistance;
  const nx = -uy;
  const ny = ux;
  const dx = nose.x - eyeMid.x;
  const dy = nose.y - eyeMid.y;
  const eyeToMouth =
    (mouthMid.x - eyeMid.x) * nx + (mouthMid.y - eyeMid.y) * ny;
  if (eyeToMouth < 1) return null;
  return {
    yaw: (dx * ux + dy * uy) / eyeDistance,
    pitch: (dx * nx + dy * ny) / eyeToMouth,
    roll: (Math.atan2(uy, ux) * 180) / Math.PI,
  };
}

/** 側傾和正面時相比是否在上限內（歪頭或轉照片不算轉頭）。 */
export const rollSteady = (neutral: HeadPose, pose: HeadPose) =>
  Math.abs(pose.roll - neutral.roll) <= MAX_ROLL_CHANGE;

/** 和正面時的角度相比，動作是否做到（前端用比伺服器嚴格 1.5 倍的門檻）。 */
export function actionReached(
  action: ChallengeAction,
  neutral: HeadPose,
  pose: HeadPose,
) {
  if (!rollSteady(neutral, pose)) return false;
  const yaw = pose.yaw - neutral.yaw;
  const pitch = pose.pitch - neutral.pitch;
  if (action === "turn_left") return yaw >= SERVER_YAW * CAPTURE_MARGIN;
  if (action === "turn_right") return yaw <= -SERVER_YAW * CAPTURE_MARGIN;
  if (action === "look_up") return pitch <= -SERVER_PITCH * CAPTURE_MARGIN;
  return pitch >= SERVER_PITCH * CAPTURE_MARGIN;
}

/** 正面拍攝要求：頭沒有明顯左右偏，也沒有歪頭（伺服器的上限是 |yaw| 0.15、|roll| 20°）。 */
export const facingCamera = (pose: HeadPose) =>
  Math.abs(pose.yaw) <= 0.08 && Math.abs(pose.roll) <= 10;

export const actionLabels: Record<ChallengeAction, string> = {
  turn_left: "慢慢把頭轉向你的左邊",
  turn_right: "慢慢把頭轉向你的右邊",
  look_up: "慢慢抬頭往上看",
  look_down: "慢慢低頭往下看",
};
