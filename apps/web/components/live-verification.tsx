"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  Camera,
  Check,
  RotateCcw,
  ScanFace,
  X,
} from "lucide-react";
import type { FaceLandmarker } from "@mediapipe/tasks-vision";
import { api } from "@/lib/api";
import {
  actionLabels,
  actionReached,
  facingCamera,
  headPose,
  rollSteady,
  type ChallengeAction,
  type HeadPose,
} from "@/lib/head-pose";

export type VerificationState = {
  id?: string;
  status: string;
  reasonCode?: string | null;
  modelName?: string | null;
  createdAt?: string;
  canVerify: boolean;
};
type Challenge = {
  challengeId: string;
  actions: ChallengeAction[];
  expiresInSeconds: number;
};
type Phase = "idle" | "starting" | "capturing" | "uploading" | "error";

// 拍下正面影格前，要連續幾張相機畫面都正對鏡頭（30 fps 約半秒），避免拍到正在移動的模糊畫面。
const STEADY_FRAMES = 15;
// 動作要連續幾張相機畫面都達標才拍下，避免一瞬間的誤判。
const REACHED_FRAMES = 3;
// 臉的寬度至少要佔畫面寬度的比例；太遠的臉伺服器會判成 FACE_TOO_SMALL。
const MIN_FACE_WIDTH = 0.22;
// 上傳的影格長邊上限（px）；後端會再縮到 1024 以內。
const FRAME_MAX_SIDE = 960;

const actionIcons: Record<ChallengeAction, typeof ArrowLeft> = {
  // 預覽畫面是鏡像的，往自己的左邊轉頭時，畫面裡的臉也往左轉，所以箭頭方向和使用者一致。
  turn_left: ArrowLeft,
  turn_right: ArrowRight,
  look_up: ArrowUp,
  look_down: ArrowDown,
};

/** 把驗證結果的代碼轉成給使用者看的說明與下一步。 */
export function verificationMessage(state?: VerificationState | null) {
  const code = state?.reasonCode ?? "";
  if (state?.status === "verified")
    return "驗證完成，你的檔案會顯示真人驗證標記。";
  if (code === "AVATAR_REQUIRED")
    return "先上傳一張本人清楚的正面照當大頭貼，才能開始真人驗證。";
  if (code === "CHALLENGE_FAILED")
    return "沒有偵測到指定的動作。請跟著提示把頭轉得明顯一點，再試一次。";
  if (code === "FACE_CHANGED_DURING_CAPTURE")
    return "拍攝過程中偵測到不同的人，請由本人獨自完成驗證。";
  if (code === "SPOOF_SUSPECTED")
    return "畫面看起來像是翻拍（例如螢幕或照片），請本人直接面對鏡頭再試一次。";
  if (code === "FACE_MISMATCH")
    return "和你的大頭貼看起來不是同一個人。請確認大頭貼是你本人的清楚正面照。";
  if (code === "AVATAR_CHANGED")
    return "你換過大頭貼了，請用新的大頭貼重新驗證。";
  if (
    code.startsWith("FRAME_") ||
    code.startsWith("IMAGE_") ||
    code === "INVALID_IMAGE"
  )
    return "拍下的畫面無法使用，請在光線充足的地方重新驗證。";
  if (code === "REFERENCE_PHOTO_UNAVAILABLE")
    return "暫時讀不到你的大頭貼，請稍後再試。你的帳號狀態不受影響。";
  if (code.startsWith("REFERENCE_"))
    return "你的大頭貼無法用來比對（沒有清楚的正臉、有多個人或臉太小）。請換一張本人清楚的正面照當大頭貼。";
  if (code.endsWith("NO_FACE_DETECTED"))
    return "畫面中找不到清楚的臉，請在光線充足的地方正對鏡頭再試一次。";
  if (code.endsWith("MULTIPLE_FACES_DETECTED"))
    return "畫面中只能有你一個人，請在沒有其他人的地方再試一次。";
  if (code.endsWith("FACE_TOO_SMALL"))
    return "臉太小了，請靠近鏡頭一點再試一次。";
  if (code === "LIVE_CAPTURE_REQUIRED")
    return "上傳的照片無法證明是本人當下拍攝，請用鏡頭即時驗證。";
  if (state?.status === "unavailable")
    return "驗證服務暫時無法使用，請稍後再試。你的帳號狀態不受影響。";
  if (state?.status === "rejected")
    return "這次沒有通過驗證，請跟著提示再試一次。";
  return "";
}

/** 相機、權限等錯誤轉成使用者看得懂的說明。 */
function cameraError(error: unknown) {
  const name = error instanceof DOMException ? error.name : "";
  if (name === "NotAllowedError")
    return "沒有取得相機權限。請在瀏覽器的網址列或設定裡允許使用相機，再試一次。";
  if (name === "NotFoundError" || name === "OverconstrainedError")
    return "找不到可以使用的相機。";
  if (name === "NotReadableError")
    return "相機正在被其他程式使用，請關閉後再試一次。";
  return "無法開啟相機，請稍後再試。";
}

async function loadLandmarker() {
  const { FaceLandmarker, FilesetResolver } =
    await import("@mediapipe/tasks-vision");
  // WASM 與模型都由網站自己提供（見 scripts/prepare-mediapipe.mjs），不連到 Google。
  const fileset = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
  const options = (delegate: "GPU" | "CPU") => ({
    baseOptions: {
      modelAssetPath: "/mediapipe/face_landmarker.task",
      delegate,
    },
    runningMode: "VIDEO" as const,
    // 設 2 才偵測得到「畫面裡有第二個人」。
    numFaces: 2,
  });
  try {
    return await FaceLandmarker.createFromOptions(fileset, options("GPU"));
  } catch {
    // 沒有 WebGL（部分手機、無頭瀏覽器）時改用 CPU。
    return FaceLandmarker.createFromOptions(fileset, options("CPU"));
  }
}

/** 從影片目前的畫面拍一張 JPEG（原始畫面，沒有鏡像）。 */
function capture(video: HTMLVideoElement) {
  const scale = Math.min(
    1,
    FRAME_MAX_SIDE / Math.max(video.videoWidth, video.videoHeight),
  );
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise<Blob>((resolve, reject) =>
    canvas.toBlob(
      (blob) =>
        blob ? resolve(blob) : reject(new Error("拍攝失敗，請再試一次。")),
      "image/jpeg",
      0.9,
    ),
  );
}

// 每按一次「開始」就是一輪，資源都掛在這一輪自己的物件上；
// 取消後馬上重新開始時，還在等待中的舊一輪醒來會發現自己不是目前這一輪，只清掉自己的資源。
type Run = {
  active: boolean;
  stream?: MediaStream;
  landmarker?: FaceLandmarker;
  frame?: number;
  timer?: ReturnType<typeof setTimeout>;
};

function release(run: Run) {
  run.active = false;
  if (run.frame) cancelAnimationFrame(run.frame);
  if (run.timer) clearTimeout(run.timer);
  run.stream?.getTracks().forEach((track) => track.stop());
  run.landmarker?.close();
  run.stream = undefined;
  run.landmarker = undefined;
}

export function LiveVerification({
  onFinished,
}: {
  onFinished: (result: VerificationState) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const session = useRef<Run>({ active: false });
  const [phase, setPhase] = useState<Phase>("idle");
  const [hint, setHint] = useState("");
  const [actions, setActions] = useState<ChallengeAction[]>([]);
  const [step, setStep] = useState(0);
  const [error, setError] = useState("");
  const [consent, setConsent] = useState(false);
  // 預覽框的長寬比跟著相機畫面走，使用者看到的就是偵測與上傳的完整範圍。
  const [aspect, setAspect] = useState<string>();

  const stop = useCallback(() => {
    release(session.current);
    session.current = { active: false };
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);
  // 離開頁面時一定要關掉相機與模型。
  useEffect(() => stop, [stop]);

  const isCurrent = (run: Run) => session.current === run && run.active;

  const fail = useCallback(
    (message: string) => {
      stop();
      setError(message);
      setPhase("error");
    },
    [stop],
  );

  async function upload(challenge: Challenge, frames: Promise<Blob>[]) {
    setPhase("uploading");
    setHint("正在確認是你本人…");
    stop();
    try {
      const data = new FormData();
      data.append("challengeId", challenge.challengeId);
      (await Promise.all(frames)).forEach((blob, index) =>
        data.append("frames", blob, `frame-${index}.jpg`),
      );
      const result = await api<VerificationState>("/onboarding/live", {
        method: "POST",
        body: data,
      });
      setPhase("idle");
      onFinished(result);
    } catch (e) {
      fail((e as Error).message);
    }
  }

  function track(run: Run, challenge: Challenge) {
    const video = videoRef.current;
    const landmarker = run.landmarker;
    if (!video || !landmarker) return;
    // 伺服器上的挑戰會過期；留 5 秒給上傳。用獨立的計時器，偵測迴圈停了也一定會觸發。
    run.timer = setTimeout(
      () => {
        if (isCurrent(run)) fail("時間到了，請重新開始驗證。");
      },
      (challenge.expiresInSeconds - 5) * 1000,
    );
    const frames: Promise<Blob>[] = [];
    let neutral: HeadPose | null = null;
    let current = 0;
    let steady = 0;
    let lastHint = "";
    // 只處理相機的新畫面：畫面更新率（rAF）通常比相機快，同一張畫面不重複偵測，steady 才是以相機畫面計數。
    let lastVideoTime = -1;
    // detectForVideo 的時間戳記必須嚴格遞增，否則 MediaPipe 會丟錯而且之後都無法再用。
    let lastTimestamp = 0;
    const say = (text: string) => {
      if (text !== lastHint) setHint((lastHint = text));
    };

    const detect = () => {
      const width = video.videoWidth;
      const height = video.videoHeight;
      const timestamp = Math.max(performance.now(), lastTimestamp + 1);
      lastTimestamp = timestamp;
      const faces = landmarker.detectForVideo(video, timestamp).faceLandmarks;
      const face = faces[0];
      const xs = face?.map((p) => p.x) ?? [];
      const faceWidth = face ? Math.max(...xs) - Math.min(...xs) : 0;
      const centerX = face ? (Math.max(...xs) + Math.min(...xs)) / 2 : 0.5;
      const pose = face ? headPose(face, width, height) : null;
      if (!face) {
        steady = 0;
        say("找不到臉，請正對鏡頭，把臉放進框裡");
      } else if (faces.length > 1) {
        steady = 0;
        say("畫面裡只能有你一個人");
      } else if (faceWidth < MIN_FACE_WIDTH) {
        steady = 0;
        say("請靠近鏡頭一點");
      } else if (centerX < 0.3 || centerX > 0.7 || !pose) {
        steady = 0;
        say("請把臉移到框的中間");
      } else if (current === 0) {
        // 第一步：正對鏡頭，保持不動，拍下正面影格並記下這時的角度當基準。
        if (!facingCamera(pose)) {
          steady = 0;
          say("請正對鏡頭，頭保持水平");
        } else if (++steady >= STEADY_FRAMES) {
          neutral = pose;
          frames.push(capture(video));
          steady = 0;
          setStep((current = 1));
        } else say("很好，保持不動…");
      } else if (neutral) {
        // 之後每一步：做到指定的動作（和正面相比轉得夠多）就拍下一張。
        const action = challenge.actions[current - 1];
        say(
          rollSteady(neutral, pose)
            ? actionLabels[action]
            : "頭保持水平，不要歪頭",
        );
        if (!actionReached(action, neutral, pose)) steady = 0;
        else if (++steady >= REACHED_FRAMES) {
          frames.push(capture(video));
          steady = 0;
          setStep((current += 1));
          if (current > challenge.actions.length) return true;
        }
      }
      return false;
    };

    const loop = () => {
      if (!isCurrent(run)) return;
      if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        try {
          if (detect()) return void upload(challenge, frames);
        } catch (e) {
          console.error(e);
          return fail("人臉偵測發生錯誤，請重新開始。");
        }
      }
      run.frame = requestAnimationFrame(loop);
    };
    run.frame = requestAnimationFrame(loop);
  }

  async function start() {
    stop();
    setError("");
    setStep(0);
    setPhase("starting");
    setHint("正在開啟相機…");
    if (!navigator.mediaDevices?.getUserMedia)
      return fail(
        "這個瀏覽器不能開啟相機。請改用 Safari 或 Chrome 開啟本站（在 LINE、Instagram 裡請選「用瀏覽器開啟」）。",
      );
    const run: Run = { active: true };
    session.current = run;
    let stage: "camera" | "model" | "challenge" = "camera";
    try {
      run.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      const video = videoRef.current;
      if (!isCurrent(run) || !video) return release(run);
      video.srcObject = run.stream;
      await video.play();
      if (!isCurrent(run)) return release(run);
      stage = "model";
      setHint("正在載入人臉偵測…");
      run.landmarker = await loadLandmarker();
      // 載入期間使用者按了取消：這一輪的模型與相機要自己關掉，不然會留在記憶體裡。
      if (!isCurrent(run)) return release(run);
      stage = "challenge";
      // 相機與模型都準備好才領挑戰：挑戰 120 秒後過期，不讓等待權限的時間吃掉。
      const challenge = await api<Challenge>("/verification/challenge", {
        method: "POST",
      });
      if (!isCurrent(run)) return release(run);
      setActions(challenge.actions);
      setPhase("capturing");
      track(run, challenge);
    } catch (e) {
      // 已經取消或換成新的一輪：只清掉自己的資源，不顯示錯誤（例如取消時 video.play() 的 AbortError）。
      if (!isCurrent(run)) return release(run);
      console.error(e);
      fail(
        stage === "camera"
          ? cameraError(e)
          : stage === "model"
            ? "人臉偵測載入失敗，請檢查網路後重新開始。"
            : (e as Error).message,
      );
    }
  }

  const running = phase === "starting" || phase === "capturing";
  const steps = ["正對鏡頭", ...actions.map((a) => actionLabels[a])];
  const ActionIcon =
    phase === "capturing" && step > 0 ? actionIcons[actions[step - 1]] : null;
  return (
    <div className="live-verification">
      <div
        className={
          running || phase === "uploading" ? "live-stage on" : "live-stage"
        }
        style={aspect ? { aspectRatio: aspect } : undefined}
      >
        <video
          ref={videoRef}
          className="live-video"
          playsInline
          muted
          onLoadedMetadata={(e) => {
            const { videoWidth, videoHeight } = e.currentTarget;
            if (videoWidth && videoHeight)
              setAspect(`${videoWidth} / ${videoHeight}`);
          }}
        />
        <div className="live-oval" aria-hidden="true" />
        {ActionIcon && (
          <span className="live-arrow" aria-hidden="true">
            <ActionIcon size={40} />
          </span>
        )}
        <p className="live-hint" role="status" aria-live="polite">
          {hint}
        </p>
      </div>
      {phase === "capturing" && (
        <ol className="live-steps">
          {steps.map((label, index) => (
            <li
              key={label}
              className={
                index < step ? "done" : index === step ? "current" : ""
              }
            >
              {index < step ? <Check size={16} /> : <span>{index + 1}</span>}
              {label}
            </li>
          ))}
        </ol>
      )}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {running ? (
        <button
          type="button"
          className="button secondary"
          onClick={() => {
            stop();
            setPhase("idle");
            setHint("");
          }}
        >
          取消
          <X size={18} />
        </button>
      ) : phase === "uploading" ? null : (
        <>
          <p className="muted">
            打開鏡頭後，先正對鏡頭，再依畫面提示轉頭或抬頭，全程約 20
            秒。畫面只用在這次驗證，不會保存影像。
          </p>
          <label className="check-label">
            <input
              type="checkbox"
              checked={consent}
              onChange={(e) => setConsent(e.target.checked)}
            />
            我同意使用鏡頭畫面進行本次真人驗證。
          </label>
          <button
            type="button"
            className="button"
            disabled={!consent}
            onClick={start}
          >
            {phase === "error" ? "重新開始" : "開始真人驗證"}
            {phase === "error" ? <RotateCcw size={18} /> : <Camera size={18} />}
          </button>
        </>
      )}
      {phase === "idle" && !error && (
        <p className="live-note muted">
          <ScanFace size={16} />{" "}
          人臉偵測在你的瀏覽器裡執行，送出的只有過程中拍下的幾張畫面。
        </p>
      )}
    </div>
  );
}
