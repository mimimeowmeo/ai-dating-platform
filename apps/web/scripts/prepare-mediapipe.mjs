// 把即時框臉需要的 MediaPipe 檔案放進 public/mediapipe/，由網站自己的網域提供，
// 使用者的瀏覽器不會連到 Google。dev 與 build 前都會執行；檔案已經正確就跳過。
// - WASM：從 node_modules/@mediapipe/tasks-vision/wasm 複製（版本跟著 package.json 鎖定的 0.10.34）。
// - 模型：face_landmarker.task 從 Google 的模型庫下載，驗證大小與 sha256，不符就中止。
import { createHash } from "node:crypto";
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const webRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const target = join(webRoot, "public", "mediapipe");
const MODEL = {
  file: "face_landmarker.task",
  url: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
  size: 3_758_596,
  sha256: "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
};

const sha256 = (buffer) => createHash("sha256").update(buffer).digest("hex");

mkdirSync(join(target, "wasm"), { recursive: true });
// 套件的 exports 沒有開放 package.json，改用它有開放的 WASM 路徑找出 wasm 資料夾。
const wasmFile = createRequire(join(webRoot, "package.json")).resolve(
  "@mediapipe/tasks-vision/vision_wasm_internal.wasm",
);
const wasmSource = dirname(wasmFile);
for (const name of readdirSync(wasmSource))
  copyFileSync(join(wasmSource, name), join(target, "wasm", name));

const modelPath = join(target, MODEL.file);
if (
  !existsSync(modelPath) ||
  sha256(readFileSync(modelPath)) !== MODEL.sha256
) {
  const response = await fetch(MODEL.url);
  if (!response.ok) throw new Error(`${MODEL.file}: HTTP ${response.status}`);
  const data = Buffer.from(await response.arrayBuffer());
  if (data.length !== MODEL.size || sha256(data) !== MODEL.sha256)
    throw new Error(`${MODEL.file}: 大小或 sha256 不符，拒絕使用`);
  writeFileSync(modelPath, data);
}
console.log("MediaPipe 檔案已就緒：public/mediapipe/");
