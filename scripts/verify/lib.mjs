// 驗收腳本共用的小工具：呼叫 API、查資料庫、印出檢查結果。
// 預設打本機 docker 的 :8080 與 heartlink-pg 的 dating 資料庫，可用環境變數覆蓋：
//   VERIFY_API_BASE（預設 http://localhost:8080/api/v1）
//   VERIFY_ORIGIN  （預設 http://localhost:8080，要與後端 WEB_ORIGIN 相同）
//   VERIFY_PG_*    （容器名／使用者／資料庫，預設 heartlink-pg / heartlink / dating）
import { execFileSync } from "node:child_process";
import { randomUUID } from "node:crypto";

export const API =
  process.env.VERIFY_API_BASE || "http://localhost:8080/api/v1";
export const ORIGIN = process.env.VERIFY_ORIGIN || "http://localhost:8080";
const PG_CONTAINER = process.env.VERIFY_PG_CONTAINER || "heartlink-pg";
const PG_USER = process.env.VERIFY_PG_USER || "heartlink";
const PG_PASSWORD = process.env.VERIFY_PG_PASSWORD || "heartlink";
const PG_DATABASE = process.env.VERIFY_PG_DATABASE || "dating";

// 測試帳號一律用這個前綴，跑完由 apps/api/test/cleanup-e2e.mjs 依 RUN_ID 清乾淨。
export const RUN_ID = process.env.E2E_RUN_ID || randomUUID();
export const testEmail = (label) => `e2e-${RUN_ID}-${label}@example.test`;

/** 直接查資料庫，用來確認 API 真的寫到正確的表與欄位。 */
export const db = (sql) =>
  execFileSync("docker", [
    "exec",
    "-e",
    `PGPASSWORD=${PG_PASSWORD}`,
    PG_CONTAINER,
    "psql",
    "-U",
    PG_USER,
    "-d",
    PG_DATABASE,
    "-tAc",
    sql,
  ])
    .toString()
    .trim();

/** 呼叫 API。form 用來送 multipart，raw 用來取原始回應（例如圖片）。 */
export async function call(
  path,
  { method = "GET", body, token, form, raw } = {},
) {
  const headers = { origin: ORIGIN };
  if (body) headers["content-type"] = "application/json";
  if (token) headers.authorization = `Bearer ${token}`;
  const res = await fetch(path.startsWith("http") ? path : `${API}${path}`, {
    method,
    headers,
    body: form ?? (body && JSON.stringify(body)),
  });
  if (raw)
    return {
      status: res.status,
      type: res.headers.get("content-type"),
      size: (await res.arrayBuffer()).byteLength,
    };
  return { status: res.status, data: await res.json().catch(() => null) };
}

const PNG =
  "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAA1BMVEXZj3op6S/DAAAACXBIWXMAAAPoAAAD6AG1e1JrAAAAG0lEQVRYw+3BgQAAAADDoPlT3+AEVQEAAAB8AxBAAAEZszF2AAAAAElFTkSuQmCC";
export const photoForm = () => {
  const form = new FormData();
  form.append(
    "file",
    new Blob([Buffer.from(PNG, "base64")], { type: "image/png" }),
    "photo.png",
  );
  return form;
};

/**
 * 建一個「可以進站」的帳號：基本資料 + 交友目標 + 小熱愛 + 一張照片。
 * 四項缺一就會被前端的完成度關卡擋在個人檔案頁。
 */
export async function makeAccount(label, overrides = {}) {
  const email = testEmail(label);
  const password = `Safe-${randomUUID()}`;
  const registered = await call("/auth/register", {
    method: "POST",
    body: { email, password },
  });
  if (registered.status !== 201)
    throw new Error(
      `註冊失敗 ${registered.status} ${JSON.stringify(registered.data)}`,
    );
  const token = registered.data.accessToken;
  const catalog = await call("/traits", { token });
  const pick = (category, n) =>
    catalog.data
      .filter((t) => t.category === category)
      .slice(0, n)
      .map((t) => t.code);
  await call("/profile", {
    method: "PUT",
    token,
    body: {
      displayName: label,
      birthDate: "1995-05-05",
      gender: "woman",
      bio: `${label} 的自我介紹：喜歡散步、看展，也想認識聊得來的朋友。`,
      city: "台北市",
      latitude: 25.033,
      longitude: 121.5654,
      heightCm: 168,
      traits: [
        ...pick("interest", 3),
        ...pick("personality", 1),
        ...pick("diet", 1),
        ...pick("value", 1),
        ...pick("lifestyle", 1),
      ],
      datingGoals: pick("dating_goal", 1),
      ...overrides,
    },
  });
  await call("/preferences", {
    method: "PUT",
    token,
    body: {
      minAge: 18,
      maxAge: 99,
      preferredGender: "any",
      maxDistanceKm: 500,
      minHeightCm: 130,
      maxHeightCm: 250,
      preferredDatingIntent: "any",
    },
  });
  const photo = await call("/profile/photos", {
    method: "POST",
    token,
    form: photoForm(),
  });
  const mine = await call("/profile", { token });
  return { email, password, token, id: mine.data.userId, photo: photo.data };
}

/** 收集檢查結果，最後用 summary() 印出通過數並決定離開碼。 */
export function checker(title) {
  const rows = [];
  console.log(`\n── ${title} ──`);
  const check = (name, ok, detail = "") => {
    rows.push({ name, ok });
    console.log(
      `${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  — ${detail}` : ""}`,
    );
    return ok;
  };
  check.summary = () => {
    const failed = rows.filter((r) => !r.ok);
    console.log(`=== ${rows.length - failed.length}/${rows.length} 通過 ===`);
    return {
      total: rows.length,
      failed: failed.length,
      names: failed.map((f) => f.name),
    };
  };
  return check;
}
