// 逐支 API 實際呼叫，並回資料庫確認寫進正確的表與欄位。
// 用法：node scripts/verify/api-endpoints.mjs
import { randomUUID } from "node:crypto";
import {
  API,
  call,
  checker,
  db,
  makeAccount,
  photoForm,
  testEmail,
} from "./lib.mjs";

const check = checker("API 端點 ↔ 資料表欄位");

// ---------- 公開端點 ----------
const health = await call("/health");
check(
  "GET /health",
  health.status === 200 && health.data.status === "ok",
  JSON.stringify(health.data),
);

// ---------- 帳號：users / sessions / preferences ----------
const email = testEmail("api");
const password = `Safe-${randomUUID()}`;
const jar = new Map();
const withCookies = async (path, options = {}) => {
  const cookie = [...jar].map(([k, v]) => `${k}=${v}`).join("; ");
  const res = await fetch(`${API}${path}`, {
    method: options.method || "GET",
    headers: {
      origin: process.env.VERIFY_ORIGIN || "http://localhost:8080",
      ...(options.body ? { "content-type": "application/json" } : {}),
      ...(options.token ? { authorization: `Bearer ${options.token}` } : {}),
      ...(cookie ? { cookie } : {}),
    },
    body: options.body && JSON.stringify(options.body),
  });
  for (const line of res.headers.getSetCookie?.() ?? []) {
    const [pair] = line.split(";");
    const i = pair.indexOf("=");
    jar.set(pair.slice(0, i), pair.slice(i + 1));
  }
  return { status: res.status, data: await res.json().catch(() => null) };
};

const registered = await withCookies("/auth/register", {
  method: "POST",
  body: { email, password },
});
let token = registered.data?.accessToken;
const userId = registered.data?.user?.id;
check(
  "POST /auth/register → users",
  registered.status === 201 &&
    db(`select count(*) from users where email='${email}'`) === "1",
);
check(
  "↳ 同時建立 sessions",
  Number(db(`select count(*) from sessions where user_id='${userId}'`)) >= 1,
);
check(
  "↳ 同時建立 preferences（預設值）",
  db(
    `select min_age||'-'||max_age||'/'||min_height_cm||'-'||max_height_cm from preferences where user_id='${userId}'`,
  ) === "18-99/130-250",
);

const me = await call("/auth/me", { token });
check(
  "GET /auth/me",
  me.status === 200 && me.data.email === email,
  me.data?.email,
);
const refreshed = await withCookies("/auth/refresh", { method: "POST" });
check(
  "POST /auth/refresh → sessions.rotated_at",
  refreshed.status === 201 && !!refreshed.data?.accessToken,
);
if (refreshed.data?.accessToken) token = refreshed.data.accessToken;

// ---------- 個人檔案：profiles / user_traits / user_photos ----------
const saved = await call("/profile", {
  method: "PUT",
  token,
  body: {
    displayName: "掃描甲",
    birthDate: "1994-02-02",
    gender: "woman",
    bio: "API 驗收帳號，檢查每個欄位都寫進對應的資料表。",
    city: "台中市",
    latitude: 24.147,
    longitude: 120.673,
    heightCm: 168,
    occupation: "工程師",
    education: "碩士",
    traits: [
      "coffee",
      "hiking",
      "reading",
      "humorous",
      "likes_seafood",
      "values_trust",
      "nine_to_five",
    ],
    datingGoals: ["serious_relationship"],
  },
});
const profileRow = db(
  `select display_name||'|'||city||'|'||height_cm||'|'||occupation||'|'||education from profiles where user_id='${userId}'`,
);
check(
  "PUT /profile → profiles",
  saved.status === 200 && profileRow === "掃描甲|台中市|168|工程師|碩士",
  profileRow,
);
const traitRow = db(
  `select string_agg(t.code, ',' order by t.code) from user_traits ut join traits t on t.id=ut.trait_id where ut.user_id='${userId}'`,
);
check(
  "↳ traits/datingGoals → user_traits",
  traitRow ===
    "coffee,hiking,humorous,likes_seafood,nine_to_five,reading,serious_relationship,values_trust",
  traitRow,
);
const mine = await call("/profile", { token });
check(
  "GET /profile",
  mine.status === 200 &&
    mine.data.heightCm === 168 &&
    mine.data.traits.length === 7,
);

const photo = await call("/profile/photos", {
  method: "POST",
  token,
  form: photoForm(),
});
const photoRow = db(
  `select storage_key||'|'||is_avatar||'|'||display_order from user_photos where id='${photo.data?.id}'`,
);
check(
  "POST /profile/photos → user_photos",
  photo.status === 201 && photoRow.includes("|true|0"),
  photoRow,
);
const signed = await call(photo.data.url.replace("/api/v1", ""), { raw: true });
const unsigned = await call(`/media/${photo.data.id}`, { raw: true });
check(
  "GET /media/:id 需要簽章",
  signed.status === 200 && unsigned.status === 403,
  `簽章 ${signed.status} / 無簽章 ${unsigned.status}`,
);

// ---------- 偏好與選項：preferences / traits ----------
const prefs = await call("/preferences", {
  method: "PUT",
  token,
  body: {
    minAge: 25,
    maxAge: 45,
    preferredGender: "any",
    maxDistanceKm: 300,
    minHeightCm: 160,
    maxHeightCm: 190,
    preferredDatingIntent: "serious_relationship",
  },
});
const prefsRow = db(
  `select min_age||'-'||max_age||'|'||preferred_gender||'|'||max_distance_km||'|'||min_height_cm||'-'||max_height_cm||'|'||preferred_dating_intent from preferences where user_id='${userId}'`,
);
check(
  "PUT /preferences → preferences",
  prefs.status === 200 &&
    prefsRow === "25-45|any|300|160-190|serious_relationship",
  prefsRow,
);
check(
  "GET /preferences",
  (await call("/preferences", { token })).data.minHeightCm === 160,
);
const traits = await call("/traits", { token });
check(
  "GET /traits ＝ traits 資料表",
  traits.data.length === Number(db("select count(*) from traits")),
  `${traits.data.length} 筆`,
);
for (const path of ["/interests", "/hobbies", "/foods"]) {
  const r = await call(path, { token });
  check(
    `GET ${path}（程式內建清單）`,
    r.status === 200 && Array.isArray(r.data),
  );
}
for (const [path, column] of [
  ["/me/interests", "interests"],
  ["/me/hobbies", "hobbies"],
  ["/me/foods", "foods"],
]) {
  const catalog = await call(path.replace("/me", ""), { token });
  const value = catalog.data[0];
  const r = await call(path, {
    method: "PUT",
    token,
    body: { values: [value] },
  });
  check(
    `PUT ${path} → profiles.${column}`,
    r.status === 200 &&
      db(
        `select array_to_string(${column}, ',') from profiles where user_id='${userId}'`,
      ) === value,
  );
}

// ---------- 驗證：verification_records ----------
check(
  "GET /verification/status",
  (await call("/verification/status", { token })).status === 200,
);
const selfie = await call("/onboarding/selfie", {
  method: "POST",
  token,
  form: photoForm(),
});
const verifyRow = db(
  `select status from verification_records where user_id='${userId}' order by created_at desc limit 1`,
);
check(
  "POST /onboarding/selfie → verification_records",
  selfie.status === 201 && verifyRow.length > 0,
  verifyRow,
);
check(
  "POST /verification/retry（不寫資料庫）",
  (await call("/verification/retry", { method: "POST", token })).status === 201,
);

// ---------- 探索與互動：likes / matches / notifications ----------
const other = await makeAccount("api-b", {
  displayName: "掃描乙",
  gender: "man",
  city: "台中市",
  latitude: 24.147,
  longitude: 120.673,
});
const discovery = await call("/discovery", { token });
check(
  "GET /discovery",
  discovery.status === 200 && discovery.data.some((c) => c.userId === other.id),
  `${discovery.data?.length} 張卡片`,
);
const publicProfile = await call(`/profile/${other.id}`, { token });
check(
  "GET /profile/:id",
  publicProfile.status === 200 && Array.isArray(publicProfile.data.traits),
);

const like = await call("/interactions", {
  method: "POST",
  token,
  body: { targetUserId: other.id, action: "like" },
});
check(
  "POST /interactions → likes",
  like.status === 201 &&
    db(
      `select action from likes where from_user_id='${userId}' and to_user_id='${other.id}'`,
    ) === "like",
);
const back = await call("/interactions", {
  method: "POST",
  token: other.token,
  body: { targetUserId: userId, action: "like" },
});
const matchId = db(
  `select id from matches where (user_a_id='${userId}' and user_b_id='${other.id}') or (user_a_id='${other.id}' and user_b_id='${userId}')`,
);
check(
  "↳ 互讚 → matches + notifications",
  back.data?.matched === true &&
    matchId.length === 36 &&
    db(
      `select count(*) from notifications where type='match' and user_id in ('${userId}','${other.id}')`,
    ) === "2",
);
check("GET /matches", (await call("/matches", { token })).data.length === 1);
check(
  "GET /matches/:id",
  (await call(`/matches/${matchId}`, { token })).status === 200,
);

// ---------- 聊天：conversations / conversation_members / messages ----------
const conversations = await call("/conversations", { token });
const conversationId = conversations.data?.[0]?.id;
check("GET /conversations", conversations.status === 200 && !!conversationId);
check(
  "POST /conversations（以 matchId 取回同一間）",
  (await call("/conversations", { method: "POST", token, body: { matchId } }))
    .data?.id === conversationId,
);
check(
  "GET /conversations/:id",
  (await call(`/conversations/${conversationId}`, { token })).status === 200,
);
const clientId = randomUUID();
const sent = await call(`/conversations/${conversationId}/messages`, {
  method: "POST",
  token,
  body: { content: "驗收訊息", clientId },
});
check(
  "POST /conversations/:id/messages → messages",
  sent.status === 201 &&
    db(
      `select sender_id||'|'||content from messages where conversation_id='${conversationId}' order by created_at desc limit 1`,
    ) === `${userId}|驗收訊息`,
);
await call(`/conversations/${conversationId}/messages`, {
  method: "POST",
  token,
  body: { content: "驗收訊息", clientId },
});
check(
  "↳ 同一個 clientId 不重複寫入",
  db(
    `select count(*) from messages where conversation_id='${conversationId}'`,
  ) === "1",
);
check(
  "GET /conversations/:id/messages",
  (await call(`/conversations/${conversationId}/messages`, { token })).data
    .length === 1,
);
const read = await call(`/conversations/${conversationId}/read`, {
  method: "POST",
  token: other.token,
});
check(
  "POST /conversations/:id/read → conversation_members.last_read_at",
  read.status === 201 &&
    db(
      `select count(*) from conversation_members where conversation_id='${conversationId}' and user_id='${other.id}' and last_read_at is not null`,
    ) === "1",
);

// ---------- 通知 ----------
const notifications = await call("/notifications", { token: other.token });
const notificationId = notifications.data?.[0]?.id;
check(
  "GET /notifications",
  notifications.status === 200 && !!notificationId,
  `${notifications.data?.length} 筆`,
);
const readNotification = await call(`/notifications/${notificationId}/read`, {
  method: "POST",
  token: other.token,
});
check(
  "POST /notifications/:id/read → notifications.read_at",
  readNotification.status === 201 &&
    db(
      `select count(*) from notifications where id='${notificationId}' and read_at is not null`,
    ) === "1",
);

// ---------- 封鎖與解除配對：blocks / matches.status ----------
const blocked = await call("/blocks", {
  method: "POST",
  token,
  body: { blockedUserId: other.id },
});
check(
  "POST /blocks → blocks + matches.status",
  blocked.status === 201 &&
    db(
      `select count(*) from blocks where user_id='${userId}' and blocked_user_id='${other.id}'`,
    ) === "1" &&
    db(`select status from matches where id='${matchId}'`) === "blocked",
);
check("GET /blocks", (await call("/blocks", { token })).data.length === 1);
check(
  "DELETE /blocks/:id",
  (await call(`/blocks/${other.id}`, { method: "DELETE", token })).status ===
    200 &&
    db(
      `select count(*) from blocks where user_id='${userId}' and blocked_user_id='${other.id}'`,
    ) === "0",
);
check(
  "DELETE /profile/photos/:id",
  (await call(`/profile/photos/${photo.data.id}`, { method: "DELETE", token }))
    .status === 200 &&
    db(`select count(*) from user_photos where id='${photo.data.id}'`) === "0",
);
check(
  "DELETE /matches/:id → matches.status=unmatched",
  (await call(`/matches/${matchId}`, { method: "DELETE", token })).status ===
    200 &&
    db(`select status from matches where id='${matchId}'`) === "unmatched",
);

// ---------- 登出 ----------
const relogin = await withCookies("/auth/login", {
  method: "POST",
  body: { email, password },
});
const freshToken = relogin.data?.accessToken;
check("POST /auth/login", relogin.status === 201 && !!freshToken);
const loggedOut = await withCookies("/auth/logout", {
  method: "POST",
  token: freshToken,
});
check(
  "POST /auth/logout → sessions.revoked_at",
  loggedOut.status === 201 &&
    Number(
      db(
        `select count(*) from sessions where user_id='${userId}' and revoked_at is not null`,
      ),
    ) >= 1,
);
check(
  "↳ 登出後 access token 失效",
  (await call("/profile", { token: freshToken })).status === 401,
);

const result = check.summary();
process.exitCode = result.failed ? 1 : 0;
