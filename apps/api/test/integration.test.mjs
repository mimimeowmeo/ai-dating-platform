import test, { after } from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { createRequire } from "node:module";
import { PrismaClient } from "@prisma/client";
import { io } from "socket.io-client";
import sharp from "sharp";
import { Client } from "minio";
import { Queue, Worker } from "bullmq";
const base = process.env.TEST_API_URL || "http://127.0.0.1:3001/api/v1";
const origin = new URL(base).origin;
const db = new PrismaClient();
const require = createRequire(import.meta.url);
require("ts-node/register/transpile-only");
// 雙向偏好篩選的程式版；探索頁用的是 SQL 版（同一個檔案），下面拿兩者的結果比對。
const { eligible } = require("../src/eligibility.ts");
const accounts = [];
const password = `Test-${randomUUID()}-safe`;
// 個人檔案的必填欄位：身高、20 字以上的自我介紹、想遇見的關係 1～2 項、
// 個性／飲食／價值觀／生活型態各至少 1 項、興趣至少 3 項。
const requiredProfile = {
  heightCm: 165,
  bio: "喜歡週末去咖啡店看書，也常常一個人出門散步拍照。",
  datingGoals: ["serious_relationship"],
  traits: [
    "humorous",
    "likes_hotpot",
    "values_communication",
    "nine_to_five",
    "coffee",
    "travel",
    "reading",
  ],
};
after(() => db.$disconnect());
async function request(
  path,
  { user, method = "GET", body, cookie, expected = 200 } = {},
) {
  const headers = {};
  if (user?.token) headers.Authorization = `Bearer ${user.token}`;
  if (body && !(body instanceof FormData))
    headers["Content-Type"] = "application/json";
  if (cookie) headers.Cookie = cookie;
  const res = await fetch(`${base}${path}`, {
    method,
    headers,
    body:
      body instanceof FormData ? body : body ? JSON.stringify(body) : undefined,
  });
  const value = await res.json();
  assert.equal(
    res.status,
    expected,
    `${method} ${path}: ${JSON.stringify(value)}`,
  );
  return {
    value,
    cookie: res.headers.get("set-cookie")?.split(";")[0],
    headers: res.headers,
  };
}
async function register() {
  const email = `phase-test-${randomUUID()}@example.test`;
  const r = await request("/auth/register", {
    method: "POST",
    body: { email, password },
    expected: 201,
  });
  const user = {
    id: r.value.user.id,
    token: r.value.accessToken,
    cookie: r.cookie,
    email,
  };
  accounts.push(user);
  return user;
}
const bucket = process.env.S3_BUCKET || "dating-media";
function storage() {
  const url = new URL(process.env.S3_ENDPOINT);
  return new Client({
    endPoint: url.hostname,
    port: Number(url.port || (url.protocol === "https:" ? 443 : 80)),
    useSSL: url.protocol === "https:",
    accessKey: process.env.S3_ACCESS_KEY,
    secretKey: process.env.S3_SECRET_KEY,
  });
}
async function cleanup() {
  const ids = accounts.map((a) => a.id);
  // 包含軟刪除的照片：測試帳號的物件要全部清掉。
  const photos = await db.photo.findMany({
    where: { userId: { in: ids } },
    select: { storageKey: true },
  });
  if (photos.length)
    await storage().removeObjects(
      bucket,
      photos.map((p) => p.storageKey),
    );
  await db.user.deleteMany({ where: { id: { in: ids } } });
  accounts.length = 0;
}
async function create(name) {
  const user = await register();
  await request("/profile", {
    method: "PUT",
    user,
    body: {
      displayName: name,
      birthDate: "1997-05-10",
      gender: "woman",
      city: "台北市",
      latitude: 25.033,
      longitude: 121.5654,
      ...requiredProfile,
      datingIntent: "serious",
      interests: ["咖啡"],
      hobbies: ["攝影"],
      foods: ["日式料理"],
    },
  });
  return user;
}
// 探索頁的候選池（SQL）要和 eligible() 逐一判斷全部使用者的結果相同，而且不限註冊時間：
// 比對 ?debug=explain 回傳的候選人數，並確認回傳的卡片都在程式版的名單裡。
async function assertPoolMatchesEligible(user) {
  const include = {
    profile: true,
    preference: true,
    traits: { include: { trait: true } },
  };
  const [me, everyone, sent, blocks, matches] = await Promise.all([
    db.user.findUnique({ where: { id: user.id }, include }),
    db.user.findMany({ where: { id: { not: user.id } }, include }),
    db.interaction.findMany({
      where: { fromUserId: user.id },
      select: { toUserId: true },
    }),
    db.block.findMany({
      where: { OR: [{ userId: user.id }, { blockedUserId: user.id }] },
    }),
    db.match.findMany({
      where: { OR: [{ userAId: user.id }, { userBId: user.id }] },
    }),
  ]);
  const skip = new Set([
    ...sent.map((s) => s.toUserId),
    ...blocks.flatMap((b) => [b.userId, b.blockedUserId]),
    ...matches.flatMap((m) => [m.userAId, m.userBId]),
  ]);
  const expected = new Set(
    everyone
      .filter((u) => !skip.has(u.id) && eligible(me, u) && eligible(u, me))
      .map((u) => u.id),
  );
  const cards = (await request("/discovery?debug=explain", { user })).value;
  assert.equal(
    cards[0]?.explain.poolSize ?? 0,
    expected.size,
    "探索頁的候選池要和 eligible() 逐一判斷全部使用者的結果一樣多",
  );
  for (const card of cards)
    assert.ok(expected.has(card.userId), "卡片上的人都要符合雙向偏好");
  return expected.size;
}
// 走前端現在的即時鏡頭流程：領挑戰 → 送正面＋每個動作各一張影格。
async function liveVerify(user, frame) {
  const challenge = (
    await request("/verification/challenge", {
      method: "POST",
      user,
      expected: 201,
    })
  ).value;
  const data = new FormData();
  data.append("challengeId", challenge.challengeId);
  for (let i = 0; i <= challenge.actions.length; i++)
    data.append(
      "frames",
      new Blob([frame], { type: "image/jpeg" }),
      `frame-${i}.jpg`,
    );
  return (
    await request("/onboarding/live", {
      method: "POST",
      user,
      body: data,
      expected: 201,
    })
  ).value;
}
// 測試影格是純色圖，重點是「不會假造驗證通過」，結果依有沒有接上人臉模型而不同：
// - 沒有模型：unavailable。AI_SERVICE_UNAVAILABLE＝連不上 ai 服務（CI 的 compose 不含它），
//   MODEL_NOT_CONFIGURED＝ai 服務有回應，但沒設定 provider。
// - 有模型（本機啟用 services/face）：影格裡沒有臉，rejected / NO_FACE_DETECTED。
function assertNotVerified(result) {
  const expected = {
    unavailable: ["AI_SERVICE_UNAVAILABLE", "MODEL_NOT_CONFIGURED"],
    rejected: ["NO_FACE_DETECTED"],
  }[result.status];
  assert.ok(
    expected?.includes(result.reasonCode),
    `未預期的結果：${result.status} / ${result.reasonCode}`,
  );
}
function connected(user) {
  const socket = io(origin, {
    auth: { token: user.token },
    transports: ["websocket"],
    reconnection: false,
  });
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      socket.close();
      reject(new Error("Socket timeout"));
    }, 5000);
    socket.once("connect", () => {
      clearTimeout(timer);
      resolve(socket);
    });
    socket.once("connect_error", (e) => {
      clearTimeout(timer);
      reject(e);
    });
  });
}
function ack(socket, event, body) {
  return new Promise((resolve, reject) => {
    socket
      .timeout(5000)
      .emit(event, body, (err, value) => (err ? reject(err) : resolve(value)));
  });
}
function event(socket, name) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${name} timeout`)), 5000);
    socket.once(name, (data) => {
      clearTimeout(timer);
      resolve(data);
    });
  });
}
function until(socket, name, match, ms = 8000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      socket.off(name, on);
      reject(new Error(`${name} timeout`));
    }, ms);
    function on(data) {
      if (!match(data)) return;
      clearTimeout(timer);
      socket.off(name, on);
      resolve(data);
    }
    socket.on(name, on);
  });
}
test(
  "真實帳號 → 偏好 → 照片 → 互讚 → Socket 聊天 → 封鎖與 session 權限",
  { timeout: 60000 },
  async () => {
    const sockets = [];
    let photo;
    try {
      await request("/health");
      await request("/profile", { expected: 401 });
      const a = await create("測試甲"),
        b = await create("測試乙"),
        c = await create("測試丙");
      await request("/auth/register", {
        method: "POST",
        body: { email: a.email, password },
        expected: 409,
      });
      await request("/auth/login", {
        method: "POST",
        body: { email: a.email, password: "definitely-wrong-password" },
        expected: 401,
      });
      await request("/profile", {
        method: "PUT",
        user: c,
        body: { displayName: "越權", isVerified: true },
        expected: 400,
      });
      await request("/preferences", {
        method: "PUT",
        user: a,
        body: {
          minAge: 40,
          maxAge: 20,
          preferredGender: "any",
          preferredDatingIntent: "any",
          maxDistanceKm: 100,
        },
        expected: 400,
      });
      const discover = (await request("/discovery", { user: a })).value;
      assert.ok(discover.some((p) => p.userId === b.id));
      assert.ok(!discover.some((p) => p.userId === a.id));
      await assertPoolMatchesEligible(a);
      // 探索頁的測試參數（docs/testing/QUERY-PARAMS.md）：不認得的參數或值回 400，舊名字提示新寫法。
      for (const [query, hint] of [
        ["rank=foo", "rank 只能是"],
        ["rank=", "rank 至少要選一個"],
        ["prefs=maybe", "prefs 只能是"],
        ["debug=verbose", "debug 只能是"],
        ["colour=red", "不認得的參數：colour"],
        ["hardfilter=false", "prefs=off"],
        ["test=true", "debug=explain"],
      ])
        assert.match(
          (await request(`/discovery?${query}`, { user: a, expected: 400 }))
            .value.message,
          new RegExp(hint),
          query,
        );
      const tagsOnly = (
        await request("/discovery?rank=tags&debug=explain", { user: a })
      ).value;
      assert.ok(
        tagsOnly.length &&
          tagsOnly.every(
            (p) => p.explain.appearance === null && p.explain.interest,
          ),
        "rank=tags 只用個人標籤",
      );
      assert.ok(
        (
          await request("/discovery?rank=face&debug=explain", { user: a })
        ).value.every((p) => p.explain.interest === null),
        "rank=face 不看個人標籤",
      );
      const bothSignals = (
        await request("/discovery?rank=face&rank=tags&debug=explain,search", {
          user: a,
        })
      ).value;
      assert.ok(
        bothSignals.length &&
          bothSignals.every((p) => p.explain.appearance && p.explain.interest),
        "rank 可以用重複參數多選；debug 可以用逗號多選",
      );
      const distantProfile = {
        displayName: "測試丙",
        birthDate: "1997-05-10",
        gender: "woman",
        city: "高雄市",
        latitude: 22.6273,
        longitude: 120.3014,
        ...requiredProfile,
        datingIntent: "serious",
        interests: [],
        hobbies: [],
        foods: [],
      };
      await request("/profile", {
        method: "PUT",
        user: c,
        body: { ...distantProfile, birthDate: "2015-01-01" },
        expected: 400,
      });
      await request("/profile", {
        method: "PUT",
        user: c,
        body: distantProfile,
      });
      assert.ok(
        !(await request("/discovery", { user: a })).value.some(
          (p) => p.userId === c.id,
        ),
      );
      // 測試參數：prefs=off 不套用偏好篩選，距離不符的人也要看得到。只看個人標籤（rank=tags）：
      // 測試帳號的標籤都一樣，會排在最前面，不會被種子帳號擠出 30 張之外。
      assert.ok(
        (
          await request("/discovery?prefs=off&rank=tags", { user: a })
        ).value.some((p) => p.userId === c.id),
        "prefs=off 要看得到距離不符的人",
      );
      const publicB = discover.find((p) => p.userId === b.id);
      for (const privateKey of [
        "email",
        "birthDate",
        "latitude",
        "longitude",
        "passwordHash",
      ])
        assert.equal(privateKey in publicB, false);
      await request("/preferences", {
        method: "PUT",
        user: b,
        body: {
          minAge: 60,
          maxAge: 70,
          preferredGender: "any",
          preferredDatingIntent: "any",
          maxDistanceKm: 100,
        },
      });
      assert.ok(
        !(await request("/discovery", { user: a })).value.some(
          (p) => p.userId === b.id,
        ),
      );
      await assertPoolMatchesEligible(a);
      await request("/preferences", {
        method: "PUT",
        user: b,
        body: {
          minAge: 18,
          maxAge: 99,
          preferredGender: "any",
          preferredDatingIntent: "any",
          maxDistanceKm: 100,
        },
      });
      const jpeg = await sharp({
        create: { width: 128, height: 128, channels: 3, background: "#c0b0a0" },
      })
        .jpeg()
        .toBuffer();
      const data = new FormData();
      data.append("file", new Blob([jpeg], { type: "image/jpeg" }), "test.jpg");
      photo = (
        await request("/profile/photos", {
          method: "POST",
          user: a,
          body: data,
          expected: 201,
        })
      ).value;
      assert.equal((await fetch(`${origin}${photo.url}`)).status, 200);
      await request(`/profile/photos/${photo.id}`, {
        method: "DELETE",
        user: b,
        expected: 404,
      });
      const invalid = new FormData();
      invalid.append(
        "file",
        new Blob(["<svg></svg>"], { type: "image/svg+xml" }),
        "bad.svg",
      );
      await request("/profile/photos", {
        method: "POST",
        user: a,
        body: invalid,
        expected: 400,
      });
      assertNotVerified(await liveVerify(a, jpeg));
      assert.equal(
        (await request("/auth/me", { user: a })).value.isVerified,
        false,
      );
      // 滑卡紀錄：探索頁會記下每張卡當下的推薦狀態；debug=explain 才附上推薦依據。
      assert.equal(
        (await request("/discovery", { user: a })).value[0].explain,
        undefined,
        "沒帶 debug=explain 不附推薦依據",
      );
      const shownB = (
        await request("/discovery?debug=explain", { user: a })
      ).value.find((p) => p.userId === b.id);
      assert.ok(shownB.explain.interest, "興趣開著要附上興趣的依據");
      assert.ok(
        shownB.explain.sharedTags.length > 0,
        "甲乙的標籤相同，要標出共同的標籤",
      );
      await Promise.all([
        request("/interactions", {
          method: "POST",
          user: a,
          body: { targetUserId: b.id, action: "like" },
          expected: 201,
        }),
        request("/interactions", {
          method: "POST",
          user: b,
          body: { targetUserId: a.id, action: "like" },
          expected: 201,
        }),
      ]);
      const matches = (await request("/matches", { user: a })).value;
      assert.equal(matches.length, 1);
      const match = matches[0],
        conversationId = match.conversationId;
      const [liked] = await db.swipeLog.findMany({
        where: { userId: a.id, targetUserId: b.id },
      });
      assert.equal(liked.action, "like");
      assert.equal(liked.source, shownB.explain.source);
      assert.equal(liked.position, shownB.explain.position);
      assert.equal(liked.interestOn, true);
      assert.ok(liked.rankingVersion);
      assert.equal(
        (
          await db.swipeLog.findFirst({
            where: { userId: b.id, targetUserId: a.id },
          })
        ).source,
        "unknown",
        "乙沒打開過探索頁，對不到當時的推薦狀態",
      );
      await request("/interactions", {
        method: "POST",
        user: a,
        body: { targetUserId: b.id, action: "like" },
        expected: 201,
      });
      assert.equal((await request("/matches", { user: a })).value.length, 1);
      assert.equal(
        await db.swipeLog.count({
          where: { userId: a.id, targetUserId: b.id },
        }),
        1,
        "已配對時再按喜歡不會寫入，也不記滑卡紀錄",
      );
      await request("/discovery/search/interactions", {
        method: "POST",
        user: a,
        body: { targetUserId: c.id, action: "pass" },
        expected: 201,
      });
      const searched = await db.swipeLog.findFirst({
        where: { userId: a.id, targetUserId: c.id },
      });
      assert.equal(searched.source, "search", "從測試用搜尋列滑的記成 search");
      assert.equal(searched.score, null);
      await request(`/conversations/${conversationId}/messages`, {
        user: c,
        expected: 404,
      });
      const sa = await connected(a),
        sb = await connected(b),
        sc = await connected(c);
      sockets.push(sa, sb, sc);
      assert.equal(
        (await ack(sa, "conversation:join", { conversationId })).ok,
        true,
      );
      assert.equal(
        (await ack(sb, "conversation:join", { conversationId })).ok,
        true,
      );
      assert.equal(
        (await ack(sc, "conversation:join", { conversationId })).ok,
        false,
      );
      const received = event(sb, "message:new");
      const dto = {
        conversationId,
        content: "你好，這是即時訊息。",
        clientId: randomUUID(),
      };
      const sent = await ack(sa, "message:send", dto);
      assert.equal(sent.ok, true);
      assert.equal((await received).id, sent.message.id);
      const duplicate = await request(
        `/conversations/${conversationId}/messages`,
        {
          method: "POST",
          user: a,
          body: { content: dto.content, clientId: dto.clientId },
          expected: 201,
        },
      );
      assert.equal(duplicate.value.id, sent.message.id);
      assert.equal(
        (
          await request(`/conversations/${conversationId}/messages`, {
            user: b,
          })
        ).value.length,
        1,
      );
      await request(`/conversations/${conversationId}/messages`, {
        method: "POST",
        user: a,
        body: { content: "不同內容", clientId: dto.clientId },
        expected: 409,
      });
      const typing = event(sb, "typing");
      assert.equal(
        (await ack(sa, "typing", { conversationId, isTyping: true })).ok,
        true,
      );
      assert.equal((await typing).isTyping, true);
      const read = event(sa, "conversation:read");
      await request(`/conversations/${conversationId}/read`, {
        method: "POST",
        user: b,
        body: {},
        expected: 201,
      });
      assert.equal((await read).userId, b.id);
      assert.ok(
        (await request("/conversations", { user: a })).value[0]
          .otherLastReadAt >= sent.message.createdAt,
      );
      assert.equal(
        (await request("/conversations", { user: b })).value[0].unreadCount,
        0,
      );
      assert.ok(
        (await request("/notifications", { user: b })).value.length >= 2,
      );
      await request("/blocks", {
        method: "POST",
        user: a,
        body: { blockedUserId: b.id },
        expected: 201,
      });
      assert.equal((await request("/matches", { user: a })).value.length, 0);
      assert.equal(
        (await request("/conversations", { user: b })).value.length,
        0,
      );
      await request(`/conversations/${conversationId}/messages`, {
        user: b,
        expected: 403,
      });
      assert.equal(
        (
          await ack(sb, "message:send", {
            conversationId,
            content: "blocked",
            clientId: randomUUID(),
          })
        ).ok,
        false,
      );
      await request(`/profile/${a.id}`, { user: b, expected: 404 });
      await request(`/blocks/${b.id}`, { method: "DELETE", user: a });
      assert.equal((await request("/matches", { user: a })).value.length, 0);
      await request("/profile", {
        method: "PUT",
        user: c,
        body: {
          ...distantProfile,
          city: "台北市",
          latitude: 25.033,
          longitude: 121.5654,
        },
      });
      await request("/interactions", {
        method: "POST",
        user: a,
        body: { targetUserId: c.id, action: "like" },
        expected: 201,
      });
      await request("/interactions", {
        method: "POST",
        user: c,
        body: { targetUserId: a.id, action: "like" },
        expected: 201,
      });
      const secondMatch = (await request("/matches", { user: a })).value[0];
      await request(`/matches/${secondMatch.id}`, {
        method: "DELETE",
        user: a,
      });
      await request(`/conversations/${secondMatch.conversationId}/messages`, {
        user: c,
        expected: 403,
      });
      const both = await Promise.all(
        [0, 1].map(() =>
          request("/auth/refresh", {
            method: "POST",
            cookie: c.cookie,
            expected: 201,
          }),
        ),
      );
      const rotated = both.filter((r) => r.cookie);
      assert.equal(rotated.length, 1);
      assert.ok(rotated[0].headers.get("set-cookie").includes("HttpOnly"));
      await request("/auth/me", { user: c });
      const overlap = await request("/auth/refresh", {
        method: "POST",
        cookie: c.cookie,
        expected: 201,
      });
      assert.equal(overlap.cookie, undefined);
      await db.session.updateMany({
        where: { userId: c.id },
        data: { rotatedAt: new Date(Date.now() - 60_000) },
      });
      await request("/auth/refresh", {
        method: "POST",
        cookie: c.cookie,
        expected: 401,
      });
      const renewed = await request("/auth/refresh", {
        method: "POST",
        cookie: rotated[0].cookie,
        expected: 201,
      });
      assert.ok(renewed.cookie);
      const oldToken = c.token;
      c.token = renewed.value.accessToken;
      c.cookie = renewed.cookie;
      await request("/auth/me", { user: c });
      await request("/auth/logout", {
        method: "POST",
        cookie: c.cookie,
        expected: 201,
      });
      await request("/auth/me", { user: c, expected: 401 });
      await request("/auth/me", { user: { token: oldToken }, expected: 401 });
      console.log(
        "已驗證：驗證不誤判、雙向偏好、互讚並發、照片權限、訊息去重、Socket 權限、封鎖、refresh 重疊期與重放保護。",
      );
    } finally {
      for (const s of sockets) s.close();
      await cleanup();
    }
  },
);
test(
  "修正回歸：session 輪替不中斷連線、線上狀態、分頁游標、上傳與錯誤訊息",
  { timeout: 60000 },
  async () => {
    const sockets = [];
    try {
      const x = await create("回歸甲"),
        y = await create("回歸乙");
      const bare = await register();
      const jpeg = await sharp({
        create: { width: 96, height: 96, channels: 3, background: "#a0b0c0" },
      })
        .jpeg()
        .toBuffer();
      const form = (buffer, name = "p.jpg", type = "image/jpeg") => {
        const data = new FormData();
        data.append("file", new Blob([buffer], { type }), name);
        return data;
      };
      const noProfile = await request("/profile/photos", {
        method: "POST",
        user: bare,
        body: form(jpeg),
        expected: 409,
      });
      assert.equal(noProfile.value.code, "PROFILE_REQUIRED");
      const upload = async () =>
        (
          await request("/profile/photos", {
            method: "POST",
            user: x,
            body: form(jpeg),
            expected: 201,
          })
        ).value;
      const p1 = await upload();
      const p2 = await upload();
      await request(`/profile/photos/${p1.id}`, { method: "DELETE", user: x });
      // 軟刪除：記錄與 MinIO 物件都留著，但取消主照片、舊網址回 404、不能再刪第二次。
      const removed = await db.photo.findUnique({ where: { id: p1.id } });
      assert.ok(removed?.deletedAt);
      assert.equal(removed.isAvatar, false);
      await storage().statObject(bucket, removed.storageKey);
      assert.equal((await fetch(`${origin}${p1.url}`)).status, 404);
      await request(`/profile/photos/${p1.id}`, {
        method: "DELETE",
        user: x,
        expected: 404,
      });
      const p3 = await upload();
      const orders = (await db.photo.findMany({ where: { userId: x.id } })).map(
        (p) => p.displayOrder,
      );
      assert.equal(new Set(orders).size, orders.length);
      const mine = (await request("/profile", { user: x })).value.photos;
      assert.deepEqual(
        mine.map((p) => p.id),
        [p2.id, p3.id],
      );
      assert.equal(mine[0].isAvatar, true);
      // 張數上限只算還在的照片：軟刪除的 p1 不佔名額，所以能補到 6 張（資料表共 7 列），第 7 張才被擋。
      for (let i = 0; i < 4; i++) await upload();
      assert.equal(await db.photo.count({ where: { userId: x.id } }), 7);
      const full = await request("/profile/photos", {
        method: "POST",
        user: x,
        body: form(jpeg),
        expected: 400,
      });
      assert.equal(full.value.code, "PHOTO_LIMIT");
      // 刪光再上傳：新照片要成為主照片（只看還在的照片），整張表只有它一張主照片，
      // 排序號碼接在所有舊照片（含軟刪除）之後、不撞號。
      for (const p of (await request("/profile", { user: x })).value.photos)
        await request(`/profile/photos/${p.id}`, { method: "DELETE", user: x });
      const fresh = await upload();
      assert.equal(fresh.isAvatar, true);
      assert.deepEqual(
        (await request("/profile", { user: x })).value.photos.map((p) => p.id),
        [fresh.id],
      );
      assert.equal(
        await db.photo.count({ where: { userId: x.id, isAvatar: true } }),
        1,
      );
      const allOrders = (
        await db.photo.findMany({ where: { userId: x.id } })
      ).map((p) => p.displayOrder);
      assert.equal(new Set(allOrders).size, allOrders.length);
      const tiny = await sharp({
        create: { width: 32, height: 32, channels: 3, background: "#000" },
      })
        .png()
        .toBuffer();
      const tooSmall = await request("/profile/photos", {
        method: "POST",
        user: x,
        body: form(tiny, "tiny.png", "image/png"),
        expected: 400,
      });
      assert.match(tooSmall.value.message, /太小/);
      const huge = await request("/profile/photos", {
        method: "POST",
        user: x,
        body: form(Buffer.alloc(9 * 1024 * 1024), "big.jpg"),
        expected: 413,
      });
      assert.equal(huge.value.code, "PAYLOAD_TOO_LARGE");
      assert.match(huge.value.message, /8 MB/);
      const bigJson = await request("/profile", {
        method: "PUT",
        user: x,
        body: { bio: "字".repeat(14000) },
        expected: 413,
      });
      assert.equal(bigJson.value.code, "PAYLOAD_TOO_LARGE");
      const malformed = await fetch(`${base}/preferences`, {
        method: "PUT",
        headers: {
          Authorization: `Bearer ${x.token}`,
          "Content-Type": "application/json",
        },
        body: "{bad json",
      });
      assert.equal(malformed.status, 400);
      assert.equal((await malformed.json()).code, "BAD_REQUEST");
      const invalid = await request("/preferences", {
        method: "PUT",
        user: x,
        body: {
          minAge: "x",
          maxAge: 30,
          preferredGender: "any",
          preferredDatingIntent: "any",
          maxDistanceKm: 100,
        },
        expected: 400,
      });
      assert.match(invalid.value.message, /最小年齡/);
      assert.doesNotMatch(invalid.value.message, /[A-Za-z]/);
      const notFound = await request("/interactions", {
        method: "POST",
        user: x,
        body: { targetUserId: randomUUID(), action: "like" },
        expected: 404,
      });
      assert.equal(notFound.value.code, "NOT_FOUND");
      // 已驗證的人重新驗證：unavailable 不動標記，rejected 會取消標記。
      await db.user.update({ where: { id: x.id }, data: { isVerified: true } });
      const verification = await liveVerify(x, jpeg);
      assertNotVerified(verification);
      assert.equal(
        (await request("/auth/me", { user: x })).value.isVerified,
        verification.status === "unavailable",
      );

      await request("/interactions", {
        method: "POST",
        user: x,
        body: { targetUserId: y.id, action: "like" },
        expected: 201,
      });
      const liked = await request("/interactions", {
        method: "POST",
        user: y,
        body: { targetUserId: x.id, action: "like" },
        expected: 201,
      });
      const conversationId = (
        await request(`/matches/${liked.value.matchId}`, { user: x })
      ).value.conversationId;
      const start = Date.parse("2026-01-01T00:00:00.000Z");
      await db.message.createMany({
        data: Array.from({ length: 52 }, (_, i) => ({
          conversationId,
          senderId: x.id,
          clientId: randomUUID(),
          content: `分頁 m${i + 1}`,
          createdAt: new Date(
            start + (i === 0 ? 0 : i <= 2 ? 1000 : (i - 1) * 1000),
          ),
        })),
      });
      const page1 = (
        await request(`/conversations/${conversationId}/messages`, { user: x })
      ).value;
      assert.equal(page1.length, 50);
      const oldest = page1[0];
      const page2 = (
        await request(
          `/conversations/${conversationId}/messages?before=${encodeURIComponent(oldest.createdAt)}&beforeId=${oldest.id}`,
          { user: x },
        )
      ).value;
      assert.equal(new Set([...page1, ...page2].map((m) => m.id)).size, 52);

      const sx = await connected(x);
      sockets.push(sx);
      assert.equal(
        (await ack(sx, "conversation:join", { conversationId })).ok,
        true,
      );
      const cameOnline = until(
        sx,
        "presence",
        (d) => d.userId === y.id && d.online === true,
      );
      const sy = await connected(y);
      sockets.push(sy);
      await cameOnline;
      const refreshed = await request("/auth/refresh", {
        method: "POST",
        cookie: y.cookie,
        expected: 201,
      });
      assert.ok(refreshed.cookie);
      const delivered = event(sy, "message:new");
      await request(`/conversations/${conversationId}/messages`, {
        method: "POST",
        user: x,
        body: { content: "換新 token 後仍收得到", clientId: randomUUID() },
        expected: 201,
      });
      assert.equal((await delivered).content, "換新 token 後仍收得到");
      assert.equal(sy.connected, true);
      const wentOffline = until(
        sx,
        "presence",
        (d) => d.userId === y.id && d.online === false,
      );
      sy.close();
      await wentOffline;
      console.log(
        "已驗證：未建檔不可上傳、照片排序、圖片／大小／格式錯誤訊息、404、驗證不被取消、分頁游標、線上狀態、輪替後連線不中斷。",
      );
    } finally {
      for (const s of sockets) s.close();
      await cleanup();
    }
  },
);
// 原本這裡有一個「Node BullMQ producer → Python worker 真實互通」測試。
// ai-worker 服務已移除（見 docker-compose.yml 的說明），佇列沒有消費者，
// 這個測試必然逾時，因此一併移除。真人驗證改由 verifySelfie 的降級路徑覆蓋。
test(
  "我按過的喜歡：等待回應、配對後、略過、雙向封鎖與解除配對",
  { timeout: 30000 },
  async () => {
    try {
      const a = await create("喜歡甲"),
        b = await create("喜歡乙"),
        c = await create("喜歡丙"),
        d = await create("喜歡丁");
      const likes = (user) => request("/likes", { user });
      const ids = (value) => value.map((l) => l.targetUserId).sort();
      assert.deepEqual(
        (await likes(a)).value,
        [],
        "還沒按過喜歡時應該是空清單",
      );
      for (const target of [b, c]) {
        await request("/interactions", {
          method: "POST",
          user: a,
          body: { targetUserId: target.id, action: "like" },
          expected: 201,
        });
      }
      await request("/interactions", {
        method: "POST",
        user: a,
        body: { targetUserId: d.id, action: "pass" },
        expected: 201,
      });
      const waiting = (await likes(a)).value;
      assert.deepEqual(
        ids(waiting),
        [b.id, c.id].sort(),
        "只列出按過喜歡的人，略過的不算",
      );
      assert.ok(
        waiting.every((l) => l.status === "waiting" && !l.conversationId),
        "對方還沒回應前都是 waiting",
      );
      assert.ok(
        waiting.every((l) => l.user?.userId === l.targetUserId),
        "每一筆的 user 要就是被按喜歡的那個人",
      );
      assert.equal(
        waiting.find((l) => l.targetUserId === b.id).user.displayName,
        "喜歡乙",
      );
      // 送出的喜歡是單向的：被按的人自己的清單不會因此多出東西。
      assert.deepEqual((await likes(b)).value, [], "被按喜歡的人清單仍是空的");
      await request("/interactions", {
        method: "POST",
        user: b,
        body: { targetUserId: a.id, action: "like" },
        expected: 201,
      });
      const afterMatch = (await likes(a)).value;
      const matched = afterMatch.find((l) => l.targetUserId === b.id);
      assert.equal(matched.status, "matched", "互相喜歡後要變成 matched");
      assert.ok(
        matched.matchId && matched.conversationId,
        "配對後要帶聊天室 id",
      );
      assert.equal(
        afterMatch.find((l) => l.targetUserId === c.id).status,
        "waiting",
      );
      // interact() 以 [id, target].sort() 決定誰是 userA；兩邊都查，每次執行都會走過兩個分支。
      const fromB = (await likes(b)).value;
      assert.deepEqual(ids(fromB), [a.id], "b 送出的喜歡只有 a");
      assert.equal(fromB[0].status, "matched");
      assert.equal(fromB[0].matchId, matched.matchId, "雙方看到同一個配對");
      assert.equal(fromB[0].conversationId, matched.conversationId);
      // 對方封鎖我。
      await request("/blocks", {
        method: "POST",
        user: c,
        body: { blockedUserId: a.id },
        expected: 201,
      });
      assert.ok(
        !ids((await likes(a)).value).includes(c.id),
        "被對方封鎖後就不再出現在送出的喜歡裡",
      );
      // 略過之後改成喜歡：interact() 會覆寫 action，d 就會進到清單。
      await request("/interactions", {
        method: "POST",
        user: a,
        body: { targetUserId: d.id, action: "like" },
        expected: 201,
      });
      assert.ok(
        ids((await likes(a)).value).includes(d.id),
        "略過後改按喜歡，對方要進到清單",
      );
      // 我封鎖對方（hidden 的另一個方向）。
      await request("/blocks", {
        method: "POST",
        user: a,
        body: { blockedUserId: d.id },
        expected: 201,
      });
      assert.deepEqual(
        ids((await likes(a)).value),
        [b.id],
        "我封鎖的人也不該出現在送出的喜歡裡",
      );
      // 解除配對：like 紀錄還在，但這段關係已結束，不該再以「等待回應」出現。
      await request(`/matches/${matched.matchId}`, {
        method: "DELETE",
        user: a,
      });
      assert.deepEqual(
        (await likes(a)).value,
        [],
        "解除配對後對方不該再出現在送出的喜歡裡",
      );
      assert.deepEqual((await likes(b)).value, [], "解除配對對雙方都生效");
      console.log("已驗證：送出的喜歡清單的狀態、略過、雙向封鎖與解除配對。");
    } finally {
      await cleanup();
    }
  },
);
test("個人檔案必填：身高、自我介紹 20 字、想遇見的關係 1～2 項、各類小熱愛", async () => {
  try {
    const user = await register();
    const valid = {
      displayName: "必填測試",
      birthDate: "1996-01-01",
      gender: "woman",
      city: "台北市",
      latitude: 25.033,
      longitude: 121.5654,
      ...requiredProfile,
    };
    const without = (code) => valid.traits.filter((c) => c !== code);
    for (const [patch, pattern] of [
      [{ heightCm: undefined }, /身高/],
      [{ heightCm: 99 }, /身高/],
      // 18 個字＋❤️：UTF-16 長度 21、code point 20，但使用者看到的只有 19 個字。
      [{ bio: `${"字".repeat(18)}❤️` }, /自我介紹至少要 20 個字/],
      [{ bio: `  ${"字".repeat(19)}  ` }, /自我介紹至少要 20 個字/],
      [{ datingGoals: [] }, /想遇見的關係請選 1～2 項/],
      [
        { datingGoals: ["serious_relationship", "friends_first", "chat_only"] },
        /想遇見的關係請選 1～2 項/,
      ],
      [{ traits: without("humorous") }, /個性至少選 1 項/],
      [{ traits: without("likes_hotpot") }, /飲食至少選 1 項/],
      [{ traits: without("values_communication") }, /價值觀至少選 1 項/],
      [{ traits: without("nine_to_five") }, /生活型態至少選 1 項/],
      [{ traits: without("reading") }, /興趣至少選 3 項/],
      [{ traits: undefined }, /我的小熱愛/],
    ]) {
      const r = await request("/profile", {
        method: "PUT",
        user,
        body: { ...valid, ...patch },
        expected: 400,
      });
      assert.match(r.value.message, pattern, JSON.stringify(patch));
    }
    assert.equal(
      await db.profile.findUnique({ where: { userId: user.id } }),
      null,
      "驗證沒過就不該建立檔案",
    );
    // 剛好 20 個字（❤️ 算一個字）可以存。
    const saved = await request("/profile", {
      method: "PUT",
      user,
      body: { ...valid, bio: `${"字".repeat(19)}❤️` },
    });
    assert.equal(saved.value.heightCm, 165);
    // 別人看到的卡片也要帶出身高。
    const other = await create("必填對象");
    assert.equal(
      (await request(`/profile/${user.id}`, { user: other })).value.heightCm,
      165,
    );
    // 身高低於偏好拉桿下限（130）的人：預設偏好停在兩端＝不限，雙方都要看得到彼此；
    // 對方把下限調高，才會被濾掉。
    await request("/profile", {
      method: "PUT",
      user,
      body: { ...valid, heightCm: 120 },
    });
    const sees = async (viewer, target) =>
      (await request("/discovery", { user: viewer })).value.some(
        (c) => c.userId === target.id,
      );
    assert.ok(await sees(other, user), "身高 120 的人在預設偏好下要看得到");
    assert.ok(await sees(user, other), "身高 120 的人自己的探索不能是空的");
    await request("/preferences", {
      method: "PUT",
      user: other,
      body: {
        minAge: 18,
        maxAge: 99,
        preferredGender: "any",
        preferredDatingIntent: "any",
        maxDistanceKm: 100,
        minHeightCm: 150,
        maxHeightCm: 250,
      },
    });
    assert.ok(!(await sees(other, user)), "下限調到 150 就要濾掉身高 120 的人");
    console.log("已驗證：個人檔案必填規則、卡片上的身高、偏好拉桿兩端＝不限。");
  } finally {
    await cleanup();
  }
});
// 每 200ms 讀一次，直到有值或逾時；背景工作（BullMQ）寫回資料庫需要一點時間。
async function eventually(read, timeoutMs = 10000) {
  const until = Date.now() + timeoutMs;
  for (;;) {
    const value = await read();
    if (value || Date.now() > until) return value;
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
}
// 也測「像在哪裡」（GET /discovery/explain-appearance）的存取控制與結果：本機開著真的推薦 worker 時用它
// （空白照片找不到臉，結果是 null）；沒開就先確認回 503，再模擬一個只處理這次請求的 worker。
test("照片網址要簽章才讀得到，封鎖後立即失效", { timeout: 60000 }, async () => {
  const origin = new URL(base).origin;
  const connection = {
    url: process.env.REDIS_URL,
    maxRetriesPerRequest: null,
  };
  const results = new Queue("rec-results", { connection });
  const redis = await results.client;
  let fake;
  let explainKey;
  try {
    const a = await create("照片甲");
    const b = await create("照片乙");
    const jpeg = await sharp({
      create: { width: 128, height: 128, channels: 3, background: "#ccc" },
    })
      .jpeg()
      .toBuffer();
    const data = new FormData();
    data.append("file", new Blob([jpeg], { type: "image/jpeg" }), "p.jpg");
    const uploaded = await request("/profile/photos", {
      method: "POST",
      user: b,
      body: data,
      expected: 201,
    });
    assert.match(uploaded.value.url, /\?u=[0-9a-f-]+&e=\d+&s=[0-9a-f]{32}$/);
    assert.equal(
      (await fetch(`${origin}${uploaded.value.url}`)).status,
      200,
      "自己的簽章網址應該讀得到",
    );
    assert.equal(
      (await fetch(`${base}/media/${uploaded.value.id}`)).status,
      403,
      "沒有簽章的舊網址應該被擋",
    );
    assert.equal(
      (
        await fetch(
          `${origin}${uploaded.value.url.replace(/s=.{4}/, "s=dead")}`,
        )
      ).status,
      403,
      "簽章被竄改應該被擋",
    );
    const discovery = await request("/discovery", { user: a });
    const card = discovery.value.find((c) => c.userId === b.id);
    assert.ok(card?.photos?.[0]?.url, "探索卡片要帶出照片網址");
    const seen = `${origin}${card.photos[0].url}`;
    assert.equal((await fetch(seen)).status, 200);
    const explain = (candidate, expected = 200) =>
      request(`/discovery/explain-appearance?candidate=${candidate}`, {
        user: a,
        expected,
      });
    await explain("abc", 400);
    await explain(randomUUID(), 404);
    await explain(b.id, 404); // 兩人都沒有外貌向量，這張卡沒有「最像的人」
    // 「最像的人」要從外貌向量排出來；這裡直接改 API 存在 Redis 的清單狀態，把 b 那張卡的最像的人設成 a。
    const own = new FormData();
    own.append("file", new Blob([jpeg], { type: "image/jpeg" }), "p.jpg");
    const anchorPhoto = await request("/profile/photos", {
      method: "POST",
      user: a,
      body: own,
      expected: 201,
    });
    const servedKey = `discovery:served:${a.id}`;
    await redis.hset(
      servedKey,
      b.id,
      JSON.stringify({
        ...JSON.parse(await redis.hget(servedKey, b.id)),
        anchorId: a.id,
      }),
    );
    const requestId = `${uploaded.value.id}:${anchorPhoto.value.id}`;
    explainKey = `rec:explain:${requestId}`;
    const box = [40, 80, 30, 10];
    const fakeResult = {
      similarity: 0.9,
      candidate: { width: 128, height: 128 },
      anchor: { width: 128, height: 128 },
      regions: [
        {
          name: "lips",
          drop: 0.02,
          dropCandidate: 0.01,
          dropAnchor: 0.03,
          candidate: [box],
          anchor: [box],
        },
      ],
    };
    const real = (await redis.exists("rec-worker:online")) > 0;
    if (!real) {
      assert.equal((await explain(b.id, 503)).value.code, "REC_WORKER_OFFLINE");
      fake = new Worker(
        "rec-jobs",
        async (job) => {
          if (job.data.requestId !== requestId) return;
          assert.ok(job.data.candidate && job.data.anchor, "要送兩張照片");
          await results.add("explain-appearance", {
            requestId,
            result: fakeResult,
          });
        },
        { connection },
      );
      await fake.waitUntilReady();
      await redis.set("rec-worker:online", "1", "EX", 60);
    }
    assert.equal((await explain(b.id)).value.status, "pending");
    const ready = await eventually(async () => {
      const state = (await explain(b.id)).value;
      return state.status === "ready" ? state : null;
    }, 30000);
    assert.ok(ready, "worker 算好後要拿到結果");
    assert.equal(ready.anchorUserId, a.id);
    assert.match(
      ready.candidatePhotoUrl,
      new RegExp(`/media/${uploaded.value.id}\\?`),
    );
    assert.match(
      ready.anchorPhotoUrl,
      new RegExp(`/media/${anchorPhoto.value.id}\\?`),
    );
    const expected = real ? null : fakeResult;
    assert.deepEqual(ready.result, expected);
    await results.add("explain-appearance", {
      requestId,
      result: {
        ...fakeResult,
        similarity: 0.1,
        regions: [{ ...fakeResult.regions[0], name: "ears" }],
      },
    });
    await new Promise((resolve) => setTimeout(resolve, 1500));
    assert.deepEqual(
      (await explain(b.id)).value.result,
      expected,
      "格式不對的 worker 結果不能寫進快取",
    );
    await request("/blocks", {
      method: "POST",
      user: b,
      body: { blockedUserId: a.id },
      expected: 201,
    });
    assert.equal(
      (await fetch(seen)).status,
      404,
      "被封鎖後原本的照片網址要失效",
    );
    assert.ok(
      !(
        await request("/discovery?prefs=off&rank=tags", { user: a })
      ).value.some((c) => c.userId === b.id),
      "prefs=off 也不能看到封鎖自己的人",
    );
    await explain(b.id, 404);
    console.log(
      `已驗證（${real ? "真的" : "模擬的"}推薦 worker）：照片簽章、竄改與封鎖後的存取控制；「像在哪裡」的權限與結果。`,
    );
  } finally {
    if (fake) {
      await redis.del("rec-worker:online");
      await fake.close();
    }
    if (explainKey) await redis.del(explainKey);
    await results.close();
    await cleanup();
  }
});
// 外貌向量：推薦 worker 在線才把照片放進 rec-jobs；結果寫回（找不到臉也留一列、沒有向量）；
// 刪照片 → 向量一起硬刪、下一張遞補成主照片再排一次；刪掉之後才到的結果不能寫回。
// 本機開著真的推薦 worker 時用它；沒開（例如 CI）就在這裡模擬一個（含在線訊號），只處理這個測試的照片。
test(
  "外貌向量：worker 在線才排照片，結果寫回，刪照片一起刪",
  { timeout: 60000 },
  async () => {
    const connection = {
      url: process.env.REDIS_URL,
      maxRetriesPerRequest: null,
    };
    const jobs = new Queue("rec-jobs", { connection });
    const results = new Queue("rec-results", { connection });
    const redis = await jobs.client;
    const mine = new Set();
    const received = [];
    let fake;
    try {
      const user = await create("外貌甲");
      const upload = async () => {
        const image = await sharp({
          create: {
            width: 1600,
            height: 1200,
            channels: 3,
            background: "#ccc",
          },
        })
          .jpeg()
          .toBuffer();
        const data = new FormData();
        data.append("file", new Blob([image], { type: "image/jpeg" }), "p.jpg");
        const photo = (
          await request("/profile/photos", {
            method: "POST",
            user,
            body: data,
            expected: 201,
          })
        ).value;
        mine.add(photo.id);
        return photo;
      };
      const row = (photoId) =>
        db.appearanceEmbedding.findUnique({
          where: { photoId },
          select: { userId: true, modelVersion: true },
        });
      const stored = (photoId, modelVersion) =>
        eventually(async () => {
          const found = await row(photoId);
          return found?.modelVersion === modelVersion ? found : null;
        });
      const empty = async (photoId) =>
        (
          await db.$queryRaw`SELECT embedding IS NULL AS empty FROM appearance_embeddings WHERE photo_id = ${photoId}::uuid`
        )[0]?.empty;
      const real = (await redis.exists("rec-worker:online")) > 0;
      const first = await upload();
      if (real)
        assert.ok(
          await eventually(() => row(first.id), 30000),
          "推薦 worker 的結果要寫回",
        );
      else {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        assert.equal(
          await jobs.getJob(`appearance-${first.id}`),
          undefined,
          "推薦 worker 沒在跑時不能把照片放進 Redis",
        );
        fake = new Worker(
          "rec-jobs",
          async (job) => {
            if (!mine.has(job.data.photoId)) return;
            const meta = await sharp(
              Buffer.from(job.data.image, "base64"),
            ).metadata();
            received.push({
              photoId: job.data.photoId,
              size: [meta.width, meta.height],
              format: meta.format,
            });
            await results.add("embed-appearance", {
              photoId: job.data.photoId,
              modelVersion: "integration-test-no-face",
              embedding: null,
            });
          },
          { connection },
        );
        await fake.waitUntilReady();
        await redis.set("rec-worker:online", "1", "EX", 60);
      }
      const second = await upload();
      await request(`/profile/photos/${first.id}`, {
        method: "DELETE",
        user,
      });
      assert.equal(await row(first.id), null, "刪照片要一起刪掉外貌向量");
      assert.ok(
        await eventually(() => row(second.id), 30000),
        "遞補成主照片後要再排一次",
      );
      assert.equal(
        await empty(second.id),
        true,
        "找不到臉也要留一列（沒有向量），補排才不會一直重送",
      );
      if (!real)
        assert.deepEqual(
          received.filter((job) => job.photoId === second.id),
          [{ photoId: second.id, size: [1600, 1200], format: "jpeg" }],
          "送去 worker 的是存在物件儲存的 JPEG 原檔",
        );
      const result = {
        photoId: second.id,
        modelVersion: "integration-test",
        embedding: Array.from({ length: 512 }, (_, i) => (i === 0 ? 1 : 0)),
      };
      await results.add("embed-appearance", result);
      assert.deepEqual(await stored(second.id, "integration-test"), {
        userId: user.id,
        modelVersion: "integration-test",
      });
      assert.equal(await empty(second.id), false);
      await request(`/profile/photos/${second.id}`, {
        method: "DELETE",
        user,
      });
      assert.equal(await row(second.id), null, "刪照片要一起刪掉外貌向量");
      await results.add("embed-appearance", result);
      await new Promise((resolve) => setTimeout(resolve, 1500));
      assert.equal(await row(second.id), null, "照片刪掉後才到的結果不能寫回");
      console.log(
        `已驗證（${real ? "真的" : "模擬的"}推薦 worker）：外貌向量排程、寫回與刪除。`,
      );
    } finally {
      if (fake) {
        await redis.del("rec-worker:online");
        await fake.close();
      }
      await jobs.close();
      await results.close();
      await cleanup();
    }
  },
);
test(
  "AI 推薦回覆：權限、請求紀錄、訊息來源標記",
  { timeout: 60000 },
  async () => {
    try {
      const a = await create("AI甲"),
        b = await create("AI乙"),
        c = await create("AI丙");
      for (const [from, to] of [
        [a, b],
        [b, a],
      ])
        await request("/interactions", {
          method: "POST",
          user: from,
          body: { targetUserId: to.id, action: "like" },
          expected: 201,
        });
      const conversationId = (await request("/conversations", { user: a }))
        .value[0].id;
      // 不是聊天室成員的人按推薦，連聊天室存在與否都不該知道。
      await request(`/conversations/${conversationId}/reply-suggestions`, {
        method: "POST",
        user: c,
        expected: 404,
      });
      await request(`/conversations/${conversationId}/messages`, {
        method: "POST",
        user: b,
        body: { content: "嗨嗨，你週末都在做什麼？", clientId: randomUUID() },
        expected: 201,
      });
      // AI 服務沒啟動時（CI 的 compose 不含 ai 服務）要回 503，而不是 500 或假資料。
      const asked = await fetch(
        `${base}/conversations/${conversationId}/reply-suggestions`,
        { method: "POST", headers: { Authorization: `Bearer ${a.token}` } },
      );
      const suggested = await asked.json();
      const requestRow = await db.aiSuggestionRequest.findFirst({
        where: { conversationId, requesterId: a.id },
        orderBy: { createdAt: "desc" },
        include: { suggestions: true },
      });
      assert.ok(requestRow, "不論成功或失敗都要留下請求紀錄（規格 9）");
      if (asked.status === 201) {
        assert.ok(suggested.suggestions.length <= 5, "一次最多 5 則");
        // 沒有依據時不硬湊：可以是 0 則，這時要附上固定提示（規格 4.1）。
        if (suggested.suggestions.length === 0)
          assert.equal(suggested.notice, "沒有可推薦的句子");
        assert.equal(requestRow.status, suggested.status);
        assert.ok(requestRow.modelName, "要記下實際回應的模型");
        assert.equal(
          requestRow.suggestions.filter((s) => s.rank > 0).length,
          suggested.suggestions.length,
          "回傳的推薦都要存檔",
        );
        assert.equal(suggested.canRegenerate, false, "「換一批」目前鎖住");
      } else {
        assert.equal(asked.status, 503);
        assert.ok(
          ["AI_UNAVAILABLE", "AI_NOT_CONFIGURED"].includes(suggested.code),
          `未預期的錯誤代碼：${suggested.code}`,
        );
        assert.equal(requestRow.status, "error");
        assert.ok(requestRow.errorCode, "要記下失敗代碼");
      }
      // 同一輪只產生一次（「換一批」先鎖起來）。一輪＝聊天室的最後一則訊息相同。
      // 不依賴 AI 服務：直接放一批「這一輪已經產生過」的推薦，再放一筆更新的失敗紀錄。
      const turnMessage = await db.message.findFirst({
        where: { conversationId },
        orderBy: { createdAt: "desc" },
      });
      const earlier = await db.aiSuggestionRequest.create({
        data: {
          conversationId,
          requesterId: a.id,
          mode: "reply",
          status: "partial",
          notice: "只找到 2 則合適的建議",
          lastMessageId: turnMessage.id,
          suggestions: {
            create: [
              { rank: 2, text: "妳週末都怎麼過呢", intent: "question" },
              { rank: 1, text: "我週末大多在打球，妳呢", intent: "answer" },
              { rank: 0, text: "被刪掉的候選", rejectedReason: "DUPLICATE" },
            ],
          },
        },
        include: { suggestions: true },
      });
      await db.aiSuggestionRequest.create({
        data: {
          conversationId,
          requesterId: a.id,
          mode: "unknown",
          status: "error",
          errorCode: "LLM_UNAVAILABLE",
          lastMessageId: turnMessage.id,
        },
      });
      const requestsBefore = await db.aiSuggestionRequest.count({
        where: { conversationId },
      });
      const again = await request(
        `/conversations/${conversationId}/reply-suggestions`,
        { method: "POST", user: a, expected: 201 },
      );
      const rankOf = (rank) => earlier.suggestions.find((s) => s.rank === rank);
      assert.equal(
        again.value.requestId,
        earlier.id,
        "同一輪再按要回傳同一批；後來失敗的請求不算用掉這一輪",
      );
      assert.deepEqual(
        again.value.suggestions.map((s) => s.id),
        [rankOf(1).id, rankOf(2).id],
        "依名次排好，被刪掉的候選不回傳；id 沿用，送出時才能標記來源",
      );
      assert.equal(again.value.status, "partial");
      assert.equal(again.value.mode, "reply");
      assert.equal(again.value.notice, "只找到 2 則合適的建議");
      assert.equal(again.value.canRegenerate, false);
      assert.equal(
        await db.aiSuggestionRequest.count({ where: { conversationId } }),
        requestsBefore,
        "回傳同一批時不呼叫 AI，也不新增請求紀錄",
      );
      // 鎖住的是「自己」這一輪：對方按推薦不會拿到 A 的那一批。
      const partnerAsked = await fetch(
        `${base}/conversations/${conversationId}/reply-suggestions`,
        { method: "POST", headers: { Authorization: `Bearer ${b.token}` } },
      );
      const partnerBody = await partnerAsked.json();
      if (partnerAsked.status === 201)
        assert.notEqual(partnerBody.requestId, earlier.id);
      else assert.equal(partnerAsked.status, 503);
      // 以下的來源標記不依賴 AI 服務：直接放一則推薦進資料庫，模擬剛剛產生過。
      const stored = await db.aiSuggestionRequest.create({
        data: {
          conversationId,
          requesterId: a.id,
          mode: "reply",
          status: "ok",
          suggestions: {
            create: [
              {
                rank: 1,
                text: "我週末通常會去爬山耶，妳呢",
                intent: "answer",
                styleTarget: "partner",
                styleDistance: 0.12,
              },
            ],
          },
        },
        include: { suggestions: true },
      });
      const suggestion = stored.suggestions[0];
      const send = (user, content, suggestionId) =>
        request(`/conversations/${conversationId}/messages`, {
          method: "POST",
          user,
          body: {
            content,
            clientId: randomUUID(),
            ...(suggestionId ? { suggestionId } : {}),
          },
          expected: 201,
        });
      const verbatim = await send(a, suggestion.text, suggestion.id);
      const verbatimOrigin = await db.messageOrigin.findUnique({
        where: { messageId: verbatim.value.id },
      });
      assert.equal(verbatimOrigin.origin, "ai_verbatim");
      assert.equal(verbatimOrigin.similarity, 1);
      assert.equal(
        (await db.aiSuggestion.findUnique({ where: { id: suggestion.id } }))
          .chosenAt instanceof Date,
        true,
        "被採用的推薦要記下時間，才能算採用率",
      );
      const edited = await send(a, `${suggestion.text}哈哈`, suggestion.id);
      const editedOrigin = await db.messageOrigin.findUnique({
        where: { messageId: edited.value.id },
      });
      assert.equal(editedOrigin.origin, "ai_edited");
      assert.ok(
        editedOrigin.similarity >= 0.5 && editedOrigin.similarity < 0.95,
        `改寫後的相似度要落在 0.5～0.95：${editedOrigin.similarity}`,
      );
      const rewritten = await send(a, "我其實都在家裡耍廢", suggestion.id);
      const rewrittenOrigin = await db.messageOrigin.findUnique({
        where: { messageId: rewritten.value.id },
      });
      assert.equal(
        rewrittenOrigin.origin,
        "human",
        "改到看不出原樣就算真人寫的",
      );
      assert.equal(
        rewrittenOrigin.suggestionId,
        suggestion.id,
        "仍保留出處，之後才能評估修改幅度",
      );
      // 不能拿別人的推薦替自己的訊息貼標籤。
      await request(`/conversations/${conversationId}/messages`, {
        method: "POST",
        user: b,
        body: {
          content: "借用別人的推薦",
          clientId: randomUUID(),
          suggestionId: suggestion.id,
        },
        expected: 404,
      });
      await request(`/conversations/${conversationId}/messages`, {
        method: "POST",
        user: a,
        body: {
          content: "多帶一個欄位",
          clientId: randomUUID(),
          extraField: "nope",
        },
        expected: 400,
      });
      const plain = await send(b, "那下次一起去啊");
      assert.equal(
        await db.messageOrigin.findUnique({
          where: { messageId: plain.value.id },
        }),
        null,
        "一般訊息不寫來源紀錄，查詢時視為 human",
      );
      // 有人傳出新訊息就是新的一輪：A 再按不會拿到舊的那一批。
      const nextTurn = await fetch(
        `${base}/conversations/${conversationId}/reply-suggestions`,
        { method: "POST", headers: { Authorization: `Bearer ${a.token}` } },
      );
      const nextBody = await nextTurn.json();
      if (nextTurn.status === 201)
        assert.notEqual(nextBody.requestId, earlier.id, "新的一輪要重新產生");
      else assert.equal(nextTurn.status, 503);
      console.log(
        `已驗證：推薦權限、請求紀錄（${asked.status === 201 ? "AI 服務可用" : "AI 服務不可用時回 503"}）、ai_verbatim／ai_edited／human 判定與越權保護。`,
      );
    } finally {
      await cleanup();
    }
  },
);
