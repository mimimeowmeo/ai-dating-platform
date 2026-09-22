import test, { after } from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { PrismaClient } from "@prisma/client";
import { io } from "socket.io-client";
import sharp from "sharp";
import { Client } from "minio";
const base = process.env.TEST_API_URL || "http://127.0.0.1:3001/api/v1";
const origin = new URL(base).origin;
const db = new PrismaClient();
const accounts = [];
const password = `Test-${randomUUID()}-safe`;
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
async function cleanup() {
  const ids = accounts.map((a) => a.id);
  const photos = await db.photo.findMany({
    where: { userId: { in: ids } },
    select: { storageKey: true },
  });
  if (photos.length) {
    const url = new URL(process.env.S3_ENDPOINT);
    await new Client({
      endPoint: url.hostname,
      port: Number(url.port || (url.protocol === "https:" ? 443 : 80)),
      useSSL: url.protocol === "https:",
      accessKey: process.env.S3_ACCESS_KEY,
      secretKey: process.env.S3_SECRET_KEY,
    }).removeObjects(
      process.env.S3_BUCKET || "dating-media",
      photos.map((p) => p.storageKey),
    );
  }
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
      bio: "自動化整合測試帳號",
      datingIntent: "serious",
      interests: ["咖啡"],
      hobbies: ["攝影"],
      foods: ["日式料理"],
    },
  });
  return user;
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
      const distantProfile = {
        displayName: "測試丙",
        birthDate: "1997-05-10",
        gender: "woman",
        city: "高雄市",
        latitude: 22.6273,
        longitude: 120.3014,
        bio: "自動化整合測試帳號",
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
      const selfie = new FormData();
      selfie.append(
        "file",
        new Blob([jpeg], { type: "image/jpeg" }),
        "selfie.jpg",
      );
      const verification = (
        await request("/onboarding/selfie", {
          method: "POST",
          user: a,
          body: selfie,
          expected: 201,
        })
      ).value;
      assert.equal(verification.status, "unavailable");
      // 重點是「不會假造驗證通過」；兩種代碼都代表沒有真人辨識模型：
      // AI_SERVICE_UNAVAILABLE＝連不上 ai 服務（CI 的 compose 不含它），
      // MODEL_NOT_CONFIGURED＝ai 服務有回應，但沒掛上辨識模型（本機開著 ai 服務時）。
      assert.ok(
        ["AI_SERVICE_UNAVAILABLE", "MODEL_NOT_CONFIGURED"].includes(
          verification.reasonCode,
        ),
        `未預期的代碼：${verification.reasonCode}`,
      );
      assert.equal(
        (await request("/auth/me", { user: a })).value.isVerified,
        false,
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
      await request("/interactions", {
        method: "POST",
        user: a,
        body: { targetUserId: b.id, action: "like" },
        expected: 201,
      });
      assert.equal((await request("/matches", { user: a })).value.length, 1);
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
      await db.user.update({ where: { id: x.id }, data: { isVerified: true } });
      const verification = await request("/onboarding/selfie", {
        method: "POST",
        user: x,
        body: form(jpeg, "selfie.jpg"),
        expected: 201,
      });
      assert.equal(verification.value.status, "unavailable");
      assert.equal(
        (await request("/auth/me", { user: x })).value.isVerified,
        true,
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
test("照片網址要簽章才讀得到，封鎖後立即失效", { timeout: 20000 }, async () => {
  const origin = new URL(base).origin;
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
    console.log("已驗證：照片簽章、竄改與封鎖後的存取控制。");
  } finally {
    await cleanup();
  }
});
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
      } else {
        assert.equal(asked.status, 503);
        assert.ok(
          ["AI_UNAVAILABLE", "AI_NOT_CONFIGURED"].includes(suggested.code),
          `未預期的錯誤代碼：${suggested.code}`,
        );
        assert.equal(requestRow.status, "error");
        assert.ok(requestRow.errorCode, "要記下失敗代碼");
      }
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
      console.log(
        `已驗證：推薦權限、請求紀錄（${asked.status === 201 ? "AI 服務可用" : "AI 服務不可用時回 503"}）、ai_verbatim／ai_edited／human 判定與越權保護。`,
      );
    } finally {
      await cleanup();
    }
  },
);
