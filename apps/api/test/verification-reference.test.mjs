// 真人驗證 API（Profiles 的 verification／verifySelfie／retry／createChallenge／verifyLive）單元測試。
// 用假的資料庫、假的物件儲存、假的 Redis 與假的 fetch，不需要 AI 服務，也不需要正在執行的 API。
// 要確認的規則：
// - 只拿使用者的第一張主照片（大頭貼）比對；沒有大頭貼就不能驗證（409 AVATAR_REQUIRED）。
// - 大頭貼會縮圖後放進 referenceImages 送給 AI 服務。
// - 大頭貼讀不到或太小時不呼叫 AI 服務，只記 unavailable，不改使用者的 isVerified。
// - 狀態 API 與重新驗證 API 都會回 canVerify，前端依它決定要不要顯示真人驗證區塊。
// - 即時鏡頭：挑戰是隨機 2 個動作、120 秒失效、只能用一次；影格依挑戰順序放進 liveCapture。
// - 只有 verified／rejected 會改 isVerified；證據不完整的 verified、比對途中換了大頭貼都不採信。
// - 刪掉大頭貼會取消驗證標記，狀態 API 改回 AVATAR_CHANGED。

// node 內建的測試框架：test 定義一個測試案例，afterEach 在每個案例結束後執行。
import test, { afterEach } from "node:test";
// 嚴格模式的斷言函式（比較時不做型別轉換）。
import assert from "node:assert/strict";
// Readable.from 可以把一段資料包成串流，用來假裝 MinIO 回傳的檔案串流。
import { Readable } from "node:stream";
// 在 ES module 裡取得 require，用來載入 TypeScript 原始碼。
import { createRequire } from "node:module";
// 影像處理套件：產生測試用 JPEG、讀取縮圖後的尺寸。
import sharp from "sharp";
// 建立 require 函式（以這個測試檔的位置為基準解析路徑）。
const require = createRequire(import.meta.url);
// 註冊 ts-node：之後 require 的 .ts 檔會即時轉成 JavaScript（只轉譯、不做型別檢查，速度快）。
require("ts-node/register/transpile-only");
// 載入要測的 Profiles 類別。
const { Profiles } = require("../src/profiles.ts");

// 先記住真正的 fetch，測試裡會換成假的。
const realFetch = globalThis.fetch;
// 每個測試結束後把 fetch 換回來，避免影響下一個測試。
afterEach(() => {
  globalThis.fetch = realFetch;
});

/** 產生一張指定寬高、單色的 JPEG，回傳檔案內容（Buffer）。 */
const jpeg = (width, height) =>
  // 用 sharp 憑空建立一張圖。
  sharp({
    // create：指定尺寸、3 個色彩通道（RGB）與背景色。
    create: {
      width,
      height,
      channels: 3,
      background: { r: 50, g: 80, b: 100 },
    },
  })
    // 輸出成 JPEG。
    .jpeg()
    // 取得檔案內容。
    .toBuffer();

/**
 * 假的 Prisma 資料庫：只實作真人驗證會用到的方法，並記下每一次呼叫與寫入，讓測試檢查。
 * avatar：photo.findFirst 要回傳的照片（null 代表使用者沒有照片）。
 * avatarLater：第 2 次以後查照片要回傳的照片；用來模擬「呼叫 AI 期間換了大頭貼」。沒給就一直是 avatar。
 * latest：verification.findFirst 要回傳的最近一筆驗證紀錄（null 代表從沒驗證過）。
 * isVerified：user.findUnique 回傳的使用者驗證標記。
 */
function fakeDb({
  avatar = null,
  avatarLater = avatar,
  latest = null,
  isVerified = false,
} = {}) {
  // 記錄區：查詢照片的參數、建立的紀錄、更新紀錄的內容、更新使用者的內容、交易裡拿的鎖。
  const calls = {
    photoQueries: [],
    created: [],
    verification: [],
    user: [],
    locks: 0,
  };
  const db = {
    // 把記錄區也回傳出去，測試才讀得到。
    calls,
    photo: {
      // 查照片：記下查詢條件，第一次回傳 avatar，之後回傳 avatarLater。
      findFirst: async (args) => {
        calls.photoQueries.push(args);
        return calls.photoQueries.length === 1 ? avatar : avatarLater;
      },
    },
    verification: {
      // 建立驗證紀錄：記下內容，回傳固定的紀錄編號。
      create: async (args) => {
        calls.created.push(args.data);
        return { id: "record-1" };
      },
      // 更新驗證紀錄：記下寫入的結果。
      update: async (args) => calls.verification.push(args.data),
      // 查最近一筆驗證紀錄：回傳指定的紀錄。
      findFirst: async () => latest,
    },
    user: {
      // 更新使用者（isVerified）：記下寫入的內容。
      update: async (args) => calls.user.push(args.data),
      // 查使用者的驗證標記。
      findUnique: async () => ({ isVerified }),
    },
    // 交易裡的 pg_advisory_xact_lock：只記次數。
    $executeRaw: async () => {
      calls.locks += 1;
      return 0;
    },
    // 互動式交易：直接把自己當成交易連線（tx）交給回呼函式。
    $transaction: (fn) => fn(db),
  };
  return db;
}

/**
 * 假的基礎設施：次數限制只記下被呼叫的 key（不會真的擋），物件儲存用傳進來的函式取檔，
 * Redis 只實作挑戰會用到的 set 與 getdel。
 * challenge：getdel 要回傳的挑戰內容（物件，會轉成 JSON 字串）；不給代表挑戰不存在或已過期。
 */
function fakeInfra(getObject, { challenge } = {}) {
  // 記下每一次次數限制的 key，測試可以確認有沒有被計算。
  const limits = [];
  // 記下 Redis 的寫入（set）與讀取刪除（getdel）。
  const redisCalls = { set: [], getdel: [] };
  return {
    limits,
    redisCalls,
    // 次數限制：只記錄，不擋。
    limit: async (key) => limits.push(key),
    // 物件儲存：getObject 由各個測試決定要回傳檔案還是丟錯。
    storage: { getObject },
    redis: {
      // 寫入：記下所有參數（key、內容、EX、秒數）。
      set: async (...args) => redisCalls.set.push(args),
      // 讀出並刪除：記下 key，回傳指定的挑戰（JSON 字串）或 null。
      getdel: async (key) => {
        redisCalls.getdel.push(key);
        return challenge ? JSON.stringify(challenge) : null;
      },
    },
  };
}

/** 模擬 multer 解析出的上傳檔案：只需要 buffer 與 size 兩個欄位。 */
const upload = (buffer) => ({ buffer, size: buffer.length });

/** 沒有接模型時 AI 服務的標準回應。 */
const NOT_CONFIGURED = {
  status: "unavailable",
  reasonCode: "MODEL_NOT_CONFIGURED",
  modelName: null,
  modelVersion: null,
};

/** 證據完整的 verified 回應（有模型名稱、版本與兩個分數）。 */
const VERIFIED = {
  status: "verified",
  reasonCode: "VERIFICATION_PASSED",
  modelName: "yunet+sface+minifasnet",
  modelVersion: "policy-3",
  livenessScore: 0.95,
  faceMatchScore: 0.8,
};

/**
 * 假的 AI 服務：把 fetch 換成假的，記下每一次送出的請求內容。
 * response：要回傳的 JSON，預設是「沒有接模型」；status：HTTP 狀態碼，預設 200。
 */
function fakeAi(response = NOT_CONFIGURED, status = 200) {
  // 收集每一次請求的 JSON 內容。
  const bodies = [];
  // 用假的 fetch 取代全域的 fetch。
  globalThis.fetch = async (_url, init) => {
    // 把請求內容解析成物件後存起來。
    bodies.push(JSON.parse(init.body));
    // 回傳指定的回應，內容是 JSON。
    return new Response(JSON.stringify(response), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  };
  // 回傳收集區，測試用它檢查送出了什麼。
  return bodies;
}

/** 用一張 128x128 的自拍呼叫 verifySelfie。 */
async function verify(db, infra) {
  // 產生自拍檔案。
  const selfie = await jpeg(128, 128);
  // 模擬 multer 解析出的上傳檔案：只需要 buffer 與 size 兩個欄位。
  return new Profiles(db, infra).verifySelfie("user-1", {
    buffer: selfie,
    size: selfie.length,
  });
}

// 測試共用的大頭貼資料：照片編號與 storageKey（照片在 MinIO 的路徑）。
const avatar = { id: "photo-1", storageKey: "photos/user-1/avatar.jpg" };

test("只取第一張主照片：主照片優先，再依排序與上傳時間", async () => {
  // 準備一張 1200x900 的照片當大頭貼檔案。
  const stored = await jpeg(1200, 900);
  // 換上假的 AI 服務。
  fakeAi();
  // 使用者有大頭貼。
  const db = fakeDb({ avatar });
  // 物件儲存回傳這張照片。
  await verify(
    db,
    fakeInfra(async () => Readable.from([stored])),
  );
  // 查照片的條件：只查自己還沒刪除的照片，主照片排最前，再依 displayOrder、createdAt。
  assert.deepEqual(db.calls.photoQueries[0], {
    where: { userId: "user-1", deletedAt: null },
    orderBy: [
      { isAvatar: "desc" },
      { displayOrder: "asc" },
      { createdAt: "asc" },
    ],
  });
});

test("大頭貼縮到 800px 內，當作 referenceImages 送給 AI", async () => {
  // 1200x900 的大頭貼，縮完應該是 800x600。
  const stored = await jpeg(1200, 900);
  // 換上假的 AI 服務，拿到請求收集區。
  const bodies = fakeAi();
  // 使用者有大頭貼。
  const db = fakeDb({ avatar });
  // 呼叫驗證。
  await verify(
    db,
    fakeInfra(async () => Readable.from([stored])),
  );
  // 只送出一次請求。
  assert.equal(bodies.length, 1);
  // 參照照片只有一張（第一張主照片）。
  assert.equal(bodies[0].referenceImages.length, 1);
  // 格式是 JPEG。
  assert.equal(bodies[0].referenceImages[0].mimeType, "image/jpeg");
  // 把 base64 還原成檔案，讀出尺寸。
  const meta = await sharp(
    Buffer.from(bodies[0].referenceImages[0].imageBase64, "base64"),
  ).metadata();
  // 等比例縮到長邊 800px。
  assert.deepEqual([meta.width, meta.height], [800, 600]);
  // AI 服務的回應被寫進驗證紀錄。
  assert.equal(db.calls.verification[0].reasonCode, "MODEL_NOT_CONFIGURED");
  // unavailable 不改使用者的 isVerified。
  assert.deepEqual(db.calls.user, []);
});

test("沒有大頭貼：回 409 AVATAR_REQUIRED，不計次數、不建紀錄、不呼叫 AI", async () => {
  // 換上假的 AI 服務。
  const bodies = fakeAi();
  // 使用者一張照片都沒有。
  const db = fakeDb({ avatar: null });
  // 物件儲存不該被用到，被呼叫就讓測試失敗。
  const infra = fakeInfra(async () => assert.fail("不該讀物件儲存"));
  // verifySelfie 應該丟出 409 錯誤，代碼是 AVATAR_REQUIRED。
  await assert.rejects(verify(db, infra), (error) => {
    // HTTP 狀態碼是 409。
    assert.equal(error.getStatus(), 409);
    // 錯誤代碼是 AVATAR_REQUIRED。
    assert.equal(error.getResponse().code, "AVATAR_REQUIRED");
    return true;
  });
  // 沒有用掉驗證次數。
  assert.deepEqual(infra.limits, []);
  // 沒有建立驗證紀錄。
  assert.deepEqual(db.calls.created, []);
  // 沒有呼叫 AI 服務。
  assert.equal(bodies.length, 0);
});

test("大頭貼讀不到時不呼叫 AI，只記 unavailable，不動 isVerified", async () => {
  // 換上假的 AI 服務。
  const bodies = fakeAi();
  // 使用者有大頭貼。
  const db = fakeDb({ avatar });
  // 物件儲存故障：讀檔時丟錯。
  await verify(
    db,
    fakeInfra(async () => {
      throw new Error("storage down");
    }),
  );
  // 沒有呼叫 AI 服務。
  assert.equal(bodies.length, 0);
  // 驗證紀錄是 unavailable。
  assert.equal(db.calls.verification[0].status, "unavailable");
  // 原因代碼寫明是參照照片讀不到。
  assert.equal(
    db.calls.verification[0].reasonCode,
    "REFERENCE_PHOTO_UNAVAILABLE",
  );
  // 不改使用者的 isVerified。
  assert.deepEqual(db.calls.user, []);
});

test("縮圖後短邊不到 64px 的大頭貼不送出，代碼寫明是照片太小", async () => {
  // 上傳時通過 64px 檢查的 1600x100 橫幅照，縮到 800px 後只剩 800x50。
  const stored = await jpeg(1600, 100);
  // 換上假的 AI 服務。
  const bodies = fakeAi();
  // 使用者有大頭貼。
  const db = fakeDb({ avatar });
  // 物件儲存回傳這張橫幅照。
  await verify(
    db,
    fakeInfra(async () => Readable.from([stored])),
  );
  // 沒有呼叫 AI 服務。
  assert.equal(bodies.length, 0);
  // 驗證紀錄是 unavailable。
  assert.equal(db.calls.verification[0].status, "unavailable");
  // 原因代碼寫明是照片太小。
  assert.equal(
    db.calls.verification[0].reasonCode,
    "REFERENCE_PHOTO_TOO_SMALL",
  );
  // 不改使用者的 isVerified。
  assert.deepEqual(db.calls.user, []);
});

test("狀態 API：有大頭貼才 canVerify，沒有時說明缺大頭貼，也不回傳分數", async () => {
  // 不會用到物件儲存。
  const infra = fakeInfra(async () => assert.fail("不該讀物件儲存"));
  // 情況一：沒有照片、沒有紀錄 → 請使用者先上傳大頭貼。
  assert.deepEqual(await new Profiles(fakeDb(), infra).verification("user-1"), {
    status: "not_started",
    reasonCode: "AVATAR_REQUIRED",
    canVerify: false,
  });
  // 情況二：有大頭貼、沒有紀錄 → 可以開始驗證。
  assert.deepEqual(
    await new Profiles(fakeDb({ avatar }), infra).verification("user-1"),
    { status: "not_started", canVerify: true },
  );
  // 情況三：有大頭貼、也有一筆驗證紀錄（紀錄裡有分數）。
  const createdAt = new Date("2026-09-24T00:00:00Z");
  const latest = {
    id: "record-1",
    status: "unavailable",
    reasonCode: "LIVE_CAPTURE_REQUIRED",
    modelName: "yunet+sface+minifasnet",
    modelVersion: "policy-1",
    livenessScore: 0.9,
    faceMatchScore: 0.8,
    createdAt,
  };
  // 回傳紀錄內容加上 canVerify，兩個分數不在回傳內容裡。
  assert.deepEqual(
    await new Profiles(fakeDb({ avatar, latest }), infra).verification(
      "user-1",
    ),
    {
      id: "record-1",
      status: "unavailable",
      reasonCode: "LIVE_CAPTURE_REQUIRED",
      modelName: "yunet+sface+minifasnet",
      modelVersion: "policy-1",
      createdAt,
      canVerify: true,
    },
  );
});

test("重新驗證 API：沒有大頭貼回 AVATAR_REQUIRED，有大頭貼回 LIVE_CAPTURE_REQUIRED", async () => {
  // 不會用到物件儲存。
  const infra = fakeInfra(async () => assert.fail("不該讀物件儲存"));
  // 沒有大頭貼：還不能驗證。
  assert.deepEqual(await new Profiles(fakeDb(), infra).retry("user-1"), {
    status: "not_started",
    reasonCode: "AVATAR_REQUIRED",
    canVerify: false,
  });
  // 有大頭貼：請使用者用鏡頭重新驗證（上傳自拍無法通過）。
  assert.deepEqual(
    await new Profiles(fakeDb({ avatar }), infra).retry("user-1"),
    {
      status: "not_started",
      reasonCode: "LIVE_CAPTURE_REQUIRED",
      canVerify: true,
    },
  );
  // 不設每小時上限：不計算重新驗證的次數。
  assert.deepEqual(infra.limits, []);
});

// 四種動作挑戰；必須和 profiles.ts 的 challengeActions 相同。
const ACTIONS = ["turn_left", "turn_right", "look_up", "look_down"];
// 測試用的挑戰編號（UUID 格式）。
const CHALLENGE_ID = "5f0c3a2e-8d4b-4c61-9a57-2b7e1f6d9c30";

/** 期待呼叫會丟出指定狀態碼與代碼的錯誤。 */
async function rejectsWith(promise, status, code) {
  await assert.rejects(promise, (error) => {
    // HTTP 狀態碼相符。
    assert.equal(error.getStatus(), status);
    // 錯誤代碼相符。
    assert.equal(error.getResponse().code, code);
    return true;
  });
}

test("領挑戰：沒有大頭貼回 409，不計次數、不寫 Redis", async () => {
  // 使用者沒有照片。
  const infra = fakeInfra(async () => assert.fail("不該讀物件儲存"));
  // 應該回 409 AVATAR_REQUIRED。
  await rejectsWith(
    new Profiles(fakeDb(), infra).createChallenge("user-1"),
    409,
    "AVATAR_REQUIRED",
  );
  // 沒有用掉領挑戰的次數。
  assert.deepEqual(infra.limits, []);
  // 沒有寫入 Redis。
  assert.deepEqual(infra.redisCalls.set, []);
});

test("領挑戰：隨機 2 個不同動作，存進 Redis 並設定 120 秒後失效", async () => {
  // 使用者有大頭貼。
  const infra = fakeInfra(async () => assert.fail("不該讀物件儲存"));
  // 領一份挑戰。
  const result = await new Profiles(fakeDb({ avatar }), infra).createChallenge(
    "user-1",
  );
  // 挑戰編號是 UUID。
  assert.match(result.challengeId, /^[0-9a-f-]{36}$/);
  // 有 2 個動作，而且不重複。
  assert.equal(new Set(result.actions).size, 2);
  // 每個動作都是四種之一。
  assert.ok(result.actions.every((action) => ACTIONS.includes(action)));
  // 有效秒數是 120。
  assert.equal(result.expiresInSeconds, 120);
  // 不設每小時上限：不計算領挑戰的次數。
  assert.deepEqual(infra.limits, []);
  // Redis 的寫入：key、內容（誰的挑戰、哪些動作）、EX、120 秒。
  assert.deepEqual(infra.redisCalls.set, [
    [
      `verify-challenge:${result.challengeId}`,
      JSON.stringify({ userId: "user-1", actions: result.actions }),
      "EX",
      120,
    ],
  ]);
});

test("即時驗證：正面影格當自拍、動作影格依挑戰順序放進 liveCapture", async () => {
  // 大頭貼。
  const stored = await jpeg(1200, 900);
  // 三張尺寸不同的影格，才分得出哪一張被放到哪裡：縮到 1024px 內後分別是 1024x768、768x1024、800x800。
  const frames = await Promise.all([
    jpeg(1200, 900),
    jpeg(900, 1200),
    jpeg(800, 800),
  ]);
  // 換上假的 AI 服務。
  const bodies = fakeAi();
  // 使用者有大頭貼。
  const db = fakeDb({ avatar });
  // Redis 裡有這位使用者的挑戰：先向左轉，再抬頭。
  const infra = fakeInfra(async () => Readable.from([stored]), {
    challenge: { userId: "user-1", actions: ["turn_left", "look_up"] },
  });
  // 送出正面 + 2 個動作，共 3 張影格。
  await new Profiles(db, infra).verifyLive("user-1", frames.map(upload), {
    challengeId: CHALLENGE_ID,
  });
  // 挑戰被讀出並刪除（只能用一次）。
  assert.deepEqual(infra.redisCalls.getdel, [
    `verify-challenge:${CHALLENGE_ID}`,
  ]);
  // 不設每小時上限：不計算驗證次數。
  assert.deepEqual(infra.limits, []);
  // 送出一次請求。
  assert.equal(bodies.length, 1);
  // 動作影格的動作與挑戰順序相同。
  assert.deepEqual(
    bodies[0].liveCapture.frames.map((f) => f.action),
    ["turn_left", "look_up"],
  );
  // 請求裡帶著挑戰編號。
  assert.equal(bodies[0].liveCapture.challengeId, CHALLENGE_ID);
  // 參照照片只有一張（第一張主照片）。
  assert.equal(bodies[0].referenceImages.length, 1);
  // 讀出一張 base64 影像的尺寸。
  const size = async (base64) => {
    const meta = await sharp(Buffer.from(base64, "base64")).metadata();
    return [meta.width, meta.height];
  };
  // 第 1 張上傳的影格是正面影格（當自拍），縮到 1024px 內。
  assert.deepEqual(await size(bodies[0].imageBase64), [1024, 768]);
  // 第 2、3 張依序是 turn_left、look_up 的動作影格。
  assert.deepEqual(
    await Promise.all(
      bodies[0].liveCapture.frames.map((f) => size(f.imageBase64)),
    ),
    [
      [768, 1024],
      [800, 800],
    ],
  );
  // AI 服務的回應被寫進驗證紀錄。
  assert.equal(db.calls.verification[0].reasonCode, "MODEL_NOT_CONFIGURED");
});

test("即時驗證：挑戰不存在、過期或不是自己的，回 409，不建紀錄也不呼叫 AI", async () => {
  // 影格。
  const frame = await jpeg(640, 480);
  // 三張影格。
  const frames = [upload(frame), upload(frame), upload(frame)];
  for (const challenge of [
    // 挑戰不存在（過期或已用過）。
    undefined,
    // 別人的挑戰。
    { userId: "someone-else", actions: ["turn_left", "look_up"] },
  ]) {
    // 換上假的 AI 服務。
    const bodies = fakeAi();
    // 使用者有大頭貼。
    const db = fakeDb({ avatar });
    // 應該回 409 CHALLENGE_EXPIRED。
    await rejectsWith(
      new Profiles(
        db,
        fakeInfra(async () => null, { challenge }),
      ).verifyLive("user-1", frames, { challengeId: CHALLENGE_ID }),
      409,
      "CHALLENGE_EXPIRED",
    );
    // 沒有建立驗證紀錄。
    assert.deepEqual(db.calls.created, []);
    // 沒有呼叫 AI 服務。
    assert.equal(bodies.length, 0);
  }
});

test("即時驗證：缺挑戰編號回 400，影格數量不對回 400，沒有大頭貼回 409", async () => {
  // 影格。
  const frame = await jpeg(640, 480);
  // 有效的挑戰內容。
  const challenge = { userId: "user-1", actions: ["turn_left", "look_up"] };
  // 缺少挑戰編號（或格式不是 UUID）。
  await rejectsWith(
    new Profiles(
      fakeDb({ avatar }),
      fakeInfra(async () => null, { challenge }),
    ).verifyLive("user-1", [upload(frame)], { challengeId: "not-a-uuid" }),
    400,
    "CHALLENGE_REQUIRED",
  );
  // 挑戰要 2 個動作，卻只送了 2 張影格（應該是 3 張）。
  await rejectsWith(
    new Profiles(
      fakeDb({ avatar }),
      fakeInfra(async () => null, { challenge }),
    ).verifyLive("user-1", [upload(frame), upload(frame)], {
      challengeId: CHALLENGE_ID,
    }),
    400,
    "INVALID_FRAMES",
  );
  // 送太多張（4 張）或完全沒有影格，同樣回 400。
  for (const files of [
    [upload(frame), upload(frame), upload(frame), upload(frame)],
    undefined,
  ])
    await rejectsWith(
      new Profiles(
        fakeDb({ avatar }),
        fakeInfra(async () => null, { challenge }),
      ).verifyLive("user-1", files, { challengeId: CHALLENGE_ID }),
      400,
      "INVALID_FRAMES",
    );
  // 沒有大頭貼：在次數限制之前就回 409。
  const infra = fakeInfra(async () => null, { challenge });
  await rejectsWith(
    new Profiles(fakeDb(), infra).verifyLive("user-1", [upload(frame)], {
      challengeId: CHALLENGE_ID,
    }),
    409,
    "AVATAR_REQUIRED",
  );
  // 沒有用掉驗證次數，也沒有動到挑戰。
  assert.deepEqual(infra.limits, []);
  assert.deepEqual(infra.redisCalls.getdel, []);
});

/** 用 3 張影格跑一次即時驗證（挑戰：向左轉、抬頭），回傳假資料庫。 */
async function live(response, dbOptions = {}, status = 200) {
  // 大頭貼與影格。
  const stored = await jpeg(640, 480);
  const frame = await jpeg(640, 480);
  // 換上假的 AI 服務，回傳指定的結果。
  fakeAi(response, status);
  // 使用者有大頭貼。
  const db = fakeDb({ avatar, ...dbOptions });
  // Redis 裡有這位使用者的挑戰。
  const infra = fakeInfra(async () => Readable.from([stored]), {
    challenge: { userId: "user-1", actions: ["turn_left", "look_up"] },
  });
  // 送出 3 張影格。
  await new Profiles(db, infra).verifyLive(
    "user-1",
    [upload(frame), upload(frame), upload(frame)],
    { challengeId: CHALLENGE_ID },
  );
  return db;
}

test("即時驗證：證據完整的 verified 才設 isVerified=true；rejected 設回 false", async () => {
  // AI 回 verified，而且模型名稱、版本、兩個分數都有。
  const passed = await live(VERIFIED);
  // 紀錄寫入 verified。
  assert.equal(passed.calls.verification[0].status, "verified");
  // 使用者被標成已驗證。
  assert.deepEqual(passed.calls.user, [{ isVerified: true }]);
  // 寫入前有拿使用者鎖。
  assert.equal(passed.calls.locks, 1);
  // AI 回 rejected：取消已驗證。
  const rejected = await live({
    ...VERIFIED,
    status: "rejected",
    reasonCode: "FACE_MISMATCH",
  });
  // 使用者的已驗證標記被取消。
  assert.deepEqual(rejected.calls.user, [{ isVerified: false }]);
});

test("即時驗證：verified 但缺分數，改記 INVALID_PROVIDER_RESPONSE，不動 isVerified", async () => {
  // AI 回 verified，卻沒有活體分數。
  const db = await live({ ...VERIFIED, livenessScore: null });
  // 紀錄改成 unavailable／INVALID_PROVIDER_RESPONSE。
  assert.equal(db.calls.verification[0].status, "unavailable");
  assert.equal(
    db.calls.verification[0].reasonCode,
    "INVALID_PROVIDER_RESPONSE",
  );
  // 沒有改使用者的驗證標記。
  assert.deepEqual(db.calls.user, []);
});

test("即時驗證：呼叫 AI 期間換了大頭貼，結果不採信，記 AVATAR_CHANGED", async () => {
  // 第 2 次查照片（交易裡重新確認）時，大頭貼已經換成另一張。
  const db = await live(VERIFIED, {
    avatarLater: { id: "photo-2", storageKey: "photos/user-1/other.jpg" },
  });
  // 紀錄是 unavailable／AVATAR_CHANGED。
  assert.equal(db.calls.verification[0].status, "unavailable");
  assert.equal(db.calls.verification[0].reasonCode, "AVATAR_CHANGED");
  // 沒有把使用者標成已驗證。
  assert.deepEqual(db.calls.user, []);
});

test("即時驗證：AI 服務拒收影像（400）時記下它的錯誤碼，不當成服務故障", async () => {
  // AI 服務回 400，錯誤碼標出是第幾類影像的問題。
  const db = await live(
    { code: "FRAME_IMAGE_TOO_LARGE", message: "影像格式、內容或尺寸不符要求" },
    {},
    400,
  );
  // 紀錄是 unavailable 加上 AI 服務的錯誤碼。
  assert.equal(db.calls.verification[0].status, "unavailable");
  assert.equal(db.calls.verification[0].reasonCode, "FRAME_IMAGE_TOO_LARGE");
  // 沒有改使用者的驗證標記。
  assert.deepEqual(db.calls.user, []);
  // 錯誤碼格式不對（例如夾帶任意文字）時不採用，維持 AI_SERVICE_UNAVAILABLE。
  const odd = await live({ code: "bad code <script>" }, {}, 400);
  assert.equal(odd.calls.verification[0].reasonCode, "AI_SERVICE_UNAVAILABLE");
});

test("狀態 API：通過驗證後換了大頭貼（isVerified 已取消），回 AVATAR_CHANGED", async () => {
  // 最近一筆是 verified，但使用者的驗證標記已經被取消。
  const latest = {
    id: "record-1",
    status: "verified",
    reasonCode: "VERIFICATION_PASSED",
    modelName: "yunet+sface+minifasnet",
    modelVersion: "policy-3",
    createdAt: new Date("2026-09-24T00:00:00Z"),
  };
  const infra = fakeInfra(async () => assert.fail("不該讀物件儲存"));
  // 不能再顯示成 verified。
  assert.deepEqual(
    await new Profiles(
      fakeDb({ avatar, latest, isVerified: false }),
      infra,
    ).verification("user-1"),
    { status: "not_started", reasonCode: "AVATAR_CHANGED", canVerify: true },
  );
  // 標記還在時照常回傳紀錄。
  assert.equal(
    (
      await new Profiles(
        fakeDb({ avatar, latest, isVerified: true }),
        infra,
      ).verification("user-1")
    ).status,
    "verified",
  );
});

test("刪掉大頭貼（軟刪除）會取消驗證標記；刪其他照片不會", async () => {
  // 照片資料：一張大頭貼、一張一般照片。
  const photos = {
    avatar: { id: "11111111-1111-4111-8111-111111111111", isAvatar: true },
    other: { id: "22222222-2222-4222-8222-222222222222", isAvatar: false },
  };
  for (const [name, photo] of Object.entries(photos)) {
    // 使用者更新與照片更新的記錄。
    const userUpdates = [];
    const photoUpdates = [];
    // 假的資料庫：查得到這張照片，刪除後還剩一張可以遞補。
    const db = {
      photo: {
        findFirst: async () => ({ ...photo, storageKey: `photos/${name}.jpg` }),
        update: async (args) => photoUpdates.push(args),
      },
      user: { update: async (args) => userUpdates.push(args.data) },
      $executeRaw: async () => 0,
      $transaction: (fn) => fn(db),
    };
    await new Profiles(
      db,
      fakeInfra(async () => null),
    ).removePhoto("user-1", photo.id);
    // 刪除是軟刪除：寫入 deletedAt 並取消主照片。
    assert.equal(photoUpdates[0].where.id, photo.id);
    assert.ok(photoUpdates[0].data.deletedAt instanceof Date);
    // 只有刪大頭貼時才取消驗證標記。
    assert.deepEqual(
      userUpdates,
      name === "avatar" ? [{ isVerified: false }] : [],
    );
  }
});
