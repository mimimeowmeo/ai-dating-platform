import { Injectable } from "@nestjs/common";
import {
  createHmac,
  randomInt,
  randomUUID,
  timingSafeEqual,
} from "node:crypto";
import type { Prisma } from "@prisma/client";
import sharp, { type Metadata, type Sharp } from "sharp";
import { z } from "zod";
import {
  Database,
  Infrastructure,
  config,
  fail,
  parse,
  ageAt,
  catalogs,
  uuid,
} from "./core";
const gender = z.enum(["woman", "man", "nonbinary"]);
// 送給 AI 服務的身分參照影像：base64 編碼的 JPEG（使用者的第一張主照片，縮圖後）。
type ReferenceImage = { imageBase64: string; mimeType: "image/jpeg" };
// 即時鏡頭動作挑戰可以出的動作；必須和 services/face/app/policy.py 的 CHALLENGE_ACTIONS 相同。
// 方向以使用者自己為準：turn_left 是往自己的左邊轉頭，look_up 是抬頭。
const challengeActions = [
  "turn_left",
  "turn_right",
  "look_up",
  "look_down",
] as const;
// 單一動作的型別：上面四個字串之一。
type ChallengeAction = (typeof challengeActions)[number];
// 一次挑戰要做幾個動作。
const CHALLENGE_ACTION_COUNT = 2;
// 每位使用者每小時最多送出幾次真人驗證（上傳自拍與即時鏡頭共用）。
const VERIFY_PER_HOUR = 5;
// 挑戰的有效秒數：從領到挑戰到送出影格，超過就要重新開始。
const CHALLENGE_TTL_SECONDS = 120;
// AI 服務對每張動作影格的上限（1 MiB），和 services/ai 的 MAX_FRAME_BYTES 相同。
const MAX_FRAME_BYTES = 1024 * 1024;
// 挑戰在 Redis 裡的 key。
const challengeKey = (challengeId: string) => `verify-challenge:${challengeId}`;
// 解析 Redis 裡存的挑戰；格式不對（例如被手動改過）就回傳 null，當作沒有挑戰。
function parseChallenge(stored: string) {
  try {
    // 用 zod 檢查內容：使用者編號與動作清單。
    const parsed = z
      .object({
        // 領取挑戰的使用者。
        userId: z.string(),
        // 動作清單，只能是四種動作之一，至少一個。
        actions: z.array(z.enum(challengeActions)).min(1),
      })
      // safeParse 不會丟錯。
      .safeParse(JSON.parse(stored));
    // 格式正確就回傳內容，否則 null。
    return parsed.success ? parsed.data : null;
  } catch {
    // 不是合法的 JSON：當作沒有挑戰。
    return null;
  }
}
// 送給 AI 服務的影像內容：自拍（或正面影格），以及即時鏡頭時的動作影格。
type VerificationCapture = {
  // 自拍或正面影格，base64 編碼的 JPEG。
  imageBase64: string;
  // 格式固定是 JPEG。
  mimeType: "image/jpeg";
  // 即時鏡頭的動作挑戰；上傳自拍時沒有這個欄位。
  liveCapture?: {
    // 挑戰編號。
    challengeId: string;
    // 每個動作一張影格。
    frames: {
      action: ChallengeAction;
      imageBase64: string;
      mimeType: "image/jpeg";
    }[];
  };
};
const intent = z.enum(["serious", "casual", "friendship"]);
const tags = (category: keyof typeof catalogs) =>
  z
    .array(
      z
        .string()
        .refine((v) => catalogs[category].includes(v), "請選擇清單中的標籤"),
    )
    .max(8)
    .transform((v) => [...new Set(v)]);
// 建檔與編輯都要填齊：身高、自我介紹、想遇見的關係 1～2 項，以及下面各類小熱愛的最低數量。
const bioMinLength = 20;
// 以使用者看到的字（grapheme）計算：❤️、👍🏻、國旗都算一個字，跟畫面上的計數一致。
const graphemes = new Intl.Segmenter("zh-Hant", { granularity: "grapheme" });
const charCount = (text: string) => [...graphemes.segment(text)].length;
// 探索偏好的身高拉桿範圍；停在兩端代表不限（見 Social.eligible）。
export const preferenceHeightRange = [130, 250] as const;
const datingGoalMessage = "想遇見的關係請選 1～2 項";
const traitMinimums: Record<string, { title: string; min: number }> = {
  personality: { title: "個性", min: 1 },
  diet: { title: "飲食", min: 1 },
  value: { title: "價值觀", min: 1 },
  lifestyle: { title: "生活型態", min: 1 },
  interest: { title: "興趣", min: 3 },
};
const profileInput = z
  .object({
    displayName: z.string().trim().min(1).max(40),
    birthDate: z
      .string()
      .regex(/^\d{4}-\d{2}-\d{2}$/)
      .refine((v) => {
        const d = new Date(v);
        return (
          Number.isFinite(+d) &&
          d.toISOString().slice(0, 10) === v &&
          ageAt(d) >= 18 &&
          ageAt(d) <= 120
        );
      }, "請填入有效生日，須年滿 18 歲"),
    gender,
    bio: z
      .string()
      .trim()
      .max(1000)
      .refine(
        (v) => charCount(v) >= bioMinLength,
        `自我介紹至少要 ${bioMinLength} 個字`,
      ),
    city: z.string().trim().min(1).max(80),
    latitude: z.number().min(-90).max(90),
    longitude: z.number().min(-180).max(180),
    heightCm: z.number().int().min(100).max(250),
    // traits 資料表的代碼：dating_goal 放 datingGoals，其餘類別放 traits。
    traits: z.array(z.string().trim().min(1)).max(40),
    datingGoals: z
      .array(z.string().trim().min(1))
      .min(1, datingGoalMessage)
      .max(2, datingGoalMessage),
    // 以下為舊欄位，仍接受（匯入資料與既有測試會送），但畫面已改用 traits。
    datingIntent: intent.optional(),
    occupation: z.string().max(80).nullable().optional(),
    education: z.string().max(80).nullable().optional(),
    interests: tags("interests").optional(),
    hobbies: tags("hobbies").optional(),
    foods: tags("foods").optional(),
  })
  .strict();
const preferencesInput = z
  .object({
    minAge: z.number().int().min(18).max(130),
    maxAge: z.number().int().min(18).max(130),
    preferredGender: z.enum(["woman", "man", "nonbinary", "any"]),
    maxDistanceKm: z.number().int().min(1).max(20000),
    // 沒送就是拉桿的兩端（不限）；舊版前端與匯入腳本不會帶這兩個欄位。
    minHeightCm: z
      .number()
      .int()
      .min(preferenceHeightRange[0])
      .max(preferenceHeightRange[1])
      .default(preferenceHeightRange[0]),
    maxHeightCm: z
      .number()
      .int()
      .min(preferenceHeightRange[0])
      .max(preferenceHeightRange[1])
      .default(preferenceHeightRange[1]),
    // "any" 或 traits 表裡 dating_goal 的代碼，實際值在 savePreferences 檢查。
    preferredDatingIntent: z.string().trim().min(1),
  })
  .strict()
  .refine((v) => v.minAge <= v.maxAge, "年齡下限不得大於上限")
  .refine((v) => v.minHeightCm <= v.maxHeightCm, "身高下限不得大於上限");
// <img> 不會帶 Authorization header，所以照片網址改用短效簽章，並綁定觀看者。
const MEDIA_WINDOW_MS = 6 * 60 * 60 * 1000;
const mediaSignature = (photoId: string, viewerId: string, expires: number) =>
  createHmac("sha256", config.JWT_SECRET)
    .update(`${photoId}.${viewerId}.${expires}`)
    .digest("hex")
    .slice(0, 32);
export const photoView = (p: any, viewerId: string) => {
  // 對齊到固定時段，同一段時間內網址不變，瀏覽器才快取得住。
  const expires =
    Math.ceil((Date.now() + MEDIA_WINDOW_MS) / MEDIA_WINDOW_MS) *
    MEDIA_WINDOW_MS;
  const signature = mediaSignature(p.id, viewerId, expires);
  return {
    id: p.id,
    url: `/api/v1/media/${p.id}?u=${viewerId}&e=${expires}&s=${signature}`,
    isAvatar: p.isAvatar,
    displayOrder: p.displayOrder,
  };
};
// user_traits 讀出來的選擇，依類別拆成「交友目標」與「其他喜好」。
export const traitCodes = (user: any, datingGoal: boolean) =>
  (user?.traits || [])
    .filter((t: any) => (t.trait.category === "dating_goal") === datingGoal)
    .map((t: any) => t.trait.code);
export function card(user: any, viewerId: string) {
  if (!user?.profile) return null;
  const p = user.profile;
  return {
    userId: user.id,
    displayName: p.displayName,
    age: ageAt(p.birthDate),
    gender: p.gender,
    bio: p.bio,
    city: p.city,
    heightCm: p.heightCm ?? null,
    datingIntent: p.datingIntent,
    interests: p.interests,
    hobbies: p.hobbies,
    foods: p.foods,
    photos: (user.photos || []).map((p: any) => photoView(p, viewerId)),
    isVerified: user.isVerified,
    traits: traitCodes(user, false),
    datingGoals: traitCodes(user, true),
  };
}
export const userInclude = {
  profile: true,
  preference: true,
  // 軟刪除的照片留在資料表裡，但所有畫面與 API 都不再出現。
  photos: {
    where: { deletedAt: null },
    orderBy: [{ displayOrder: "asc" as const }, { createdAt: "asc" as const }],
  },
  traits: {
    include: { trait: true },
    orderBy: { traitId: "asc" as const },
  },
};
@Injectable()
export class Profiles {
  constructor(
    private db: Database,
    private infra: Infrastructure,
  ) {}
  async mine(id: string) {
    const user = await this.db.user.findUniqueOrThrow({
      where: { id },
      include: userInclude,
    });
    return user.profile
      ? {
          ...user.profile,
          birthDate: user.profile.birthDate.toISOString().slice(0, 10),
          photos: user.photos.map((p: any) => photoView(p, id)),
          traits: traitCodes(user, false),
          datingGoals: traitCodes(user, true),
        }
      : null;
  }
  async save(id: string, body: unknown) {
    const { traits, datingGoals, ...dto } = parse(profileInput, body);
    const data = { ...dto, birthDate: new Date(dto.birthDate) };
    const selected = await this.traitIds(traits, datingGoals);
    await this.db.$transaction(async (tx) => {
      await tx.profile.upsert({
        where: { userId: id },
        create: { userId: id, ...data },
        update: data,
      });
      await tx.userTrait.deleteMany({ where: { userId: id } });
      await tx.userTrait.createMany({
        data: selected.map((traitId) => ({ userId: id, traitId })),
      });
    });
    return this.mine(id);
  }
  // 只接受 traits 表裡的代碼，確認類別放對位置，且各類都選到最低數量。
  private async traitIds(traits: string[], goals: string[]) {
    const wanted = [...new Set([...traits, ...goals])];
    const rows = wanted.length
      ? await this.db.trait.findMany({ where: { code: { in: wanted } } })
      : [];
    const byCode = new Map(rows.map((t) => [t.code, t]));
    const ids: number[] = [];
    for (const [codes, datingGoal] of [
      [traits, false],
      [goals, true],
    ] as const)
      for (const code of new Set(codes)) {
        const trait = byCode.get(code);
        if (!trait || (trait.category === "dating_goal") !== datingGoal)
          return fail(400, "INVALID_TRAIT", "請選擇清單中的選項。");
        ids.push(trait.id);
      }
    const counts = new Map<string, number>();
    for (const code of new Set(traits)) {
      const category = byCode.get(code)!.category;
      counts.set(category, (counts.get(category) ?? 0) + 1);
    }
    const missing = Object.entries(traitMinimums)
      .filter(([category, rule]) => (counts.get(category) ?? 0) < rule.min)
      .map(([, rule]) => `${rule.title}至少選 ${rule.min} 項`);
    if (missing.length)
      return fail(400, "TRAITS_REQUIRED", `${missing.join("、")}。`);
    return ids;
  }
  // 選項清單（畫面用來顯示標籤與可選項目）。
  async traitCatalog() {
    const rows = await this.db.trait.findMany({ orderBy: { id: "asc" } });
    return rows.map((t) => ({
      category: t.category,
      code: t.code,
      label: t.labelZh || t.code,
    }));
  }
  async blocked(a: string, b: string) {
    return !!(await this.db.block.findFirst({
      where: {
        OR: [
          { userId: a, blockedUserId: b },
          { userId: b, blockedUserId: a },
        ],
      },
    }));
  }
  async public(id: string, target: string) {
    uuid(target);
    if (await this.blocked(id, target))
      return fail(404, "NOT_FOUND", "找不到此個人檔案。");
    const u = await this.db.user.findUnique({
      where: { id: target },
      include: userInclude,
    });
    if (!u?.profile) return fail(404, "NOT_FOUND", "找不到此個人檔案。");
    return card(u, id);
  }
  async preferences(id: string) {
    return this.db.preference.findUniqueOrThrow({ where: { userId: id } });
  }
  async savePreferences(id: string, body: unknown) {
    const data = parse(preferencesInput, body);
    if (data.preferredDatingIntent !== "any") {
      const goal = await this.db.trait.findFirst({
        where: { category: "dating_goal", code: data.preferredDatingIntent },
      });
      if (!goal) return fail(400, "INVALID_TRAIT", "請選擇清單中的關係期待。");
    }
    return this.db.preference.upsert({
      where: { userId: id },
      create: { userId: id, ...data },
      update: data,
    });
  }
  async setTags(id: string, category: keyof typeof catalogs, body: unknown) {
    const dto = parse(z.object({ values: tags(category) }).strict(), body);
    if (!(await this.db.profile.findUnique({ where: { userId: id } })))
      return fail(409, "PROFILE_REQUIRED", "請先完成個人資料。");
    const p = await this.db.profile.update({
      where: { userId: id },
      data: { [category]: dto.values },
    });
    return p[category];
  }
  async image(file: Express.Multer.File | undefined) {
    if (!file || file.size > 8 * 1024 * 1024)
      return fail(
        400,
        "INVALID_IMAGE",
        "請選擇 8 MB 以內的 JPEG、PNG 或 WebP 圖片。",
      );
    let input: Sharp;
    let meta: Metadata;
    try {
      input = sharp(file.buffer, {
        limitInputPixels: 24_000_000,
        animated: false,
      });
      meta = await input.metadata();
    } catch {
      return fail(
        400,
        "INVALID_IMAGE",
        "無法讀取圖片，請重新選擇 JPEG、PNG 或 WebP。",
      );
    }
    if (
      !["jpeg", "png", "webp"].includes(meta.format || "") ||
      (meta.pages || 1) > 1
    )
      return fail(400, "INVALID_IMAGE", "只支援 JPEG、PNG 或 WebP 靜態圖片。");
    if (!meta.width || !meta.height || meta.width < 64 || meta.height < 64)
      return fail(400, "INVALID_IMAGE", "圖片太小，長和寬都至少需要 64 像素。");
    if (meta.width * meta.height > 24_000_000)
      return fail(
        400,
        "INVALID_IMAGE",
        "圖片解析度太高，請選擇 2400 萬像素以內的圖片。",
      );
    try {
      return await input
        .rotate()
        .resize({
          width: 1600,
          height: 1600,
          fit: "inside",
          withoutEnlargement: true,
        })
        .jpeg({ quality: 88 })
        .toBuffer();
    } catch {
      return fail(
        400,
        "INVALID_IMAGE",
        "無法讀取圖片，請重新選擇 JPEG、PNG 或 WebP。",
      );
    }
  }
  async upload(id: string, file: Express.Multer.File | undefined) {
    await this.infra.limit(`upload:${id}`, 20, 3600);
    if (!(await this.db.profile.findUnique({ where: { userId: id } })))
      return fail(409, "PROFILE_REQUIRED", "請先儲存個人檔案，再加入照片。");
    const buffer = await this.image(file);
    const key = `photos/${id}/${randomUUID()}.jpg`;
    await this.infra.storage.putObject(
      config.S3_BUCKET,
      key,
      buffer,
      buffer.length,
      { "Content-Type": "image/jpeg" },
    );
    try {
      const photo = await this.db.$transaction(async (tx) => {
        await tx.$executeRaw`SELECT pg_advisory_xact_lock(hashtext(${id}))`;
        const photos = await tx.photo.findMany({ where: { userId: id } });
        // 上限與主照片只算還在的照片；排序號碼看全部，才不會跟軟刪除的舊照片撞號。
        const active = photos.filter((p) => !p.deletedAt);
        if (active.length >= 6)
          return fail(400, "PHOTO_LIMIT", "最多可上傳 6 張照片。");
        return tx.photo.create({
          data: {
            userId: id,
            storageKey: key,
            mimeType: "image/jpeg",
            isAvatar: active.length === 0,
            displayOrder:
              Math.max(-1, ...photos.map((p) => p.displayOrder)) + 1,
          },
        });
      });
      return photoView(photo, id);
    } catch (error) {
      await this.infra.storage.removeObject(config.S3_BUCKET, key);
      throw error;
    }
  }
  // 刪除照片（含刪掉主照片、改由下一張接替的「替換主照片」）一律是軟刪除：
  // 只寫 deleted_at 並取消主照片，照片記錄與 MinIO 物件都保留。
  // 取消 isAvatar 之後，任何用 isAvatar 找主照片的地方都不會拿到已刪除的照片。
  async removePhoto(id: string, photoId: string) {
    uuid(photoId);
    await this.db.$transaction(async (tx) => {
      await tx.$executeRaw`SELECT pg_advisory_xact_lock(hashtext(${id}))`;
      // 在鎖裡才查：同一張照片同時刪兩次時，第二次回 404 而不是重複標記。
      const photo = await tx.photo.findFirst({
        where: { id: photoId, userId: id, deletedAt: null },
      });
      if (!photo) return fail(404, "NOT_FOUND", "找不到照片。");
      await tx.photo.update({
        where: { id: photoId },
        data: { deletedAt: new Date(), isAvatar: false },
      });
      if (photo.isAvatar) {
        // 真人驗證比對的是這張大頭貼；換掉之後驗證標記就不再成立，要重新驗證。
        await tx.user.update({ where: { id }, data: { isVerified: false } });
        const first = await tx.photo.findFirst({
          where: { userId: id, deletedAt: null },
          orderBy: [{ displayOrder: "asc" }, { createdAt: "asc" }],
        });
        if (first)
          await tx.photo.update({
            where: { id: first.id },
            data: { isAvatar: true },
          });
      }
    });
    return { ok: true };
  }
  async media(
    photoId: string,
    query: { u?: string; e?: string; s?: string } = {},
  ) {
    uuid(photoId);
    const viewerId = query.u ?? "";
    const expires = Number(query.e);
    const signature = query.s ?? "";
    const expected = Number.isFinite(expires)
      ? mediaSignature(photoId, viewerId, expires)
      : "";
    const matches =
      expected.length > 0 &&
      signature.length === expected.length &&
      timingSafeEqual(Buffer.from(signature), Buffer.from(expected));
    if (!matches || expires < Date.now())
      return fail(403, "FORBIDDEN", "照片連結已失效，請重新整理頁面。");
    const photo = await this.db.photo.findUnique({ where: { id: photoId } });
    // 軟刪除的照片物件還在 MinIO，但舊網址（例如別人頁面的快取）一律當作不存在。
    if (!photo || photo.deletedAt)
      return fail(404, "NOT_FOUND", "找不到照片。");
    // 封鎖之後就看不到對方的照片。
    if (
      photo.userId !== viewerId &&
      (await this.blocked(viewerId, photo.userId))
    )
      return fail(404, "NOT_FOUND", "找不到照片。");
    return this.infra.storage.getObject(config.S3_BUCKET, photo.storageKey);
  }
  /**
   * 找出這位使用者做真人驗證時要比對的「第一張主照片」（大頭貼）。
   *
   * 挑選規則（依序）：
   * 1. 標記為主照片（isAvatar = true）的照片優先。第一張上傳的照片會自動成為主照片；
   *    主照片被刪掉時，排在最前面的下一張會遞補（見 removePhoto）。
   * 2. 同樣是主照片、或都不是主照片時，依使用者排的順序（displayOrder）取最前面的一張。
   * 3. 順序也一樣時，取最早上傳的一張，讓結果永遠固定。
   *
   * 第 2、3 條是保險：資料庫沒有「每人只能有一張主照片」的限制，匯入的舊資料也可能沒有標主照片。
   * 使用者一張照片都沒有時回傳 null，呼叫端據此擋下真人驗證。
   *
   * db：預設用一般連線；runVerification 在交易裡重新確認大頭貼時傳入交易用的連線（tx）。
   */
  async avatarPhoto(id: string, db: Prisma.TransactionClient = this.db) {
    // 只查這位使用者自己的照片；id 來自登入 token，查不到別人的照片。
    return db.photo.findFirst({
      // 條件：照片的擁有者是這位使用者，而且沒有被刪除（刪除照片是軟刪除，記錄還留著）。
      where: { userId: id, deletedAt: null },
      // 排序：主照片排最前面（布林值由大到小，true 在前），再依使用者排的順序，最後依上傳時間。
      orderBy: [
        { isAvatar: "desc" },
        { displayOrder: "asc" },
        { createdAt: "asc" },
      ],
    });
  }
  /**
   * GET /verification/status：回傳使用者目前的真人驗證狀態，以及現在能不能開始驗證。
   *
   * 回傳欄位：
   * - status、reasonCode、modelName、modelVersion、createdAt：最近一筆驗證紀錄的內容；
   *   從沒驗證過時 status 是 not_started。
   * - canVerify：已經上傳第一張照片（大頭貼）時是 true。前端依它決定要不要顯示真人驗證區塊，
   *   因為真人驗證就是拿自拍和這張大頭貼比對，沒有大頭貼就無從驗證。
   * - 從沒驗證過、也還沒上傳照片時，reasonCode 是 AVATAR_REQUIRED，告訴前端缺的是大頭貼。
   *
   * - 最近一筆是 verified，但之後刪掉或換掉了大頭貼（使用者的 isVerified 已被取消）時，
   *   回 not_started 與 AVATAR_CHANGED：通過驗證的是舊照片，新的大頭貼要重新驗證。
   *
   * 刻意不回傳活體分數與比對分數：分數只存在資料庫，避免被拿來反覆試探驗證門檻。
   */
  async verification(id: string) {
    // 三個查詢互不相依，同時送出以節省等待時間。
    const [v, avatar, user] = await Promise.all([
      // 查詢一：最近一筆驗證紀錄（依建立時間由新到舊排序，取第一筆）。
      this.db.verification.findFirst({
        // 條件：紀錄屬於這位使用者。
        where: { userId: id },
        // 排序：最新的在最前面。
        orderBy: { createdAt: "desc" },
      }),
      // 查詢二：第一張主照片；沒有照片時是 null。
      this.avatarPhoto(id),
      // 查詢三：使用者目前的「已驗證」標記。
      this.db.user.findUnique({ where: { id }, select: { isVerified: true } }),
    ]);
    // 有大頭貼才能做真人驗證。
    const canVerify = avatar !== null;
    // 通過驗證之後換過大頭貼：驗證標記已經取消，不能再顯示成 verified。
    if (v?.status === "verified" && !user?.isVerified)
      return { status: "not_started", reasonCode: "AVATAR_CHANGED", canVerify };
    // 有驗證紀錄：回傳紀錄的內容，加上現在能不能驗證。
    if (v)
      return {
        // 紀錄編號。
        id: v.id,
        // 驗證狀態：pending、verified、rejected、unavailable 之一。
        status: v.status,
        // 原因代碼，例如 LIVE_CAPTURE_REQUIRED、FACE_MISMATCH。
        reasonCode: v.reasonCode,
        // 做判斷的模型名稱；沒有接模型時是 null。
        modelName: v.modelName,
        // 模型與判定政策的版本；沒有接模型時是 null。
        modelVersion: v.modelVersion,
        // 這次驗證的時間。
        createdAt: v.createdAt,
        // 現在能不能再驗證一次（例如使用者後來把照片全刪了，就不能）。
        canVerify,
      };
    // 沒有任何驗證紀錄、但已經有大頭貼：可以開始驗證。
    if (canVerify) return { status: "not_started", canVerify };
    // 沒有任何驗證紀錄、也還沒有大頭貼：請使用者先上傳照片。
    return { status: "not_started", reasonCode: "AVATAR_REQUIRED", canVerify };
  }
  /**
   * 把大頭貼轉成要送給 AI 服務的「身分參照」影像。
   *
   * 1. 從 MinIO（物件儲存）讀出照片原檔。
   * 2. 用 sharp 依 EXIF 轉正、等比例縮到長邊 800px 內、轉成 JPEG（品質 85）。
   *    AI 服務對參照照片的上限是 1 MiB，800px 的 JPEG 遠小於這個上限。
   * 3. 縮完短邊不到 64px（例如很寬的橫幅照）就丟出 REFERENCE_PHOTO_TOO_SMALL。
   *    AI 服務會拒收短邊小於 64px 的影像；在這裡先擋，驗證紀錄上的原因代碼才看得出問題在哪。
   *
   * 讀檔失敗（檔案不見、MinIO 故障）或圖片壞掉時，錯誤直接往外丟，
   * 由 verifySelfie 記成 REFERENCE_PHOTO_UNAVAILABLE。
   */
  async referenceImage(photo: { storageKey: string }): Promise<ReferenceImage> {
    // 向 MinIO 取得照片的讀取串流；檔案會一段一段傳回來。
    const stream = await this.infra.storage.getObject(
      // 照片所在的 bucket（儲存桶），由環境變數 S3_BUCKET 設定。
      config.S3_BUCKET,
      // 照片在 bucket 裡的路徑，上傳時存在資料庫的 storageKey。
      photo.storageKey,
    );
    // 用來收集每一段讀到的資料。
    const chunks: Buffer[] = [];
    // 逐段讀完整個串流，把每一段放進陣列。
    for await (const chunk of stream) chunks.push(chunk as Buffer);
    // 把所有片段接成完整的檔案，交給 sharp 處理，拿回處理後的內容（data）與尺寸（info）。
    const { data, info } = await sharp(Buffer.concat(chunks), {
      // 最多 2400 萬像素，和上傳照片時的限制相同，防止超大圖片耗盡記憶體。
      limitInputPixels: 24_000_000,
    })
      // 依 EXIF 的方向資訊轉正（手機直拍的照片常常存成橫的）。
      .rotate()
      // 等比例縮小到長、寬都不超過 800px；本來就比較小的照片不放大。
      .resize({
        // 寬度上限 800px。
        width: 800,
        // 高度上限 800px。
        height: 800,
        // inside：整張照片塞進 800x800 的框內，保持原比例、不裁切。
        fit: "inside",
        // 小於 800px 的照片維持原尺寸，放大只會讓畫面變糊。
        withoutEnlargement: true,
      })
      // 統一轉成 JPEG、品質 85：檔案小，細節也足夠做人臉比對。
      .jpeg({ quality: 85 })
      // resolveWithObject：同時回傳檔案內容與輸出尺寸，下一步要檢查尺寸。
      .toBuffer({ resolveWithObject: true });
    // 短邊不到 64px 時 AI 服務會拒收，先在這裡丟出明確的原因代碼。
    if (Math.min(info.width, info.height) < 64)
      throw new Error("REFERENCE_PHOTO_TOO_SMALL");
    // 轉成 base64 文字才能放進 JSON 請求；格式固定是 JPEG。
    return { imageBase64: data.toString("base64"), mimeType: "image/jpeg" };
  }
  /**
   * POST /onboarding/selfie：上傳一張自拍檔，和自己的第一張大頭貼比對（舊流程，前端已改用即時鏡頭）。
   *
   * 只有一張上傳的檔案，無法證明是活人當下拍攝；自架 provider 全部通過也只回
   * unavailable / LIVE_CAPTURE_REQUIRED。要通過驗證請走即時鏡頭（createChallenge + verifyLive）。
   *
   * 流程：
   * 1. 沒有大頭貼就回 409 AVATAR_REQUIRED；這一步放在次數限制之前，還沒上傳照片的人不會白白用掉次數。
   * 2. 每位使用者每小時最多驗證 5 次（和即時鏡頭共用），超過回 429。
   * 3. 檢查並整理自拍：格式、大小、轉正、縮到 1600px、轉成 JPEG（見 image）。
   * 4. 交給 runVerification 呼叫 AI 服務並記錄結果。
   */
  async verifySelfie(id: string, file: Express.Multer.File | undefined) {
    // 步驟 1：找出第一張主照片（大頭貼）。
    const avatar = await this.avatarPhoto(id);
    // 沒有大頭貼：回 409，請使用者先上傳照片。不建立紀錄、不呼叫 AI 服務。
    if (!avatar)
      return fail(
        409,
        "AVATAR_REQUIRED",
        "請先上傳至少一張大頭貼，再進行真人驗證。",
      );
    // 步驟 2：次數限制，同一位使用者每 3600 秒（1 小時）最多 5 次，超過會丟出 429。
    await this.infra.limit(`verify:${id}`, VERIFY_PER_HOUR, 3600);
    // 步驟 3：檢查並整理自拍；不合格會直接丟出 400 與中文說明。
    const buffer = await this.image(file);
    // 步驟 4：只送自拍（沒有 liveCapture），交給共用流程呼叫 AI 服務並記錄結果。
    return this.runVerification(id, avatar, {
      // 自拍轉成 base64 文字。
      imageBase64: buffer.toString("base64"),
      // 自拍已整理成 JPEG。
      mimeType: "image/jpeg",
    });
  }
  /**
   * POST /verification/challenge：開始即時鏡頭驗證前，向伺服器領一份「動作挑戰」。
   *
   * - 從「向左轉、向右轉、抬頭、低頭」隨機挑 2 個不同的動作，順序也是隨機的。
   *   動作由伺服器決定、每次都不一樣，對著鏡頭播放事先錄好的影片無法剛好照著做。
   *   限制：伺服器無法證明影格是鏡頭當下拍的；繞過前端直接呼叫 API、送出事先準備好的影格（注入攻擊）擋不住。
   * - 挑戰存在 Redis，120 秒後自動失效，而且只能用一次（verifyLive 讀出後立刻刪除）。
   * - 沒有大頭貼回 409 AVATAR_REQUIRED；每位使用者每小時最多領 10 次，超過回 429。
   *   這個小時的驗證次數（5 次）已經用完時也先回 429，不讓使用者做完整段鏡頭流程才失敗。
   *
   * 回傳 { challengeId, actions, expiresInSeconds }，前端依 actions 的順序引導使用者做動作。
   */
  async createChallenge(id: string) {
    // 先確認有大頭貼：沒有就無從比對，不必開始挑戰。
    const avatar = await this.avatarPhoto(id);
    // 沒有大頭貼：回 409，請使用者先上傳照片。
    if (!avatar)
      return fail(
        409,
        "AVATAR_REQUIRED",
        "請先上傳至少一張大頭貼，再進行真人驗證。",
      );
    // 驗證次數（每小時 5 次）已經用完：現在就回 429，不要讓使用者做完整段鏡頭流程才在送出時失敗。
    // 這裡只讀取計數、不增加，真正扣次數的是 verifyLive。
    if (
      Number(await this.infra.redis.get(`rate:verify:${id}`)) >= VERIFY_PER_HOUR
    )
      return fail(429, "RATE_LIMIT", "操作太頻繁，請稍後再試。");
    // 次數限制：同一位使用者每 3600 秒（1 小時）最多 10 次，超過會丟出 429。
    await this.infra.limit(`challenge:${id}`, 10, 3600);
    // 可以抽的動作（複製一份，抽過的就移除，才不會抽到重複的）。
    const pool = [...challengeActions];
    // 這次挑戰要做的動作，依抽出的順序排列。
    const actions: ChallengeAction[] = [];
    // 抽到指定的數量為止。
    while (actions.length < CHALLENGE_ACTION_COUNT)
      // randomInt 用作業系統的安全亂數，挑一個位置取出，並從剩下的動作中移除。
      actions.push(pool.splice(randomInt(pool.length), 1)[0]);
    // 挑戰編號：隨機的 UUID，別人猜不到。
    const challengeId = randomUUID();
    // 存進 Redis：記下是誰的挑戰、要做哪些動作；EX 設定幾秒後自動刪除。
    await this.infra.redis.set(
      // key 帶上挑戰編號，一份挑戰一個 key。
      challengeKey(challengeId),
      // 內容存成 JSON 文字。
      JSON.stringify({ userId: id, actions }),
      // EX：以秒為單位的有效期限。
      "EX",
      // 有效秒數。
      CHALLENGE_TTL_SECONDS,
    );
    // 回傳給前端：挑戰編號、動作清單、還有幾秒可以完成。
    return {
      challengeId,
      actions,
      expiresInSeconds: CHALLENGE_TTL_SECONDS,
    };
  }
  /**
   * 整理一張即時鏡頭拍下的影格：先用 image 做完整檢查與轉正，再縮到 1024px 內。
   * AI 服務對每張動作影格的上限是 1 MiB，超過就回 400，請使用者重拍。
   */
  async frame(file: Express.Multer.File) {
    // 先照一般照片的規則檢查（格式、大小、像素）並轉正、縮到 1600px、轉成 JPEG。
    const normalized = await this.image(file);
    // 再縮到長、寬都不超過 1024px（不放大），轉成 JPEG 品質 88：人臉偵測與比對都夠用，也讓請求變小。
    const resized = await sharp(normalized)
      // 等比例縮小，整張塞進 1024x1024 的框內。
      .resize({
        width: 1024,
        height: 1024,
        fit: "inside",
        withoutEnlargement: true,
      })
      // 轉成 JPEG、品質 88。
      .jpeg({ quality: 88 })
      // 取得檔案內容。
      .toBuffer();
    // 超過 AI 服務的單張上限（1 MiB）就請使用者重拍。
    if (resized.length > MAX_FRAME_BYTES)
      return fail(400, "INVALID_IMAGE", "影格檔案太大，請重新開始驗證。");
    // 回傳整理好的 JPEG。
    return resized;
  }
  /**
   * POST /onboarding/live：送出即時鏡頭拍下的影格，完成動作挑戰並和第一張大頭貼比對。
   *
   * 請求是 multipart 表單：
   * - challengeId：createChallenge 發的挑戰編號。
   * - frames：依序是「正面影格」＋每個動作各一張（順序和挑戰的 actions 相同）。
   *
   * 流程：
   * 1. 沒有大頭貼回 409 AVATAR_REQUIRED（在次數限制之前）。
   * 2. 每位使用者每小時最多驗證 5 次（和上傳自拍共用），超過回 429。
   * 3. 從 Redis 讀出並立刻刪除挑戰（只能用一次）；不存在、過期或不是自己的，回 409 CHALLENGE_EXPIRED。
   * 4. 影格數量必須等於「動作數 + 1」，否則回 400 INVALID_FRAMES。
   * 5. 逐張整理影格（見 frame）。
   * 6. 交給 runVerification：正面影格當自拍，動作影格放進 liveCapture，由 provider 重新檢查每個動作。
   */
  async verifyLive(
    id: string,
    files: Express.Multer.File[] | undefined,
    body: unknown,
  ) {
    // 步驟 1：找出第一張主照片（大頭貼）。
    const avatar = await this.avatarPhoto(id);
    // 沒有大頭貼：回 409，請使用者先上傳照片。
    if (!avatar)
      return fail(
        409,
        "AVATAR_REQUIRED",
        "請先上傳至少一張大頭貼，再進行真人驗證。",
      );
    // 步驟 2：次數限制，和上傳自拍共用同一個計數（每小時 5 次）。
    await this.infra.limit(`verify:${id}`, VERIFY_PER_HOUR, 3600);
    // 步驟 3：從表單取出挑戰編號，必須是 UUID 格式。
    const parsed = z
      // 表單欄位 challengeId 是 UUID 字串。
      .object({ challengeId: z.string().uuid() })
      // safeParse 不會丟錯，格式不對時 success 是 false。
      .safeParse(body);
    // 沒有挑戰編號或格式不對：請使用者重新開始。
    if (!parsed.success)
      return fail(400, "CHALLENGE_REQUIRED", "缺少驗證挑戰，請重新開始驗證。");
    // 挑戰編號。
    const { challengeId } = parsed.data;
    // GETDEL：讀出挑戰的同時就刪除，同一份挑戰不能送第二次（防止重送同一組影格）。
    const stored = await this.infra.redis.getdel(challengeKey(challengeId));
    // 把存的 JSON 解析回來；格式不對時當作沒有挑戰。
    const challenge = stored ? parseChallenge(stored) : null;
    // 挑戰不存在（過期或用過）或不是這位使用者的：回 409，請重新開始。
    if (!challenge || challenge.userId !== id)
      return fail(
        409,
        "CHALLENGE_EXPIRED",
        "驗證已逾時或已使用，請重新開始驗證。",
      );
    // 步驟 4：影格數量必須是「正面 1 張 + 每個動作 1 張」。
    if (!files || files.length !== challenge.actions.length + 1)
      return fail(400, "INVALID_FRAMES", "影格數量不正確，請重新開始驗證。");
    // 步驟 5：逐張整理影格；第一張是正面，其餘依序對應每個動作。
    const [neutral, ...actionFrames] = await Promise.all(
      // 每張都經過 frame 的檢查與縮圖。
      files.map((file) => this.frame(file)),
    );
    // 步驟 6：交給共用流程呼叫 AI 服務並記錄結果。
    return this.runVerification(id, avatar, {
      // 正面影格當作「自拍」，用來和大頭貼比對，也當作動作角度的基準。
      imageBase64: neutral.toString("base64"),
      // 影格已整理成 JPEG。
      mimeType: "image/jpeg",
      // 即時鏡頭的動作挑戰。
      liveCapture: {
        // 挑戰編號，方便對照。
        challengeId,
        // 每張動作影格配上它對應的動作。
        frames: actionFrames.map((frame, index) => ({
          // 第 index 個動作。
          action: challenge.actions[index],
          // 影格轉成 base64。
          imageBase64: frame.toString("base64"),
          // 影格已整理成 JPEG。
          mimeType: "image/jpeg" as const,
        })),
      },
    });
  }
  /**
   * 上傳自拍與即時鏡頭共用的後半段：建立紀錄 → 準備大頭貼參照 → 呼叫 AI 服務 → 寫入結果。
   *
   * 1. 在資料庫建立一筆 pending（處理中）的驗證紀錄；它的 id 同時當作這次請求的編號送給 AI 服務。
   * 2. 把大頭貼轉成參照影像；失敗就記 unavailable，不呼叫 AI 服務。
   * 3. 把影像送到 AI 服務（15 秒逾時），並嚴格檢查回應格式。
   * 4. 把結果寫回驗證紀錄。只有 verified、rejected 會改使用者的 isVerified；
   *    unavailable（沒接模型、逾時、服務故障、參照照片有問題、影像被 AI 服務拒收）只記錄，不改變既有的驗證狀態。
   *    寫入前在同一把使用者鎖（和刪照片相同）裡重新確認大頭貼沒換過；換過就改記 unavailable／AVATAR_CHANGED，
   *    避免把舊照片的結果套到新照片上。
   * 5. 回傳最新的驗證狀態（同 GET /verification/status）。
   */
  async runVerification(
    id: string,
    avatar: { id: string; storageKey: string },
    capture: VerificationCapture,
  ) {
    // 步驟 1：先建立一筆驗證紀錄，status 預設是 pending（處理中）。
    const record = await this.db.verification.create({
      // 紀錄屬於這位使用者；影像一律整理成 JPEG，所以格式固定。影像本身不存檔。
      data: { userId: id, mimeType: "image/jpeg" },
    });
    // AI 服務回應的格式規則：不符合就當作沒有收到有效回應。
    const responseSchema = z.object({
      // 判定結果只能是這三種。
      status: z.enum(["verified", "rejected", "unavailable"]),
      // 原因代碼，例如 FACE_MISMATCH。
      reasonCode: z.string(),
      // 模型名稱；沒有接模型時是 null。
      modelName: z.string().nullable(),
      // 模型版本；沒有接模型時是 null。
      modelVersion: z.string().nullable(),
      // 活體（防偽）分數，0 到 1；可能沒有。
      livenessScore: z.number().min(0).max(1).nullable().optional(),
      // 人臉比對分數，0 到 1；可能沒有。
      faceMatchScore: z.number().min(0).max(1).nullable().optional(),
    });
    // 預設結果：AI 服務連不上。後面任何一步失敗，就會停在這個保守的結果。
    let result: z.infer<typeof responseSchema> = {
      // unavailable：無法判斷，不影響使用者原本的驗證狀態。
      status: "unavailable",
      // 預設原因：AI 服務連不上。
      reasonCode: "AI_SERVICE_UNAVAILABLE",
      // 沒有模型做判斷。
      modelName: null,
      // 沒有模型版本。
      modelVersion: null,
    };
    // 步驟 2：準備參照影像；null 代表準備失敗，就不呼叫 AI 服務。
    let reference: ReferenceImage | null = null;
    try {
      // 把大頭貼讀出來、縮圖、轉成 base64。
      reference = await this.referenceImage(avatar);
    } catch (error) {
      // 照片太小時，referenceImage 丟出的錯誤訊息就是 REFERENCE_PHOTO_TOO_SMALL。
      const tooSmall =
        error instanceof Error && error.message === "REFERENCE_PHOTO_TOO_SMALL";
      // 記成 unavailable：問題出在參照照片，不是使用者本人，所以不改驗證狀態。
      result = {
        // 其他欄位沿用預設值（unavailable、沒有模型）。
        ...result,
        // 太小記 REFERENCE_PHOTO_TOO_SMALL；其他狀況（讀不到、檔案壞掉）記 REFERENCE_PHOTO_UNAVAILABLE。
        reasonCode: tooSmall
          ? "REFERENCE_PHOTO_TOO_SMALL"
          : "REFERENCE_PHOTO_UNAVAILABLE",
      };
    }
    // 步驟 3：參照影像準備好了，才送去 AI 服務。
    if (reference) {
      try {
        // 呼叫 AI 服務的內部 API（只在容器網路內開放）。
        const res = await fetch(
          // AI 服務的網址由環境變數 AI_INTERNAL_URL 設定。
          `${config.AI_INTERNAL_URL}/internal/ai/face/verify`,
          {
            // 用 POST 送出影像。
            method: "POST",
            headers: {
              // 請求內容是 JSON。
              "Content-Type": "application/json",
              // 內部服務之間的通行碼；AI 服務會先驗證它，才讀取內容。
              "X-Internal-Token": config.AI_INTERNAL_TOKEN,
            },
            // 請求內容：影像、請求編號、參照照片，有即時鏡頭時再加上動作影格。
            body: JSON.stringify({
              // 自拍（或即時鏡頭的正面影格）與它的格式，以及有的話 liveCapture。
              ...capture,
              // 用驗證紀錄的 id 當請求編號，方便對照。
              requestId: record.id,
              // 參照照片：只放第一張主照片這一張。
              referenceImages: [reference],
            }),
            // 15 秒還沒回應就放棄，維持預設的 unavailable；不讓使用者一直等。
            signal: AbortSignal.timeout(15000),
          },
        );
        // 400：AI 服務拒收影像（格式、大小不符），回應的 code 會標出是哪一張（REFERENCE_、FRAME_ 前綴）。
        // 記成 unavailable 加上這個代碼，前端才能請使用者重拍，而不是說服務故障。
        if (res.status === 400) {
          // 讀出錯誤碼；讀不到或格式不對就維持預設的 AI_SERVICE_UNAVAILABLE。
          const code = z
            // AI 服務的錯誤回應是 {code, message}。
            .object({
              // 只接受大寫英文、數字、底線組成的代碼，不把任意文字寫進資料庫。
              code: z.string().regex(/^[A-Z][A-Z0-9_]{0,79}$/),
            })
            // safeParse 不會丟錯。
            .safeParse(await res.json());
          // 格式正確才採用。
          if (code.success)
            // 其他欄位沿用預設值（unavailable、沒有模型）。
            result = { ...result, reasonCode: code.data.code };
        }
        // 只有 HTTP 2xx 才讀取結果；其他狀態碼維持預設的 unavailable。
        if (res.ok) {
          // 用上面的格式規則檢查回應；safeParse 失敗時不會丟錯，只回傳 success=false。
          const parsed = responseSchema.safeParse(await res.json());
          // 格式正確才採用 AI 服務的結果。
          if (parsed.success) {
            // 採用 AI 服務回傳的判定。
            result = parsed.data;
            // 再多一層保護：宣稱 verified，就一定要有模型名稱、版本與兩個分數，缺任何一項都不採信。
            if (
              // 判定是「通過」。
              result.status === "verified" &&
              // 但缺少模型名稱、模型版本、活體分數或比對分數其中之一。
              (!result.modelName ||
                !result.modelVersion ||
                result.livenessScore == null ||
                result.faceMatchScore == null)
            )
              // 證據不完整的「通過」改記成 unavailable，絕不因此把使用者標成已驗證。
              result = {
                // 無法判斷。
                status: "unavailable",
                // 原因：AI 服務的回應不完整。
                reasonCode: "INVALID_PROVIDER_RESPONSE",
                // 不採信的結果不記模型名稱。
                modelName: null,
                // 也不記模型版本。
                modelVersion: null,
              };
          }
        }
      } catch {
        // 連線失敗、逾時或回應不是 JSON：維持預設的 unavailable / AI_SERVICE_UNAVAILABLE。
      }
    }
    // 步驟 4：用交易（transaction）一起寫入：紀錄與使用者狀態要嘛都更新成功，要嘛都不更新。
    await this.db.$transaction(async (tx) => {
      // 和刪除照片用同一把使用者鎖：寫入期間大頭貼不會被換掉。
      await tx.$executeRaw`SELECT pg_advisory_xact_lock(hashtext(${id}))`;
      // 重新查目前的大頭貼。
      const current = await this.avatarPhoto(id, tx);
      // 呼叫 AI 服務期間大頭貼被刪掉或換掉：結果是針對舊照片的，不能拿來改驗證狀態。
      if (result.status !== "unavailable" && current?.id !== avatar.id)
        result = {
          // 無法判斷。
          status: "unavailable",
          // 原因：比對途中換了大頭貼。
          reasonCode: "AVATAR_CHANGED",
          // 不採信的結果不記模型名稱。
          modelName: null,
          // 也不記模型版本。
          modelVersion: null,
        };
      // 把結果寫回剛剛建立的驗證紀錄（包含分數，只存在資料庫）。
      await tx.verification.update({ where: { id: record.id }, data: result });
      // 只有「通過」或「未通過」才算有明確結論，才更新使用者的「已驗證」標記。
      if (result.status !== "unavailable")
        await tx.user.update({
          // 更新這位使用者。
          where: { id },
          // verified 設為已驗證；rejected 取消已驗證。
          data: { isVerified: result.status === "verified" },
        });
    });
    // 步驟 5：回傳最新的驗證狀態給前端（不含分數）。
    return this.verification(id);
  }
  /**
   * POST /verification/retry：使用者按「重新驗證」時呼叫。
   *
   * 不寫資料庫，只回傳「接下來要做什麼」：真正的驗證走即時鏡頭（POST /verification/challenge → POST /onboarding/live）。
   * 判斷方式和 GET /verification/status 一致：還沒上傳大頭貼時回 AVATAR_REQUIRED 與 canVerify=false。
   */
  async retry(id: string) {
    // 同一位使用者每 3600 秒（1 小時）最多 10 次，超過會丟出 429。
    await this.infra.limit(`retry:${id}`, 10, 3600);
    // 查第一張主照片，判斷現在能不能驗證。
    const avatar = await this.avatarPhoto(id);
    // 沒有大頭貼：請使用者先上傳照片。
    if (!avatar)
      return {
        // 狀態：尚未開始。
        status: "not_started",
        // 原因：缺大頭貼。
        reasonCode: "AVATAR_REQUIRED",
        // 現在還不能驗證。
        canVerify: false,
      };
    // 有大頭貼：請使用者用鏡頭重新驗證。
    return {
      // 狀態：尚未開始。
      status: "not_started",
      // 原因：需要即時鏡頭驗證（上傳自拍無法通過）。
      reasonCode: "LIVE_CAPTURE_REQUIRED",
      // 可以開始驗證。
      canVerify: true,
    };
  }
}
