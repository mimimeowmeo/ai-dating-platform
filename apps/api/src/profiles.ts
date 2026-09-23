import { Injectable } from "@nestjs/common";
import { createHmac, randomUUID, timingSafeEqual } from "node:crypto";
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
  async verification(id: string) {
    const v = await this.db.verification.findFirst({
      where: { userId: id },
      orderBy: { createdAt: "desc" },
    });
    return v
      ? {
          id: v.id,
          status: v.status,
          reasonCode: v.reasonCode,
          modelName: v.modelName,
          modelVersion: v.modelVersion,
          createdAt: v.createdAt,
        }
      : { status: "not_started" };
  }
  async verifySelfie(id: string, file: Express.Multer.File | undefined) {
    await this.infra.limit(`verify:${id}`, 5, 3600);
    const buffer = await this.image(file);
    const record = await this.db.verification.create({
      data: { userId: id, mimeType: "image/jpeg" },
    });
    const responseSchema = z.object({
      status: z.enum(["verified", "rejected", "unavailable"]),
      reasonCode: z.string(),
      modelName: z.string().nullable(),
      modelVersion: z.string().nullable(),
      livenessScore: z.number().min(0).max(1).nullable().optional(),
      faceMatchScore: z.number().min(0).max(1).nullable().optional(),
    });
    let result: z.infer<typeof responseSchema> = {
      status: "unavailable",
      reasonCode: "AI_SERVICE_UNAVAILABLE",
      modelName: null,
      modelVersion: null,
    };
    try {
      const res = await fetch(
        `${config.AI_INTERNAL_URL}/internal/ai/face/verify`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Internal-Token": config.AI_INTERNAL_TOKEN,
          },
          body: JSON.stringify({
            imageBase64: buffer.toString("base64"),
            mimeType: "image/jpeg",
            requestId: record.id,
          }),
          signal: AbortSignal.timeout(15000),
        },
      );
      if (res.ok) {
        const parsed = responseSchema.safeParse(await res.json());
        if (parsed.success) {
          result = parsed.data;
          if (
            result.status === "verified" &&
            (!result.modelName ||
              !result.modelVersion ||
              result.livenessScore == null ||
              result.faceMatchScore == null)
          )
            result = {
              status: "unavailable",
              reasonCode: "INVALID_PROVIDER_RESPONSE",
              modelName: null,
              modelVersion: null,
            };
        }
      }
    } catch {}
    const decided = result.status !== "unavailable";
    await this.db.$transaction([
      this.db.verification.update({ where: { id: record.id }, data: result }),
      ...(decided
        ? [
            this.db.user.update({
              where: { id },
              data: { isVerified: result.status === "verified" },
            }),
          ]
        : []),
    ]);
    return this.verification(id);
  }
  async retry(id: string) {
    await this.infra.limit(`retry:${id}`, 10, 3600);
    return { status: "not_started", reasonCode: "SELFIE_REQUIRED" };
  }
}
