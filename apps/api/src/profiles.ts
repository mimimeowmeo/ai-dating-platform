import { Injectable } from "@nestjs/common";
import { randomUUID } from "node:crypto";
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
    bio: z.string().trim().max(1000).default(""),
    city: z.string().trim().min(1).max(80),
    latitude: z.number().min(-90).max(90),
    longitude: z.number().min(-180).max(180),
    datingIntent: intent,
    heightCm: z.number().int().min(100).max(250).nullable().optional(),
    occupation: z.string().max(80).nullable().optional(),
    education: z.string().max(80).nullable().optional(),
    interests: tags("interests").optional(),
    hobbies: tags("hobbies").optional(),
    foods: tags("foods").optional(),
  })
  .strict();
const preferencesInput = z
  .object({
    minAge: z.number().int().min(18).max(99),
    maxAge: z.number().int().min(18).max(99),
    preferredGender: z.enum(["woman", "man", "nonbinary", "any"]),
    maxDistanceKm: z.number().int().min(1).max(20000),
    preferredDatingIntent: z.enum(["serious", "casual", "friendship", "any"]),
  })
  .strict()
  .refine((v) => v.minAge <= v.maxAge, "年齡下限不得大於上限");
export const photoView = (p: any) => ({
  id: p.id,
  url: `/api/v1/media/${p.id}`,
  isAvatar: p.isAvatar,
  displayOrder: p.displayOrder,
});
export function card(user: any) {
  if (!user?.profile) return null;
  const p = user.profile;
  return {
    userId: user.id,
    displayName: p.displayName,
    age: ageAt(p.birthDate),
    gender: p.gender,
    bio: p.bio,
    city: p.city,
    datingIntent: p.datingIntent,
    interests: p.interests,
    hobbies: p.hobbies,
    foods: p.foods,
    photos: (user.photos || []).map(photoView),
    isVerified: user.isVerified,
  };
}
export const userInclude = {
  profile: true,
  preference: true,
  photos: {
    orderBy: [{ displayOrder: "asc" as const }, { createdAt: "asc" as const }],
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
          photos: user.photos.map(photoView),
        }
      : null;
  }
  async save(id: string, body: unknown) {
    const dto = parse(profileInput, body);
    const data = { ...dto, birthDate: new Date(dto.birthDate) };
    await this.db.profile.upsert({
      where: { userId: id },
      create: { userId: id, ...data },
      update: data,
    });
    return this.mine(id);
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
    return card(u);
  }
  async preferences(id: string) {
    return this.db.preference.findUniqueOrThrow({ where: { userId: id } });
  }
  async savePreferences(id: string, body: unknown) {
    const data = parse(preferencesInput, body);
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
        if (photos.length >= 6)
          return fail(400, "PHOTO_LIMIT", "最多可上傳 6 張照片。");
        return tx.photo.create({
          data: {
            userId: id,
            storageKey: key,
            mimeType: "image/jpeg",
            isAvatar: photos.length === 0,
            displayOrder:
              Math.max(-1, ...photos.map((p) => p.displayOrder)) + 1,
          },
        });
      });
      return photoView(photo);
    } catch (error) {
      await this.infra.storage.removeObject(config.S3_BUCKET, key);
      throw error;
    }
  }
  async removePhoto(id: string, photoId: string) {
    uuid(photoId);
    const photo = await this.db.photo.findFirst({
      where: { id: photoId, userId: id },
    });
    if (!photo) return fail(404, "NOT_FOUND", "找不到照片。");
    await this.db.$transaction(async (tx) => {
      await tx.$executeRaw`SELECT pg_advisory_xact_lock(hashtext(${id}))`;
      await tx.photo.delete({ where: { id: photoId } });
      if (photo.isAvatar) {
        const first = await tx.photo.findFirst({
          where: { userId: id },
          orderBy: [{ displayOrder: "asc" }, { createdAt: "asc" }],
        });
        if (first)
          await tx.photo.update({
            where: { id: first.id },
            data: { isAvatar: true },
          });
      }
    });
    await this.infra.storage.removeObject(config.S3_BUCKET, photo.storageKey);
    return { ok: true };
  }
  async media(photoId: string) {
    uuid(photoId);
    const photo = await this.db.photo.findUnique({ where: { id: photoId } });
    if (!photo) return fail(404, "NOT_FOUND", "找不到照片。");
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
